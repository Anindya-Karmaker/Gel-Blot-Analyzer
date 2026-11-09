"""Gel Blot Analyzer — AI control bridge (MCP).

Lets any MCP-capable AI client (Claude Desktop / Claude Code, LM Studio, Cursor, Ollama-based
clients, ...) operate the running app, LIVE, in the user's window.

How it fits together
--------------------
* A small HTTP server on 127.0.0.1 (token-protected) speaks MCP "Streamable HTTP" at /mcp
  (JSON responses, no SSE). Clients that support HTTP MCP can connect to it directly.
* gba_mcp_server.py is a zero-dependency stdio MCP server for clients that launch a command;
  it relays to this endpoint (and can start the app if it isn't running).
* The HTTP thread NEVER touches Qt. Every tool runs on the GUI thread through a queued signal
  (_Dispatcher). Actions that open a modal dialog (exec()) keep running inside the dialog's
  nested event loop; the worker notices the new dialog and returns status="dialog_open" so the
  AI can answer it — the same way a person would.
* Live preview: a purple AI cursor moves/ripples over the viewer for clicks and drags, every
  control the AI touches is revealed (tab switched, scrolled into view) and outlined before it
  changes, and the AI Control panel logs every call. The user can Pause or switch to View-only
  at any time; a configurable step delay slows the AI down so it can be followed.

Rules this module follows (see the project memory notes):
* No QLayout.itemAt() walks and no Python eventFilter (stale-wrapper segfault in PySide 6.10).
* paintEvent overrides never let an exception escape.
* Only one AI action runs at a time.
"""

import base64
import json
import os
import re
import secrets
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from PySide6.QtCore import (QBuffer, QByteArray, QEvent, QIODevice, QObject, QPointF, QRectF,
                            Qt, QTimer, Signal)
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QGuiApplication, QImage, QKeyEvent,
                           QKeySequence, QMouseEvent, QPainter, QPen, QAction, QShortcut)
from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider,
                               QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPlainTextEdit, QPushButton, QRadioButton, QScrollArea,
                               QScrollBar, QSpinBox, QStackedWidget, QTabBar, QTableView,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget,
                               QFileDialog, QMenu)

try:
    import shiboken6
    _is_valid = shiboken6.isValid
    _cpp_ptr = lambda o: shiboken6.getCppPointer(o)[0]
except Exception:  # pragma: no cover
    _is_valid = lambda o: o is not None
    _cpp_ptr = id

import gba_mcp_schema as S

AI_COLOR = QColor("#7C3AED")          # violet: everything the AI draws on screen
OVERLAY_OBJECT_NAME = "GBA_AIOverlay"
PANEL_OBJECT_NAME = "GBA_AIControlPanel"


class ToolError(Exception):
    """A user-facing tool failure (returned to the AI as isError, never a crash)."""


# =============================================================================================
# Settings (port, token, preferences) — also read by gba_mcp_server.py
# =============================================================================================
_DEFAULTS = {
    "port": S.DEFAULT_PORT,
    "token": "",
    "enabled": False,              # OFF until the user turns it on in Settings
    "step_delay_ms": 250,
    "show_overlays": True,
    "permission": "full",          # "full" | "view"
}


def load_settings():
    data = dict(_DEFAULTS)
    try:
        with open(S.settings_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data.update({k: raw[k] for k in raw if k in _DEFAULTS or k in ("active_port", "pid")})
            data["enabled"] = bool(raw.get("enabled", False))
    except Exception:
        pass
    if not data.get("token"):
        data["token"] = secrets.token_urlsafe(24)
        save_settings(data)
    return data


def save_settings(data):
    path = S.settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)       # the token is a credential for this machine's user
        except Exception:
            pass
    except Exception:
        pass


# =============================================================================================
# GUI-thread dispatcher
# =============================================================================================
class _Job:
    __slots__ = ("fn", "is_async", "event", "result", "error", "before_ids", "started",
                 "name", "jid", "finished_at", "reported")

    _seq = 0

    def __init__(self, fn, is_async=False, name=""):
        _Job._seq += 1
        self.jid = _Job._seq
        self.fn = fn
        self.is_async = is_async
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.before_ids = set()
        self.started = False
        self.name = name
        self.finished_at = None
        self.reported = False

    def finish(self, value=None):
        if not self.event.is_set():
            self.result = value
            self.finished_at = time.time()
            self.event.set()

    def fail(self, err):
        if not self.event.is_set():
            self.error = err
            self.finished_at = time.time()
            self.event.set()


class _Dispatcher(QObject):
    submit = Signal(object)
    log = Signal(str, str)           # level, text
    activity = Signal(str)           # short "what the AI is doing" text for the status bar

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.submit.connect(self._run, Qt.QueuedConnection)

    def _run(self, job):
        b = self.bridge
        job.started = True
        try:
            job.before_ids = set(b._window_ids_now())
        except Exception:
            job.before_ids = set()
        b.ai_depth += 1
        try:
            if job.is_async:
                job.fn(job)               # the fn calls job.finish()/job.fail() later
            else:
                job.finish(job.fn())
        except ToolError as e:
            job.fail(e)
        except Exception as e:  # noqa: BLE001 - report every failure to the AI
            job.fail(ToolError("%s: %s\n%s" % (type(e).__name__, e,
                                               "".join(traceback.format_exc(limit=6)))))
        finally:
            b.ai_depth -= 1


# =============================================================================================
# On-screen overlays (live preview)
# =============================================================================================
class _HighlightFrame(QWidget):
    """Violet outline drawn over the control the AI is about to use."""

    def __init__(self, target, caption=""):
        top = target.window()
        super().__init__(top)
        self.setObjectName(OVERLAY_OBJECT_NAME)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.caption = caption
        r = target.rect()
        tl = target.mapTo(top, r.topLeft())
        pad = 4
        cap_h = 18 if caption else 0
        self.setGeometry(tl.x() - pad, tl.y() - pad - cap_h, r.width() + 2 * pad,
                         r.height() + 2 * pad + cap_h)
        self._cap_h = cap_h
        self.raise_()
        self.show()
        QTimer.singleShot(1300, self._done)

    def _done(self):
        try:
            self.hide()
            self.deleteLater()
        except Exception:
            pass

    def paintEvent(self, _ev):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(AI_COLOR, 3)
            p.setPen(pen)
            p.setBrush(QColor(124, 58, 237, 28))
            rect = QRectF(1.5, 1.5 + self._cap_h, self.width() - 3, self.height() - 3 - self._cap_h)
            p.drawRoundedRect(rect, 6, 6)
            if self.caption:
                f = QFont(self.font())
                f.setPointSizeF(max(8.0, f.pointSizeF() * 0.85))
                f.setBold(True)
                p.setFont(f)
                fm = QFontMetrics(f)
                w = min(self.width(), fm.horizontalAdvance(self.caption) + 12)
                p.setPen(Qt.NoPen)
                p.setBrush(AI_COLOR)
                p.drawRoundedRect(QRectF(0, 0, w, self._cap_h), 4, 4)
                p.setPen(Qt.white)
                p.drawText(QRectF(6, 0, w - 6, self._cap_h), Qt.AlignVCenter | Qt.AlignLeft,
                           self.caption)
            p.end()
        except Exception:
            pass


class _AiCursor(QWidget):
    """Violet pointer + click ripple drawn over the image viewer."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self.setObjectName(OVERLAY_OBJECT_NAME)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.pos_ = None
        self.pressed = False
        self.ripple_t0 = None
        self.trail = []
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade)
        self._anim = QTimer(self)
        self._anim.setInterval(30)
        self._anim.timeout.connect(self.update)
        self.hide()

    def move_to(self, pt, pressed=False, ripple=False):
        try:
            parent = self.parentWidget()
            self.setGeometry(0, 0, parent.width(), parent.height())
            self.pos_ = QPointF(pt)
            self.pressed = pressed
            if pressed:
                self.trail.append(QPointF(pt))
                self.trail = self.trail[-400:]
            else:
                self.trail = []
            if ripple:
                self.ripple_t0 = time.time()
                self._anim.start()
            self.raise_()
            self.show()
            self.update()
            self._hide_timer.start(2200)
        except Exception:
            pass

    def _fade(self):
        self._anim.stop()
        self.trail = []
        self.hide()

    def paintEvent(self, _ev):
        try:
            if self.pos_ is None:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            if len(self.trail) > 1:
                p.setPen(QPen(QColor(124, 58, 237, 120), 2, Qt.DashLine))
                for a, b in zip(self.trail, self.trail[1:]):
                    p.drawLine(a, b)
            if self.ripple_t0 is not None:
                t = time.time() - self.ripple_t0
                if t < 0.6:
                    r = 6 + 34 * t / 0.6
                    alpha = int(220 * (1 - t / 0.6))
                    p.setPen(QPen(QColor(124, 58, 237, alpha), 3))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(self.pos_, r, r)
                else:
                    self.ripple_t0 = None
                    self._anim.stop()
            x, y = self.pos_.x(), self.pos_.y()
            # arrow pointer with its tip on the point
            from PySide6.QtGui import QPolygonF
            poly = QPolygonF([QPointF(x, y), QPointF(x, y + 20), QPointF(x + 5, y + 15),
                              QPointF(x + 9, y + 24), QPointF(x + 12, y + 22),
                              QPointF(x + 8, y + 14), QPointF(x + 15, y + 14)])
            p.setPen(QPen(Qt.white, 1.5))
            p.setBrush(AI_COLOR if not self.pressed else QColor("#5B21B6"))
            p.drawPolygon(poly)
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(max(7.5, f.pointSizeF() * 0.8))
            p.setFont(f)
            p.setPen(Qt.NoPen)
            p.setBrush(AI_COLOR)
            p.drawRoundedRect(QRectF(x + 14, y + 20, 22, 14), 3, 3)
            p.setPen(Qt.white)
            p.drawText(QRectF(x + 14, y + 20, 22, 14), Qt.AlignCenter, "AI")
            p.end()
        except Exception:
            pass


# =============================================================================================
# The bridge
# =============================================================================================
_IMAGE_EXT_FILTERS = {
    ".png": "PNG Files (*.png)", ".tif": "TIFF Files (*.tif *.tiff)",
    ".tiff": "TIFF Files (*.tif *.tiff)", ".jpg": "JPEG Files (*.jpg *.jpeg)",
    ".jpeg": "JPEG Files (*.jpg *.jpeg)", ".bmp": "BMP Files (*.bmp)",
    ".pptx": "PowerPoint Files (*.pptx)",
}

_REGION_ANSWERS = [(r"hidden|visible", ["Show Markers", "Show", "Yes"])]

_INPUT_TYPES = (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox, QLineEdit,
                QTextEdit, QPlainTextEdit, QTabWidget, QListWidget, QTableView, QGroupBox)


class _ImageResult:
    def __init__(self, png, meta):
        self.png = png
        self.meta = meta


class AIBridge(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.settings = load_settings()
        self.ai_depth = 0
        self.paused = False
        self.server = None
        self.server_thread = None
        self.port = None
        self.last_client = ""
        self.file_answers = []            # queued answers for the next file chooser(s)
        self.file_dialog_requests = []    # choosers the AI triggered without an answer
        self._call_lock = threading.Lock()
        self._inflight = {}               # jid -> _Job still running on the GUI thread
        self._bg_finished = []
        self._win_seq = 0
        self._ctl_seq = 0
        self._cursor = None
        self._pin = None
        self.panel = None                  # AIActivityWindow (created on demand)
        self.settings_group = None
        self.settings_widgets = {}
        self.dispatcher = _Dispatcher(self)
        self.dispatcher.log.connect(self._on_log)
        self.dispatcher.activity.connect(self._on_activity)
        self._log_lines = []
        self._status_label = None
        self._patch_file_dialogs()

    # ----------------------------------------------------------------- server lifecycle
    def is_running(self):
        return self.server is not None

    def start(self):
        if self.server is not None:
            return True
        base = int(self.settings.get("port") or S.DEFAULT_PORT)
        last_err = None
        for port in range(base, base + S.PORT_TRIES):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(self))
                srv.daemon_threads = True
                self.server = srv
                self.port = port
                break
            except OSError as e:
                last_err = e
        if self.server is None:
            self._on_log("error", "Could not start AI control server: %s" % last_err)
            return False
        self.server_thread = threading.Thread(target=self.server.serve_forever,
                                              name="GBA-MCP-HTTP", daemon=True)
        self.server_thread.start()
        self.settings["active_port"] = self.port
        self.settings["pid"] = os.getpid()
        save_settings(self.settings)
        self._on_log("info", "AI control listening on %s" % self.endpoint_url())
        self._refresh_indicator()
        return True

    def stop(self):
        srv = self.server
        if srv is None:
            return
        self.server = None
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            pass
        self.settings.pop("active_port", None)
        self.settings.pop("pid", None)
        save_settings(self.settings)
        self._on_log("info", "AI control stopped.")
        self._refresh_indicator()

    def endpoint_url(self):
        return "http://127.0.0.1:%d/mcp" % (self.port or int(self.settings.get("port") or S.DEFAULT_PORT))

    def regenerate_token(self):
        self.settings["token"] = secrets.token_urlsafe(24)
        save_settings(self.settings)

    # ----------------------------------------------------------------- logging / indicator
    def _on_log(self, level, text):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), text)
        self._log_lines.append((level, line))
        self._log_lines = self._log_lines[-800:]
        if self.panel is not None and _is_valid(self.panel):
            self.panel.append_log(level, line)
        if level == "info":
            self._refresh_indicator()

    def _on_activity(self, text):
        try:
            if self._status_label is not None and _is_valid(self._status_label):
                self._status_label.setText("● AI: " + text if text else self._idle_text())
        except Exception:
            pass

    def _idle_text(self):
        if not self.is_running():
            return ""
        if self.paused:
            return "● AI paused"
        return "● AI ready"

    def _refresh_indicator(self):
        try:
            if self._status_label is not None and _is_valid(self._status_label):
                t = self._idle_text()
                self._status_label.setText(t)
                self._status_label.setVisible(bool(t))
            if self.panel is not None and _is_valid(self.panel):
                self.panel.refresh()
            w = self.settings_widgets
            if w and _is_valid(w.get("status")):
                w["status"].setText(self.state_html())
                for key, val in (("enable", bool(self.settings.get("enabled"))), ("pause", self.paused)):
                    cb = w.get(key)
                    if cb is not None and _is_valid(cb) and cb.isChecked() != val:
                        cb.blockSignals(True)
                        cb.setChecked(val)
                        cb.blockSignals(False)
                for key in ("perm", "delay", "overlays", "pause"):
                    if w.get(key) is not None and _is_valid(w[key]):
                        w[key].setEnabled(bool(self.settings.get("enabled")))
        except Exception:
            pass

    # ----------------------------------------------------------------- user-side controls
    def state_html(self):
        if not self.settings.get("enabled"):
            return ("<span style='color:#9CA3AF'>●</span> Off — AI apps cannot connect. Tick the box "
                    "above to allow them.")
        if not self.is_running():
            return "<span style='color:#DC2626'>●</span> Enabled, but the server could not start (see Activity Log)."
        client = (" — last client: %s" % self.last_client) if self.last_client else ""
        paused = " — <b>PAUSED</b>" if self.paused else ""
        return ("<span style='color:#15803D'>●</span> On — listening on %s%s%s"
                % (self.endpoint_url(), paused, client))

    def set_enabled(self, on):
        self.settings["enabled"] = bool(on)
        save_settings(self.settings)
        if on:
            self.start()
        else:
            self.stop()
        self._refresh_indicator()

    def set_setting(self, key, value):
        self.settings[key] = value
        save_settings(self.settings)
        self._refresh_indicator()

    def set_paused(self, v):
        v = bool(v)
        if v == self.paused:
            return
        self.paused = v
        self._on_log("info", "AI control %s by the user." % ("PAUSED" if v else "resumed"))
        self._refresh_indicator()

    def show_settings(self):
        win = self.win
        try:
            win.open_settings_tab()
        except Exception:
            tw = getattr(win, "tab_widget", None)
            if tw is not None:
                tw.setCurrentIndex(tw.count() - 1)
        g = self.settings_group
        if g is not None and _is_valid(g):
            QTimer.singleShot(80, lambda: _is_valid(g) and self.reveal(g, "AI settings"))

    def show_activity(self):
        if self.panel is None or not _is_valid(self.panel):
            self.panel = AIActivityWindow(self)
        self.panel.refresh()
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def _info(self, title, text):
        QMessageBox.information(self.win, title, text)

    def connect_client(self, name, path, found):
        if not found:
            QGuiApplication.clipboard().setText(json.dumps({"mcpServers": {S.SERVER_NAME: server_entry()}},
                                                           indent=2))
            self._info("Connect %s" % name,
                       "%s was not found on this computer.\n\nThe settings were copied to the "
                       "clipboard — paste them into its MCP configuration (mcpServers)." % name)
            return
        if _is_msix() and path.lower().startswith(os.environ.get("APPDATA", "~").lower()) \
                and not os.path.exists(path):
            QGuiApplication.clipboard().setText(json.dumps({"mcpServers": {S.SERVER_NAME: server_entry()}},
                                                           indent=2))
            self._info("Connect %s" % name,
                       "Open %s once and use its 'Edit Config' option so its settings file exists, "
                       "then click this button again.\n\n(The settings were also copied to the "
                       "clipboard.)" % name)
            return
        try:
            written = install_client_config(path)
        except ToolError as e:
            QMessageBox.warning(self.win, "Connect %s" % name, str(e))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.win, "Connect %s" % name, "Could not update %s:\n%s" % (path, e))
            return
        self._on_log("info", "Added Gel Blot Analyzer to %s (%s)." % (name, written))
        extra = "" if self.settings.get("enabled") else (
            "\n\nAI control is still OFF — tick 'Allow AI assistants to control Gel Blot Analyzer' "
            "when you want the AI to connect.")
        self._info("Connect %s" % name,
                   "Gel Blot Analyzer was added to %s.\n\nRestart %s, then ask it something like "
                   '"Use Gel Blot Analyzer to show me the open image".%s' % (name, name, extra))

    def copy_claude_code(self):
        QGuiApplication.clipboard().setText(claude_code_command())
        self._info("Claude Code", "This command was copied to the clipboard — paste it into a "
                                  "terminal:\n\n" + claude_code_command())

    def show_other_clients(self):
        dlg = QDialog(self.win)
        dlg.setObjectName(PANEL_OBJECT_NAME)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.setWindowTitle("Connect Ollama / other AI apps")
        dlg.resize(640, 520)
        lay = QVBoxLayout(dlg)
        txt = QPlainTextEdit(other_clients_text(self))
        txt.setReadOnly(True)
        f2 = QFont("Menlo" if sys.platform == "darwin" else "Consolas")
        f2.setStyleHint(QFont.Monospace)
        txt.setFont(f2)
        lay.addWidget(txt, 1)
        row = QHBoxLayout()
        cp = QPushButton("Copy All")
        cp.clicked.connect(lambda: QGuiApplication.clipboard().setText(txt.toPlainText()))
        row.addWidget(cp)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.close)
        row.addWidget(bb)
        lay.addLayout(row)
        dlg.show()

    def ask_regenerate_token(self):
        if QMessageBox.question(self.win, "New token",
                                "Generate a new token? Apps connected by URL (Claude Code over HTTP, "
                                "Open WebUI...) must be updated. One-click connections keep working."
                                ) == QMessageBox.Yes:
            self.regenerate_token()
            self._on_log("info", "New token generated.")

    # ----------------------------------------------------------------- file-dialog patch
    def _patch_file_dialogs(self):
        """Let the AI answer file choosers (which are native and otherwise unreachable).
        Only consulted when an answer is queued or when the chooser was opened by an
        AI-initiated action; the user's own clicks always get the real dialog."""
        if getattr(QFileDialog, "_gba_patched", False):
            QFileDialog._gba_bridge = self
            return
        QFileDialog._gba_bridge = self
        QFileDialog._gba_patched = True

        def make(name, orig):
            def patched(*a, **k):
                b = getattr(QFileDialog, "_gba_bridge", None)
                if b is not None and (b.file_answers or b.ai_depth > 0):
                    caption = k.get("caption", a[1] if len(a) > 1 else "")
                    if b.file_answers:
                        path, filt = b.file_answers.pop(0)
                        b.dispatcher.log.emit("info", "   file chooser '%s' answered: %s" % (caption, path))
                        if name == "getOpenFileNames":
                            return ([path], filt)
                        if name == "getExistingDirectory":
                            return path
                        return (path, filt)
                    b.file_dialog_requests.append(str(caption))
                    b.dispatcher.log.emit("warn", "   file chooser '%s' skipped (no answer_file_dialog)" % caption)
                    if name == "getOpenFileNames":
                        return ([], "")
                    if name == "getExistingDirectory":
                        return ""
                    return ("", "")
                return orig(*a, **k)
            return staticmethod(patched)

        for name in ("getOpenFileName", "getSaveFileName", "getOpenFileNames", "getExistingDirectory"):
            orig = getattr(QFileDialog, name)
            setattr(QFileDialog, name, make(name, orig))

    # ----------------------------------------------------------------- GUI execution
    def gui(self, fn, timeout=20.0):
        """Run fn() on the GUI thread and return its value (quick, read-only work)."""
        job = _Job(fn)
        self.dispatcher.submit.emit(job)
        if not job.event.wait(timeout):
            raise ToolError("The app is busy (timed out after %.0fs). Try again or call wait." % timeout)
        if job.error is not None:
            raise job.error
        return job.result

    def action(self, fn, name, timeout=30.0, is_async=False, auto_answers=None, auto_ok_info=False):
        """Run a state-changing action on the GUI thread, detecting dialogs it opens.

        auto_answers: list of (regex, [button texts]) — dialogs whose title/text match are
        answered automatically (e.g. an overwrite confirmation the caller already approved).
        """
        job = _Job(fn, is_async=is_async, name=name)
        self.dispatcher.submit.emit(job)
        t0 = time.time()
        answered = set()
        probe_pending = None
        if self.settings.get("auto_ok_info"):
            auto_ok_info = True
        while True:
            if job.event.wait(0.12):
                break
            if time.time() - t0 > timeout:
                self._inflight[job.jid] = job
                return {"status": "running", "job": job.jid,
                        "note": "Still working after %.0fs. Call wait to collect the result." % timeout}
            if not job.started:
                continue
            # Probe for dialogs this action opened (runs inside the dialog's event loop).
            if probe_pending is None or probe_pending.event.is_set():
                if probe_pending is not None and probe_pending.error is None:
                    dialogs = [d for d in (probe_pending.result or [])
                               if d["window_id"] not in job.before_ids and d["window_id"] not in answered]
                    if dialogs:
                        d = dialogs[0]
                        if auto_ok_info and d.get("kind") == "QMessageBox" and d.get("buttons") == ["OK"]:
                            auto_answers = list(auto_answers or []) + [(r"[\s\S]", ["OK"])]
                        hit = False
                        for rx, buttons in (auto_answers or []):
                            if re.search(rx, d.get("title", "") + "\n" + d.get("text", ""), re.I):
                                try:
                                    self.gui(lambda wid=d["window_id"], bt=buttons:
                                             self._press_dialog_button(wid, bt)[0].click(), timeout=5)
                                    answered.add(d["window_id"])
                                    self.dispatcher.log.emit("info", "   auto-answered '%s'" % d.get("title"))
                                    hit = True
                                except ToolError as e:
                                    self.dispatcher.log.emit("warn", "   auto-answer failed: %s"
                                                             % str(e).split("\n")[0])
                                break
                        if not hit:
                            self._inflight[job.jid] = job
                            return {"status": "dialog_open", "job": job.jid, "dialog": d,
                                    "note": "The action is waiting on this dialog. Answer it with "
                                            "respond_to_dialog / set_control / click_control "
                                            "(window_id=%s), then call wait or continue." % d["window_id"]}
                probe_pending = _Job(self._modal_probe)
                self.dispatcher.submit.emit(probe_pending)
        if job.error is not None:
            raise job.error
        out = {"status": "done"}
        if job.result is not None:
            out["result"] = job.result
        new_wins = self.gui(lambda: [w for w in self._windows_info()
                                     if w["window_id"] not in job.before_ids])
        if new_wins:
            out["new_windows"] = new_wins
        return out

    def _modal_probe(self):
        return [w for w in self._windows_info() if w.get("modal")]

    def _collect_background(self):
        done = []
        for jid, job in list(self._inflight.items()):
            if job.event.is_set():
                self._inflight.pop(jid, None)
                rec = {"job": jid, "action": job.name}
                if job.error is not None:
                    rec["error"] = str(job.error).split("\n")[0]
                elif job.result is not None:
                    rec["result"] = job.result
                done.append(rec)
        return done

    def step_delay(self):
        try:
            return max(0, int(self.settings.get("step_delay_ms", 250)))
        except Exception:
            return 250

    def overlays_on(self):
        return bool(self.settings.get("show_overlays", True))

    # ----------------------------------------------------------------- windows & controls
    def _wid(self, w):
        if w is self.win:
            return "main"
        v = w.property("aiWinId")
        if not v:
            self._win_seq += 1
            v = "w%d" % self._win_seq
            w.setProperty("aiWinId", v)
        return str(v)

    def _top_windows(self):
        out = []
        for w in QApplication.topLevelWidgets():
            try:
                if not _is_valid(w) or not w.isVisible():
                    continue
                if w.objectName() in (PANEL_OBJECT_NAME, OVERLAY_OBJECT_NAME):
                    continue
                if isinstance(w, QMenu):
                    continue
                wt = w.windowType()
                if wt in (Qt.ToolTip, Qt.Popup, Qt.SplashScreen):
                    continue
                out.append(w)
            except RuntimeError:
                continue
        return out

    def _window_ids_now(self):
        return [self._wid(w) for w in self._top_windows()]

    def _windows_info(self):
        out = []
        for w in self._top_windows():
            out.append(self._window_info(w))
        return out

    def _window_info(self, w):
        modal = bool(w.isModal() or w.windowModality() != Qt.NonModal)
        info = {"window_id": self._wid(w), "title": w.windowTitle() or "",
                "kind": type(w).__name__, "modal": modal}
        if w is self.win:
            info["kind"] = "main window"
            info["modal"] = False
        if isinstance(w, QMessageBox):
            info["text"] = _plain(w.text())
            if w.informativeText():
                info["text"] += "\n" + _plain(w.informativeText())
            info["buttons"] = [_btn_text(b) for b in w.findChildren(QPushButton) if b.isVisibleTo(w)]
        elif isinstance(w, QDialog):
            texts = []
            for lab in w.findChildren(QLabel)[:6]:
                try:
                    t = _plain(lab.text())
                    if t and lab.isVisibleTo(w):
                        texts.append(t)
                except Exception:
                    pass
            if texts:
                info["text"] = "\n".join(texts)[:600]
            btns = []
            for bb in w.findChildren(QDialogButtonBox):
                btns += [_btn_text(b) for b in bb.buttons() if b.isVisibleTo(w)]
            if not btns:
                btns = [_btn_text(b) for b in w.findChildren(QPushButton)
                        if b.isVisibleTo(w) and b.text()][:12]
            info["buttons"] = btns
        return info

    def _window_by_id(self, wid):
        if not wid or wid == "main":
            return self.win
        for w in self._top_windows():
            if self._wid(w) == wid:
                return w
        raise ToolError("No open window with window_id '%s'. Call list_windows." % wid)

    def _front_dialog(self):
        dlgs = [w for w in self._top_windows() if w is not self.win]
        if not dlgs:
            raise ToolError("No dialog is open.")
        active = QApplication.activeModalWidget()
        if active is not None and active in dlgs:
            return active
        modal = [w for w in dlgs if w.isModal() or w.windowModality() != Qt.NonModal]
        return (modal or dlgs)[-1]

    def _attr_map(self, owner):
        m = {}
        try:
            items = list(vars(owner).items())
        except TypeError:
            return m
        for name, val in items:
            if name.startswith("__") or not isinstance(val, QWidget):
                continue
            try:
                if not _is_valid(val):
                    continue
                key = _cpp_ptr(val)
            except Exception:
                continue
            if key not in m or (m[key].startswith("_") and not name.startswith("_")) \
                    or (len(name) < len(m[key]) and not name.startswith("_")):
                m[key] = name
        return m

    def _control_id(self, w, top, attr_map):
        name = attr_map.get(_cpp_ptr(w))
        wid = self._wid(top)
        if name:
            return name if wid == "main" else "%s.%s" % (wid, name)
        v = w.property("aiCtlId")
        if not v:
            self._ctl_seq += 1
            v = "%s#%d" % (wid, self._ctl_seq)
            w.setProperty("aiCtlId", v)
        return str(v)

    def _resolve_control(self, cid):
        cid = str(cid or "").strip()
        if not cid:
            raise ToolError("control_id is required.")
        if "#" in cid:
            for top in self._top_windows():
                for w in top.findChildren(QWidget):
                    try:
                        if w.property("aiCtlId") == cid:
                            # PySide invalidates child wrappers of a C++-created window (message
                            # boxes...) when the window's wrapper is released: keep it alive.
                            self._pin = top
                            return w
                    except RuntimeError:
                        continue
            raise ToolError("Control '%s' no longer exists (its window may have closed). "
                            "Call list_controls again." % cid)
        if "." in cid:
            wid, attr = cid.split(".", 1)
            owner = self._window_by_id(wid)
        else:
            owner, attr = self.win, cid
        w = getattr(owner, attr, None)
        if not isinstance(w, QWidget) or not _is_valid(w):
            raise ToolError("Unknown control '%s'. Call list_controls to see valid ids." % cid)
        return w

    def resolve_for_action(self, cid):
        w = self._resolve_control(cid)
        top = w.window()
        self._pin = top      # see _resolve_control
        _check_user_only(top)
        if top.objectName() == PANEL_OBJECT_NAME or _in_ai_settings(w):
            raise ToolError("AI-control settings can only be changed by the user.")
        return w

    def _is_listable(self, w, top):
        if w.objectName() == OVERLAY_OBJECT_NAME:
            return False
        if isinstance(w, (QScrollBar, QTabBar)):
            return False
        if _in_ai_settings(w):
            return False
        p = w.parentWidget()
        if isinstance(p, (QAbstractSpinBox, QComboBox)):
            return False
        # hidden explicitly (or inside a hidden container) — but pages of inactive tabs count;
        # widgets of child WINDOWS (dialogs parented to this window) belong to those windows.
        cur = w
        while cur is not None and cur is not top:
            if cur.isWindow():
                return False
            par = cur.parentWidget()
            if cur.isHidden() and not isinstance(par, QStackedWidget):
                return False
            cur = par
        return cur is top

    def _tab_path(self, w, top):
        tabs = []
        cur = w
        while cur is not None and cur is not top:
            par = cur.parentWidget()
            if isinstance(par, QStackedWidget) and isinstance(par.parentWidget(), QTabWidget):
                tw = par.parentWidget()
                idx = tw.indexOf(cur)
                if idx >= 0:
                    tabs.append(tw.tabText(idx).replace("&", ""))
            cur = par
        return " > ".join(reversed(tabs))

    def _group_title(self, w, top):
        cur = w.parentWidget()
        while cur is not None and cur is not top:
            if isinstance(cur, QGroupBox) and cur.title():
                return cur.title().replace("&", "")
            cur = cur.parentWidget()
        return ""

    def _nearby_label(self, w):
        """Caption for an unlabeled input: the buddy QLabel, a QFormLayout label, or the
        nearest QLabel to its left on the same row (pure geometry — no layout walking)."""
        par = w.parentWidget()
        if par is None:
            return ""
        try:
            lay = par.layout()
            if isinstance(lay, QFormLayout):
                lab = lay.labelForField(w)
                if isinstance(lab, QLabel):
                    return _plain(lab.text())
        except Exception:
            pass
        best, best_d = "", 1e9
        try:
            g = w.geometry()
            cy = g.center().y()
            for lab in par.findChildren(QLabel, options=Qt.FindDirectChildrenOnly):
                if lab.buddy() is w:
                    return _plain(lab.text())
                if lab.isHidden():
                    continue
                lg = lab.geometry()
                if lg.right() <= g.left() + 2 and abs(lg.center().y() - cy) <= max(10, g.height() * 0.6):
                    d = g.left() - lg.right()
                    if d < best_d:
                        best, best_d = _plain(lab.text()), d
            if not best:     # label directly above (stacked forms)
                for lab in par.findChildren(QLabel, options=Qt.FindDirectChildrenOnly):
                    if lab.isHidden():
                        continue
                    lg = lab.geometry()
                    if 0 <= g.top() - lg.bottom() <= 14 and abs(lg.left() - g.left()) <= 30:
                        best = _plain(lab.text())
                        break
        except Exception:
            pass
        return best[:80]

    def _describe(self, w, top, attr_map, include_values=True):
        kind = _kind(w)
        d = {"id": self._control_id(w, top, attr_map), "kind": kind}
        label = ""
        if isinstance(w, QAbstractButton):
            label = _btn_text(w)
        elif isinstance(w, QGroupBox):
            label = w.title().replace("&", "")
        if not label:
            label = w.accessibleName() or self._nearby_label(w)
        tip = (w.toolTip() or "").strip()
        if not label and tip:
            label = _plain(tip.split("\n")[0])[:80]
        d["label"] = label
        tab = self._tab_path(w, top)
        if tab:
            d["tab"] = tab
        grp = self._group_title(w, top)
        if grp:
            d["group"] = grp
        if tip:
            d["tooltip"] = _plain(tip)[:220]
        if not w.isEnabled():
            d["enabled"] = False
        if include_values:
            d.update(_value_of(w))
        return d

    # ----------------------------------------------------------------- reveal / highlight
    def reveal(self, w, caption=""):
        """Bring w on screen: switch every enclosing tab, scroll it into view, outline it."""
        cur = w
        while cur is not None:
            par = cur.parentWidget()
            if isinstance(par, QStackedWidget) and isinstance(par.parentWidget(), QTabWidget):
                tw = par.parentWidget()
                idx = tw.indexOf(cur)
                if idx >= 0 and tw.currentIndex() != idx:
                    tw.setCurrentIndex(idx)
            cur = par
        cur = w.parentWidget()
        while cur is not None:
            if isinstance(cur, QScrollArea):
                try:
                    cur.ensureWidgetVisible(w, 20, 20)
                except Exception:
                    pass
            cur = cur.parentWidget()
        top = w.window()
        if top is not None and top.isMinimized():
            top.showNormal()
        if self.overlays_on() and w.isVisible():
            _HighlightFrame(w, caption)

    def cursor(self):
        lv = getattr(self.win, "live_view_label", None)
        if lv is None:
            return None
        if self._cursor is None or not _is_valid(self._cursor):
            self._cursor = _AiCursor(lv)
        return self._cursor

    # ----------------------------------------------------------------- viewer mapping
    def _viewer_geom(self):
        win = self.win
        img = getattr(win, "image", None)
        lv = getattr(win, "live_view_label", None)
        if img is None or img.isNull() or lv is None:
            raise ToolError("No image is loaded. Use open_image first.")
        iw, ih = float(img.width()), float(img.height())
        lw, lh = float(lv.width()), float(lv.height())
        if min(iw, ih, lw, lh) <= 0:
            raise ToolError("The image viewer is collapsed; enlarge the window/viewer.")
        s = min(lw / iw, lh / ih)
        ox, oy = (lw - iw * s) / 2.0, (lh - ih * s) / 2.0
        z = lv.zoom_level if lv.zoom_level > 1e-9 else 1.0
        return lv, s, ox, oy, z, lv.pan_offset

    def img_to_widget(self, x, y):
        lv, s, ox, oy, z, pan = self._viewer_geom()
        return QPointF((ox + float(x) * s) * z + pan.x(), (oy + float(y) * s) * z + pan.y())

    def _send_mouse(self, lv, etype, pos, button, buttons, mods):
        gp = lv.mapToGlobal(pos)
        ev = QMouseEvent(etype, pos, QPointF(gp), button, buttons, mods)
        QApplication.sendEvent(lv, ev)

    # ----------------------------------------------------------------- tool dispatch
    def call_tool(self, name, args):
        """Entry point for one MCP tools/call. Returns (content_list, is_error)."""
        args = args or {}
        fn = getattr(self, "t_" + str(name), None)
        if fn is None or name not in {t["name"] for t in S.TOOLS}:
            return [_text({"error": "Unknown tool '%s'." % name})], True
        if self.paused and name not in ("wait",):
            return [_text({"error": "The user has PAUSED AI control in the app (AI Control panel). "
                                    "Wait for them to resume."})], True
        if self.settings.get("permission") == "view" and name not in S.READ_ONLY_TOOLS:
            return [_text({"error": "The user set AI control to VIEW ONLY: '%s' changes the app and "
                                    "is not allowed. Read-only tools still work." % name})], True
        summary = _summarize_args(args)
        self.dispatcher.log.emit("call", "▶ %s(%s)" % (name, summary))
        self.dispatcher.activity.emit("%s %s" % (name, summary)[:90])
        t0 = time.time()
        with self._call_lock:
            try:
                res = fn(args)
                is_error = False
            except ToolError as e:
                res, is_error = {"error": str(e)}, True
            except Exception as e:  # noqa: BLE001
                res, is_error = {"error": "%s: %s" % (type(e).__name__, e),
                                 "trace": traceback.format_exc(limit=5)}, True
        ms = (time.time() - t0) * 1000
        if isinstance(res, _ImageResult):
            content = [{"type": "image", "data": base64.b64encode(res.png).decode("ascii"),
                        "mimeType": "image/png"}, _text(res.meta)]
            brief = "image %s" % res.meta.get("returned_size", "")
        else:
            if isinstance(res, dict) and not is_error and name not in ("wait",):
                bg = self._collect_background()
                if bg:
                    res["finished_in_background"] = bg
                if self.file_dialog_requests:
                    res["skipped_file_choosers"] = list(self.file_dialog_requests)
                    res["hint"] = "Call answer_file_dialog(path) before the click that opens a file chooser."
                    self.file_dialog_requests = []
            content = [_text(res)]
            brief = (res.get("error") if is_error else res.get("status", "ok")) if isinstance(res, dict) else "ok"
        self.dispatcher.log.emit("error" if is_error else "ok",
                                 "%s %s  (%d ms)  %s" % ("✗" if is_error else "✓", name, ms,
                                                         str(brief).split("\n")[0][:160]))
        self.dispatcher.activity.emit("")
        return content, is_error

    # =========================================================================== tools
    # ---- observe -------------------------------------------------------------------
    def t_get_status(self, a):
        def run():
            win = self.win
            img = getattr(win, "image", None)
            st = {"app": "Gel Blot Analyzer",
                  "version": _app_global(win, "APP_VERSION", "?"),
                  "ai_control": {"permission": self.settings.get("permission"),
                                 "step_delay_ms": self.step_delay()}}
            if img is not None and not img.isNull():
                st["image"] = {"path": getattr(win, "image_path", None),
                               "width": img.width(), "height": img.height(),
                               "format": _fmt_name(img), "bit_depth": _bit_depth(img),
                               "grayscale": bool(img.isGrayscale())}
            else:
                st["image"] = None
            st["unsaved_changes"] = bool(getattr(win, "is_modified", False))
            lv = getattr(win, "live_view_label", None)
            if lv is not None:
                st["viewer"] = {"zoom": round(float(lv.zoom_level), 3),
                                "size": [lv.width(), lv.height()]}
            tw = getattr(win, "tab_widget", None)
            if tw is not None:
                st["current_tab"] = tw.tabText(tw.currentIndex())
                st["tabs"] = [tw.tabText(i) for i in range(tw.count())]
            ms = getattr(win, "mode_status_label", None)
            st["armed_tool"] = ms.text() if ms is not None and ms.text() else None
            st["status_bar"] = win.statusBar().currentMessage()
            st["undo_steps"] = max(0, len(getattr(win, "undo_stack", []) or []) - 1)
            st["redo_steps"] = len(getattr(win, "redo_stack", []) or [])
            st["windows"] = self._windows_info()
            return st
        st = self.gui(run)
        inflight = [{"job": j.jid, "action": j.name} for j in self._inflight.values() if not j.event.is_set()]
        if inflight:
            st["actions_waiting_on_dialogs"] = inflight
        return st

    def t_get_annotations(self, a):
        def run():
            win = self.win
            cfg = win.get_current_config()
            out = {}
            lm = cfg.get("marker_positions", {})
            out["left_markers"] = {"x": getattr(win, "left_marker_shift_added", 0),
                                   "items": [{"y": _r(p), "label": l} for p, l in lm.get("left", [])]}
            out["right_markers"] = {"x": getattr(win, "right_marker_shift_added", 0),
                                    "items": [{"y": _r(p), "label": l} for p, l in lm.get("right", [])]}
            out["top_markers"] = {"y": getattr(win, "top_marker_shift_added", 0),
                                  "items": [{"x": _r(p), "label": l} for p, l in lm.get("top", [])]}
            out["custom_text"] = [dict(index=i, **{k: (_r(v) if k in ("x", "y") else v) for k, v in m.items()})
                                  for i, m in enumerate(cfg.get("custom_markers", []))]
            shapes = []
            for i, s in enumerate(cfg.get("custom_shapes", [])):
                d = {"index": i}
                for k, v in s.items():
                    d[k] = _jsonable(v)
                shapes.append(d)
            out["shapes"] = shapes
            regions = []
            for d in cfg.get("multi_lane_definitions", []):
                regions.append({"lane_id": d.get("id"), "type": d.get("type"),
                                "points_image": _jsonable(d.get("points_image"))})
            out["lane_regions"] = regions
            out["padding_fields"] = cfg.get("adding_white_space")
            out["font"] = cfg.get("font_options")
            out["image_adjustments"] = _jsonable(cfg.get("image_adjustments"))
            cb = getattr(win, "combo_box", None)
            out["marker_preset"] = cb.currentText() if cb is not None else None
            out["marker_presets"] = sorted((getattr(win, "presets_data", {}) or {}).keys())
            out["preset_values"] = [str(v) for v in getattr(win, "marker_values", []) or []]
            out["top_label_list"] = [str(v) for v in getattr(win, "top_label", []) or []]
            return out
        return self.gui(run)

    def t_get_analysis_results(self, a):
        def run():
            win = self.win
            cfg = win.get_current_config()
            res = _jsonable(cfg.get("densitometry_results", {}))
            out = {"densitometry": res,
                   "standards": {
                       "quantities_text": _safe_text(getattr(win, "standard_protein_values", None)),
                       "areas_text": _safe_text(getattr(win, "standard_protein_areas_text", None)),
                       "quantity_area_pairs": _jsonable(getattr(win, "quantities_peak_area_dict", {}))}}
            # Percent of lane per band (what the results window shows)
            pct = {}
            for lane, areas in (res.get("multi_lane_peak_areas") or {}).items():
                tot = sum(areas) or 0.0
                pct[lane] = [round(100.0 * x / tot, 3) if tot else 0.0 for x in areas]
            if pct:
                out["percent_of_lane"] = pct
            return out
        return self.gui(run)

    def t_screenshot(self, a):
        target = a.get("target") or "image"
        max_size = int(a.get("max_size") or 1280)

        def run():
            win = self.win
            meta = {"target": target}
            hidden = self._hide_overlays()
            try:
                if target == "image":
                    img = win._render_annotated_gel_qimage(render_scale=1, draw_densitometry_boxes=True)
                    if img is None or img.isNull():
                        raise ToolError("No image is loaded.")
                    iw, ih = img.width(), img.height()
                    out = _fit(img, max_size)
                    k = out.width() / float(iw)
                    if a.get("grid"):
                        out = _draw_grid(out, k, iw, ih, a.get("grid_step"))
                    meta.update({"image_size": [iw, ih], "returned_size": [out.width(), out.height()],
                                 "scale": round(k, 5),
                                 "note": "Image coordinates = returned pixel / scale. Lane regions "
                                         "(if any) are drawn as boxes."})
                elif target == "viewer":
                    lv = win.live_view_label
                    out = _fit(lv.grab().toImage(), max_size)
                    meta["returned_size"] = [out.width(), out.height()]
                elif target == "window":
                    out = _fit(win.grab().toImage(), max_size)
                    meta["returned_size"] = [out.width(), out.height()]
                else:
                    w = self._window_by_id(a["window_id"]) if a.get("window_id") else self._front_dialog()
                    out = _fit(w.grab().toImage(), max_size)
                    meta.update({"window_id": self._wid(w), "title": w.windowTitle(),
                                 "returned_size": [out.width(), out.height()]})
                return _ImageResult(_png(out), meta)
            finally:
                for w in hidden:
                    try:
                        w.show()
                    except Exception:
                        pass
        return self.gui(run, timeout=60)

    def _hide_overlays(self):
        hidden = []
        for top in QApplication.topLevelWidgets():
            try:
                for w in top.findChildren(QWidget, OVERLAY_OBJECT_NAME):
                    if w.isVisible():
                        w.hide()
                        hidden.append(w)
            except RuntimeError:
                continue
        return hidden

    def t_list_windows(self, a):
        return {"windows": self.gui(self._windows_info)}

    def t_list_controls(self, a):
        def run():
            top = self._window_by_id(a.get("window_id") or "main")
            amap = self._attr_map(top)
            tab_f = (a.get("tab") or "").lower()
            q = (a.get("query") or "").lower()
            inc_labels = bool(a.get("include_labels"))
            inc_dis = a.get("include_disabled", True)
            limit = int(a.get("limit") or 250)
            out, total = [], 0
            types = _INPUT_TYPES + ((QLabel,) if inc_labels else ())
            for w in top.findChildren(QWidget):
                try:
                    if not isinstance(w, types):
                        continue
                    if isinstance(w, QGroupBox) and not w.isCheckable():
                        continue
                    if isinstance(w, QLabel) and (not w.text().strip() or isinstance(w.parentWidget(), QAbstractButton)):
                        continue
                    if not self._is_listable(w, top):
                        continue
                    if not inc_dis and not w.isEnabled():
                        continue
                    d = self._describe(w, top, amap)
                    if tab_f and tab_f not in d.get("tab", "").lower():
                        continue
                    if q:
                        hay = " ".join(str(d.get(k, "")) for k in ("id", "label", "tooltip", "group", "tab", "text")).lower()
                        if q not in hay:
                            continue
                    total += 1
                    if len(out) < limit:
                        out.append(d)
                except RuntimeError:
                    continue
            res = {"window_id": self._wid(top), "title": top.windowTitle(), "controls": out}
            if total > len(out):
                res["truncated"] = "%d of %d shown — narrow with tab/query or raise limit." % (len(out), total)
            return res
        return self.gui(run, timeout=30)

    def t_read_table(self, a):
        def run():
            w = self._resolve_control(a["control_id"])
            if not isinstance(w, QTableView):
                raise ToolError("'%s' is not a table." % a["control_id"])
            m = w.model()
            if m is None:
                return {"rows": []}
            cols = m.columnCount()
            hdr = [str(m.headerData(c, Qt.Horizontal) or "") for c in range(cols)]
            rows = []
            for r in range(min(m.rowCount(), int(a.get("max_rows") or 500))):
                rows.append([_cell(m.index(r, c).data()) for c in range(cols)])
            return {"columns": hdr, "rows": rows, "row_count": m.rowCount()}
        return self.gui(run)

    def t_list_menu_actions(self, a):
        def run():
            out, seen = [], set()
            for act in self._all_actions():
                key = _act_name(act)
                if not key or key in seen:
                    continue
                seen.add(key)
                d = {"action": key, "enabled": act.isEnabled()}
                if act.menu() is not None:
                    continue
                if act.isCheckable():
                    d["checked"] = act.isChecked()
                sc = act.shortcut().toString(QKeySequence.NativeText)
                if sc:
                    d["shortcut"] = sc
                tip = (act.toolTip() or "").split("\n")[0]
                if tip and tip.replace("&", "") != key:
                    d["tooltip"] = tip[:160]
                out.append(d)
            return {"actions": out}
        return self.gui(run)

    def _all_actions(self):
        acts = []
        mb = self.win.menuBar()
        for top in mb.actions():
            m = top.menu()
            if m is not None:
                for act in m.actions():
                    if not act.isSeparator():
                        acts.append(act)
        for act in self.win.findChildren(QAction):
            if not act.isSeparator() and act.text():
                acts.append(act)
        return acts

    # ---- generic UI -------------------------------------------------------------------
    def _paced(self, job, w, caption, do, resolve=None):
        """Reveal+highlight w now, perform `do` after the step delay (live preview).

        `resolve` (optional) re-fetches the widget when the delay ends and `do` then receives
        it: Python wrappers of C++-created widgets (message-box buttons...) are invalidated by
        PySide whenever the same object is fetched again, so they must not be held across
        the delay."""
        self.reveal(w, caption)
        delay = self.step_delay()

        def go():
            self.ai_depth += 1
            try:
                job.finish(do(resolve()) if resolve is not None else do())
            except ToolError as e:
                job.fail(e)
            except Exception as e:  # noqa: BLE001
                job.fail(ToolError("%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc(limit=8))))
            finally:
                self.ai_depth -= 1
        if delay:
            QTimer.singleShot(delay, go)
        else:
            go()

    def t_set_control(self, a):
        cid, value = a.get("control_id"), a.get("value")

        def start(job):
            w = self.resolve_for_action(cid)
            if not w.isEnabled():
                raise ToolError("Control '%s' is disabled right now." % cid)
            self._paced(job, w, "AI: set", lambda ww: _set_value(ww, value),
                        resolve=lambda: self.resolve_for_action(cid))
        res = self.action(start, "set_control", is_async=True)
        res["control"] = self.gui(lambda: _value_of(self._resolve_control(cid))) \
            if res.get("status") == "done" and _alive(self, cid) else None
        return self._with_status(res)

    def t_click_control(self, a):
        cid = a.get("control_id")

        def start(job):
            w = self.resolve_for_action(cid)
            if not isinstance(w, (QAbstractButton, QGroupBox)):
                raise ToolError("'%s' is a %s, not a button; use set_control." % (cid, _kind(w)))
            if not w.isEnabled():
                raise ToolError("Button '%s' is disabled right now." % cid)

            def do(ww):
                if isinstance(ww, QGroupBox):
                    ww.setChecked(not ww.isChecked())
                    return {"checked": ww.isChecked()}
                checkable = ww.isCheckable()
                ptr = _cpp_ptr(ww)
                ww.click()
                if checkable:
                    try:
                        again = self._resolve_control(cid)
                        if _cpp_ptr(again) == ptr:
                            return {"checked": again.isChecked()}
                    except (ToolError, RuntimeError):
                        pass
                return None
            self._paced(job, w, "AI: click", do, resolve=lambda: self.resolve_for_action(cid))
        return self._with_status(self.action(start, "click_control", is_async=True, timeout=45,
                                             auto_ok_info=bool(a.get("_auto_ok")),
                                             auto_answers=a.get("_auto_answers")))

    def t_select_tab(self, a):
        name = (a.get("tab") or "").lower()

        def run():
            top = self._window_by_id(a.get("window_id") or "main")
            for tw in top.findChildren(QTabWidget):
                for i in range(tw.count()):
                    if name in tw.tabText(i).replace("&", "").lower():
                        self.reveal(tw)
                        tw.setCurrentIndex(i)
                        return {"tab": tw.tabText(i)}
            raise ToolError("No tab matching '%s'." % a.get("tab"))
        return self._with_status(self.action(run, "select_tab"))

    def t_trigger_menu_action(self, a):
        want = _norm(a.get("action"))

        def start(job):
            cand = [act for act in self._all_actions() if _norm(_act_name(act)) == want]
            if not cand:
                cand = [act for act in self._all_actions() if want and want in _norm(_act_name(act))]
            if not cand:
                raise ToolError("No menu/toolbar command named '%s'. Call list_menu_actions." % a.get("action"))
            act = cand[0]
            if not act.isEnabled():
                raise ToolError("'%s' is disabled right now." % _act_name(act))
            self.win.statusBar().showMessage("AI ▸ " + _act_name(act), 2500)

            def do():
                if act.isCheckable() and "checked" in a:
                    if act.isChecked() != bool(a["checked"]):
                        act.trigger()
                else:
                    act.trigger()
                return {"action": _act_name(act), **({"checked": act.isChecked()} if act.isCheckable() else {})}
            delay = self.step_delay()
            QTimer.singleShot(delay, lambda: self._finish_with(job, do))
        return self._with_status(self.action(start, "trigger_menu_action", is_async=True, timeout=45))

    def _finish_with(self, job, do):
        self.ai_depth += 1
        try:
            job.finish(do())
        except ToolError as e:
            job.fail(e)
        except Exception as e:  # noqa: BLE001
            job.fail(ToolError("%s: %s" % (type(e).__name__, e)))
        finally:
            self.ai_depth -= 1

    def _press_dialog_button(self, wid, texts):
        w = self._window_by_id(wid) if wid else self._front_dialog()
        self._pin = w        # keep the dialog's wrapper alive while its buttons are used
        _check_user_only(w)
        if isinstance(texts, str):
            texts = [texts]
        wanted = [_norm(t) for t in texts]
        # ONE fetch, used immediately (see _paced about invalidated wrappers)
        vis = [b for b in w.findChildren(QPushButton) if b.isVisibleTo(w)]
        for want in wanted:
            for b in vis:
                if _norm(_btn_text(b)) == want:
                    return b, w
        for want in wanted:
            for b in vis:
                if want and want in _norm(_btn_text(b)):
                    return b, w
        raise ToolError("No button %s in '%s'. Available: %s"
                        % (texts, w.windowTitle(), [_btn_text(b) for b in vis]))

    def t_respond_to_dialog(self, a):
        wid, text = a.get("window_id"), a.get("button")

        def start(job):
            b, w = self._press_dialog_button(wid, [text])
            label = _btn_text(b)
            wid_now = self._wid(w)
            info = {"pressed": label, "dialog": w.windowTitle() or wid_now}
            self._paced(job, b, "AI: " + label, lambda bb: (bb.click(), info)[1],
                        resolve=lambda: self._press_dialog_button(wid_now, [label])[0])
        res = self.action(start, "respond_to_dialog", is_async=True, timeout=60)
        self._settle(res)
        return self._with_status(res)

    def t_close_window(self, a):
        def run():
            w = self._window_by_id(a.get("window_id"))
            if w is self.win:
                raise ToolError("The AI cannot close the main window.")
            _check_user_only(w)
            if isinstance(w, QDialog):
                w.reject()
            else:
                w.close()
            return {"closed": a.get("window_id")}
        res = self.action(run, "close_window")
        self._settle(res)
        return self._with_status(res)

    def t_press_keys(self, a):
        keys = a.get("keys") or ""

        def run():
            seq = QKeySequence(keys)
            if seq.isEmpty():
                raise ToolError("Could not parse keys '%s'." % keys)
            top = self._window_by_id(a.get("window_id")) if a.get("window_id") else (
                QApplication.activeModalWidget() or self.win)
            # 1) a command bound to this shortcut
            for act in top.findChildren(QAction) + (self._all_actions() if top is self.win else []):
                if act.isEnabled() and any(s == seq for s in act.shortcuts()):
                    act.trigger()
                    return {"triggered": _act_name(act)}
            for sc in top.findChildren(QShortcut):
                if sc.isEnabled() and sc.key() == seq:
                    sc.activated.emit()
                    return {"triggered_shortcut": keys}
            # 2) plain key events to the focused widget (or the window)
            comb = seq[0]
            key = comb.key()
            mods = comb.keyboardModifiers()
            target = QApplication.focusWidget()
            if target is None or target.window() is not top:
                target = top
            kv = key.value if hasattr(key, "value") else int(key)
            plain = mods in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)
            text = chr(kv) if (plain and 32 <= kv < 127) else ""
            if text and mods == Qt.KeyboardModifier.NoModifier:
                text = text.lower()
            for et in (QEvent.KeyPress, QEvent.KeyRelease):
                QApplication.sendEvent(target, QKeyEvent(et, kv, mods, text))
            return {"sent": keys, "to": type(target).__name__}
        return self._with_status(self.action(run, "press_keys"))

    def t_answer_file_dialog(self, a):
        path = os.path.expanduser(str(a.get("path") or ""))
        if not path:
            raise ToolError("path is required.")
        filt = a.get("file_filter") or _IMAGE_EXT_FILTERS.get(os.path.splitext(path)[1].lower(), "")
        self.file_answers.append((path, filt))
        return {"queued": path, "pending_answers": len(self.file_answers)}

    # ---- viewer ----------------------------------------------------------------------
    def _mods(self, names):
        m = Qt.KeyboardModifier.NoModifier
        table = {"shift": Qt.ShiftModifier, "ctrl": Qt.ControlModifier,
                 "alt": Qt.AltModifier, "meta": Qt.MetaModifier}
        for n in names or []:
            m |= table.get(str(n).lower(), Qt.KeyboardModifier.NoModifier)
        return m

    def t_viewer_click(self, a):
        x, y = float(a["x"]), float(a["y"])
        btn = Qt.RightButton if a.get("button") == "right" else Qt.LeftButton
        mods = self._mods(a.get("modifiers"))
        dbl = bool(a.get("double_click"))

        def start(job):
            pos = self.img_to_widget(x, y)
            lv = self.win.live_view_label
            cur = self.cursor() if self.overlays_on() else None
            if cur is not None:
                cur.move_to(pos)

            def do():
                if cur is not None:
                    cur.move_to(pos, pressed=True, ripple=True)
                self._send_mouse(lv, QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton, mods)
                self._send_mouse(lv, QEvent.MouseButtonPress, pos, btn, btn, mods)
                self._send_mouse(lv, QEvent.MouseButtonRelease, pos, btn, Qt.NoButton, mods)
                if dbl:
                    self._send_mouse(lv, QEvent.MouseButtonDblClick, pos, btn, btn, mods)
                    self._send_mouse(lv, QEvent.MouseButtonRelease, pos, btn, Qt.NoButton, mods)
                if cur is not None and _is_valid(cur):
                    cur.move_to(pos)
                return {"clicked_image_xy": [x, y]}
            delay = self.step_delay()
            QTimer.singleShot(delay, lambda: self._finish_with(job, do))
        return self._with_status(self.action(start, "viewer_click", is_async=True, timeout=45,
                                             auto_ok_info=bool(a.get("_auto_ok"))))

    def t_viewer_drag(self, a):
        pts = [(float(p[0]), float(p[1])) for p in (a.get("points") or [])]
        if len(pts) < 2:
            raise ToolError("points needs at least 2 [x, y] pairs.")
        dur = int(a.get("duration_ms", 400))
        btn = Qt.RightButton if a.get("button") == "right" else Qt.LeftButton
        mods = self._mods(a.get("modifiers"))
        return self._with_status(self.action(lambda job: self._drag(job, pts, dur, btn, mods),
                                             "viewer_drag", is_async=True, timeout=45 + dur / 1000.0,
                                             auto_ok_info=bool(a.get("_auto_ok"))))

    def _drag(self, job, pts, dur, btn, mods, after=None):
        """Animated press → moves → release on the viewer (GUI thread)."""
        lv = self.win.live_view_label
        wpts = [self.img_to_widget(x, y) for x, y in pts]
        # interpolate so tools that rely on many move events see a smooth drag
        path = []
        for (p0, p1) in zip(wpts, wpts[1:]):
            n = max(2, int(max(abs(p1.x() - p0.x()), abs(p1.y() - p0.y())) / 6))
            for i in range(n):
                t = i / float(n)
                path.append(QPointF(p0.x() + (p1.x() - p0.x()) * t, p0.y() + (p1.y() - p0.y()) * t))
        path.append(wpts[-1])
        cur = self.cursor() if self.overlays_on() else None
        steps = len(path)
        interval = int(dur / steps) if dur > 0 else 0
        state = {"i": 0}

        def press():
            if cur is not None:
                cur.move_to(path[0], pressed=True, ripple=True)
            self._send_mouse(lv, QEvent.MouseMove, path[0], Qt.NoButton, Qt.NoButton, mods)
            self._send_mouse(lv, QEvent.MouseButtonPress, path[0], btn, btn, mods)

        def step():
            i = state["i"] = state["i"] + 1
            if i < steps:
                if cur is not None and _is_valid(cur):
                    cur.move_to(path[i], pressed=True)
                self._send_mouse(lv, QEvent.MouseMove, path[i], Qt.NoButton, btn, mods)
                return False
            self._send_mouse(lv, QEvent.MouseButtonRelease, path[-1], btn, Qt.NoButton, mods)
            if cur is not None and _is_valid(cur):
                cur.move_to(path[-1])
            return True

        def run_steps():
            # executed under ai_depth via _finish_with-like guard
            try:
                self.ai_depth += 1
                done = step()
            except Exception as e:  # noqa: BLE001
                self.ai_depth -= 1
                job.fail(ToolError("%s: %s" % (type(e).__name__, e)))
                return
            self.ai_depth -= 1
            if done:
                if after is not None:
                    self._finish_with(job, after)
                else:
                    job.finish({"dragged_image_points": [list(p) for p in pts]})
            else:
                QTimer.singleShot(interval, run_steps)

        def begin():
            try:
                self.ai_depth += 1
                press()
            except Exception as e:  # noqa: BLE001
                job.fail(ToolError("%s: %s" % (type(e).__name__, e)))
                return
            finally:
                self.ai_depth -= 1
            QTimer.singleShot(interval, run_steps)

        if cur is not None:
            cur.move_to(path[0])
        QTimer.singleShot(self.step_delay(), begin)

    def t_set_view(self, a):
        def run():
            win = self.win
            lv = win.live_view_label
            if a.get("reset"):
                win.reset_view()
            else:
                lv0, s, ox, oy, z, pan = self._viewer_geom()
                if a.get("zoom") is not None:
                    lv.zoom_level = max(lv.MIN_ZOOM, min(lv.MAX_ZOOM, float(a["zoom"])))
                z = lv.zoom_level
                if a.get("center_x") is not None and a.get("center_y") is not None:
                    lx = ox + float(a["center_x"]) * s
                    ly = oy + float(a["center_y"]) * s
                    lv.pan_offset = QPointF(lv.width() / 2.0 - lx * z, lv.height() / 2.0 - ly * z)
                for act in ("pan_left_action", "pan_right_action", "pan_up_action", "pan_down_action"):
                    if hasattr(win, act):
                        getattr(win, act).setEnabled(lv.is_view_transformed())
                win.update_live_view()
            return {"zoom": round(lv.zoom_level, 3), "pan": [lv.pan_offset.x(), lv.pan_offset.y()]}
        return self._with_status(self.action(run, "set_view"))

    # ---- files ----------------------------------------------------------------------
    def t_open_image(self, a):
        path = os.path.abspath(os.path.expanduser(str(a.get("path") or "")))
        if not os.path.isfile(path):
            raise ToolError("File not found: %s" % path)

        def run():
            win = self.win
            if not win.prompt_save_if_needed():
                return {"opened": False, "note": "Cancelled at the unsaved-changes prompt."}
            win.open_image_from_path(path)
            img = win.image
            if img is None or img.isNull():
                raise ToolError("The app could not open %s" % path)
            return {"opened": path, "width": img.width(), "height": img.height(),
                    "bit_depth": _bit_depth(img)}
        return self._with_status(self.action(run, "open_image", timeout=120))

    def t_save_image(self, a):
        path = os.path.abspath(os.path.expanduser(str(a.get("path") or "")))
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            raise ToolError("path must end in .png, .tif, .jpg or .bmp")
        if not os.path.isdir(os.path.dirname(path)):
            raise ToolError("Folder does not exist: %s" % os.path.dirname(path))
        self.file_answers = [(path, _IMAGE_EXT_FILTERS[ext])]
        answers = [(r"^Saved\b|saved successfully", ["OK"])]
        if a.get("overwrite"):
            answers.insert(0, (r"exist|overwrite|replace", ["Yes", "Overwrite", "Replace", "Save", "OK"]))

        def run():
            ok = self.win.save_image()
            base = os.path.splitext(path)[0].replace("_original", "").replace("_modified", "")
            sfx = ".tif" if ext == ".tiff" else ext
            files = [p for p in (base + "_original" + sfx, base + "_modified" + sfx, base + "_config.txt")
                     if os.path.exists(p)]
            return {"saved": bool(ok), "files": files}
        res = self.action(run, "save_image", timeout=180, auto_answers=answers)
        self.file_answers = []
        return self._with_status(res)

    def t_export_pptx(self, a):
        path = os.path.abspath(os.path.expanduser(str(a.get("path") or "")))
        if not path.lower().endswith(".pptx"):
            path += ".pptx"
        if os.path.exists(path) and not a.get("overwrite"):
            raise ToolError("%s exists. Pass overwrite=true to replace it." % path)
        self.file_answers = [(path, _IMAGE_EXT_FILTERS[".pptx"])]
        answers = [(r"exist|overwrite|replace", ["Yes", "Replace", "OK"]),
                   (r"export(ed)?\b.*(success|saved)|saved", ["OK"])]

        def run():
            self.win.save_image_pptx()
            return {"exported": os.path.exists(path), "path": path}
        res = self.action(run, "export_pptx", timeout=180, auto_answers=answers)
        self.file_answers = []
        return self._with_status(res)

    # ---- editing --------------------------------------------------------------------
    def t_history(self, a):
        n = int(a.get("steps") or 1)
        act = a.get("action")

        def run():
            win = self.win
            fn = win.undo_action_m if act == "undo" else win.redo_action_m
            for _ in range(n):
                fn()
            return {act: n, "undo_steps": max(0, len(win.undo_stack) - 1), "redo_steps": len(win.redo_stack)}
        return self._with_status(self.action(run, "history"))

    _OPS = {"invert": "invert_image", "flip_vertical": "flip_vertical",
            "flip_horizontal": "flip_horizontal", "grayscale": "convert_to_black_and_white",
            "to_8bit": "convert_to_8bit_depth", "optimize_levels": "optimize_levels",
            "reset_adjustments": "reset_all_adjustments", "reset_image": None}

    def t_image_operation(self, a):
        op = a.get("operation")
        if op not in self._OPS:
            raise ToolError("Unknown operation '%s'." % op)

        def run():
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            if op == "reset_image":
                mb = QMessageBox(QMessageBox.Warning, "AI request — Reset Image",
                                 "The AI wants to reset the image and remove ALL edits and "
                                 "annotations. Allow?", QMessageBox.Yes | QMessageBox.No, win)
                mb.setDefaultButton(QMessageBox.No)
                mb.setProperty("gbaUserOnly", True)
                r = mb.exec()
                mb.deleteLater()
                if r != QMessageBox.Yes:
                    return {"reset": False, "note": "Declined."}
                win.reset_image()
            else:
                getattr(win, self._OPS[op])()
            img = win.image
            return {"operation": op, "image": [img.width(), img.height()] if img is not None else None}
        return self._with_status(self.action(run, "image_operation", timeout=90))

    def t_adjust_levels(self, a):
        def start(job):
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            cb = getattr(win, "adjustment_context_combo", None)
            if cb is not None and cb.currentText() != "Main Image":
                cb.setCurrentText("Main Image")
            todo = []
            if a.get("black_point") is not None:
                todo.append((win.black_point_slider, int(round(float(a["black_point"])))))
            if a.get("white_point") is not None:
                todo.append((win.white_point_slider, int(round(float(a["white_point"])))))
            if a.get("gamma") is not None:
                todo.append((win.gamma_slider, int(round(float(a["gamma"]) * 100))))
            if not todo:
                raise ToolError("Give black_point, white_point and/or gamma.")

            def do():
                for s, v in todo:
                    _set_value(s, v)
                return {"black_point": win.black_point_slider.value(),
                        "white_point": win.white_point_slider.value(),
                        "gamma": win.gamma_slider.value() / 100.0,
                        "max_value": win.black_point_slider.maximum()}
            self._paced(job, todo[0][0], "AI: levels", do)
        return self._with_status(self.action(start, "adjust_levels", is_async=True, timeout=60))

    def t_crop(self, a):
        x, y, w, h = (float(a[k]) for k in ("x", "y", "width", "height"))
        if w < 2 or h < 2:
            raise ToolError("width and height must be at least 2 px.")

        def start(job):
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            img = win.image
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(img.width()), x + w), min(float(img.height()), y + h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                raise ToolError("The crop rectangle is outside the image (%dx%d)." % (img.width(), img.height()))
            self.reveal(win.draw_crop_rect_button, "AI: crop")
            if not win.crop_rectangle_mode:
                win.draw_crop_rect_button.click()

            def apply():
                win.apply_crop_button.click()
                im = win.image
                return {"cropped_to": [im.width(), im.height()]}
            self._drag(job, [(x0, y0), (x1, y1)], 500, Qt.LeftButton, Qt.KeyboardModifier.NoModifier,
                       after=apply)
        return self._with_status(self.action(start, "crop", is_async=True, timeout=90))

    def t_set_padding(self, a):
        def start(job):
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            first = win.recommend_button if a.get("recommended") else win.left_padding_input

            def do():
                if a.get("recommended"):
                    win.recommended_values()
                for side in ("left", "right", "top", "bottom"):
                    if a.get(side) is not None:
                        getattr(win, side + "_padding_input").setText(str(int(a[side])))
                vals = {s: getattr(win, s + "_padding_input").text() for s in ("left", "right", "top", "bottom")}
                if a.get("apply", True):
                    win.finalize_image()
                im = win.image
                return {"padding": vals, "applied": bool(a.get("apply", True)),
                        "image": [im.width(), im.height()]}
            self._paced(job, first, "AI: padding", do)
        return self._with_status(self.action(start, "set_padding", is_async=True, timeout=90))

    def t_set_marker_preset(self, a):
        name = a.get("preset") or ""

        def start(job):
            cb = self.win.combo_box
            items = [cb.itemText(i) for i in range(cb.count())]
            match = [t for t in items if t.lower() == name.lower()] or \
                    [t for t in items if name.lower() in t.lower()]
            if not match:
                raise ToolError("No preset '%s'. Available: %s" % (name, items))

            def do():
                cb.setCurrentText(match[0])
                return {"preset": match[0],
                        "values": [str(v) for v in getattr(self.win, "marker_values", []) or []],
                        "top_labels": [str(v) for v in getattr(self.win, "top_label", []) or []]}
            self._paced(job, cb, "AI: preset", do)
        return self._with_status(self.action(start, "set_marker_preset", is_async=True))

    def t_place_markers(self, a):
        side = a.get("side")
        positions = [float(p) for p in a.get("positions") or []]
        labels = a.get("labels")
        win = self.win
        if side not in ("left", "right", "top"):
            raise ToolError("side must be left, right or top.")

        def prep():
            if not win._has_image():
                raise ToolError("No image is loaded.")
            img = win.image
            if side == "top":
                cy = a.get("y")
                if cy is None:
                    cy = getattr(win, "top_marker_shift_added", 0) if win.top_markers else img.height() * 0.03
                clicks = [(p, float(cy)) for p in positions]
            else:
                cx = a.get("x")
                if cx is None:
                    cur = getattr(win, side + "_marker_shift_added", 0)
                    cx = cur if getattr(win, side + "_markers") else (
                        img.width() * 0.03 if side == "left" else img.width() * 0.97)
                clicks = [(float(cx), p) for p in positions]
            if a.get("clear_existing"):
                win.reset_marker(side, "reset")
            getattr(win, "enable_%s_marker_mode" % side)()
            return clicks
        clicks = self.gui(prep)
        placed = []
        for (cx, cy) in clicks:
            r = self.t_viewer_click({"x": cx, "y": cy, "_auto_ok": True})
            if r.get("status") != "done":
                r["placed_so_far"] = placed
                return r
            placed.append([cx, cy])

        def finish():
            lst = getattr(win, side + "_markers")
            if labels:
                key_pos = positions
                for i, (pos, lab) in enumerate(list(lst)):
                    for j, p in enumerate(key_pos):
                        if j < len(labels) and abs(pos - p) < 1.0:
                            lst[i] = (pos, str(labels[j]))
            win._cancel_all_interaction_modes()
            win.update_live_view()
            axis = "y" if side != "top" else "x"
            return {"side": side, "markers": [{axis: _r(p), "label": l} for p, l in lst]}
        return self._with_status(self.action(finish, "place_markers"))

    def t_add_text(self, a):
        def run():
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            col = QColor(a.get("color") or "#000000")
            if not col.isValid():
                raise ToolError("Invalid color '%s'." % a.get("color"))
            font = a.get("font") or win.custom_font_type_dropdown.currentText()
            size = int(a.get("font_size") or win.custom_font_size_spinbox.value())
            win.save_state()
            win.custom_markers = getattr(win, "custom_markers", [])
            win.custom_markers.append([float(a["x"]), float(a["y"]), str(a["text"]), col, font, size,
                                       bool(a.get("bold")), bool(a.get("italic"))])
            win.is_modified = True
            win.update_live_view()
            self._point_at(a["x"], a["y"])
            return {"index": len(win.custom_markers) - 1, "font": font, "font_size": size}
        return self._with_status(self.action(run, "add_text"))

    def t_add_shape(self, a):
        typ = a.get("type")

        def run():
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            col = QColor(a.get("color") or "#000000")
            if not col.isValid():
                raise ToolError("Invalid color '%s'." % a.get("color"))
            shape = {"type": typ, "color": col.name(), "thickness": float(a.get("thickness") or 2)}
            if typ in ("line", "arrow"):
                try:
                    shape["start"] = (float(a["x1"]), float(a["y1"]))
                    shape["end"] = (float(a["x2"]), float(a["y2"]))
                except KeyError:
                    raise ToolError("line/arrow needs x1, y1, x2, y2.")
                px, py = shape["end"]
            elif typ == "rectangle":
                try:
                    shape["rect"] = (float(a["x"]), float(a["y"]), float(a["width"]), float(a["height"]))
                except KeyError:
                    raise ToolError("rectangle needs x, y, width, height.")
                px, py = shape["rect"][0], shape["rect"][1]
            else:
                raise ToolError("type must be line, arrow or rectangle.")
            win.custom_shapes.append(shape)
            win.save_state()
            win.is_modified = True
            win.update_live_view()
            self._point_at(px, py)
            return {"index": len(win.custom_shapes) - 1}
        return self._with_status(self.action(run, "add_shape"))

    def _point_at(self, x, y):
        try:
            if self.overlays_on():
                c = self.cursor()
                if c is not None:
                    c.move_to(self.img_to_widget(x, y), ripple=True)
        except ToolError:
            pass

    def t_remove_custom_items(self, a):
        kind = a.get("kind")
        idx = sorted({int(i) for i in (a.get("indices") or [])}, reverse=True)

        def run():
            win = self.win
            win.save_state()
            removed = {"text": 0, "shape": 0}
            for k, attr in (("text", "custom_markers"), ("shape", "custom_shapes")):
                if kind not in (k, "all"):
                    continue
                lst = getattr(win, attr, [])
                if kind == "all" and not idx:
                    removed[k] = len(lst)
                    lst.clear()
                    continue
                for i in idx:
                    if 0 <= i < len(lst):
                        lst.pop(i)
                        removed[k] += 1
            win.is_modified = True
            win.update_live_view()
            return {"removed": removed, "remaining": {"text": len(win.custom_markers),
                                                      "shape": len(win.custom_shapes)}}
        return self._with_status(self.action(run, "remove_custom_items"))

    # ---- analysis -------------------------------------------------------------------
    def t_define_lane_regions(self, a):
        lanes = a.get("lanes") or []
        win = self.win
        if a.get("clear_existing"):
            def clear():
                win.save_state()
                win.multi_lane_definitions = []
                win.update_live_view()
            self.gui(clear)
        drawn = []
        for i, lane in enumerate(lanes):
            if "corners" in lane:
                pts = lane["corners"]
                if len(pts) != 4:
                    raise ToolError("lane %d: corners needs 4 [x, y] points." % i)
                r = self.t_click_control({"control_id": "btn_define_quad", "_auto_ok": True,
                                          "_auto_answers": _REGION_ANSWERS})
                if r.get("status") != "done":
                    r["lanes_drawn"] = drawn
                    return r
                for (x, y) in pts:
                    r = self.t_viewer_click({"x": x, "y": y, "_auto_ok": True})
                    if r.get("status") != "done":
                        r["lanes_drawn"] = drawn
                        return r
            else:
                try:
                    x, y, w, h = (float(lane[k]) for k in ("x", "y", "width", "height"))
                except KeyError:
                    raise ToolError("lane %d needs x, y, width, height (or corners)." % i)
                r = self.t_click_control({"control_id": "btn_define_rec", "_auto_ok": True,
                                          "_auto_answers": _REGION_ANSWERS})
                if r.get("status") != "done":
                    r["lanes_drawn"] = drawn
                    return r
                r = self.t_viewer_drag({"points": [[x, y], [x + w, y + h]], "duration_ms": 350,
                                        "_auto_ok": True})
                if r.get("status") != "done":
                    r["lanes_drawn"] = drawn
                    return r
            drawn.append(i)

        def summary():
            win._cancel_all_interaction_modes()
            win.update_live_view()
            cfg = win.get_current_config()
            return {"lane_regions": [{"lane_id": d.get("id"), "type": d.get("type"),
                                      "points_image": _jsonable(d.get("points_image"))}
                                     for d in cfg.get("multi_lane_definitions", [])]}
        return self._with_status(self.action(summary, "define_lane_regions"))

    def t_analyze_lanes(self, a):
        btn = "btn_process_std" if a.get("as") == "standard" else "btn_analyze_sample"
        r = self.t_click_control({"control_id": btn})
        if r.get("status") == "done":
            r["note"] = "If no dialog opened, check status_bar / define regions first."
        return r

    def t_wait(self, a):
        ms = int(a.get("ms", 1000))
        deadline = time.time() + ms / 1000.0
        while time.time() < deadline:
            if self._inflight and all(j.event.is_set() for j in self._inflight.values()):
                break
            time.sleep(0.1)
        bg = self._collect_background()
        res = {"waited_ms": ms, "finished": bg,
               "still_running": [{"job": j.jid, "action": j.name} for j in self._inflight.values()]}
        try:
            res["windows"] = self.gui(self._windows_info, timeout=5)
        except ToolError:
            res["windows"] = "app busy"
        return res

    # ---- helpers ----------------------------------------------------------------------
    def _settle(self, res, max_s=2.5):
        """After a dialog was answered, let the action that was waiting on it continue and
        report what happened next: it finished, or it opened another dialog."""
        before = set()
        try:
            before = {d["window_id"] for d in self.gui(self._modal_probe, timeout=3)}
        except ToolError:
            pass
        waiting = [j for j in self._inflight.values() if not j.event.is_set()]
        deadline = time.time() + (max_s if waiting else 0.3)
        while time.time() < deadline:
            time.sleep(0.12)
            if waiting and all(j.event.is_set() for j in waiting):
                break
            try:
                modal = [d for d in self.gui(self._modal_probe, timeout=2) if d["window_id"] not in before]
            except ToolError:
                continue
            if modal:
                res["next_dialog"] = modal[0]
                res["note"] = "Another dialog opened — answer it next."
                break
        bg = self._collect_background()
        if bg:
            res["finished_in_background"] = bg

    def _with_status(self, res):
        try:
            def snap():
                win = self.win
                d = {"status_bar": win.statusBar().currentMessage()}
                ms = getattr(win, "mode_status_label", None)
                if ms is not None and ms.text():
                    d["armed_tool"] = ms.text()
                return d
            res.update(self.gui(snap, timeout=3))
        except ToolError:
            pass
        return res


# =============================================================================================
# HTTP / MCP endpoint
# =============================================================================================
_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def openapi_spec(bridge):
    """OpenAPI 3.1 description of every tool, for OpenAPI tool servers (Open WebUI and other
    Ollama / OpenAI-function-calling front-ends that don't speak MCP)."""
    paths = {}
    for t in S.TOOLS:
        schema = dict(t["inputSchema"])
        schema.setdefault("type", "object")
        paths["/tools/" + t["name"]] = {"post": {
            "operationId": t["name"],
            "summary": t["description"].split(". ")[0][:120],
            "description": t["description"],
            "requestBody": {"required": False,
                            "content": {"application/json": {"schema": schema}}},
            "responses": {"200": {"description": "Tool result",
                                  "content": {"application/json": {"schema": {"type": "object"}}}}},
        }}
    return {
        "openapi": "3.1.0",
        "info": {"title": S.SERVER_TITLE + " tools", "version": "1.0.0",
                 "description": S.SERVER_INSTRUCTIONS},
        "servers": [{"url": "http://127.0.0.1:%d" % (bridge.port or S.DEFAULT_PORT)}],
        "paths": paths,
        "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}},
        "security": [{"bearer": []}],
    }


def _openapi_result(content, is_err, include_image):
    out = {"ok": not is_err}
    for c in content:
        if c["type"] == "text":
            try:
                out["result"] = json.loads(c["text"])
            except ValueError:
                out["result"] = c["text"]
        elif c["type"] == "image":
            if include_image:
                out["image"] = "data:%s;base64,%s" % (c["mimeType"], c["data"])
            else:
                # Text-only models: a base64 image would flood the context window. Keep a
                # copy on disk and say where it is instead.
                try:
                    d = os.path.join(S.state_dir(), "screenshots")
                    os.makedirs(d, exist_ok=True)
                    p = os.path.join(d, "shot_%d.png" % int(time.time() * 1000))
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(c["data"]))
                    out["image_saved_to"] = p
                except Exception:
                    pass
                out["image_note"] = ("Image omitted for text-only tool calling; send "
                                     "\"_include_image\": true to receive it as a data URL.")
    return out


def _make_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GelBlotAnalyzerMCP/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):      # keep the console quiet
            pass

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and _LOCAL_ORIGIN.match(origin):
                return {"Access-Control-Allow-Origin": origin, "Vary": "Origin",
                        "Access-Control-Allow-Headers": "Authorization, Content-Type, X-GBA-Token, "
                                                        "Mcp-Session-Id, Mcp-Protocol-Version",
                        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                        "Access-Control-Expose-Headers": "Mcp-Session-Id"}
            return {}

        def _send(self, code, body=None, ctype="application/json", extra=None):
            data = b"" if body is None else (body if isinstance(body, bytes) else
                                              json.dumps(body).encode("utf-8"))
            self.send_response(code)
            if data:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            hdrs = dict(self._cors())
            hdrs.update(extra or {})
            for k, v in hdrs.items():
                self.send_header(k, v)
            self.end_headers()
            if data:
                self.wfile.write(data)

        def _origin_ok(self):
            origin = self.headers.get("Origin")
            return not origin or bool(_LOCAL_ORIGIN.match(origin))

        def _authorized(self):
            if not self._origin_ok():
                return False
            tok = bridge.settings.get("token", "")
            auth = self.headers.get("Authorization", "")
            given = ""
            if auth.lower().startswith("bearer "):
                given = auth[7:].strip()
            given = given or self.headers.get("X-GBA-Token", "")
            if not given:
                q = parse_qs(urlparse(self.path).query)
                given = (q.get("token") or [""])[0]
            return bool(tok) and secrets.compare_digest(given, tok)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            return json.loads(raw) if raw.strip() else None

        def do_OPTIONS(self):            # CORS preflight (browser-based tool servers)
            if not self._origin_ok():
                return self._send(403)
            return self._send(204)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/health":
                if not self._authorized():
                    return self._send(401, {"error": "unauthorized"})
                return self._send(200, {"app": S.SERVER_TITLE, "ok": True, "pid": os.getpid(),
                                        "paused": bridge.paused})
            if p == "/openapi.json":
                if not self._origin_ok():
                    return self._send(403, {"error": "forbidden origin"})
                return self._send(200, openapi_spec(bridge))
            if p == "/mcp":
                return self._send(405, {"error": "SSE stream not offered; POST JSON-RPC to /mcp."},
                                  extra={"Allow": "POST"})
            return self._send(404, {"error": "not found"})

        def do_DELETE(self):
            return self._send(200 if self._authorized() else 401)

        def do_POST(self):
            path = urlparse(self.path).path
            if path.startswith("/tools/"):
                return self._post_tool(path[len("/tools/"):])
            if path != "/mcp":
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32001, "message": "Missing or wrong token. "
                                                  "Copy the connection details from Settings > "
                                                  "AI Assistant Control."}})
            try:
                msg = self._read_json()
            except Exception:
                return self._send(400, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32700, "message": "Parse error"}})
            ua = self.headers.get("User-Agent", "")
            if ua:
                bridge.last_client = ua[:80]
            if isinstance(msg, list):
                out = [r for r in (handle_rpc(bridge, m) for m in msg) if r is not None]
                return self._send(200, out) if out else self._send(202)
            resp = handle_rpc(bridge, msg)
            if resp is None:
                return self._send(202)
            return self._send(200, resp, extra={"Mcp-Session-Id": "gba-%d" % os.getpid()})

        def _post_tool(self, name):
            """OpenAPI-style call: POST /tools/<name> with the arguments as the JSON body."""
            if not self._authorized():
                return self._send(401, {"ok": False, "error": "Missing or wrong token (Bearer)."})
            try:
                args = self._read_json() or {}
                if not isinstance(args, dict):
                    raise ValueError
            except Exception:
                return self._send(400, {"ok": False, "error": "Body must be a JSON object."})
            include_image = bool(args.pop("_include_image", False))
            if not bridge.last_client:
                bridge.last_client = "OpenAPI client"
            content, is_err = bridge.call_tool(name, args)
            return self._send(200, _openapi_result(content, is_err, include_image))
    return Handler


def handle_rpc(bridge, msg):
    """Shared MCP JSON-RPC handler (used by the HTTP endpoint)."""
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if mid is None:                       # notification
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    if method == "initialize":
        pv = params.get("protocolVersion")
        info = params.get("clientInfo") or {}
        if info.get("name"):
            bridge.last_client = "%s %s" % (info.get("name"), info.get("version", ""))
            bridge.dispatcher.log.emit("info", "Client connected: %s" % bridge.last_client)
        return ok({"protocolVersion": pv if pv in S.SUPPORTED_PROTOCOL_VERSIONS else S.LATEST_PROTOCOL_VERSION,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": S.SERVER_NAME, "title": S.SERVER_TITLE, "version": "1.0.0"},
                   "instructions": S.SERVER_INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": S.public_tools()})
    if method == "tools/call":
        content, is_err = bridge.call_tool(params.get("name"), params.get("arguments") or {})
        return ok({"content": content, "isError": bool(is_err)})
    if method in ("resources/list",):
        return ok({"resources": []})
    if method in ("prompts/list",):
        return ok({"prompts": []})
    if method == "resources/templates/list":
        return ok({"resourceTemplates": []})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found: %s" % method}}


# =============================================================================================
# Connecting AI apps (plug and play)
# =============================================================================================
def _is_msix():
    """True when running from a Microsoft Store / MSIX package (installed under the protected,
    versioned WindowsApps folder that other programs can't launch by path)."""
    return sys.platform == "win32" and "\\windowsapps\\" in os.path.abspath(sys.executable).lower()


def stdio_launch_spec():
    """(command, args) an AI app should run to reach this installation over stdio MCP."""
    if getattr(sys, "frozen", False):
        if _is_msix():
            alias = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps",
                                 S.APP_ALIAS + ".exe")
            return (alias if os.path.exists(alias) else S.APP_ALIAS), ["--mcp-stdio"]
        return sys.executable, ["--mcp-stdio"]
    here = os.path.dirname(os.path.abspath(__file__))
    return sys.executable, [os.path.join(here, "gba_mcp_server.py")]


def _client_targets():
    """Known AI apps with a JSON `mcpServers` config: (name, config path, detected?)."""
    home = os.path.expanduser("~")
    out = []
    if sys.platform == "darwin":
        claude = os.path.join(home, "Library", "Application Support", "Claude",
                              "claude_desktop_config.json")
    elif sys.platform == "win32":
        claude = os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude_desktop_config.json")
        # Claude Desktop installed from an MSIX package keeps its config in its own container
        import glob
        pk = glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Packages", "Claude_*",
                                    "LocalCache", "Roaming", "Claude"))
        if pk and not os.path.isdir(os.path.dirname(claude)):
            claude = os.path.join(pk[0], "claude_desktop_config.json")
    else:
        claude = os.path.join(home, ".config", "Claude", "claude_desktop_config.json")
    out.append(("Claude Desktop", claude, os.path.isdir(os.path.dirname(claude))))
    lm = os.path.join(home, ".lmstudio", "mcp.json")
    out.append(("LM Studio", lm, os.path.isdir(os.path.dirname(lm))))
    cur = os.path.join(home, ".cursor", "mcp.json")
    out.append(("Cursor", cur, os.path.isdir(os.path.dirname(cur))))
    return out


def server_entry():
    cmd, args = stdio_launch_spec()
    return {"command": cmd, "args": args}


def install_client_config(path):
    """Add/refresh our entry in a client's mcpServers JSON. Keeps everything else, writes a
    one-time backup, and never overwrites a file it can't parse. Returns a message."""
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            data = json.loads(txt) if txt.strip() else {}
        except Exception as e:
            raise ToolError("%s is not valid JSON (%s). It was left untouched — fix it or add the "
                            "entry by hand." % (path, e))
        if not isinstance(data, dict):
            raise ToolError("%s has an unexpected format; left untouched." % path)
        bak = path + ".bak-gel-blot-analyzer"
        if not os.path.exists(bak):
            try:
                with open(bak, "w", encoding="utf-8") as f:
                    f.write(txt)
            except Exception:
                pass
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[S.SERVER_NAME] = server_entry()
    data["mcpServers"] = servers
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-gba"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


def _quote(s):
    return '"%s"' % s if (" " in s or "\\" in s) else s


def claude_code_command():
    cmd, args = stdio_launch_spec()
    return "claude mcp add %s -- %s" % (S.SERVER_NAME, " ".join(_quote(x) for x in [cmd] + args))


def other_clients_text(bridge):
    port = bridge.port or int(bridge.settings.get("port") or S.DEFAULT_PORT)
    tok = bridge.settings.get("token", "")
    return (
        "GEL BLOT ANALYZER — AI CONNECTION DETAILS\n"
        "(AI Assistant Control must be ON in Settings.)\n\n"
        "1) Apps that launch an MCP server (LM Studio, Claude Desktop, Cursor, mcphost, ollmcp,\n"
        "   Continue, Cline, ...) — add this to the app's mcpServers config:\n\n"
        + json.dumps({"mcpServers": {S.SERVER_NAME: server_entry()}}, indent=2) +
        "\n\n2) Apps that connect to an MCP URL (Claude Code, VS Code, LM Studio, ...):\n"
        "   URL:    http://127.0.0.1:%d/mcp\n"
        "   Header: Authorization: Bearer %s\n\n"
        "3) Ollama with Open WebUI (or any OpenAPI tool-server client):\n"
        "   Settings > Tools > Add tool server\n"
        "   URL:    http://127.0.0.1:%d          (spec: /openapi.json)\n"
        "   Auth:   Bearer  %s\n"
        "   Use a model that supports tool calling (e.g. Qwen 3, Llama 3.1+, Mistral Small).\n"
        % (port, tok, port, tok))


# =============================================================================================
# Settings tab group + activity window
# =============================================================================================
SETTINGS_GROUP_OBJECT_NAME = "GBA_AISettingsGroup"


class AIActivityWindow(QDialog):
    """Live log of what the AI is doing, with Pause. Single reused instance (never re-created,
    so it can't leak). Its controls are never reachable by the AI."""

    def __init__(self, bridge):
        super().__init__(bridge.win)
        self.bridge = bridge
        self.setObjectName(PANEL_OBJECT_NAME)
        self.setWindowTitle("AI Activity")
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        self.state_lbl = QLabel()
        self.state_lbl.setWordWrap(True)
        self.state_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.state_lbl)
        self.pause_cb = QCheckBox("Pause the AI (its requests are refused until unticked)")
        self.pause_cb.setChecked(bridge.paused)
        self.pause_cb.toggled.connect(bridge.set_paused)
        lay.addWidget(self.pause_cb)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        f2 = QFont("Menlo" if sys.platform == "darwin" else "Consolas")
        f2.setStyleHint(QFont.Monospace)
        self.log.setFont(f2)
        lay.addWidget(self.log, 1)
        for level, line in bridge._log_lines:
            self.append_log(level, line)
        row = QHBoxLayout()
        sb = QPushButton("AI Settings…")
        sb.clicked.connect(bridge.show_settings)
        row.addWidget(sb)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.hide)
        row.addWidget(bb)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        self.state_lbl.setText(self.bridge.state_html())
        self.pause_cb.blockSignals(True)
        self.pause_cb.setChecked(self.bridge.paused)
        self.pause_cb.blockSignals(False)

    def append_log(self, level, line):
        color = {"error": "#DC2626", "warn": "#B45309", "ok": "#15803D",
                 "call": "#7C3AED"}.get(level, "")
        esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.log.appendHtml("<span style='color:%s'>%s</span>" % (color, esc) if color else esc)


def build_settings_group(bridge):
    """The 'AI Assistant Control (MCP)' group shown in the Settings tab. Changes apply
    immediately (like Autosave). OFF by default."""
    st = bridge.settings
    g = QGroupBox("AI Assistant Control (MCP)")
    g.setObjectName(SETTINGS_GROUP_OBJECT_NAME)
    lay = QVBoxLayout(g)
    lay.setSpacing(8)

    w = {}
    w["enable"] = QCheckBox("Allow AI assistants to control Gel Blot Analyzer")
    w["enable"].setChecked(bool(st.get("enabled")))
    w["enable"].setToolTip(
        "Starts a private local server (this computer only, token-protected) that lets AI apps —\n"
        "Claude, LM Studio, Ollama front-ends, Cursor, VS Code... — operate the program through\n"
        "the Model Context Protocol. Everything the AI does is shown live. Off by default.")
    lay.addWidget(w["enable"])
    w["status"] = QLabel()
    w["status"].setWordWrap(True)
    w["status"].setTextInteractionFlags(Qt.TextSelectableByMouse)
    lay.addWidget(w["status"])

    form = QFormLayout()
    w["perm"] = QComboBox()
    w["perm"].addItems(["Full control", "View only (screenshots and reading results)"])
    w["perm"].setCurrentIndex(1 if st.get("permission") == "view" else 0)
    form.addRow("Permission:", w["perm"])
    w["delay"] = QSpinBox()
    w["delay"].setRange(0, 3000)
    w["delay"].setSingleStep(50)
    w["delay"].setSuffix(" ms")
    w["delay"].setValue(bridge.step_delay())
    w["delay"].setToolTip("Pause before each click or change so you can follow along. 0 = full speed.")
    form.addRow("Step delay:", w["delay"])
    lay.addLayout(form)
    w["overlays"] = QCheckBox("Show the AI cursor and control highlights (live preview)")
    w["overlays"].setChecked(bridge.overlays_on())
    lay.addWidget(w["overlays"])
    w["pause"] = QCheckBox("Pause the AI")
    w["pause"].setChecked(bridge.paused)
    lay.addWidget(w["pause"])

    lay.addWidget(QLabel("<b>Connect an AI app</b> (one click — then restart that app):"))
    row1 = QHBoxLayout()
    w["connect"] = {}
    for name, path, found in _client_targets():
        b = QPushButton(name)
        b.setToolTip(("Add Gel Blot Analyzer to %s's MCP settings:\n%s" % (name, path)) if found
                     else "%s was not found on this computer — the settings will be copied to "
                          "the clipboard instead." % name)
        b.clicked.connect(lambda _=False, n=name, p=path, f=found: bridge.connect_client(n, p, f))
        row1.addWidget(b)
        w["connect"][name] = b
    lay.addLayout(row1)
    row2 = QHBoxLayout()
    cc = QPushButton("Claude Code")
    cc.setToolTip("Copy the 'claude mcp add' command to the clipboard.")
    cc.clicked.connect(bridge.copy_claude_code)
    row2.addWidget(cc)
    other = QPushButton("Ollama / other apps…")
    other.setToolTip("Connection details for any other MCP app, and the OpenAPI tool-server URL "
                     "for Ollama + Open WebUI.")
    other.clicked.connect(bridge.show_other_clients)
    row2.addWidget(other)
    lay.addLayout(row2)
    row3 = QHBoxLayout()
    act = QPushButton("Activity Log…")
    act.clicked.connect(bridge.show_activity)
    row3.addWidget(act)
    tok = QPushButton("New Token")
    tok.setToolTip("Revoke access for every connected AI app at once (URL-based apps must be "
                   "reconnected; one-click connections keep working).")
    tok.clicked.connect(bridge.ask_regenerate_token)
    row3.addWidget(tok)
    row3.addStretch(1)
    lay.addLayout(row3)
    note = QLabel("The AI works in this window: a violet AI cursor shows its clicks and every "
                  "control it changes is outlined first. See MCP_SETUP_GUIDE for details.")
    note.setWordWrap(True)
    note.setStyleSheet("color: palette(mid);")
    lay.addWidget(note)

    w["enable"].toggled.connect(bridge.set_enabled)
    w["perm"].currentIndexChanged.connect(
        lambda i: bridge.set_setting("permission", "view" if i == 1 else "full"))
    w["delay"].valueChanged.connect(lambda v: bridge.set_setting("step_delay_ms", int(v)))
    w["overlays"].toggled.connect(lambda v: bridge.set_setting("show_overlays", bool(v)))
    w["pause"].toggled.connect(bridge.set_paused)
    bridge.settings_widgets = w
    bridge.settings_group = g
    bridge._refresh_indicator()
    return g


# =============================================================================================
# Installation into the main window
# =============================================================================================
def install(win):
    """Create the bridge, put its group in the Settings tab, add the Tools menu entries and the
    status-bar indicator. The server starts only if the USER enabled AI control in Settings."""
    bridge = AIBridge(win)
    win.ai_bridge = bridge

    container = getattr(win, "ai_settings_container", None)
    if container is not None:
        try:
            container.addWidget(build_settings_group(bridge))
        except Exception:
            traceback.print_exc()

    act = QAction("AI Assistant Control (MCP)…", win)
    act.setToolTip("Let an AI assistant operate the app (Model Context Protocol), with live preview.")
    act.triggered.connect(bridge.show_settings)
    win.ai_control_action = act
    act2 = QAction("AI Activity Log…", win)
    act2.triggered.connect(bridge.show_activity)
    win.ai_activity_action = act2
    try:
        for top in win.menuBar().actions():
            if top.text().replace("&", "") == "Tools" and top.menu() is not None:
                top.menu().addSeparator()
                top.menu().addAction(act)
                top.menu().addAction(act2)
                break
    except Exception:
        pass

    lab = QLabel("")
    lab.setStyleSheet("color:#7C3AED; font-weight:600; padding:0 6px;")
    lab.setToolTip("AI assistant control — Tools > AI Activity Log… for details.")
    lab.setVisible(False)
    try:
        win.statusBar().addPermanentWidget(lab)
    except Exception:
        pass
    bridge._status_label = lab

    if bridge.settings.get("enabled"):
        bridge.start()
    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(bridge.stop)
    return bridge


# =============================================================================================
# small helpers
# =============================================================================================
def _in_ai_settings(w):
    cur = w
    while cur is not None:
        if cur.objectName() == "GBA_AISettingsGroup":
            return True
        cur = cur.parentWidget()
    return False


def _check_user_only(w):
    if w is not None and w.property("gbaUserOnly"):
        raise ToolError("This confirmation is reserved for the user — wait for them to answer it.")


def _text(obj):
    return {"type": "text", "text": json.dumps(obj, indent=1, ensure_ascii=False, default=str)}


def _plain(s):
    s = re.sub(r"<br\s*/?>", "\n", str(s or ""), flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return s.strip()


def _btn_text(b):
    return (b.text() or b.toolTip().split("\n")[0] or b.accessibleName() or "").replace("&", "").strip()


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s or "").replace("&", "").lower())


def _act_name(act):
    return re.sub(r"\s*[▶…]+\s*$", "", (act.text() or "").replace("&", "")).replace("...", "").strip()


def _r(v):
    try:
        return round(float(v), 2)
    except Exception:
        return v


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, QColor):
            return v.name()
        try:
            return float(v)
        except Exception:
            return str(v)


def _safe_text(w):
    try:
        return w.text() if w is not None else None
    except Exception:
        return None


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _alive(bridge, cid):
    try:
        bridge.gui(lambda: bridge._resolve_control(cid) and True, timeout=3)
        return True
    except ToolError:
        return False


def _app_global(win, name, default=None):
    try:
        return win.update_live_view.__globals__.get(name, default)
    except Exception:
        return default


def _fmt_name(img):
    try:
        return img.format().name
    except Exception:
        return str(img.format())


def _bit_depth(img):
    f = img.format()
    if f in (QImage.Format_Grayscale16, QImage.Format_RGBA64, QImage.Format_RGBX64,
             QImage.Format_RGBA64_Premultiplied):
        return 16
    return 8


def _fit(img, max_size):
    if max(img.width(), img.height()) <= max_size:
        return img.convertToFormat(QImage.Format_ARGB32) if img.format() != QImage.Format_ARGB32 else img
    return img.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _png(img):
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba.data())


def _draw_grid(img, k, iw, ih, step=None):
    """Coordinate grid in IMAGE px over a (scaled) render; labels on the top/left edges."""
    out = QImage(img.size(), QImage.Format_ARGB32)
    out.fill(Qt.white)
    p = QPainter(out)
    p.drawImage(0, 0, img)
    if not step:
        target = max(iw, ih) / 10.0
        for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
            step = s
            if s >= target:
                break
    step = max(1, int(step))
    f = QFont()
    f.setPixelSize(max(9, min(14, int(out.width() / 70))))
    f.setBold(True)
    p.setFont(f)
    fm = QFontMetrics(f)
    pen = QPen(QColor(255, 0, 170, 150), 1)
    for axis in ("x", "y"):
        lim = iw if axis == "x" else ih
        v = 0
        while v <= lim:
            pos = v * k
            p.setPen(pen)
            if axis == "x":
                p.drawLine(QPointF(pos, 0), QPointF(pos, out.height()))
            else:
                p.drawLine(QPointF(0, pos), QPointF(out.width(), pos))
            if v > 0:
                t = str(v)
                w = fm.horizontalAdvance(t) + 4
                h = fm.height()
                r = QRectF(pos + 2, 1, w, h) if axis == "x" else QRectF(1, pos + 1, w, h)
                p.fillRect(r, QColor(255, 255, 255, 215))
                p.setPen(QColor(190, 0, 120))
                p.drawText(r, Qt.AlignCenter, t)
            v += step
    p.end()
    return out


def _kind(w):
    if isinstance(w, QRadioButton):
        return "radio"
    if isinstance(w, QCheckBox):
        return "checkbox"
    if isinstance(w, QAbstractButton):
        return "toggle_button" if w.isCheckable() else "button"
    if isinstance(w, QAbstractSlider):
        return "slider"
    if isinstance(w, QDoubleSpinBox):
        return "number"
    if isinstance(w, QSpinBox):
        return "integer"
    if isinstance(w, QAbstractSpinBox):
        return "spinbox"
    if isinstance(w, QComboBox):
        return "combo"
    if isinstance(w, QLineEdit):
        return "text"
    if isinstance(w, (QTextEdit, QPlainTextEdit)):
        return "multiline_text"
    if isinstance(w, QTabWidget):
        return "tabs"
    if isinstance(w, QListWidget):
        return "list"
    if isinstance(w, QTableView):
        return "table"
    if isinstance(w, QGroupBox):
        return "checkable_group"
    if isinstance(w, QLabel):
        return "label"
    return type(w).__name__


def _value_of(w):
    d = {}
    try:
        if isinstance(w, QAbstractButton):
            if w.isCheckable():
                d["checked"] = w.isChecked()
        elif isinstance(w, QAbstractSlider):
            d.update(value=w.value(), min=w.minimum(), max=w.maximum())
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            d.update(value=w.value(), min=w.minimum(), max=w.maximum())
            if w.suffix():
                d["suffix"] = w.suffix().strip()
        elif isinstance(w, QAbstractSpinBox):
            d["value"] = w.text()
        elif isinstance(w, QComboBox):
            items = [w.itemText(i) for i in range(w.count())]
            d.update(value=w.currentText(), options=items[:60])
            if len(items) > 60:
                d["options_truncated"] = len(items)
            if w.isEditable():
                d["editable"] = True
        elif isinstance(w, QLineEdit):
            d["value"] = w.text()
            if w.placeholderText():
                d["placeholder"] = w.placeholderText()
            if w.isReadOnly():
                d["read_only"] = True
        elif isinstance(w, (QTextEdit, QPlainTextEdit)):
            t = w.toPlainText()
            d["value"] = t[:800] + ("…" if len(t) > 800 else "")
            if w.isReadOnly():
                d["read_only"] = True
        elif isinstance(w, QTabWidget):
            d.update(value=w.tabText(w.currentIndex()).replace("&", ""),
                     options=[w.tabText(i).replace("&", "") for i in range(w.count())])
        elif isinstance(w, QListWidget):
            items = [w.item(i).text() for i in range(min(w.count(), 60))]
            cur = w.currentItem()
            d.update(value=cur.text() if cur is not None else None, options=items)
        elif isinstance(w, QTableView):
            m = w.model()
            if m is not None:
                d.update(rows=m.rowCount(), columns=m.columnCount(),
                         note="use read_table")
        elif isinstance(w, QGroupBox):
            d["checked"] = w.isChecked()
        elif isinstance(w, QLabel):
            d["text"] = _plain(w.text())[:400]
    except RuntimeError:
        pass
    return d


def _to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "checked")
    return bool(v)


def _set_value(w, value):
    if isinstance(w, QAbstractButton):
        if w.isCheckable():
            want = _to_bool(value)
            if isinstance(w, QRadioButton):
                if want and not w.isChecked():
                    w.click()
            elif w.isChecked() != want:
                w.click()
            return {"checked": w.isChecked()}
        w.click()
        return {"clicked": True}
    if isinstance(w, QGroupBox):
        w.setChecked(_to_bool(value))
        return {"checked": w.isChecked()}
    if isinstance(w, QAbstractSlider):
        v = int(round(float(value)))
        if v < w.minimum() or v > w.maximum():
            raise ToolError("Value %s outside %d..%d" % (value, w.minimum(), w.maximum()))
        w.setSliderDown(True)
        w.setValue(v)
        w.setSliderDown(False)          # emits sliderReleased like a real drag
        return {"value": w.value()}
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        v = float(value)
        if v < w.minimum() - 1e-9 or v > w.maximum() + 1e-9:
            raise ToolError("Value %s outside %s..%s" % (value, w.minimum(), w.maximum()))
        w.setValue(int(round(v)) if isinstance(w, QSpinBox) else v)
        w.editingFinished.emit()
        return {"value": w.value()}
    if isinstance(w, QComboBox):
        idx = -1
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            idx = int(value)
        else:
            s = str(value)
            idx = w.findText(s, Qt.MatchFixedString)
            if idx < 0:
                for i in range(w.count()):
                    if s.lower() in w.itemText(i).lower():
                        idx = i
                        break
        if idx < 0 or idx >= w.count():
            if w.isEditable():
                w.setEditText(str(value))
                return {"value": w.currentText()}
            raise ToolError("'%s' is not an option. Options: %s"
                            % (value, [w.itemText(i) for i in range(w.count())][:40]))
        w.setCurrentIndex(idx)
        w.activated.emit(idx)
        w.textActivated.emit(w.itemText(idx))
        return {"value": w.currentText()}
    if isinstance(w, QLineEdit):
        if w.isReadOnly():
            raise ToolError("This field is read-only.")
        w.setText(str(value))
        w.textEdited.emit(w.text())
        w.editingFinished.emit()
        return {"value": w.text()}
    if isinstance(w, (QTextEdit, QPlainTextEdit)):
        if w.isReadOnly():
            raise ToolError("This field is read-only.")
        w.setPlainText(str(value))
        return {"value": w.toPlainText()[:200]}
    if isinstance(w, QTabWidget):
        s = str(value).lower()
        for i in range(w.count()):
            if s in w.tabText(i).replace("&", "").lower():
                w.setCurrentIndex(i)
                return {"value": w.tabText(i)}
        raise ToolError("No tab '%s'." % value)
    if isinstance(w, QListWidget):
        s = str(value).lower()
        for i in range(w.count()):
            if w.item(i).text().lower() == s or s in w.item(i).text().lower():
                w.setCurrentRow(i)
                w.itemClicked.emit(w.item(i))
                return {"value": w.item(i).text()}
        raise ToolError("No item '%s'." % value)
    raise ToolError("Controls of kind '%s' cannot be set." % _kind(w))


def _summarize_args(args):
    parts = []
    for k, v in (args or {}).items():
        s = json.dumps(v, ensure_ascii=False, default=str)
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append("%s=%s" % (k, s))
    return ", ".join(parts)
