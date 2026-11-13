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
                               QPlainTextEdit, QProgressDialog, QPushButton, QRadioButton, QScrollArea,
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

import numpy as np

import gba_mcp_schema as S
import gba_ai_gel as GEL

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


def _read_settings_file():
    try:
        with open(S.settings_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _merge_runtime_keys(data):
    """Return a copy of ``data`` whose active_port/pid never clobber ANOTHER live instance.

    Two app instances can overlap (the stdio relay launches a fresh copy while the user is
    quitting the old one). Each one keeps its own in-memory settings dict, so the old
    instance's stop() used to write back its stale copy — with active_port/pid popped —
    and erase the port the new instance had just recorded. The relay then polled a dead
    port forever ("did not become ready within 90s").

    Rule: the runtime keys are written only by the process they describe. If this process is
    not advertising itself (pid != os.getpid()), whatever another *live* process recorded on
    disk is preserved; a record left by a dead process (crash) or by this process is cleared.
    """
    out = dict(data)
    me = os.getpid()
    if out.get("pid") == me and out.get("active_port"):
        return out                                   # we are the running server: ours wins
    out.pop("active_port", None)
    out.pop("pid", None)
    disk = _read_settings_file()
    dpid, dport = disk.get("pid"), disk.get("active_port")
    if dport and dpid and dpid != me and S.pid_alive(dpid):
        out["active_port"] = dport
        out["pid"] = dpid
    return out


def save_settings(data):
    path = S.settings_path()
    try:
        data = _merge_runtime_keys(data)
    except Exception:
        pass
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


def _median3(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    return np.median(np.stack([np.r_[x[:1], x[:-1]], x, np.r_[x[1:], x[-1:]]]), axis=0)


class _Pilot(QObject):
    """Answers the app's own dialogs on behalf of a high-level AI tool (GUI thread).

    handlers: list of (match(widget) -> bool, handle(widget) -> None | "release").
    A matched top-level window is claimed once; its handler runs from a single-shot timer
    so a handler that opens a nested modal loop (wizard -> densitometry dialog) never
    blocks the polling. "release" hands the window back to the AI (it is then reported
    as dialog_open)."""

    def __init__(self, bridge, handlers, timeout_s=600):
        super().__init__(bridge.win)
        self.bridge = bridge
        self.handlers = list(handlers)
        self.claimed, self.released = set(), set()
        self.report, self.errors = {}, []
        self.t_end = time.time() + timeout_s
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._tick)

    def start(self):
        self.bridge._pilots.append(self)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        if self in self.bridge._pilots:
            self.bridge._pilots.remove(self)

    def owns(self, w):
        ptr = _cpp_ptr(w)
        if ptr in self.released:
            return False
        if ptr in self.claimed:
            return True
        for m, _ in self.handlers:
            try:
                if m(w):
                    return True
            except Exception:
                pass
        return False

    def _tick(self):
        if time.time() > self.t_end:
            self.stop()
            return
        for w in QApplication.topLevelWidgets():
            try:
                if not _is_valid(w) or not w.isVisible():
                    continue
                ptr = _cpp_ptr(w)
            except RuntimeError:
                continue
            if ptr in self.claimed or ptr in self.released:
                continue
            for m, fn in self.handlers:
                try:
                    hit = m(w)
                except Exception:
                    hit = False
                if hit:
                    self.claimed.add(ptr)
                    QTimer.singleShot(self.bridge.step_delay() + 200,
                                      lambda w=w, fn=fn, ptr=ptr: self._run(w, fn, ptr))
                    break

    def _run(self, w, fn, ptr):
        b = self.bridge
        b.ai_depth += 1
        try:
            if not _is_valid(w):
                return
            if fn(w) == "release":
                self.claimed.discard(ptr)
                self.released.add(ptr)
        except Exception as e:  # noqa: BLE001
            self.errors.append("%s: %s" % (type(e).__name__, e))
            b.dispatcher.log.emit("error", "   pilot: %s" % traceback.format_exc(limit=4))
        finally:
            b.ai_depth -= 1


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
        self._pilots = []                 # active _Pilot objects (dialogs answered by smart tools)
        self._last_inspection = None
        self._preset_warning = None
        self._last_ladder_lanes = set()
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
        if name == "LM Studio":
            # LM Studio loads mcp.json servers as plugins but enables them PER CHAT — without
            # this the model never receives the tools and claims it has no MCP access.
            extra += ("\n\nIn LM Studio, switch the tools on in each chat: click the plug "
                      "(Integrations) button next to the message box and enable "
                      "'mcp/gel-blot-analyzer'. Use a model with the Tool use badge.")
        self._info("Connect %s" % name,
                   "Gel Blot Analyzer was added to %s.\n\nRestart %s, then ask it something like "
                   '"Use Gel Blot Analyzer to show me the open image".%s' % (name, name, extra))

    def copy_claude_code(self):
        QGuiApplication.clipboard().setText(claude_code_command())
        self._info("Claude Code", "This command was copied to the clipboard — paste it into a "
                                  "terminal:\n\n" + claude_code_command())

    def show_other_clients(self):
        dlg = QDialog(self.win)
        dlg.setStyleSheet(
                    "background-color: black; color: white;"
                )
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
        """Modal dialogs the AI must answer. Progress dialogs (they close by themselves) and
        dialogs a smart tool's _Pilot is answering are not reported."""
        out = []
        for w in self._top_windows():
            try:
                if w is self.win or isinstance(w, QProgressDialog):
                    continue
                if any(p.owns(w) for p in list(self._pilots)):
                    continue
                info = self._window_info(w)
            except RuntimeError:
                continue
            if info.get("modal"):
                out.append(info)
        return out

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
        if name in _SMART_TOOLS:
            args = _clean_smart_args(args)
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
            anchor = str(a.get("anchor") or "center").lower()
            if anchor not in ("center", "left", "right"):
                raise ToolError("anchor must be center, left or right.")
            win.custom_markers = getattr(win, "custom_markers", [])
            rec = [float(a["x"]), float(a["y"]), str(a["text"]), col, font, size,
                   bool(a.get("bold")), bool(a.get("italic"))]
            if anchor != "center":
                rec.append(anchor)      # the 9th element exists only for left/right anchors
            win.custom_markers.append(rec)
            win.save_state()            # AFTER the change (undo restores the entry below)
            win.is_modified = True
            win.update_live_view()
            self._point_at(a["x"], a["y"])
            return {"index": len(win.custom_markers) - 1, "font": font, "font_size": size,
                    "anchor": anchor}
        return self._with_status(self.action(run, "add_text"))

    def t_add_table(self, a):
        def run():
            win = self.win
            if not win._has_image():
                raise ToolError("No image is loaded.")
            cells = a.get("cells")
            if (not isinstance(cells, list) or not cells
                    or not all(isinstance(r, list) for r in cells)):
                raise ToolError("cells must be a list of rows, each a list of strings, "
                                "e.g. [[\"IL-2\", \"-\", \"+\"], [\"AAT\", \"+\", \"-\"]].")
            if len(cells) > 60 or max(len(r) for r in cells) > 60:
                raise ToolError("A table can have at most 60 rows and 60 columns.")
            cells = [["" if c is None else str(c) for c in r] for r in cells]
            st = {"cells": cells}
            for key, typ in (("font", str), ("font_size", int),
                             ("thickness", float), ("first_lane", int), ("auto_pad", bool)):
                if a.get(key) is not None:
                    st[key] = typ(a[key])
            nr = len(cells); nc = max(len(r) for r in cells)

            def _idx(key, n):
                v = a.get(key) or []
                if not isinstance(v, list):
                    raise ToolError("%s must be a list of 1-based numbers." % key)
                return {int(x) - 1 for x in v if 1 <= int(x) <= n}
            b_rows, b_cols = _idx("bold_rows", nr), _idx("bold_columns", nc)
            u_rows, i_rows = _idx("underline_rows", nr), _idx("italic_rows", nr)
            if a.get("bold_first_row"):
                b_rows.add(0)
            if a.get("bold_first_col"):
                b_cols.add(0)
            st["fmt"] = [[("b" if (a.get("bold") or r in b_rows or c in b_cols) else "")
                          + ("i" if (a.get("italic") or r in i_rows) else "")
                          + ("u" if r in u_rows else "")
                          for c in range(nc)] for r in range(nr)]
            spans = []
            for m in (a.get("merge") or []):
                try:
                    spans.append([int(m["row"]) - 1, int(m["col"]) - 1,
                                  int(m.get("rowspan", 1)), int(m.get("colspan", 1))])
                except Exception:
                    raise ToolError("Each merge entry needs row and col (1-based) plus rowspan and/or "
                                    "colspan, e.g. {\"row\": 1, \"col\": 2, \"colspan\": 3}.")
            if spans:
                st["spans"] = spans
            for key, dst in (("color", "text_color"), ("line_color", "color")):
                if a.get(key):
                    c = QColor(str(a[key]))
                    if not c.isValid():
                        raise ToolError("Invalid %s '%s'." % (key, a[key]))
                    st[dst] = c.name()
            if a.get("align") is not None:
                if a["align"] not in ("left", "center", "right"):
                    raise ToolError("align must be left, center or right.")
                st["align"] = a["align"]
            if a.get("outline") is not None:
                st["border"] = bool(a["outline"])
            if a.get("inner_lines") is not None:
                st["grid"] = bool(a["inner_lines"])
            if a.get("first_column_is_labels") is not None:
                st["label_col"] = bool(a["first_column_is_labels"])
            n_lanes = len(getattr(win, "top_markers", []) or [])
            mode = a.get("column_widths_mode")
            if mode is None:
                mode = "lanes" if n_lanes >= 2 else "fit"
            if mode not in ("fit", "equal", "lanes"):
                raise ToolError("column_widths_mode must be fit, equal or lanes.")
            note = None
            if mode == "lanes" and n_lanes < 2:
                mode = "fit"
                note = ("No top (lane) labels are placed, so the columns fit their text instead of "
                        "snapping to lanes. Place top labels first (place_markers side='top', or "
                        "auto_analyze_gel) to line the table up with the lanes.")
            st["width_mode"] = mode

            def _sizes(key):
                v = a.get(key)
                if v is None:
                    return None
                if not isinstance(v, list):
                    raise ToolError("%s must be a list of numbers (null = automatic)." % key)
                return [None if x is None else float(x) for x in v]
            cw, rh = _sizes("column_widths"), _sizes("row_heights")
            if cw is not None:
                st["col_w"] = cw
            if rh is not None:
                st["row_h"] = rh
            origin = None
            if a.get("x") is not None or a.get("y") is not None:
                origin = (a.get("x"), a.get("y"))
            info = win.add_table_from_settings(st, convert=bool(a.get("convert_to_items")),
                                               origin=origin)
            if note:
                info["note"] = note
            try:
                r = info.get("rect") or {}
                self._point_at(r.get("x", 0) + r.get("width", 0) / 2.0,
                               r.get("y", 0) + r.get("height", 0) / 2.0)
            except Exception:
                pass
            return info
        return self._with_status(self.action(run, "add_table"))

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
            if not isinstance(lane, dict):
                raise ToolError("lane %d must be an object: {x, y, width, height} or {corners: [[x, y] x4]}." % i)
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

    # ---- smart gel tools (v9.8.1) ----------------------------------------------------
    # High-level tools that do a whole job the way an expert would, so that even a small
    # local model can analyse a gel/blot correctly: understand the image first
    # (inspect_gel), fix band polarity, label the ladder with the app's own Auto Lane
    # analysis, run the Auto Gel wizard end-to-end, gate out noise bands, quantify with
    # standards, and combine a gel with a western blot. The app's own dialogs are driven by
    # a _Pilot, so the numbers come from exactly the same code a person would run.

    def _analysis_region(self, region=None):
        """(gray, max_val, (x0, y0), content_box, image_size) of the ANALYSIS image (what
        densitometry measures — inversion included), cropped to `region` or to the content
        inside the transparent padding. GUI thread."""
        win = self.win
        if not win._has_image():
            raise ToolError("No image is loaded. Call open_image first.")
        adj = win._get_fully_adjusted_image_for_analysis()
        if adj is None or adj.isNull():
            raise ToolError("Could not build the analysis image.")
        pil = win.convert_qimage_to_grayscale_pil(adj)
        if pil is None:
            raise ToolError("Could not convert the image to grayscale.")
        arr = np.asarray(pil, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        mode = getattr(pil, "mode", "L")
        max_val = 65535.0 if (mode.startswith("I") or arr.max() > 255) else 255.0
        H, W = arr.shape
        try:
            margins = win._measure_transparent_margins()
        except Exception:
            margins = (0, 0, 0, 0)
        cb = GEL.content_box(arr.shape, margins)
        if region:
            try:
                x, y = float(region["x"]), float(region["y"])
                w, h = float(region["width"]), float(region["height"])
            except (KeyError, TypeError, ValueError):
                raise ToolError("region needs x, y, width, height (image pixels).")
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        else:
            x0, y0, x1, y1 = cb
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ToolError("The region is too small or outside the image (%dx%d)." % (W, H))
        return arr[y0:y1, x0:x1].copy(), max_val, (x0, y0), cb, (W, H)

    def _image_key(self):
        def run():
            win = self.win
            img = win.image
            return (getattr(win, "_image_load_serial", 0), img.width() if img else 0,
                    img.height() if img else 0, bool(getattr(win, "main_image_is_inverted", False)),
                    len(getattr(win, "undo_stack", []) or []))
        return self.gui(run)

    def _ladder_count(self):
        def run():
            vals = [str(v).strip() for v in (getattr(self.win, "marker_values", []) or [])]
            return len([v for v in vals if v])
        return self.gui(run)

    def _inspect(self, a, remember_lanes=False):
        """Read the gel. With remember_lanes=True a call WITHOUT expected_lanes re-uses the
        lane count the user gave earlier for this same image and gel region, so that lane
        numbers stay the same across tools (inspect_gel(expected_lanes=10) followed, after
        some edit, by label_mw_ladder(lane=7) must still mean the 7th of 10 lanes)."""
        region = a.get("region")
        snr = float(a.get("min_snr") or GEL.DEFAULT_SNR_MIN)
        rel = float(a.get("min_relative_intensity") if a.get("min_relative_intensity") is not None
                    else GEL.DEFAULT_REL_MIN)
        gray, max_val, origin, cb, size = self.gui(lambda: self._analysis_region(region), timeout=60)
        ladder_n = a.get("ladder_bands") or self._ladder_count() or None
        expected = a.get("expected_lanes")
        memo_key = (self.gui(lambda: getattr(self.win, "_image_load_serial", 0)),
                    gray.shape[1], gray.shape[0], json.dumps(region, sort_keys=True, default=str))
        if expected:
            self._expected_lanes_memo = (memo_key, int(expected))
        elif remember_lanes:
            memo = getattr(self, "_expected_lanes_memo", None)
            if memo and memo[0] == memo_key:
                expected = memo[1]
        res = GEL.analyze(gray, max_val, origin=origin, gel_type=a.get("gel_type") or "auto",
                          expected_lanes=expected, ladder_count=ladder_n,
                          snr_min=snr, rel_min=rel)
        res["expected_lanes_used"] = int(expected) if expected else None
        res["image_size"] = {"width": size[0], "height": size[1]}
        res["content_box"] = {"x": cb[0], "y": cb[1], "width": cb[2] - cb[0], "height": cb[3] - cb[1]}
        reg = res["region"]
        res["suggested_lane_regions"] = [
            {"x": ln["x0"], "y": reg["y"], "width": max(2, ln["x1"] - ln["x0"]), "height": reg["height"]}
            for ln in res["lanes"]]
        self._last_inspection = {"key": self._image_key(), "region": region, "result": res}
        return res

    def _cached_inspection(self, a=None):
        a = a or {}
        c = getattr(self, "_last_inspection", None)
        if (c and c.get("key") == self._image_key() and c.get("region") == a.get("region")
                and not a.get("expected_lanes")):
            return c["result"]
        return self._inspect(a, remember_lanes=True)

    def t_inspect_gel(self, a):
        res = self._inspect(a)
        gel_type = (a.get("gel_type") or "auto").lower()
        pol = res["polarity"]["polarity"]
        advice, questions = [], []
        if pol == "light_bands":
            advice.append("Bands are LIGHT on a dark background (typical of DNA/RNA gels with "
                          "EtBr/SYBR, stain-free and fluorescent gels, and raw chemiluminescence). "
                          "Densitometry needs WHITE lanes with BLACK bands: call fix_band_polarity "
                          "first (the analysis tools also do this automatically).")
        elif pol == "ambiguous":
            questions.append("I can't tell whether the bands are dark or light. Are the bands "
                             "darker than the background (e.g. Coomassie) or brighter (e.g. "
                             "ethidium-bromide DNA gel / fluorescent blot)?")
        else:
            advice.append("Bands are dark on a light background — correct polarity for analysis.")
        if gel_type in ("dna", "rna", "agarose") and pol == "dark_bands":
            advice.append("DNA/RNA gel already shows dark bands on white — it has already been "
                          "inverted; do NOT invert again.")
        W = res["image_size"]["width"]
        lad = res["ladder_lanes"]
        if lad:
            for n in lad:
                ln = res["lanes"][n - 1]
                side = "left" if ln["center_x"] < res["region"]["x"] + res["region"]["width"] / 2.0 else "right"
                advice.append("Lane %d looks like the molecular-weight ladder (%d bands). Label it with "
                              "label_mw_ladder(lane=%d, side='%s', preset=...)." % (n, ln["n_bands"], n, side))
            lc = self._ladder_count()
            if lc and any(abs(res["lanes"][n - 1]["n_bands"] - lc) > 1 for n in lad):
                questions.append("The ladder lane shows %s bands but the selected marker preset has %d "
                                 "labels. Which ladder (marker preset) was loaded on this gel?"
                                 % ("/".join(str(res["lanes"][n - 1]["n_bands"]) for n in lad), lc))
        else:
            questions.append("I could not identify a molecular-weight ladder lane. Which lane number "
                             "(1-%d, left to right) is the ladder, and which ladder/marker preset was "
                             "used? (Answer 'none' if there is no ladder.)" % res["n_lanes"])
        for c in res["standard_series_candidates"]:
            if c["confidence"] < 0.5:
                continue
            questions.append(
                "Lanes %s have a band at the same position (y≈%d) and of similar size, but with "
                "different intensities (%s× relative to the weakest; %s). This looks like a "
                "dilution series of a standard. Are these standards with known amounts? If yes, "
                "tell me the amount loaded in each of these lanes (e.g. ng or µg) and I will "
                "quantify the other samples against them with quantify_with_standards."
                % (", ".join(str(x) for x in c["lanes"]), c["y"],
                   ":".join(str(r) for r in c["ratio_to_weakest"]), c["trend"]))
            break
        if res.get("lane_count_estimated"):
            questions.insert(0, "Lane detection was uncertain, so I estimated %d lanes from the "
                                "lane spacing. Is that right (counting ladder lanes)? If not, tell me "
                                "the number and I will use expected_lanes=<number>."
                             % res["lane_count_estimated"])
        if res.get("lane_warning"):
            questions.insert(0, "I found %d lanes but the %s, so some lanes may be split or merged. "
                                "How many lanes does this gel have (counting ladder lanes)? I will "
                                "re-run with expected_lanes=<that number>."
                             % (res["n_lanes"], res["lane_warning"]))
        if not a.get("expected_lanes") and not res.get("lane_count_estimated") and not res.get("lane_warning"):
            questions.append("I found %d lanes (counting ladder lanes). Is that the right number? "
                             "(If not, tell me and I will use expected_lanes=<number>.)" % res["n_lanes"])
        empty = [ln["lane"] for ln in res["lanes"] if ln["role_guess"] == "empty"]
        if empty:
            advice.append("Lane(s) %s contain no band above the noise floor (empty or blank lanes)."
                          % ", ".join(str(e) for e in empty))
        res["advice"] = advice
        res["questions_for_user"] = questions
        # One concrete call for small models to make next.
        if pol == "light_bands":
            nxt = {"tool": "fix_band_polarity", "arguments": {"mode": "auto"}}
        elif lad:
            nxt = {"tool": "label_mw_ladder", "arguments": {"lane": lad[0]}}
        else:
            nxt = {"tool": "auto_analyze_gel", "arguments": {"densitometry": True}}
        res["recommended_next_call"] = nxt
        bits = ["%d lanes (%s)." % (res["n_lanes"], res["lane_detection"])]
        bits.append({"dark_bands": "Bands are DARK on a LIGHT background: correct for analysis, do not invert.",
                     "light_bands": "Bands are LIGHT on a DARK background: call fix_band_polarity first.",
                     "ambiguous": "Band polarity is unclear."}[pol])
        bits.append("Ladder lane(s): %s." % (", ".join(map(str, lad)) if lad else "not found"))
        bits.append("Bands per lane: %s." % ", ".join("L%d=%d" % (ln["lane"], ln["n_bands"]) for ln in res["lanes"]))
        if questions:
            bits.append("ASK THE USER: " + " | ".join(questions))
        bits.append("NEXT CALL: %s(%s)." % (nxt["tool"], ", ".join("%s=%s" % kv for kv in nxt["arguments"].items())))
        res = dict([("summary", " ".join(bits))] + list(res.items()))
        res["next_steps"] = [
            "1. If questions_for_user is not empty, ASK THE USER those questions before quantifying.",
            "2. fix_band_polarity (only if polarity is light_bands).",
            "3. label_mw_ladder for MW labels only, OR auto_analyze_gel(densitometry=true) for "
            "lanes + ladder labels + band quantification in one go.",
            "4. quantify_with_standards if the user confirms standards and gives their amounts.",
            "5. screenshot(target='image') to check, then save_image / export_pptx.",
        ]
        return res

    # ...................................................................... polarity
    def _ensure_dark_bands(self, mode="auto"):
        """Make the analysis image show DARK bands on a LIGHT background (white lanes,
        black bands). GUI thread. Returns a report dict."""
        win = self.win
        gray, max_val, origin, cb, size = self._analysis_region(None)
        before = GEL.band_polarity(gray, max_val)
        rep = {"before": before["polarity"], "background_level": before["background_level"],
               "inverted": False}
        do = (mode == "invert") or (mode == "auto" and before["polarity"] == "light_bands")
        if do:
            cbx = getattr(win, "adjustment_context_combo", None)
            if cbx is not None and cbx.currentText() != "Main Image":
                cbx.setCurrentText("Main Image")
            win.invert_image()
            gray, max_val, *_ = self._analysis_region(None)
            rep["inverted"] = True
            rep["after"] = GEL.band_polarity(gray, max_val)["polarity"]
        else:
            rep["after"] = before["polarity"]
        # The densitometry engine must then read the image as-is (dark band = signal).
        try:
            s = getattr(win, "peak_dialog_settings", None)
            if isinstance(s, dict) and s.get("is_inverted"):
                s["is_inverted"] = False
                rep["densitometry_is_inverted_reset"] = True
        except Exception:
            pass
        if rep["after"] == "light_bands":
            rep["warning"] = "Bands are still light on dark. Check the image (screenshot)."
        return rep

    def t_fix_band_polarity(self, a):
        mode = a.get("mode") or "auto"
        if mode not in ("auto", "invert", "check"):
            raise ToolError("mode must be auto, invert or check.")

        def run():
            if mode == "check":
                gray, mv, *_ = self._analysis_region(None)
                return {"polarity": GEL.band_polarity(gray, mv)}
            return self._ensure_dark_bands(mode)
        res = self._with_status(self.action(run, "fix_band_polarity", timeout=90))
        res["rule"] = ("Analysis needs white lanes with black bands. DNA/RNA (EtBr/SYBR), stain-free, "
                       "fluorescent and raw chemiluminescence images are light-on-dark and must be "
                       "inverted; Coomassie/silver gels and exported blot images usually are not.")
        return res

    # ...................................................................... dialog pilot
    def _start_pilot(self, handlers, timeout_s=600):
        p = _Pilot(self, handlers, timeout_s)
        p.start()
        return p

    def _stop_pilots(self):
        def run():
            for p in list(self._pilots):
                p.stop()
        try:
            self.gui(run, timeout=5)
        except ToolError:
            pass

    def _true_peaks(self, dlg, snr_min, rel_min, min_width, is_ladder, abs_floor=0.0):
        """Peaks of the dialog's CURRENT lane that are real bands. Returns (kept, heights)."""
        peaks = np.asarray(dlg.peaks if dlg.peaks is not None else [], dtype=int)
        prof = getattr(dlg, "profile_original_inverted", None)
        if prof is None or len(peaks) == 0 or len(prof) < 8:
            return [int(p) for p in peaks], {}
        prof = np.asarray(prof, dtype=float)
        try:
            arr = np.asarray(dlg.intensity_array_original_range, dtype=float)
            sig2 = arr if dlg.is_inverted else float(dlg.original_max_value) - arr
            raw = sig2.sum(axis=1) if sig2.ndim == 2 else sig2
            raw = _median3(raw)
            sigma = GEL.robust_noise_sigma(raw)
        except Exception:
            sigma = GEL.robust_noise_sigma(prof)
        net = prof - GEL.baseline(prof, max(15, int(len(prof) * 0.12)))
        # Slow background wander (smears, uneven staining) is invisible to the first-difference
        # estimate; the robust spread of the baseline-subtracted trace catches it.
        sigma = max(sigma, GEL.background_sigma(net))
        peaks = peaks[(peaks >= 0) & (peaks < len(prof))]
        # The dialog stores band POSITIONS (apex / centroid / FWHM midpoint ...), which need
        # not sit exactly on the trace's local maximum; measure each band at its apex.
        rad = max(2, int(len(prof) * 0.006))
        apex = np.array([int(max(0, p - rad) + np.argmax(net[max(0, p - rad):p + rad + 1]))
                         for p in peaks], dtype=int) if len(peaks) else peaks
        h = net[apex] if len(apex) else np.array([])
        try:
            import warnings
            from scipy.signal import peak_widths
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = peak_widths(net, apex, rel_height=0.5)[0]
        except Exception:
            w = np.full(len(peaks), 99.0)
        ok = (h >= snr_min * sigma) & (w >= min_width)
        if not is_ladder and ok.any():
            ok &= h >= rel_min * float(h[ok].max())
            ok &= h >= abs_floor
        return [int(p) for p in peaks[ok]], {int(p): float(v) for p, v in zip(peaks, h)}

    def _clean_peak_dialog(self, dlg, opts):
        """Noise gate + ladder-count check on every lane of an open PeakAreaDialog, using the
        dialog's own profiles. Removed peaks are deleted the same way the dialog's
        'Delete Peak' button does it, so areas/regions stay consistent."""
        snr = float(opts.get("snr_min", GEL.DEFAULT_SNR_MIN))
        rel = float(opts.get("rel_min", GEL.DEFAULT_REL_MIN))
        mw = float(opts.get("min_width", GEL.DEFAULT_MIN_WIDTH))
        report = []
        try:
            cur = dlg.current_lane_id
            dlg._save_current_lane_state()
        except Exception:
            cur = None
        # Gel-wide floor: a sample band must reach rel_min of the strongest SAMPLE band on the
        # gel, so a nearly empty lane cannot promote its own background ripples to "bands".
        gel_max = 0.0
        ladders = getattr(dlg, "mwm_lane_ids", set()) or set()
        for lane in list(getattr(dlg, "lanes_data", [])):
            if lane["id"] in ladders:
                continue
            try:
                dlg._load_lane_state(lane["id"])
                _k, _h = self._true_peaks(dlg, snr, 0.0, mw, False)
                if _k:
                    gel_max = max(gel_max, max(_h.get(p, 0.0) for p in _k))
            except Exception:
                pass
        floor = rel * gel_max if len(dlg.lanes_data) > 1 else 0.0
        for lane in list(getattr(dlg, "lanes_data", [])):
            lid = lane["id"]
            try:
                dlg._load_lane_state(lid)
                is_ladder = lid in (getattr(dlg, "mwm_lane_ids", set()) or set())
                labels = [str(v) for v in (getattr(dlg, "lane_mw_labels", {}) or {}).get(lid, [])] \
                    if is_ladder else []
                orig_prom = dlg.peak_prominence_factor
                info = {"lane": lid, "ladder": is_ladder, "detected": int(len(dlg.peaks))}
                keep, heights = self._true_peaks(dlg, snr, rel, mw, is_ladder, floor)
                prom_changed = False
                if is_ladder and labels and opts.get("fix_ladder", True):
                    n_lab = len(labels)
                    if len(keep) < n_lab:
                        for f in (0.6, 0.35, 0.2, 0.1):
                            dlg.peak_prominence_factor = max(0.002, float(orig_prom) * f)
                            dlg.detect_peaks()
                            prom_changed = True
                            keep, heights = self._true_peaks(dlg, snr, rel, mw, True)
                            if len(keep) >= n_lab:
                                break
                    info["ladder_labels"] = n_lab
                    info["ladder_bands_found"] = len(keep)
                    if len(keep) > n_lab:
                        keep, how = GEL.choose_ladder_bands(keep, heights, labels)
                        info["ladder_band_choice"] = how
                        info["needs_user_check"] = True
                        info["warning"] = (
                            "The ladder lane shows %d bands but the preset has %d values. Labels were "
                            "put on the %d bands that %s. ASK THE USER to confirm the ladder (preset) "
                            "and the labels on the screenshot; re-run with review=true or another "
                            "preset if wrong." % (info["ladder_bands_found"], n_lab, n_lab, how))
                    elif len(keep) < n_lab:
                        info["needs_user_check"] = True
                        info["warning"] = (
                            "Only %d ladder bands found but the preset has %d values: labels are "
                            "assigned top→bottom, so the lowest values may be missing or shifted. "
                            "ASK THE USER which ladder was used / check the screenshot."
                            % (len(keep), n_lab))
                current = [int(p) for p in dlg.peaks]
                new = sorted(int(p) for p in keep)
                if new != current:
                    dlg._carry_regions = dlg._snapshot_regions_by_peak()
                    dlg.peaks = np.array(new, dtype=int)
                    dlg._recalculate_all_regions()
                    dlg.regions_are_manual = True
                    info["removed_as_noise"] = len([p for p in current if p not in new])
                if prom_changed:
                    dlg.peak_prominence_factor = orig_prom
                    dlg.regions_are_manual = True
                info["bands"] = len(new)
                dlg._save_current_lane_state()
                report.append(info)
            except Exception as e:  # noqa: BLE001
                report.append({"lane": lid, "error": "%s: %s" % (type(e).__name__, e)})
        try:
            if cur is not None:
                dlg._load_lane_state(cur)
            dlg.update_plot()
            if hasattr(dlg, "_update_quality_readout"):
                dlg._update_quality_readout()
        except Exception:
            pass
        return report

    def _peak_dialog_handler(self, pilot, opts):
        PAD = _app_global(self.win, "PeakAreaDialog")

        def match(w):
            return PAD is not None and isinstance(w, PAD)

        def handle(dlg):
            if opts.get("clean", True):
                pilot.report.setdefault("band_cleanup", []).extend(self._clean_peak_dialog(dlg, opts))
            if opts.get("review"):
                pilot.report["review_dialog"] = self._wid(dlg)
                return "release"         # leave it open for the AI / user to check and press OK
            dlg.accept_and_close()
            pilot.report["densitometry_dialog"] = "accepted"
            return None
        return match, handle

    # ...................................................................... ladder (Auto Lane)
    def _lane_from_args(self, a, insp):
        """Image-px (x0, x1) of the ladder lane from lane number / x0,x1 / inspection."""
        if a.get("x0") is not None and a.get("x1") is not None:
            return float(a["x0"]), float(a["x1"]), None
        lanes = insp["lanes"]
        n = a.get("lane")
        if n is None:
            if not insp["ladder_lanes"]:
                raise ToolError("No ladder lane could be identified automatically. Ask the user which "
                                "lane is the ladder and pass lane=<number> (see inspect_gel lanes).")
            n = insp["ladder_lanes"][0]
        n = int(n)
        if not 1 <= n <= len(lanes):
            raise ToolError("lane must be 1..%d (from inspect_gel)." % len(lanes))
        ln = lanes[n - 1]
        return float(ln["x0"]), float(ln["x1"]), n

    def _apply_preset(self, preset, strict=False):
        """Select a ladder preset by (partial / word) match. Unknown names keep the current
        preset (small models invent names) unless strict."""
        cb = self.win.combo_box
        items = [cb.itemText(i) for i in range(cb.count())]
        want = str(preset or "").strip().lower()
        m = [t for t in items if t.lower() == want] or [t for t in items if want and want in t.lower()]
        if not m and want:
            words = [w for w in re.split(r"[^a-z0-9]+", want) if len(w) > 2]
            scored = sorted(((sum(w in t.lower() for w in words), t) for t in items), reverse=True)
            if scored and scored[0][0] >= max(1, len(words) // 2 + (1 if len(words) > 2 else 0)):
                m = [scored[0][1]]
        if not m:
            if strict:
                raise ToolError("No marker preset '%s'. Available: %s" % (preset, items))
            self._preset_warning = ("No ladder preset called '%s'; kept '%s'. Available presets: %s"
                                    % (preset, cb.currentText(), "; ".join(items)))
            return cb.currentText()
        cb.setCurrentText(m[0])
        return m[0]

    def t_label_mw_ladder(self, a):
        if a.get("fix_polarity", True):
            self.action(lambda: self._ensure_dark_bands("auto"), "fix_band_polarity", timeout=90)
        insp = self._cached_inspection({"region": a.get("region")})
        x0, x1, lane_no = self._lane_from_args(a, insp)
        reg = insp["region"]
        y0 = float(a["y0"]) if a.get("y0") is not None else reg["y"]
        y1 = float(a["y1"]) if a.get("y1") is not None else reg["y"] + reg["height"]
        side = a.get("side") or "auto"
        if side == "auto":
            side = "left" if (x0 + x1) / 2.0 < reg["x"] + reg["width"] / 2.0 else "right"
        if side not in ("left", "right"):
            raise ToolError("side must be left, right or auto.")
        opts = {"review": bool(a.get("review")), "snr_min": a.get("min_snr", GEL.DEFAULT_SNR_MIN),
                "rel_min": 0.0, "fix_ladder": True}
        holder = {}
        self._preset_warning = None

        def run():
            win = self.win
            if a.get("preset"):
                holder["preset"] = self._apply_preset(a["preset"])
            adj = win._get_fully_adjusted_image_for_analysis()
            W, H = adj.width(), adj.height()
            rx0, ry0 = max(0, int(round(x0))), max(0, int(round(y0)))
            rx1, ry1 = min(W, int(round(x1))), min(H, int(round(y1)))
            if rx1 - rx0 < 2 or ry1 - ry0 < 8:
                raise ToolError("Ladder lane rectangle is outside the image.")
            rect = (rx0, ry0, rx1 - rx0, ry1 - ry0)
            region = adj.copy(*rect)
            win._cancel_all_interaction_modes()
            win.auto_marker_side = side
            win._auto_lane_apply_padding = bool(a.get("padding", True))
            pilot = self._start_pilot([])
            pilot.handlers.append(self._peak_dialog_handler(pilot, opts))
            holder["pilot"] = pilot
            self.reveal(win.live_view_label, "AI: Auto Lane (MW ladder)")
            win.process_auto_lane_region(region, rect, False)
            return {"lane_rect": list(rect), "side": side}
        res = self.action(run, "label_mw_ladder", timeout=240, auto_ok_info=True)
        pilot = holder.get("pilot")
        if res.get("status") == "done":
            self._stop_pilots()
            time.sleep(0.8)              # Auto Lane applies padding on the next event-loop pass
        out = {"status": res.get("status"), "method": "Auto Lane (app's Automatic Lane Markers)",
               "ladder_lane": lane_no, "side": side, "preset": holder.get("preset")}
        if self._preset_warning:
            out["preset_warning"] = self._preset_warning
        out.update({k: v for k, v in res.items() if k not in ("status",)})
        if pilot is not None:
            out.update(pilot.report)
            if pilot.errors:
                out["pilot_errors"] = pilot.errors
        if res.get("status") == "done":
            out["markers"] = self.gui(lambda: [{"y": _r(p), "label": l}
                                               for p, l in getattr(self.win, side + "_markers")])
            if not out["markers"]:
                out["warning"] = "No MW markers were placed — no bands found in that lane. Check lane/region."
            chk = [c for c in out.get("band_cleanup", []) if c.get("warning")]
            out["summary"] = ("Labelled %d ladder bands on the %s side: %s.%s%s"
                              % (len(out["markers"]), side,
                                 ", ".join(m["label"] for m in out["markers"] if m["label"]),
                                 (" CHECK: " + chk[0]["warning"]) if chk else "",
                                 (" NOTE: " + self._preset_warning) if self._preset_warning else ""))
            out = dict([("summary", out.pop("summary"))] + list(out.items()))
        elif res.get("status") == "dialog_open":
            out["note"] = ("Noise bands were removed. Review the densitometry dialog (screenshot target="
                           "'dialog') and press OK with respond_to_dialog to place the MW labels.")
        return self._with_status(out)

    # ...................................................................... Auto Gel wizard
    def t_auto_analyze_gel(self, a):
        if a.get("fix_polarity", True):
            pol = self.action(lambda: self._ensure_dark_bands("auto"), "fix_band_polarity", timeout=90)
        else:
            pol = None
        self._preset_warning = None
        if a.get("preset"):
            self.action(lambda: self._apply_preset(a["preset"]), "set_marker_preset")
        insp = self._inspect({"region": a.get("region"), "expected_lanes": a.get("expected_lanes")},
                             remember_lanes=True)
        reg = insp["region"]
        n_insp = insp["n_lanes"]
        # Ladder lanes (1-based, in inspect_gel numbering)
        if a.get("ladder_lanes") is not None:
            ladders = [int(x) for x in a["ladder_lanes"]]
        else:
            ladders = list(insp["ladder_lanes"])
        dens = bool(a.get("densitometry", True))
        labels = a.get("lane_labels")
        # label_lanes=False means "keep the lane names the user already has". A detected
        # ladder still needs the wizard's Labels & MWM page (to mark the ladder lane and
        # place MW labels), so that page may run — but it must not rename the lanes.
        rename_lanes = bool(a.get("label_lanes", True)) or bool(labels)
        markers = bool(a.get("label_lanes", True)) or bool(ladders)
        explicit = a.get("lanes")
        if explicit:
            try:
                lanes_img = [[float(l[0]), float(l[1])] for l in explicit]
            except Exception:
                raise ToolError("lanes must be a list of [x0, x1] pairs in image pixels.")
        else:
            lanes_img = None
        expected = a.get("expected_lanes")
        if not expected:
            # lane count the user gave earlier for this image (see _inspect remember_lanes)
            expected = insp.get("expected_lanes_used")
        insp_bounds = [[ln["x0"], ln["x1"]] for ln in insp["lanes"]]
        insp_centres = [ln["center_x"] for ln in insp["lanes"]]
        opts = {"review": bool(a.get("review")),
                "snr_min": a.get("min_snr", GEL.DEFAULT_SNR_MIN),
                "rel_min": a.get("min_relative_intensity", GEL.DEFAULT_REL_MIN),
                "clean": bool(a.get("remove_noise_bands", True)), "fix_ladder": True}
        holder = {}

        def drive_wizard(pilot):
            AGD = _app_global(self.win, "AutoGelAnalysisDialog")

            def match(w):
                return AGD is not None and isinstance(w, AGD)

            def handle(wz):
                rep = pilot.report.setdefault("wizard", {})

                def fail(msg):
                    rep["error"] = msg
                    pilot.errors.append(msg)
                    wz.reject()

                def step_lanes():
                    try:
                        wz._band_invert = False           # image polarity was fixed above
                        wz._band_height = 0.0
                        wz._lock_geometry_and_go_to_lanes()
                        gx = reg["x"]

                        def set_bounds(bounds):
                            wz._lane_bounds = [[int(max(0, b0 - gx)), int(max(1, b1 - gx))] for b0, b1 in bounds]
                            wz._lane_peaks = np.array([(b[0] + b[1]) // 2 for b in wz._lane_bounds], dtype=int)
                            wz._lane_skew = [[0.0, 0.0] for _ in wz._lane_bounds]
                            wz._lane_quads = [None for _ in wz._lane_bounds]
                            try:
                                wz._p1_count_lbl.setText("Lanes: %d" % len(wz._lane_bounds))
                                wz._redraw_lanes_keep()
                            except Exception:
                                pass
                        if lanes_img:
                            set_bounds(lanes_img)
                            rep["lanes_source"] = "given by caller"
                        elif expected and len(insp_bounds) == int(expected):
                            # inspect_gel already fitted exactly that many lanes (peaks if they
                            # are regular, else a grid snapped to the gaps between lanes)
                            set_bounds(insp_bounds)
                            rep["lanes_source"] = "inspect_gel: %s" % insp["lane_detection"]
                        else:
                            wiz = [[b0 + gx, b1 + gx] for b0, b1 in wz._lane_bounds]
                            if (GEL.lane_layout_warning(wiz) and insp_bounds
                                    and not GEL.lane_layout_warning(insp_bounds)):
                                set_bounds(insp_bounds)
                                rep["lanes_source"] = ("inspect_gel (wizard lanes were uneven): %s"
                                                       % insp["lane_detection"])
                            else:
                                rep["lanes_source"] = "Auto Gel lane detection"
                        n = len(wz._lane_bounds)
                        rep["lanes"] = [[int(b0 + gx), int(b1 + gx)] for b0, b1 in wz._lane_bounds]
                        if n == 0:
                            fail("No lanes were found. Pass expected_lanes or lanes=[[x0,x1],...].")
                            return
                        QTimer.singleShot(250, lambda: step_labels(n))
                    except Exception as e:  # noqa: BLE001
                        fail("lane step failed: %s" % e)

                def step_labels(n):
                    try:
                        centres = [(b0 + b1) / 2.0 + reg["x"] for b0, b1 in wz._lane_bounds]
                        # map inspect_gel ladder lanes onto the wizard's lanes by position
                        lad_idx = set()
                        for L in ladders:
                            if lanes_img or len(insp_centres) != n:
                                if 1 <= L <= len(insp_centres):
                                    cx = insp_centres[L - 1]
                                    lad_idx.add(int(np.argmin([abs(c - cx) for c in centres])))
                            elif 1 <= L <= n:
                                lad_idx.add(L - 1)
                        rep["ladder_lanes"] = sorted(i + 1 for i in lad_idx)
                        if not markers:
                            QTimer.singleShot(0, wz._p2_go_next)     # runs the analysis (blocks)
                            return
                        wz._p2_go_next()                              # → Labels & MWM page
                        # The page is seeded from the app's current top labels; ticking an
                        # MWM box rewrites its edit ("MWM" / "L#"), so remember the seeds.
                        seeded = [e.text() for e in wz._lane_label_edits]
                        k = 0
                        for i, chk in enumerate(wz._mwm_checkboxes):
                            chk.setChecked(i in lad_idx)
                        if not rename_lanes:
                            for edt, txt in zip(wz._lane_label_edits, seeded):
                                edt.setText(txt)
                            rep["lane_labels"] = [e.text() for e in wz._lane_label_edits]
                            rep["lane_labels_kept"] = True
                            QTimer.singleShot(self.step_delay() + 250, wz._p3_go_next)
                            return
                        for i, edt in enumerate(wz._lane_label_edits):
                            if labels and i < len(labels) and str(labels[i]).strip():
                                edt.setText(str(labels[i]))
                            elif i in lad_idx:
                                edt.setText("MWM")
                            else:
                                k += 1
                                edt.setText("L%d" % k)
                        rep["lane_labels"] = [e.text() for e in wz._lane_label_edits]
                        QTimer.singleShot(self.step_delay() + 250, wz._p3_go_next)   # blocks in dialog
                    except Exception as e:  # noqa: BLE001
                        fail("label step failed: %s" % e)

                try:
                    wz._p0_markers_chk.setChecked(markers)
                    wz._p0_density_chk.setChecked(dens)
                    wz._p0_crop_chk.setChecked(bool(a.get("padding", True)))
                    wz._go_to_phase1()
                    QTimer.singleShot(400, step_lanes)
                except Exception as e:  # noqa: BLE001
                    fail("start failed: %s" % e)
                return None
            return match, handle

        def run():
            win = self.win
            adj = win._get_fully_adjusted_image_for_analysis()
            W, H = adj.width(), adj.height()
            x0, y0 = max(0, reg["x"]), max(0, reg["y"])
            x1, y1 = min(W, reg["x"] + reg["width"]), min(H, reg["y"] + reg["height"])
            rect = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            region_q = adj.copy(*rect)
            pil = win.convert_qimage_to_grayscale_pil(region_q)
            color_np = win.qimage_to_numpy(region_q)
            win._cancel_all_interaction_modes()
            pilot = self._start_pilot([])
            pilot.handlers.append(drive_wizard(pilot))
            if dens or ladders:
                pilot.handlers.append(self._peak_dialog_handler(pilot, opts))
            holder["pilot"] = pilot
            self.reveal(win.live_view_label, "AI: Auto Analyze Gel")
            win._aag_open_wizard(pil, rect, is_quad=False, color_np=color_np)
            return {"gel_region": list(rect)}
        res = self.action(run, "auto_analyze_gel", timeout=300, auto_ok_info=True)
        pilot = holder.get("pilot")
        if res.get("status") == "done":
            self._stop_pilots()
            time.sleep(1.0)              # padding + results window are applied on the next pass
        out = {"status": res.get("status"),
               "method": "Auto Analyze Gel wizard (app) + noise-gated band cleanup",
               "polarity": (pol or {}).get("result")}
        out.update({k: v for k, v in res.items() if k != "status"})
        if pilot is not None:
            out.update(pilot.report)
            if pilot.errors:
                out["pilot_errors"] = pilot.errors
        if res.get("status") == "done":
            self._last_ladder_lanes = set((out.get("wizard") or {}).get("ladder_lanes") or [])
            out["results"] = self._results_summary()
            if not ladders:
                out["note"] = ("No ladder lane was marked, so no MW labels were placed. If there is a "
                               "ladder, ask the user which lane and run label_mw_ladder or re-run with "
                               "ladder_lanes=[n].")
            cands = [c for c in insp["standard_series_candidates"] if c["confidence"] >= 0.5]
            if cands and not out["results"].get("standards"):
                c = cands[0]
                out["question_for_user"] = (
                    "Lanes %s have a similar-size band at y≈%d with intensities %s× the weakest "
                    "(looks like a standard series). If the user ALREADY told you the amounts, call "
                    "quantify_with_standards now with standards=[{lane, quantity}, ...]; otherwise "
                    "ask the user for the amount in each lane."
                    % (", ".join(map(str, c["lanes"])), c["y"], ":".join(map(str, c["ratio_to_weakest"]))))
        elif res.get("status") == "dialog_open":
            out["note"] = ("Noise bands were removed. Review the densitometry dialog, then press OK "
                           "with respond_to_dialog; afterwards call get_analysis_results.")
        if self._preset_warning:
            out["preset_warning"] = self._preset_warning
        if res.get("status") == "done":
            lanes = out["results"]["lanes"]
            wz = out.get("wizard") or {}
            bits = ["Densitometry done on %d lanes (%s)." % (len(lanes), wz.get("lanes_source", "?")),
                    "Ladder lanes: %s." % (", ".join(map(str, wz.get("ladder_lanes") or [])) or "none"),
                    "Bands per lane: %s." % ", ".join("L%s=%d" % (k, v["bands"]) for k, v in lanes.items())]
            chk = [c for c in out.get("band_cleanup", []) if c.get("warning")]
            if chk:
                bits.append("CHECK: " + chk[0]["warning"])
            if out.get("question_for_user"):
                bits.append("ASK THE USER: " + out["question_for_user"])
            if self._preset_warning:
                bits.append("NOTE: " + self._preset_warning)
            out = dict([("summary", " ".join(bits))] + list(out.items()))
        return self._with_status(out)

    def _results_summary(self):
        def run():
            win = self.win
            areas = getattr(win, "latest_multi_lane_peak_areas", {}) or {}
            qty = getattr(win, "latest_multi_lane_calculated_quantities", {}) or {}
            lanes = {}
            for lid, ar in areas.items():
                tot = float(sum(ar)) or 0.0
                lanes[str(lid)] = {"bands": len(ar), "areas": [round(float(x), 3) for x in ar],
                                   "percent_of_lane": [round(100.0 * x / tot, 2) if tot else 0.0 for x in ar],
                                   "quantities": [round(float(q), 4) for q in (qty.get(lid) or [])]}
            std = getattr(win, "quantities_peak_area_dict", {}) or {}
            return {"lanes": lanes,
                    "standards": {str(k): round(float(v), 3) for k, v in std.items()},
                    "left_markers": len(getattr(win, "left_markers", []) or []),
                    "right_markers": len(getattr(win, "right_markers", []) or []),
                    "top_labels": [str(l) for _, l in (getattr(win, "top_markers", []) or [])]}
        return self.gui(run)

    # ...................................................................... standards
    def t_quantify_with_standards(self, a):
        stds = a.get("standards") or []
        if len(stds) < 2:
            raise ToolError("Give at least 2 standards: [{lane: <lane id>, quantity: <amount>}, ...].")
        model = a.get("model") or "Linear"
        band = a.get("band") or "strongest"

        def run():
            win = self.win
            areas_by = getattr(win, "latest_multi_lane_peak_areas", {}) or {}
            if not areas_by:
                raise ToolError("No densitometry results yet. Run auto_analyze_gel (densitometry=true) "
                                "or define_lane_regions + analyze_lanes first.")
            keyed = {str(k): (k, v) for k, v in areas_by.items()}
            pairs, used = {}, []
            for s in stds:
                try:
                    lid, q = str(int(s["lane"])), float(s["quantity"])
                except (KeyError, TypeError, ValueError):
                    raise ToolError("Each standard needs lane (number) and quantity (number).")
                if lid not in keyed or not keyed[lid][1]:
                    with_bands = sorted((k for k, (_, v) in keyed.items() if v), key=lambda x: (len(x), x))
                    raise ToolError("Lane %s has no measured bands%s. Lanes with at least one band: %s. "
                                    "Pick standards among those lanes, or re-run the analysis with lower "
                                    "min_snr / min_relative_intensity if the band was removed as noise."
                                    % (lid, " (it is empty after noise cleanup)" if lid in keyed else
                                       " (no such lane)", ", ".join(with_bands) or "none"))
                ar = [float(x) for x in keyed[lid][1]]
                if band == "total":
                    area, which = sum(ar), "all bands"
                elif isinstance(band, int) or (isinstance(band, str) and band.isdigit()):
                    bi = int(band) - 1
                    if not 0 <= bi < len(ar):
                        raise ToolError("Lane %s has only %d bands." % (lid, len(ar)))
                    area, which = ar[bi], "band %d" % (bi + 1)
                else:
                    bi = int(np.argmax(ar))
                    area, which = ar[bi], "band %d (strongest)" % (bi + 1)
                pairs.setdefault(q, []).append(area)
                used.append({"lane": int(lid), "quantity": q, "area": round(area, 3), "band": which})
            std = {q: float(np.mean(v)) for q, v in pairs.items()}
            if len(std) < 2:
                raise ToolError("Need at least 2 DIFFERENT quantities.")
            if not a.get("replace", True):
                merged = dict(getattr(win, "quantities_peak_area_dict", {}) or {})
                merged.update(std)
                std = merged
            win.save_state()
            win.quantities_peak_area_dict = std
            win.latest_standard_curve_model = model
            win.standard_protein_values.setText(", ".join("%.2f" % q for q in std))
            win.standard_protein_areas_text.setText(", ".join("%.3f" % v for v in std.values()))
            win._requantify_loaded_results(only_existing=False)
            for fn in ("_refresh_sample_results_text", "_sync_table_window_standards"):
                try:
                    getattr(win, fn)()
                except Exception:
                    pass
            win.is_modified = True
            fit_fn = _app_global(win, "fit_quant_standard_curve")
            fit = fit_fn(model, list(std.keys()), list(std.values())) if fit_fn else None
            q_by = getattr(win, "latest_multi_lane_calculated_quantities", {}) or {}
            std_lanes = {u["lane"] for u in used}
            lanes = {}
            qmin, qmax = min(std), max(std)
            for lid, ar in areas_by.items():
                qs = [round(float(x), 4) for x in (q_by.get(lid) or [])]
                role = ("standard" if int(lid) in std_lanes else
                        "ladder" if int(lid) in (getattr(self, "_last_ladder_lanes", None) or set())
                        else "sample")
                lanes[str(lid)] = {"role": role,
                                   "areas": [round(float(x), 3) for x in ar], "quantities": qs,
                                   "outside_standard_range": [i + 1 for i, x in enumerate(qs)
                                                              if x < qmin or x > qmax]}
            out = {"standards_used": used, "model": model, "unit": a.get("unit") or "",
                   "lanes": lanes}
            if fit:
                out["fit"] = {"equation": fit.get("equation"), "r_squared": round(float(fit["r_squared"]), 5)}
                if fit["r_squared"] < 0.95:
                    alt = "Polynomial (Deg 2)" if "Power" in model else "Power (Log-Log)"
                    out["warning"] = ("Standard curve R² = %.3f is poor. Usually the highest standards are "
                                      "saturated (see is_saturated in get_analysis_results): leave them out, "
                                      "or try model='%s'." % (fit["r_squared"], alt))
            unit = (" " + out["unit"]) if out["unit"] else ""
            samp = ["lane %s: %s" % (k, ", ".join("%g%s" % (q, unit) for q in v["quantities"]))
                    for k, v in lanes.items() if v["role"] == "sample" and v["quantities"]]
            out = dict([("summary", "Standard curve %s (R²=%s). Sample amounts — %s.%s"
                         % (model, out.get("fit", {}).get("r_squared", "?"), "; ".join(samp) or "no sample lanes",
                            (" WARNING: " + out["warning"]) if out.get("warning") else ""))]
                       + list(out.items()))
            return out
        return self._with_status(self.action(run, "quantify_with_standards"))

    # ...................................................................... combine gel + blot
    def t_combine_images(self, a):
        path = os.path.abspath(os.path.expanduser(str(a.get("path") or "")))
        if not os.path.isfile(path):
            raise ToolError("File not found: %s" % path)
        placement = a.get("placement") or "below"
        if placement not in ("below", "above", "right", "left"):
            raise ToolError("placement must be below, above, right or left.")
        gap = max(0, int(a.get("gap") or 20))
        ext = os.path.splitext(path)[1].lower()
        self.file_answers = []

        def run():
            win = self.win
            if not win._has_image():
                raise ToolError("Open the first image (the gel) before combining.")
            ov = QImage(path)
            if ov.isNull():
                try:
                    from PIL import Image as _PIL
                    ov = win.numpy_to_qimage(np.array(_PIL.open(path)))
                except Exception as e:  # noqa: BLE001
                    raise ToolError("Could not read %s: %s" % (path, e))
            bw, bh = ov.width(), ov.height()
            img = win.image
            W, H = img.width(), img.height()
            l, r, t, b = (int(v) for v in win._measure_transparent_margins())
            cw, ch = W - l - r, H - t - b
            if a.get("scale_percent"):
                s = float(a["scale_percent"]) / 100.0
            elif a.get("fit", True):
                s = (cw / float(bw)) if placement in ("below", "above") else (ch / float(bh))
            else:
                s = 1.0
            s_req = s
            s = max(0.10, min(3.0, s))
            sw, sh = int(np.ceil(bw * s)), int(np.ceil(bh * s))
            nl, nr, nt, nb = l, r, t, b
            if placement == "below":
                nb = b + gap + sh
            elif placement == "above":
                nt = t + gap + sh
            elif placement == "right":
                nr = r + gap + sw
            else:
                nl = l + gap + sw
            for side, v in (("left", nl), ("right", nr), ("top", nt), ("bottom", nb)):
                getattr(win, side + "_padding_input").setText(str(int(v)))
            win.finalize_image()
            img = win.image
            W2, H2 = img.width(), img.height()
            cx0, cy0 = nl, nt
            cx1, cy1 = W2 - nr, H2 - nb
            if placement == "below":
                px, py = cx0 + (cw - sw) / 2.0, cy1 + b + gap
            elif placement == "above":
                px, py = cx0 + (cw - sw) / 2.0, cy0 - t - gap - sh
            elif placement == "right":
                px, py = cx1 + r + gap, cy0 + (ch - sh) / 2.0
            else:
                px, py = cx0 - l - gap - sw, cy0 + (ch - sh) / 2.0
            px, py = int(round(px)), int(round(py))
            # load the second image as Overlay 2 through the app's own loader
            self.file_answers = [(path, _IMAGE_EXT_FILTERS.get(ext, ""))]
            win.load_overlay_image()
            self.file_answers = []
            if getattr(win, "image2_original", None) is None or win.image2_original.isNull():
                raise ToolError("The app could not load %s as an overlay." % path)
            # polarity of the second image: show dark bands on light, like the gel
            pol_mode = a.get("band_polarity") or "auto"
            pol = None
            if pol_mode != "as_is":
                arr = win.qimage_to_numpy(win.image2_original)
                if arr is not None:
                    g = arr[..., :3].mean(axis=2) if arr.ndim == 3 else arr
                    mv = 65535.0 if arr.dtype == np.uint16 else 255.0
                    pol = GEL.band_polarity(np.asarray(g, dtype=float), mv)["polarity"]
                    want_inv = (pol == "light_bands") if pol_mode == "auto" else (pol_mode == "invert")
                    win.image2_adjustments["is_inverted"] = bool(want_inv)
                    try:
                        win._update_overlay_preview(2)
                    except Exception:
                        pass
            steps = win.OVERLAY_RESIZE_STEPS_PER_PERCENT
            win.image2_resize_slider.setValue(int(round(s * 100 * steps)))
            win._update_overlay_slider_ranges()
            win.image2_left_slider.setValue(px)
            win.image2_top_slider.setValue(py)
            win.image2_position = (px, py)
            cbx = getattr(win, "adjustment_context_combo", None)
            if cbx is not None:
                cbx.setCurrentText("Main Image")
            win.is_in_dedicated_edit_mode = False
            win.update_live_view()
            out = {"placed": path, "placement": placement, "position": [px, py],
                   "scale_percent": round(s * 100, 2), "size_on_canvas": [sw, sh],
                   "canvas": [W2, H2], "padding": {"left": nl, "right": nr, "top": nt, "bottom": nb},
                   "second_image_polarity": pol,
                   "second_image_inverted": bool(win.image2_adjustments.get("is_inverted"))}
            if abs(s - s_req) > 1e-6:
                out["warning"] = "Scale clamped to %.0f%% (allowed 10-300%%)." % (s * 100)
            if a.get("rasterize"):
                win.finalize_combined_image()
                out["rasterized"] = True
                out["canvas"] = [win.image.width(), win.image.height()]
            return out
        res = self.action(run, "combine_images", timeout=180, auto_ok_info=True)
        self.file_answers = []
        res["note"] = ("The second image is a separate layer (Overlap Images tab ▸ Image 2) until "
                       "rasterized; save_image / export_pptx include it either way. Annotate it with "
                       "add_text / add_shape (its y-range is in 'position' + 'size_on_canvas').")
        return self._with_status(res)

    # ...................................................................... guide
    def t_get_workflow_guide(self, a):
        topic = (a.get("task") or "overview").lower()
        return {"task": topic, "guide": S.WORKFLOW_GUIDES.get(topic, S.WORKFLOW_GUIDES["overview"]),
                "topics": sorted(S.WORKFLOW_GUIDES)}

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
    # _jsonable first: default=str alone turned numpy arrays into truncated "[1 2 ... 9]"
    # strings and numpy dict keys into a TypeError.
    return {"type": "text", "text": json.dumps(_jsonable(obj), indent=1, ensure_ascii=False,
                                                default=str)}


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
        if isinstance(v, (set, frozenset)):
            return [_jsonable(x) for x in sorted(v, key=repr)]
        if hasattr(v, "tolist"):
            try:
                return _jsonable(v.tolist())
            except Exception:
                pass
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


def _clean_smart_args(a):
    """Small local models often send numbers as strings and fill every optional field with
    placeholder values. Coerce what is meaningful, drop what is clearly not."""
    a = dict(a or {})

    def num(k, lo, hi, as_int=False):
        v = a.get(k)
        if v is None or v == "":
            a.pop(k, None)
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            a.pop(k, None)
            return
        if not (lo <= v <= hi):
            a.pop(k, None)               # out of range = placeholder, use the default
            return
        a[k] = int(round(v)) if as_int else v
    num("expected_lanes", 2, 60, True)
    num("lane", 1, 60, True)
    num("min_snr", 2, 100)
    num("min_relative_intensity", 0.0005, 0.5)
    num("gap", 0, 5000, True)
    num("scale_percent", 10, 300)
    for k in ("x0", "x1", "y0", "y1"):
        num(k, -1e6, 1e6)
    for k in ("ladder_lanes",):
        v = a.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = [v]
        if isinstance(v, str):
            v = re.findall(r"\d+", v)          # "[8, 1]", "1 and 8", "1,8" ...
        if isinstance(v, list):
            out = []
            for x in v:
                try:
                    out.append(int(float(x)))
                except (TypeError, ValueError):
                    pass
            a[k] = out
        elif v is not None:
            a.pop(k, None)
    for k in ("densitometry", "padding", "review", "fix_polarity", "label_lanes", "fit",
              "rasterize", "remove_noise_bands", "replace"):
        v = a.get(k)
        if isinstance(v, str):
            a[k] = v.strip().lower() in ("1", "true", "yes", "on")
    st = a.get("standards")
    if isinstance(st, str):                  # "[{lane: 2, quantity: 250}, ...]" written as text
        try:
            st = json.loads(st)
        except ValueError:
            st = [{"lane": int(l), "quantity": float(q)} for l, q in
                  re.findall(r"lane\W*(\d+)\W+(?:quantity|qty|amount)\W*([0-9.]+)", st, re.I)]
        a["standards"] = st
    if isinstance(a.get("standards"), dict):     # {"2": 250, "3": 500}
        a["standards"] = [{"lane": k, "quantity": v} for k, v in a["standards"].items()]
    for k in ("preset", "region", "lanes", "lane_labels"):
        if a.get(k) in ("", None, [], {}, "none", "None", "null"):
            a.pop(k, None)
    r = a.get("region")
    if r is not None and not (isinstance(r, dict) and all(k in r for k in ("x", "y", "width", "height"))):
        a.pop("region", None)
    return a


_SMART_TOOLS = {"inspect_gel", "fix_band_polarity", "label_mw_ladder", "auto_analyze_gel",
                "quantify_with_standards", "combine_images", "get_workflow_guide"}


def _summarize_args(args):
    parts = []
    for k, v in (args or {}).items():
        s = json.dumps(_jsonable(v), ensure_ascii=False, default=str)
        if len(s) > 40:
            s = s[:37] + "…"
        parts.append("%s=%s" % (k, s))
    return ", ".join(parts)
