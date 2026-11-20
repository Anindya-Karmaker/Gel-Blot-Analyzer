#!/usr/bin/env python3
"""Gel Blot Analyzer — stdio MCP server.

This is the command an AI client launches. It speaks MCP over stdin/stdout (newline-delimited
JSON-RPC) and relays every request to the running app's local AI-control endpoint
(http://127.0.0.1:<port>/mcp, token from ~/.gel_blot_analyzer/ai_control.json).
If the app is not running it starts it (with AI control enabled) on the first tool call.

Standard library only, Python 3.8+ — works with any interpreter, and the packaged app runs it
with `GelBlotAnalyzer --mcp-stdio`.

Client configuration examples
-----------------------------
Claude Desktop (claude_desktop_config.json) / LM Studio (mcp.json) / Cursor (mcp.json):
    {"mcpServers": {"gel-blot-analyzer": {
        "command": "/path/to/python3",
        "args": ["/path/to/gba_mcp_server.py"]}}}
Packaged app on macOS:
        "command": "/Applications/Gel Blot Analyzer.app/Contents/MacOS/Gel Blot Analyzer",
        "args": ["--mcp-stdio"]

Environment overrides: GBA_PORT, GBA_TOKEN, GBA_APP (command used to start the app),
GBA_NO_LAUNCH=1 (never start the app).
"""

import json
import os
import socket
import subprocess
# Everything the relay will ever use is imported NOW. A packaged app loads modules from its
# executable on first use; if the app is rebuilt/updated while this relay keeps running, a
# lazy import later fails -- the health probe then looked like "app not running" and a
# second copy of the app was launched. (http.client / email are what urllib pulls in lazily.)
import email.message  # noqa: F401
import email.parser  # noqa: F401
import http.client  # noqa: F401
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import gba_mcp_schema as S
except ImportError:  # run from another working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gba_mcp_schema as S

_out_lock = threading.Lock()
_launched = {"proc": None, "at": 0.0}
_launch_lock = threading.Lock()
_cancelled = set()                  # request ids the client cancelled (answer suppressed)
_cancel_lock = threading.Lock()
try:                                # keep-alive notifications while a tool call runs
    PROGRESS_EVERY_S = max(0.5, float(os.environ.get("GBA_PROGRESS_S") or 10.0))
except ValueError:
    PROGRESS_EVERY_S = 10.0
HTTP_TIMEOUT_S = 900                # the app answers long before this (tools time out first)


def _write(msg):
    try:
        data = json.dumps(S.json_safe(msg), ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False, default=str)
    except Exception as e:  # noqa: BLE001 - never send a line the client can't parse
        mid = msg.get("id") if isinstance(msg, dict) else None
        data = json.dumps({"jsonrpc": "2.0", "id": mid,
                           "error": {"code": -32603, "message": "Unserialisable reply: %s" % e}})
    with _out_lock:
        try:
            sys.stdout.write(data + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass                     # the client has gone; main() exits when stdin closes


def _log(text):
    """stderr is shown in the client's MCP log; stdout is reserved for the protocol."""
    try:
        sys.stderr.write("[gel-blot-analyzer] %s\n" % text)
        sys.stderr.flush()
    except Exception:
        pass


def _settings():
    try:
        with open(S.settings_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


_resolved = {"port": None}          # port that last answered /health (see _app_alive)


def _endpoint():
    st = _settings()
    port = int(os.environ.get("GBA_PORT") or _resolved["port"] or st.get("active_port")
               or st.get("port") or S.DEFAULT_PORT)
    token = os.environ.get("GBA_TOKEN") or st.get("token") or ""
    return port, token


def _candidate_ports(st):
    """Recorded port first, then every port the app may fall back to (DEFAULT_PORT..+PORT_TRIES-1)."""
    out = []
    for p in (os.environ.get("GBA_PORT"), _resolved["port"], st.get("active_port"), st.get("port")):
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if p > 0 and p not in out:
            out.append(p)
    base = int(st.get("port") or S.DEFAULT_PORT)
    for b in (base, S.DEFAULT_PORT):
        for p in range(b, b + S.PORT_TRIES):
            if p not in out:
                out.append(p)
    return out


class _ProbeBroken(Exception):
    """The probe itself failed (not a refused/closed port): never treat as 'not running'."""


_NET_ERRORS = (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError,
               http.client.HTTPException, ValueError)


def _probe(port, token, timeout=1.5):
    """Return the pid reported by /health on this port, or None when nothing (of ours) answers.
    Raises _ProbeBroken when the probe could not even run (e.g. an import failed)."""
    try:
        req = urllib.request.Request("http://127.0.0.1:%d/health" % port, method="GET")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("User-Agent", "gba-mcp-stdio/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8") or "{}")
        if r.status == 200 and isinstance(body, dict) and body.get("ok") and body.get("app") == S.SERVER_TITLE:
            return int(body.get("pid") or -1)
    except _NET_ERRORS:
        pass
    except Exception as e:  # noqa: BLE001
        raise _ProbeBroken("%s: %s" % (type(e).__name__, e))
    return None


def _running_app_pids():
    """PIDs of Gel Blot Analyzer windows already running (any copy, any install), excluding
    relays (--mcp-stdio). Used so the relay never starts a second copy of the app."""
    me = os.getpid()
    out = []
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'Gel*Blot*' -or "
                  "$_.Name -like 'gelblotanalyzer*' -or ($_.CommandLine -like '*Gel_blot_analyzer.py*') } | "
                  "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
            txt = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=0x08000000).stdout
            for line in txt.splitlines():
                pid, _, cmd = line.partition("\t")
                if "--mcp-stdio" in cmd:
                    continue
                try:
                    out.append(int(pid.strip()))
                except ValueError:
                    pass
        else:
            txt = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in txt.splitlines():
                line = line.strip()
                pid, _, cmd = line.partition(" ")
                if "--mcp-stdio" in cmd:
                    continue
                if ("Gel Blot Analyzer" in cmd and "/MacOS/" in cmd) or "Gel_blot_analyzer.py" in cmd \
                        or cmd.endswith("/gelblotanalyzer"):
                    try:
                        out.append(int(pid))
                    except ValueError:
                        pass
    except Exception:
        return []
    return [p for p in out if p != me]


def _http(method, path, body=None, timeout=HTTP_TIMEOUT_S):
    port, token = _endpoint()
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/json, text/event-stream")
    req.add_header("User-Agent", "gba-mcp-stdio/1.0")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw.decode("utf-8")) if raw else None)


def _app_alive():
    try:
        return _app_alive_impl()
    except _ProbeBroken as e:
        _log("health probe failed: %s" % e)
        raise


def _app_alive_impl():
    """Find the running app. The recorded port is tried first; if it is dead (the record can
    be stale after a crash, or after two instances overlapped) every port the app may have
    fallen back to is probed with the token. Prefers the instance whose pid matches the
    record, otherwise the first one that answers."""
    st = _settings()
    token = os.environ.get("GBA_TOKEN") or st.get("token") or ""
    if not token:
        return False
    rec_pid = st.get("pid")
    try:
        rec_pid = int(rec_pid) if rec_pid else None
    except (TypeError, ValueError):
        rec_pid = None
    first = None
    for port in _candidate_ports(st):
        pid = _probe(port, token)
        if pid is None:
            continue
        if rec_pid is None or pid == rec_pid or not S.pid_alive(rec_pid):
            _set_resolved(port)
            return True
        if first is None:
            first = port
    if first is not None:                      # recorded pid is alive but answers nowhere: use what works
        _set_resolved(first)
        return True
    _resolved["port"] = None
    return False


def _set_resolved(port):
    if _resolved["port"] != port:
        _resolved["port"] = port
        _log("app endpoint: http://127.0.0.1:%d" % port)


def _launch_command():
    env_cmd = os.environ.get("GBA_APP")
    if env_cmd:
        return env_cmd, True
    here = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False):
        if sys.platform == "win32" and "\\windowsapps\\" in sys.executable.lower():
            # MSIX: start through the App Execution Alias, not the protected install path.
            return [S.APP_ALIAS], False
        return [sys.executable], False
    script = os.path.join(here, "Gel_blot_analyzer.py")
    if os.path.isfile(script):
        return [sys.executable, script], False
    return None, False


def _ensure_app(wait_s=90):
    with _launch_lock:
        return _ensure_app_locked(wait_s)


_DISABLED_MSG = ("AI control is turned OFF in Gel Blot Analyzer. Ask the user to open "
                 "Gel Blot Analyzer > Settings > AI Assistant Control (MCP) and tick "
                 "'Allow AI assistants to control Gel Blot Analyzer'. (It is off by default "
                 "and only the user can turn it on.)")


_STALE_MSG = ("This connection to Gel Blot Analyzer is out of date (the app was updated or "
              "rebuilt while it was connected). Restart the AI app, or reconnect the server "
              "(Claude Code: /mcp), and try again. The app itself was NOT restarted.")


def _ensure_app_locked(wait_s):
    try:
        return _ensure_app_checked(wait_s)
    except _ProbeBroken:
        return False, _STALE_MSG


def _ensure_app_checked(wait_s):
    if _app_alive():
        return True, ""
    if not _settings().get("enabled") and not os.environ.get("GBA_PORT"):
        return False, _DISABLED_MSG
    if os.environ.get("GBA_NO_LAUNCH"):
        return False, ("Gel Blot Analyzer is not running (or AI control is off). Open the app and "
                       "turn on Settings > AI Assistant Control (MCP).")
    rec_pid = _settings().get("pid")
    if rec_pid and S.pid_alive(rec_pid) and rec_pid != os.getpid():
        # The app that advertised itself is still alive but did not answer /health: it is busy
        # (a long analysis) or restarting its server. Launching a second copy would only make
        # two windows fight over the port — wait for it instead.
        _log("app (pid %s) is alive but not answering yet; waiting" % rec_pid)
        deadline = time.time() + min(wait_s, 30)
        while time.time() < deadline:
            if _app_alive():
                return True, ""
            time.sleep(0.5)
        return False, ("Gel Blot Analyzer is running but not answering (it may be busy with a long "
                       "operation or showing a message box). Check the app window, then try again.")
    running = _running_app_pids()
    if running and (_launched["proc"] is None or _launched["proc"].pid not in running):
        # A copy of the app is open but did not answer: never start another one.
        _log("Gel Blot Analyzer is already running (pid %s); waiting instead of launching"
             % ", ".join(map(str, running)))
        deadline = time.time() + min(wait_s, 20)
        while time.time() < deadline:
            if _app_alive():
                return True, ""
            time.sleep(0.5)
        return False, ("Gel Blot Analyzer is open but its AI control is not answering. In the app, "
                       "check Settings > AI Assistant Control (MCP): it must be ON and not paused. "
                       "If it is on, turn it off and on again.")
    proc = _launched["proc"]
    if proc is None or proc.poll() is not None:
        cmd, shell = _launch_command()
        if not cmd:
            return False, "Gel Blot Analyzer is not running and could not be located; please open it."
        _log("starting the app: %s" % cmd)
        try:
            kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      shell=shell, cwd=os.path.dirname(os.path.abspath(__file__)))
            if os.name == "nt":
                kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | NEW_PROCESS_GROUP
            else:
                kw["start_new_session"] = True
            _launched["proc"] = subprocess.Popen(cmd, **kw)
            _launched["at"] = time.time()
        except Exception as e:
            return False, "Could not start Gel Blot Analyzer: %s" % e
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _app_alive():
            return True, ""
        p = _launched["proc"]
        if p is not None and p.poll() is not None:
            return False, "Gel Blot Analyzer exited during startup (code %s)." % p.returncode
        time.sleep(0.5)
    if not _settings().get("enabled"):
        return False, _DISABLED_MSG
    return False, ("Gel Blot Analyzer did not become ready within %ds. Check that it is open and "
                   "that Settings > AI Assistant Control is turned on." % wait_s)


def _tool_error(mid, text):
    return {"jsonrpc": "2.0", "id": mid,
            "result": {"content": [{"type": "text", "text": text}], "isError": True}}


def _progress_pump(token, stop, name):
    """notifications/progress every PROGRESS_EVERY_S while a call runs, so clients that time
    out idle requests (and reset that timer on progress) wait for long analyses."""
    n = 0
    t0 = time.time()
    while not stop.wait(PROGRESS_EVERY_S):
        n += 1
        _write({"jsonrpc": "2.0", "method": "notifications/progress",
                "params": {"progressToken": token, "progress": n,
                           "message": "%s still running (%ds)" % (name, int(time.time() - t0))}})


def _refused(e):
    r = getattr(e, "reason", e)
    return isinstance(r, (ConnectionRefusedError, ConnectionResetError)) or \
        (isinstance(r, OSError) and getattr(r, "errno", None) in (61, 111, 10061))


def _forward(msg):
    mid = msg.get("id")
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
    token = meta.get("progressToken")
    stop = threading.Event()
    if token is not None:
        threading.Thread(target=_progress_pump, args=(token, stop, params.get("name") or "tool"),
                         daemon=True).start()
    try:
        return _forward_once(msg, mid)
    finally:
        stop.set()


def _forward_once(msg, mid):
    ok, why = _ensure_app()
    if not ok:
        return _tool_error(mid, why)
    lost = ("Lost connection to Gel Blot Analyzer: %s. If the app is open, call get_status to "
            "check whether the last action ran.")
    for attempt in (0, 1):
        try:
            _status, body = _http("POST", "/mcp", msg)
        except urllib.error.HTTPError as e:      # the app answered with an error status
            try:
                detail = json.loads(e.read().decode("utf-8") or "{}")
                err = detail.get("error") if isinstance(detail, dict) else None
                text = err.get("message") if isinstance(err, dict) else (err or str(e))
            except Exception:
                text = str(e)
            if e.code == 401:
                text += " (the connection token changed: restart the AI client or reconnect it " \
                        "from Settings > AI Assistant Control)"
            return _tool_error(mid, "Gel Blot Analyzer refused the request (%s): %s" % (e.code, text))
        except (socket.timeout, TimeoutError):
            return _tool_error(mid, "Gel Blot Analyzer is still working on this request after %ds. "
                                    "Call get_status or wait, then continue." % HTTP_TIMEOUT_S)
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
                return _tool_error(mid, "Gel Blot Analyzer is still working on this request. "
                                        "Call get_status or wait, then continue.")
            _resolved["port"] = None             # re-discover the endpoint
            # Refused = the request never reached the app (it restarted, maybe on another
            # port): rediscover and resend once. Anything else might already have run.
            if attempt == 0 and _refused(e):
                ok, why = _ensure_app(wait_s=30)
                if ok:
                    continue
                return _tool_error(mid, why)
            return _tool_error(mid, lost % getattr(e, "reason", e))
        except Exception as e:  # noqa: BLE001 - reset by peer mid-reply, bad JSON, ...
            _resolved["port"] = None
            return _tool_error(mid, lost % e)
        if not isinstance(body, dict):
            return _tool_error(mid, "Gel Blot Analyzer sent an empty reply; try again.")
        body["id"] = mid
        return body
    return _tool_error(mid, "Could not reach Gel Blot Analyzer.")


def handle(msg):
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params")
    if not isinstance(params, dict):
        params = {}
    if mid is None:                                   # notifications need no answer
        if method == "notifications/cancelled":
            rid = params.get("requestId")
            if rid is not None:
                with _cancel_lock:
                    _cancelled.add(_id_key(rid))
                _log("request %s cancelled by the client (the app finishes the current step)" % rid)
        return None
    if "result" in msg or "error" in msg:
        return None                                   # a reply to us (we send no requests)
    if not isinstance(method, str):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32600, "message": "Invalid request: no method"}}
    if method == "initialize":
        pv = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": pv if pv in S.SUPPORTED_PROTOCOL_VERSIONS else S.LATEST_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": S.SERVER_NAME, "title": S.SERVER_TITLE, "version": S.SERVER_VERSION},
            "instructions": S.instructions()}}
    if method in ("ping", "logging/setLevel"):
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        # Served locally so the client sees every tool even before the app is open.
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": S.public_tools()}}
    if method == "tools/call":
        name = params.get("name")
        if not name or not isinstance(name, str):
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": "tools/call needs params.name (a tool name)"}}
        real = S.resolve_tool_name(name)
        if real is None:                              # answer locally, even if the app is closed
            sugg = S.suggest_tools(name)
            return _tool_error(mid, "Unknown tool '%s'.%s Call tools/list for the full list."
                               % (name, (" Did you mean: %s?" % ", ".join(sugg)) if sugg else ""))
        return _forward(msg)
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"resources": []}}
    if method == "resources/templates/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"resourceTemplates": []}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"prompts": []}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found: %s" % method}}


def _id_key(mid):
    return json.dumps(mid, sort_keys=True)


def _safe_handle(msg):
    try:
        resp = handle(msg)
    except Exception as e:  # noqa: BLE001 - always answer a request
        mid = msg.get("id") if isinstance(msg, dict) else None
        if isinstance(msg, dict) and mid is None:
            return None
        _log("internal error: %s: %s" % (type(e).__name__, e))
        resp = {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32603, "message": "Internal error: %s: %s" % (type(e).__name__, e)}}
    if isinstance(resp, dict) and resp.get("id") is not None:
        with _cancel_lock:
            k = _id_key(resp["id"])
            if k in _cancelled:                       # client gave up on it: stay silent
                _cancelled.discard(k)
                return None
    return resp


def _serve_one(line):
    try:
        msg = json.loads(line.lstrip("\ufeff"))
    except Exception:
        _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
        return
    if isinstance(msg, list):
        if not msg:
            _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Empty batch"}})
            return
        out = [r for r in (_safe_handle(m) for m in msg) if r is not None]
        if out:
            _write(out)
        return
    resp = _safe_handle(msg)
    if resp is not None:
        _write(resp)


def _prewarm():
    """Run one probe against a closed port so every lazily imported module urllib needs is
    loaded while the executable on disk still matches this process."""
    try:
        _probe(9, "x", timeout=0.2)
    except Exception:
        pass
    try:
        S.normalize_arguments("get_status", {})
        S.suggest_tools("x")
    except Exception:
        pass


def main():
    # Binary-safe, UTF-8 stdio on every platform (Windows defaults to the ANSI code page).
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    except Exception:
        pass
    _prewarm()
    _log("stdio MCP server ready (app endpoint port %d)" % _endpoint()[0])
    workers = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # tools/call can take a while (dialogs, analysis): handle each request on its own
        # thread so pings and cancellations are still answered meanwhile.
        t = threading.Thread(target=_serve_one, args=(line,), daemon=True)
        t.start()
        workers = [w for w in workers if w.is_alive()] + [t]
    # stdin closed: let requests already sent finish and answer (a tool call can legitimately
    # take minutes, e.g. waiting on the app to start), then exit.
    for t in workers:
        t.join(timeout=600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
