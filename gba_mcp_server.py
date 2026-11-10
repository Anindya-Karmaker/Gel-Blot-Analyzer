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
import subprocess
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


def _write(msg):
    data = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
    with _out_lock:
        sys.stdout.write(data + "\n")
        sys.stdout.flush()


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


def _probe(port, token, timeout=1.5):
    """Return the pid reported by /health on this port, or None when nothing (of ours) answers."""
    try:
        req = urllib.request.Request("http://127.0.0.1:%d/health" % port, method="GET")
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("User-Agent", "gba-mcp-stdio/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8") or "{}")
        if r.status == 200 and isinstance(body, dict) and body.get("ok") and body.get("app") == S.SERVER_TITLE:
            return int(body.get("pid") or -1)
    except Exception:
        pass
    return None


def _http(method, path, body=None, timeout=330):
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


def _ensure_app_locked(wait_s):
    if _app_alive():
        return True, ""
    if not _settings().get("enabled") and not os.environ.get("GBA_PORT"):
        return False, _DISABLED_MSG
    if os.environ.get("GBA_NO_LAUNCH"):
        return False, ("Gel Blot Analyzer is not running (or AI control is off). Open the app and "
                       "turn on Settings > AI Assistant Control (MCP).")
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


def _forward(msg):
    mid = msg.get("id")
    ok, why = _ensure_app()
    if not ok:
        return _tool_error(mid, why)
    try:
        _status, body = _http("POST", "/mcp", msg)
        return body
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
            text = (detail.get("error") or {}).get("message") if isinstance(detail.get("error"), dict) \
                else detail.get("error")
        except Exception:
            text = str(e)
        return _tool_error(mid, "Gel Blot Analyzer refused the request (%s): %s" % (e.code, text))
    except Exception as e:
        _resolved["port"] = None               # re-discover the endpoint on the next call
        return _tool_error(mid, "Lost connection to Gel Blot Analyzer: %s" % e)


def handle(msg):
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}}
    mid = msg.get("id")
    method = msg.get("method")
    if mid is None:
        return None                                   # notifications need no answer
    params = msg.get("params") or {}
    if method == "initialize":
        pv = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": pv if pv in S.SUPPORTED_PROTOCOL_VERSIONS else S.LATEST_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": S.SERVER_NAME, "title": S.SERVER_TITLE, "version": "1.0.0"},
            "instructions": S.SERVER_INSTRUCTIONS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        # Served locally so the client sees every tool even before the app is open.
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": S.public_tools()}}
    if method == "tools/call":
        return _forward(msg)
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"resources": []}}
    if method == "resources/templates/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"resourceTemplates": []}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"prompts": []}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found: %s" % method}}


def _serve_one(line):
    try:
        msg = json.loads(line)
    except Exception:
        _write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
        return
    if isinstance(msg, list):
        out = [r for r in (handle(m) for m in msg) if r is not None]
        if out:
            _write(out)
        return
    resp = handle(msg)
    if resp is not None:
        _write(resp)


def main():
    # Binary-safe, UTF-8 stdio on every platform (Windows defaults to the ANSI code page).
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    except Exception:
        pass
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
