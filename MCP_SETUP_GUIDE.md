# Gel Blot Analyzer — AI Assistant Control (MCP) Setup Guide

Gel Blot Analyzer can be operated by an AI assistant through the **Model Context Protocol (MCP)**,
the open standard used by Claude, LM Studio, Cursor, VS Code and many local-model apps. It also
offers an **OpenAPI tool server**, so Ollama users with Open WebUI can connect too.

The AI works **live in your window**, so you can watch every step and pause it at any time.

**AI control is OFF by default.** Nothing can connect until you turn it on yourself.

---

## 1. Quick start (2 steps)

1. **Turn it on.** Go to **Settings ▸ AI Assistant Control (MCP)** and tick
   **Allow AI assistants to control Gel Blot Analyzer**.
   - The status line changes to `● On — listening on http://127.0.0.1:47615/mcp`.
   - The **Tools ▸ AI Assistant Control (MCP)…** menu item jumps straight to this group.
2. **Connect your AI app with one click.** In the same group, click the button for your app.
   Then restart that app.

| Button | What it does |
|---|---|
| **Claude Desktop** | Adds Gel Blot Analyzer to Claude Desktop's settings automatically |
| **LM Studio** | Adds it to LM Studio's `mcp.json` automatically |
| **Cursor** | Adds it to Cursor's `mcp.json` automatically |
| **Claude Code** | Copies a ready-made `claude mcp add …` command to paste into a terminal |
| **Ollama / other apps…** | Shows the connection details for any other app, including Open WebUI |

When a button writes to an app's settings file:

- Your existing entries are kept.
- A one-time backup is saved next to the file (`*.bak-gel-blot-analyzer`).
- A file it can't read is never overwritten.
- If the app isn't installed, the settings are copied to the clipboard instead.

That's it. Ask the AI something like *"Use Gel Blot Analyzer to show me the open image."*

If Gel Blot Analyzer isn't open when the AI first needs it, it starts automatically. This only
happens if AI control is turned on. When it is off, the AI is told to ask you to turn it on.

---

## 2. Settings reference (Settings ▸ AI Assistant Control)

| Setting | What it does |
|---|---|
| **Allow AI assistants to control Gel Blot Analyzer** | Master switch. Off by default. Turning it off disconnects every AI immediately. |
| **Permission** | *Full control*, or *View only*: the AI can take screenshots and read results, but can't change anything. |
| **Step delay** | Pause before each click or change (default 250 ms) so you can follow along. Set to 0 for full speed. |
| **Show the AI cursor and control highlights** | Shows the live preview: a violet **AI** pointer for clicks and drags on the image, and an outline around each control just before the AI changes it. |
| **Pause the AI** | Refuses all AI requests until you untick it. |
| **Activity Log…** | Opens a live, timestamped list of every AI action and its result. Also available from **Tools ▸ AI Activity Log…**. |
| **New Token** | Revokes access for apps connected by URL (see 3.2 and 3.3). One-click connections keep working. |

The status bar shows what the AI is doing, for example `● AI: set_control gamma_slider`.

**The AI can never change these settings.** They are hidden from it, and it can't answer some
confirmations for you, such as *Reset Image*.

---

## 3. Connecting other apps by hand

### 3.1 Apps that launch an MCP server

This covers LM Studio, Claude Desktop, Cursor, mcphost, ollmcp, Continue, Cline and similar
apps. Add this to the app's `mcpServers` configuration. **Ollama / other apps…** shows the exact
version for your installation.

| Installation | Configuration |
|---|---|
| **macOS app** | `{"mcpServers": {"gel-blot-analyzer": {"command": "/Applications/Gel Blot Analyzer.app/Contents/MacOS/Gel Blot Analyzer", "args": ["--mcp-stdio"]}}}` |
| **Windows (Microsoft Store)** | `{"mcpServers": {"gel-blot-analyzer": {"command": "gelblotanalyzer", "args": ["--mcp-stdio"]}}}` |
| **Windows (installer / portable)** | `"command": "C:\\…\\Gel Blot Analyzer.exe", "args": ["--mcp-stdio"]` |
| **Running from source** | `"command": "<python>", "args": ["…/gba_mcp_server.py"]` (any Python 3.8+) |

VS Code (`.vscode/mcp.json`) uses `"servers"` instead of `"mcpServers"` and needs
`"type": "stdio"` added to the entry.

### 3.2 Apps that connect to an MCP URL

This covers Claude Code, VS Code, LM Studio and others.

| | |
|---|---|
| URL | `http://127.0.0.1:47615/mcp` |
| Header | `Authorization: Bearer <token>` (token shown in **Ollama / other apps…**) |

Claude Code over URL:

```bash
claude mcp add --transport http gel-blot-analyzer http://127.0.0.1:47615/mcp --header "Authorization: Bearer <token>"
```

### 3.3 Ollama

Ollama runs models but doesn't connect to tools by itself. Use one of these front-ends with a
**model that supports tool calling** (for example Qwen 3, Llama 3.1 or later, or Mistral Small).
For screenshots to be useful, pick a model that also supports **vision**.

**Option A: Open WebUI (OpenAPI tool server, no MCP needed)**

1. In Open WebUI, go to **Settings ▸ Tools ▸ ➕ Add tool server**.
2. Set **URL** to `http://127.0.0.1:47615`. The spec is served at `/openapi.json`.
3. Set **Auth** to *Bearer* and paste the token from **Ollama / other apps…**.
4. Enable the tool server in your chat.

Notes:

- Images are not sent to text-only models. Screenshots are saved under
  `~/.gel_blot_analyzer/screenshots/` and the file path is returned instead.
- Requests from browser pages are accepted only from `localhost` / `127.0.0.1` origins. Open
  Open WebUI as `http://localhost:…`.
- If Open WebUI runs in Docker and is added as an *admin* (server-side) tool server, use
  `http://host.docker.internal:47615`.

**Option B: any MCP-capable Ollama client** (mcphost, ollmcp, Continue, Cline, …)

Use the `mcpServers` block from 3.1.

### 3.4 Reference

| Item | Value |
|---|---|
| stdio | `gba_mcp_server.py`, or the app with `--mcp-stdio` |
| Streamable HTTP | `POST /mcp` (JSON responses) |
| OpenAPI | `GET /openapi.json`, then `POST /tools/<tool_name>` with the arguments as JSON |
| Auth | `Authorization: Bearer <token>`. Alternatives: `X-GBA-Token: <token>`, or `?token=` |
| Health check | `GET /health` |
| MCP protocol versions | 2025-06-18, 2025-03-26, 2024-11-05 |
| Port | 47615. If it is busy, the next free port up to 47624 is used and recorded for the connector. |

Environment overrides for the connector:

| Variable | Effect |
|---|---|
| `GBA_PORT` | Use this port |
| `GBA_TOKEN` | Use this token |
| `GBA_APP` | Command used to start the app |
| `GBA_NO_LAUNCH=1` | Never start the app |

---

## 4. What the AI can do

All coordinates are **image pixels** (origin at the top-left of the current image). The AI
usually starts with `screenshot(target="image", grid=true)`, which returns the figure with a
labelled pixel grid.

| Area | Tools |
|---|---|
| **Look** | `get_status`, `get_annotations`, `get_analysis_results`, `screenshot`, `list_windows` |
| **Files** | `open_image`, `save_image` (with config), `export_pptx` |
| **Image** | `image_operation` (invert, flip, grayscale, 8-bit, optimize levels, reset), `adjust_levels`, `crop`, `set_padding` |
| **Annotate** | `set_marker_preset`, `place_markers`, `add_text`, `add_shape`, `remove_custom_items`, `history` |
| **Quantify** | `define_lane_regions`, `analyze_lanes`, `get_analysis_results`, `read_table` |
| **Any control** | `list_controls`, `set_control`, `click_control`, `select_tab`, `list_menu_actions`, `trigger_menu_action`, `press_keys` |
| **Viewer** | `viewer_click`, `viewer_drag`, `set_view` |
| **Dialogs** | `respond_to_dialog`, `close_window`, `answer_file_dialog`, `wait` |

**Every control is reachable.** The AI can use every button, slider, checkbox, list and text
field in every window. If a control is on another tab, that tab is brought to the front and the
control is scrolled into view first.

**Dialogs.** When an action opens a dialog, such as a confirmation, the densitometry window or
the Auto Analyze wizard, the AI gets its text and buttons and answers it the way you would.

---

## 5. Example prompts

- *"Open ~/Gels/western_0412.tif, invert it if the bands are white, optimize the levels, and show
  me the result."*
- *"Put MW markers at the ladder bands on the left lane and label the lanes Ctrl, siRNA-1,
  siRNA-2, siRNA-3."*
- *"Add a red arrow pointing at the band near 55 kDa in lane 3, then add padding for the
  labels."*
- *"Draw lane regions for lanes 2–6, run densitometry and give me the % of lane for the main band
  in each lane."*
- *"Save it as ~/Figures/fig2b.tif and export an editable PowerPoint next to it."*

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| The AI says "AI control is turned OFF" | Tick **Allow AI assistants…** in **Settings ▸ AI Assistant Control (MCP)**. |
| The app doesn't appear in the AI app | Restart the AI app after clicking its Connect button, then check its MCP or developer log. On macOS the command must point inside the `.app` (`…/Contents/MacOS/Gel Blot Analyzer`). |
| "Missing or wrong token" | Apps connected by URL need the current token from **Ollama / other apps…**. **New Token** invalidates the old one. |
| "PAUSED" or "VIEW ONLY" errors | Untick **Pause the AI**, or set **Permission** to **Full control**. |
| Clicks land in the wrong place | Ask the AI to use `screenshot(target="image", grid=true)`. Coordinates are image pixels, not screen pixels. |
| A file chooser opened and nothing happened | Native file dialogs can't be operated remotely. The AI must call `answer_file_dialog(path)` first. `open_image`, `save_image` and `export_pptx` already do this. |
| A local model calls tools badly | Use a larger model with tool calling, and ask for a few steps at a time. |

---

## 7. Security

- The server listens on **127.0.0.1 only** (this computer).
- Every request needs the secret token. It is stored in `~/.gel_blot_analyzer/ai_control.json`,
  readable only by your account.
- Browser requests from other web origins are rejected.
- Turning the switch off, or quitting the app, ends all AI access.
- Use **View only** for advice-only sessions.

---

## 8. For developers

### Files

| File | Role |
|---|---|
| `gba_mcp_schema.py` | Tool catalogue, server instructions, port and alias constants (no Qt). |
| `gba_ai_bridge.py` | In-app side: MCP and OpenAPI HTTP endpoints, GUI-thread dispatcher, tools (`t_<name>`), live-preview overlays, the Settings group, the Activity window and the one-click client connectors. |
| `gba_mcp_server.py` | The stdio connector that AI apps launch. Standard library only. |
| `Gel_blot_analyzer.py` | `--mcp-stdio` runs only the connector. The Settings tab provides `ai_settings_container`. `gba_ai_bridge.install()` is called after the window is created. |

### Adding a tool

1. Add its schema to `TOOLS` in `gba_mcp_schema.py`.
2. Add a `t_<name>(self, args)` method to `AIBridge`.
3. Run Qt work through `self.gui(fn)` for reads, or `self.action(fn, name)` for changes. `action`
   detects dialogs automatically.

Don't hold widget wrappers across a delay. Re-resolve them, and keep the owning window pinned in
`self._pin`, because PySide invalidates child wrappers of C++-created dialogs.

### Testing

Always run offscreen tests with `HOME` pointed at a temporary folder. The app saves window
geometry and settings from the small virtual display into the real user's config otherwise.

### Microsoft Store (MSIX) build

Declare an **App Execution Alias** in the package manifest. AI apps can't launch executables
inside the protected, versioned `WindowsApps` folder, and that path changes with every update.

```xml
<Package ... xmlns:uap3="http://schemas.microsoft.com/appx/manifest/uap/windows10/3"
             xmlns:desktop="http://schemas.microsoft.com/appx/manifest/desktop/windows10">
  ...
  <Application Id="GelBlotAnalyzer" Executable="Gel Blot Analyzer\Gel Blot Analyzer.exe"
               EntryPoint="Windows.FullTrustApplication">
    <Extensions>
      <uap3:Extension Category="windows.appExecutionAlias">
        <uap3:AppExecutionAlias>
          <desktop:ExecutionAlias Alias="gelblotanalyzer.exe" />
        </uap3:AppExecutionAlias>
      </uap3:Extension>
    </Extensions>
  </Application>
</Package>
```

The alias name must match `APP_ALIAS` in `gba_mcp_schema.py` (`gelblotanalyzer`). The app
detects that it is running from `WindowsApps` and then writes the alias, not the exe path, into
client configurations. The connector also uses the alias to start the app.

**Settings files under MSIX.**

- `~/.gel_blot_analyzer`, `~/.lmstudio` and `~/.cursor` live in the user profile, which MSIX does
  not virtualize, so all processes share them.
- MSIX does redirect *new* files created in `%APPDATA%`. If Claude Desktop's config file doesn't
  exist yet, the Connect button asks the user to open Claude Desktop's *Edit Config* once and
  then click again.

Test `--mcp-stdio` on the Windows build. It is a windowed (no-console) executable, so the app
reattaches stdin and stdout itself when they are missing.
