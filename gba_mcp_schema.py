"""Gel Blot Analyzer — MCP tool catalogue (pure data, standard library only).

Shared by the two halves of the AI-control feature:

  * gba_ai_bridge.py   — runs INSIDE the app and executes the tools on the Qt GUI thread
                         (also serves MCP directly over Streamable HTTP at /mcp).
  * gba_mcp_server.py  — tiny stdio MCP server that AI clients launch; it forwards every
                         request to the running app.

Keeping the catalogue here (no Qt imports) lets the stdio server list every tool even
while the app is still starting, and guarantees both transports advertise the same tools.
"""

import os

SERVER_NAME = "gel-blot-analyzer"
SERVER_TITLE = "Gel Blot Analyzer"
DEFAULT_PORT = 47615
# Windows Store (MSIX) builds: App Execution Alias declared in the package manifest, so AI apps
# can launch the connector without knowing the versioned, protected WindowsApps install path.
APP_ALIAS = "gelblotanalyzer"
PORT_TRIES = 10
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]


def state_dir():
    return os.path.join(os.path.expanduser("~"), ".gel_blot_analyzer")


def settings_path():
    """Port, token and panel preferences. Read by the stdio server to find the app."""
    return os.path.join(state_dir(), "ai_control.json")


def pid_alive(pid):
    """True if a process with this id is still running (never signals it).

    Used to decide whether the ``active_port``/``pid`` recorded in ai_control.json still
    describes a live app instance. On Windows ``os.kill(pid, 0)`` would TERMINATE the
    process, so the handle-based query is used there.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == 259          # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


SERVER_INSTRUCTIONS = """\
You operate Gel Blot Analyzer, a desktop app for gel / western-blot images: molecular-weight
(MW) ladder labels, lane labels, arrows, band densitometry and quantification. Every action
happens live in the user's window (purple AI cursor).

ALWAYS START WITH: get_status -> (open_image) -> inspect_gel. inspect_gel tells you the band
polarity, lanes, ladder lane, real bands per lane, possible standards, and QUESTIONS to ask
the user. If questions_for_user is not empty, ask the user before quantifying.

GOLDEN RULES
 1. Analysis needs WHITE lanes with BLACK bands. DNA/RNA agarose gels (EtBr/SYBR), stain-free
    and fluorescent gels and raw chemiluminescence blots show LIGHT bands on a DARK background:
    call fix_band_polarity (it inverts only when needed; never invert twice).
 2. MW labels: use label_mw_ladder (the app's Auto Lane analysis). It finds the ladder bands
    and writes the preset's values top->bottom. Choose the preset first (set_marker_preset or
    preset=...). Do NOT click markers by hand unless the user asks.
 3. Densitometry / lane quantification: use auto_analyze_gel (the app's Auto Analyze Gel
    wizard). It finds lanes, labels the ladder, measures every band and removes noise peaks
    (only bands >= 5x the noise and >= 1% of the lane's strongest band are kept).
 4. Standards: if inspect_gel reports a standard-series candidate (same band position, similar
    size, different intensity), ASK the user whether those lanes are standards and their
    amounts. Then quantify_with_standards(standards=[{lane, quantity}, ...]).
 5. Gel + western blot in one figure: combine_images(path, placement='below' | 'right' | ...)
    adds the padding and places the second image for you.
 6. Check your work with screenshot(target='image') and read numbers with get_analysis_results.

COORDINATES are IMAGE pixels (origin top-left, see get_status image.width/height);
screenshot(target='image', grid=true) shows a pixel grid. Negative values reach the padding.

LOW-LEVEL CONTROL (only when the smart tools don't cover it): list_controls -> set_control /
click_control; list_menu_actions -> trigger_menu_action; viewer_click / viewer_drag for armed
tools. When a result says status='dialog_open', answer that dialog (respond_to_dialog) before
anything else. Call answer_file_dialog(path) before a click that opens a file chooser.
Unsure what to do? Call get_workflow_guide(task).
"""

WORKFLOW_GUIDES = {
    "overview": [
        "get_status; if image is null -> open_image(path).",
        "inspect_gel(gel_type='protein'|'dna'|'rna'|'western').",
        "Ask the user every item in questions_for_user (ladder lane? standards and amounts?).",
        "fix_band_polarity if inspect_gel says polarity = light_bands.",
        "Only MW labels needed -> label_mw_ladder(lane=<ladder lane>, preset=<ladder name>).",
        "Densitometry needed -> auto_analyze_gel(densitometry=true, ladder_lanes=[...], lane_labels=[...]).",
        "Standards confirmed -> quantify_with_standards(standards=[{lane: n, quantity: q}, ...]).",
        "screenshot(target='image') to check, then save_image(path) or export_pptx(path).",
    ],
    "protein_gel": [
        "SDS-PAGE (Coomassie/silver): bands are dark on light - no inversion needed.",
        "Stain-free gels are light on dark - fix_band_polarity first.",
        "set_marker_preset to the ladder that was loaded (e.g. 'Precision Plus Protein').",
        "label_mw_ladder(lane=<ladder lane>) or auto_analyze_gel(...) for densitometry.",
    ],
    "dna_gel": [
        "DNA/RNA agarose gels (EtBr, SYBR, GelRed) are LIGHT bands on a DARK background.",
        "fix_band_polarity -> lanes become white and bands black (required for densitometry).",
        "set_marker_preset to the DNA ladder (e.g. '1 kb', 'GeneRuler'); labels are in bp.",
        "label_mw_ladder(lane=<ladder lane>) then auto_analyze_gel for band intensities.",
    ],
    "western_blot": [
        "Raw chemiluminescence/fluorescence exports are light on dark -> fix_band_polarity.",
        "A prestained ladder is often invisible on the chemiluminescence image: if inspect_gel finds "
        "no ladder, ask the user for the ladder lane, or label from the colorimetric/marker image.",
        "auto_analyze_gel(densitometry=true) for band intensities; normalise to a loading control by "
        "dividing each lane's target area by its loading-control area (get_analysis_results).",
    ],
    "densitometry": [
        "auto_analyze_gel(densitometry=true, expected_lanes=<n if known>, ladder_lanes=[...]).",
        "Noise is removed automatically: min_snr (default 5) and min_relative_intensity (default 0.01).",
        "Raise min_snr (e.g. 8) for very noisy blots; lower min_relative_intensity to 0.002 for "
        "purity/impurity analysis where faint bands matter.",
        "review=true leaves the densitometry dialog open so you can screenshot it before OK.",
        "get_analysis_results for areas, % of lane, saturation flags.",
        "Saturated bands (is_saturated) under-report - tell the user to re-image with shorter exposure.",
    ],
    "standards": [
        "Standards = same protein/DNA loaded at known, different amounts (dilution series).",
        "inspect_gel lists standard_series_candidates: same band position, similar size, intensity changes.",
        "ASK the user: 'Are lanes X-Y standards? What amount is in each lane?' Never guess amounts.",
        "Run densitometry first (auto_analyze_gel), then "
        "quantify_with_standards(standards=[{lane: 2, quantity: 50}, {lane: 3, quantity: 100}, ...], unit='ng').",
        "Use model='Power (Log-Log)' if R2 < 0.95 with 'Linear' (slightly saturating standards).",
        "Sample amounts outside the standards' range are flagged: report them as estimates.",
    ],
    "combine": [
        "Open the gel first (open_image), finish its labels.",
        "combine_images(path=<blot file>, placement='below') puts the blot under the gel at the gel's "
        "width, adding the bottom padding automatically; placement='right' puts it side by side at the "
        "gel's height. gap sets the white space in px.",
        "band_polarity='auto' shows the blot with dark bands on light, like the gel.",
        "rasterize=true merges it into one image (needed before analysing the blot itself).",
        "Add labels for the blot with add_text / add_shape (coordinates from the result's position).",
        "A lane table (what was loaded per lane) goes above the gel with add_table.",
    ],
}

_XY = {"type": "number"}

TOOLS = [
    # ------------------------------------------------------------------ smart gel tools
    {
        "name": "inspect_gel",
        "description": "START HERE for any gel or blot. Reads the image like an expert: band "
                       "polarity (dark or light bands, i.e. whether inversion is needed), lanes "
                       "with their x-ranges, real bands per lane (noise removed), which lane is the "
                       "MW ladder, empty lanes, and possible standard (dilution) series. Returns "
                       "advice, questions_for_user (ASK them) and next_steps. Read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gel_type": {"type": "string", "enum": ["auto", "protein", "dna", "rna", "western"],
                             "default": "auto"},
                "expected_lanes": {"type": "integer", "minimum": 2,
                                   "description": "ONLY if the user said how many lanes. Otherwise omit."},
            },        },
        "readOnly": True,
    },
    {
        "name": "fix_band_polarity",
        "description": "Make the image show BLACK bands on WHITE lanes, as densitometry requires. "
                       "mode='auto' inverts only if the bands are currently light on dark (DNA/RNA "
                       "gels, stain-free, fluorescence, raw chemiluminescence); 'check' only reports; "
                       "'invert' always toggles.",
        "inputSchema": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["auto", "check", "invert"], "default": "auto"}}},
    },
    {
        "name": "label_mw_ladder",
        "description": "Label the molecular-weight ladder with the app's Auto Lane analysis: finds "
                       "the ladder bands in one lane and places the preset's MW values beside them "
                       "(top to bottom), then adds padding so labels fit. Fixes polarity first and "
                       "removes noise peaks; if the band count differs from the preset it retries / "
                       "warns. Usually only lane (from inspect_gel) and preset are needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lane": {"type": "integer", "minimum": 1,
                         "description": "Ladder lane number from inspect_gel (default: detected ladder)."},
                "x0": _XY, "x1": _XY,
                "y0": _XY, "y1": _XY,
                "side": {"type": "string", "enum": ["auto", "left", "right"], "default": "auto",
                         "description": "Which side the labels go (auto = nearest edge)."},
                "preset": {"type": "string", "description": "Ladder preset name (see get_annotations "
                           "marker_presets). Default: the current preset."},
                "padding": {"type": "boolean", "default": True},
                "review": {"type": "boolean", "default": False,
                           "description": "Leave the band dialog open for checking before OK."},
                "fix_polarity": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "auto_analyze_gel",
        "description": "Full analysis with the app's Auto Analyze Gel wizard (call with NO arguments unless the user gave values): detects lanes, labels "
                       "lanes (top labels), places MW labels from the ladder lane(s), and (if "
                       "densitometry=true) measures every band of every lane. Polarity is fixed first "
                       "and noise peaks are removed, so only true bands are reported. Results open "
                       "in the results window; numbers are returned.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "densitometry": {"type": "boolean", "default": True},
                "ladder_lanes": {"type": "array", "items": {"type": "integer"},
                                 "description": "Ladder lane numbers (inspect_gel numbering). Default: "
                                                "auto-detected. [] = no ladder."},
                "lane_labels": {"type": "array", "items": {"type": "string"},
                                "description": "Top label for each lane, left to right."},
                "label_lanes": {"type": "boolean", "default": True},
                "preset": {"type": "string", "description": "Ladder preset name."},
                "expected_lanes": {"type": "integer", "minimum": 1},
                "lanes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                          "description": "Optional explicit lanes [[x0,x1],...] in image px."},
                "padding": {"type": "boolean", "default": True},
                "min_snr": {"type": "number", "default": 5,
                            "description": "Bands must be this many times above the lane noise."},
                "min_relative_intensity": {"type": "number", "default": 0.01,
                                           "description": "Sample bands below this fraction of the "
                                                          "lane's strongest band are dropped."},
                "remove_noise_bands": {"type": "boolean", "default": True},
                "review": {"type": "boolean", "default": False},
                "fix_polarity": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "quantify_with_standards",
        "description": "Quantify all bands against standards of known amount, after densitometry. "
                       "standards = [{lane: <lane id from results>, quantity: <amount>}, ...] (>= 2 "
                       "different amounts). Uses each standard lane's strongest band (band='total' "
                       "for the whole lane, or a band number). Returns the curve (R²) and every "
                       "lane's calculated amounts. ONLY use amounts the user gave you.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "standards": {"type": "array", "minItems": 2, "items": {
                    "type": "object", "properties": {"lane": {"type": "integer"},
                                                     "quantity": {"type": "number"}},
                    "required": ["lane", "quantity"]}},
                "unit": {"type": "string", "description": "e.g. ng, ug, pmol (for the report)."},
                "model": {"type": "string", "enum": ["Linear", "Power (Log-Log)", "Polynomial (Deg 2)",
                                                     "Polynomial (Deg 3)", "4-PL Sigmoidal"],
                          "default": "Linear"},
                "band": {"description": "'strongest' (default), 'total', or a band number (1 = top)."},
                "replace": {"type": "boolean", "default": True,
                            "description": "false = add to the existing standards."},
            },
            "required": ["standards"],
        },
    },
    {
        "name": "combine_images",
        "description": "Put a second image (e.g. the western blot) next to the current one (e.g. the "
                       "gel) in one figure: adds the needed padding and places it below / above / "
                       "right / left, scaled to the same width (below/above) or height (left/right). "
                       "Shows its bands dark on light automatically.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "placement": {"type": "string", "enum": ["below", "above", "right", "left"],
                              "default": "below"},
                "gap": {"type": "integer", "minimum": 0, "default": 20},
                "fit": {"type": "boolean", "default": True},
                "scale_percent": {"type": "number", "minimum": 10, "maximum": 300},
                "band_polarity": {"type": "string", "enum": ["auto", "as_is", "invert"],
                                  "default": "auto"},
                "rasterize": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_workflow_guide",
        "description": "Step-by-step recipe for a task: overview, protein_gel, dna_gel, western_blot, "
                       "densitometry, standards, combine.",
        "inputSchema": {"type": "object", "properties": {"task": {"type": "string", "enum": [
            "overview", "protein_gel", "dna_gel", "western_blot", "densitometry", "standards",
            "combine"]}}},
        "readOnly": True,
    },
    # ------------------------------------------------------------------ observe
    {
        "name": "get_status",
        "description": "Overview of the app: loaded image (path, size, bit depth), viewer zoom, "
                       "current tab, armed tool, status-bar message, open windows/dialogs, and "
                       "whether there are unsaved changes. Call this first.",
        "inputSchema": {"type": "object", "properties": {}},
        "readOnly": True,
    },
    {
        "name": "get_annotations",
        "description": "All annotation state in image-pixel coordinates: left/right/top MW marker "
                       "positions + labels, custom text items, shapes (lines/arrows/rectangles), "
                       "densitometry lane regions, padding, fonts, image adjustments and the list "
                       "of available marker presets. Indices returned here are what "
                       "remove_custom_items expects.",
        "inputSchema": {"type": "object", "properties": {}},
        "readOnly": True,
    },
    {
        "name": "get_analysis_results",
        "description": "Densitometry results: per-lane band areas, % of lane, peak positions and "
                       "saturation flags, standard-curve model and calculated quantities.",
        "inputSchema": {"type": "object", "properties": {}},
        "readOnly": True,
    },
    {
        "name": "screenshot",
        "description": "Look at the app. target='image' (default): the current image with all "
                       "annotations, exactly as it will be saved, optionally with a labelled pixel "
                       "grid in IMAGE coordinates (best for deciding where to click/place). "
                       "'viewer': the on-screen viewer. 'window': the whole main window. "
                       "'dialog': the front-most dialog (or window_id).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["image", "viewer", "window", "dialog"],
                           "default": "image"},
                "window_id": {"type": "string", "description": "For target='dialog': which window."},
                "grid": {"type": "boolean", "default": False,
                         "description": "image target only: overlay a coordinate grid."},
                "grid_step": {"type": "integer", "minimum": 10,
                              "description": "Grid spacing in image px (auto if omitted)."},
                "max_size": {"type": "integer", "minimum": 200, "maximum": 4000, "default": 1280,
                             "description": "Longest side of the returned PNG."},
            },
        },
        "readOnly": True,
    },
    {
        "name": "list_windows",
        "description": "Visible windows and dialogs with their window_id, title, modality and "
                       "(for message boxes) text and buttons.",
        "inputSchema": {"type": "object", "properties": {}},
        "readOnly": True,
    },
    {
        "name": "list_controls",
        "description": "List interactive controls (buttons, checkboxes, sliders, spin boxes, combo "
                       "boxes, text fields, tables, tabs) of a window with their id, label, tab, "
                       "current value, range/options and enabled state. Filter with tab and/or "
                       "query (case-insensitive substring of label/tooltip/id).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window_id": {"type": "string", "description": "Default: the main window ('main')."},
                "tab": {"type": "string", "description": "Only controls on this tab (substring)."},
                "query": {"type": "string"},
                "include_labels": {"type": "boolean", "default": False,
                                   "description": "Also list read-only text labels (results, hints)."},
                "include_disabled": {"type": "boolean", "default": True},
                "limit": {"type": "integer", "default": 250, "minimum": 1, "maximum": 2000},
            },
        },
        "readOnly": True,
    },
    {
        "name": "read_table",
        "description": "Read the full contents of a table control (e.g. the densitometry results "
                       "tables) as rows of cells.",
        "inputSchema": {
            "type": "object",
            "properties": {"control_id": {"type": "string"},
                           "max_rows": {"type": "integer", "default": 500}},
            "required": ["control_id"],
        },
        "readOnly": True,
    },
    {
        "name": "list_menu_actions",
        "description": "Menu-bar and toolbar commands (File/Edit/View/Tools...) with their names, "
                       "shortcuts and enabled/checked state.",
        "inputSchema": {"type": "object", "properties": {}},
        "readOnly": True,
    },
    # ------------------------------------------------------------------ generic UI
    {
        "name": "set_control",
        "description": "Set a control's value exactly as a user would, firing its normal handlers. "
                       "Sliders/spin boxes: number. Checkboxes/toggle buttons: true/false. Combo "
                       "boxes: item text (or integer index). Text fields: string. Tab widgets: tab "
                       "name. The control's tab is brought to front first.",
        "inputSchema": {
            "type": "object",
            "properties": {"control_id": {"type": "string"},
                           "value": {"description": "number | boolean | string"}},
            "required": ["control_id", "value"],
        },
    },
    {
        "name": "click_control",
        "description": "Click a button (push/tool/radio/checkbox) by control_id. If the click opens "
                       "a dialog the result reports it (status='dialog_open').",
        "inputSchema": {
            "type": "object",
            "properties": {"control_id": {"type": "string"}},
            "required": ["control_id"],
        },
    },
    {
        "name": "select_tab",
        "description": "Bring a tab to the front (main window: 'Image and Contrast', 'Transform', "
                       "'Markers', 'Overlap Images', 'Analysis', 'Settings').",
        "inputSchema": {
            "type": "object",
            "properties": {"tab": {"type": "string"}, "window_id": {"type": "string"}},
            "required": ["tab"],
        },
    },
    {
        "name": "trigger_menu_action",
        "description": "Run a menu/toolbar command by its name (from list_menu_actions), e.g. "
                       "'Undo', 'Auto Analyze Gel', 'Automatic Lane Markers', 'Reset View'.",
        "inputSchema": {
            "type": "object",
            "properties": {"action": {"type": "string"},
                           "checked": {"type": "boolean",
                                       "description": "For checkable actions: desired state."}},
            "required": ["action"],
        },
    },
    {
        "name": "respond_to_dialog",
        "description": "Press a button in a dialog or message box by its text (e.g. 'OK', 'Yes', "
                       "'Cancel', 'Apply', 'Save'). Defaults to the front-most dialog.",
        "inputSchema": {
            "type": "object",
            "properties": {"button": {"type": "string"}, "window_id": {"type": "string"}},
            "required": ["button"],
        },
    },
    {
        "name": "close_window",
        "description": "Close a dialog/window (equivalent to its close button / Esc).",
        "inputSchema": {"type": "object", "properties": {"window_id": {"type": "string"}},
                        "required": ["window_id"]},
    },
    {
        "name": "press_keys",
        "description": "Press a key or shortcut, e.g. 'Escape' (cancel the armed tool), 'Delete', "
                       "'Ctrl+Z'. Shortcuts that belong to a command are run directly.",
        "inputSchema": {
            "type": "object",
            "properties": {"keys": {"type": "string"}, "window_id": {"type": "string"}},
            "required": ["keys"],
        },
    },
    {
        "name": "answer_file_dialog",
        "description": "Pre-answer the NEXT file chooser the app opens (open or save) with this path "
                       "(or folder). Use right before a click that opens a file chooser.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"},
                           "file_filter": {"type": "string",
                                           "description": "Optional filter text to report as chosen, e.g. 'PNG'."}},
            "required": ["path"],
        },
    },
    # ------------------------------------------------------------------ viewer
    {
        "name": "viewer_click",
        "description": "Click on the image viewer at IMAGE coordinates — used by the armed tool "
                       "(placing markers, quad corners, predicting MW, selecting items...).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": _XY, "y": _XY,
                "button": {"type": "string", "enum": ["left", "right"], "default": "left"},
                "double_click": {"type": "boolean", "default": False},
                "modifiers": {"type": "array", "items": {"type": "string",
                              "enum": ["shift", "ctrl", "alt", "meta"]}},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "viewer_drag",
        "description": "Press, move through the given IMAGE-coordinate points, and release — draws "
                       "rectangles/lines with the armed tool or moves items in Move/Resize mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "points": {"type": "array", "minItems": 2,
                           "items": {"type": "array", "items": {"type": "number"},
                                     "minItems": 2, "maxItems": 2}},
                "duration_ms": {"type": "integer", "default": 400, "minimum": 0, "maximum": 10000},
                "button": {"type": "string", "enum": ["left", "right"], "default": "left"},
                "modifiers": {"type": "array", "items": {"type": "string",
                              "enum": ["shift", "ctrl", "alt", "meta"]}},
            },
            "required": ["points"],
        },
    },
    {
        "name": "set_view",
        "description": "Zoom/pan the viewer (does not change the image). reset=true returns to "
                       "the fitted 100% view.",
        "inputSchema": {
            "type": "object",
            "properties": {"zoom": {"type": "number", "minimum": 0.1, "maximum": 20},
                           "center_x": _XY, "center_y": _XY,
                           "reset": {"type": "boolean"}},
        },
    },
    # ------------------------------------------------------------------ files
    {
        "name": "open_image",
        "description": "Open an image file (TIFF/PNG/JPG/BMP...). A matching *_config.txt next to "
                       "it restores its saved annotations/analysis.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                        "required": ["path"]},
    },
    {
        "name": "save_image",
        "description": "Save like File > Save with Config: writes <base>_original, <base>_modified "
                       "(annotated) and <base>_config.txt. The extension of path picks the format "
                       "(.tif/.png/.jpg/.bmp). If files exist the app asks to overwrite — answer "
                       "with respond_to_dialog, or pass overwrite=true.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"},
                           "overwrite": {"type": "boolean", "default": False}},
            "required": ["path"],
        },
    },
    {
        "name": "export_pptx",
        "description": "Export to an editable PowerPoint (.pptx) file.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"},
                                                          "overwrite": {"type": "boolean", "default": False}},
                        "required": ["path"]},
    },
    # ------------------------------------------------------------------ editing
    {
        "name": "history",
        "description": "Undo or redo the last change(s).",
        "inputSchema": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["undo", "redo"]},
                           "steps": {"type": "integer", "default": 1, "minimum": 1, "maximum": 50}},
            "required": ["action"],
        },
    },
    {
        "name": "image_operation",
        "description": "One-click image operations: invert, flip_vertical, flip_horizontal, "
                       "grayscale, to_8bit, optimize_levels (auto black/white points), "
                       "reset_adjustments (levels/gamma/filters back to neutral), "
                       "reset_image (discard ALL edits and annotations — asks first).",
        "inputSchema": {
            "type": "object",
            "properties": {"operation": {"type": "string", "enum": [
                "invert", "flip_vertical", "flip_horizontal", "grayscale", "to_8bit",
                "optimize_levels", "reset_adjustments", "reset_image"]}},
            "required": ["operation"],
        },
    },
    {
        "name": "adjust_levels",
        "description": "Set display levels: black_point / white_point in raw pixel values (0-255 "
                       "for 8-bit, 0-65535 for 16-bit) and gamma (1.0 = neutral). Omitted values "
                       "are left unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {"black_point": {"type": "number"}, "white_point": {"type": "number"},
                           "gamma": {"type": "number", "minimum": 0.01, "maximum": 10}},
        },
    },
    {
        "name": "crop",
        "description": "Crop the image to a rectangle in IMAGE coordinates (drawn with the real "
                       "crop tool, then applied). Annotations are kept and re-positioned.",
        "inputSchema": {
            "type": "object",
            "properties": {"x": _XY, "y": _XY, "width": _XY, "height": _XY},
            "required": ["x", "y", "width", "height"],
        },
    },
    {
        "name": "set_padding",
        "description": "Add white space around the image for labels. Give left/right/top/bottom in "
                       "px, or recommended=true to let the app compute what the labels need. "
                       "apply=false only fills the fields.",
        "inputSchema": {
            "type": "object",
            "properties": {"left": {"type": "integer", "minimum": 0},
                           "right": {"type": "integer", "minimum": 0},
                           "top": {"type": "integer", "minimum": 0},
                           "bottom": {"type": "integer", "minimum": 0},
                           "recommended": {"type": "boolean", "default": False},
                           "apply": {"type": "boolean", "default": True}},
        },
    },
    {
        "name": "set_marker_preset",
        "description": "Choose a molecular-weight ladder preset (names in get_annotations -> "
                       "marker_presets). Sets the labels used by place_markers.",
        "inputSchema": {"type": "object", "properties": {"preset": {"type": "string"}},
                        "required": ["preset"]},
    },
    {
        "name": "place_markers",
        "description": "MANUAL placement (prefer label_mw_ladder). Place standard MW markers by clicking with the real marker tool. side "
                       "'left'/'right': positions are band y-values (one per ladder band, top to "
                       "bottom) and x is the ladder lane's x. side 'top': positions are lane "
                       "x-values and y is the label row. Labels come from the active preset / "
                       "top-label list, in order. clear_existing replaces that side.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "side": {"type": "string", "enum": ["left", "right", "top"]},
                "positions": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                "x": {"type": "number", "description": "left/right: x of the click (ladder lane)."},
                "y": {"type": "number", "description": "top: y of the click."},
                "labels": {"type": "array", "items": {"type": "string"},
                           "description": "Optional labels to use instead of the preset/top list."},
                "clear_existing": {"type": "boolean", "default": False},
            },
            "required": ["side", "positions"],
        },
    },
    {
        "name": "add_text",
        "description": "Add a custom text label (or arrow glyph like '←') at IMAGE coordinates. "
                       "x,y is the text ANCHOR: the middle of the text (anchor='center', default), "
                       "of its left edge ('left' - use for '← name' labels so the arrow tip sits at "
                       "x and labels line up in a column) or of its right edge ('right'). "
                       "Color as '#RRGGBB' or a name.",
        "inputSchema": {
            "type": "object",
            "properties": {"x": _XY, "y": _XY, "text": {"type": "string"},
                           "anchor": {"type": "string", "enum": ["center", "left", "right"],
                                      "default": "center"},
                           "color": {"type": "string", "default": "#000000"},
                           "font": {"type": "string"},
                           "font_size": {"type": "integer", "minimum": 2, "maximum": 400},
                           "bold": {"type": "boolean", "default": False},
                           "italic": {"type": "boolean", "default": False}},
            "required": ["x", "y", "text"],
        },
    },
    {
        "name": "add_shape",
        "description": "Add a line, arrow (x1,y1 -> x2,y2, head at x2,y2) or rectangle "
                       "(x,y,width,height) in IMAGE coordinates.",
        "inputSchema": {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": ["line", "arrow", "rectangle"]},
                           "x1": _XY, "y1": _XY, "x2": _XY, "y2": _XY,
                           "x": _XY, "y": _XY, "width": _XY, "height": _XY,
                           "color": {"type": "string", "default": "#000000"},
                           "thickness": {"type": "number", "default": 2, "minimum": 0.5}},
            "required": ["type"],
        },
    },
    {
        "name": "add_table",
        "description": "Add a table to the figure - e.g. a +/- grid above the gel saying what was "
                       "loaded in each lane. cells is a list of rows (each a list of strings). By "
                       "default, when top (lane) labels exist, the columns snap to the lanes: the "
                       "first column holds the row labels and each further column sits over one "
                       "lane, starting at first_lane (2 skips a ladder in lane 1). Without x/y the "
                       "table goes above the top labels and top padding is added if there is no "
                       "room. One undo step. The table is ONE object (a rectangle shape; remove it "
                       "with remove_custom_items kind='shape'); convert_to_items=true makes "
                       "separate text items and lines instead.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cells": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                          "description": "Rows of cell texts, e.g. [[\"IL-2\",\"-\",\"+\"],[\"AAT\",\"+\",\"-\"]]. "
                                         "Empty string = empty cell."},
                "column_widths_mode": {"type": "string", "enum": ["lanes", "fit", "equal"],
                                       "description": "lanes = one column per lane (needs top labels); "
                                                      "fit = fit the text; equal = all as wide as the "
                                                      "widest. Default: lanes if top labels exist, else fit."},
                "first_column_is_labels": {"type": "boolean", "default": True,
                                           "description": "lanes mode: first column is row labels, not a lane."},
                "first_lane": {"type": "integer", "minimum": 1,
                               "description": "lanes mode: lane number of the first lane column "
                                              "(default 1, or 2 when lane 1 is labeled as a ladder)."},
                "x": _XY, "y": _XY,
                "font": {"type": "string"},
                "font_size": {"type": "integer", "minimum": 2, "maximum": 150},
                "bold": {"type": "boolean", "description": "Bold every cell."},
                "italic": {"type": "boolean", "description": "Italic every cell."},
                "bold_rows": {"type": "array", "items": {"type": "integer", "minimum": 1},
                              "description": "Rows to make bold (1-based), e.g. [1] for a header row."},
                "bold_columns": {"type": "array", "items": {"type": "integer", "minimum": 1},
                                 "description": "Columns to make bold (1-based), e.g. [1] for the row labels."},
                "italic_rows": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "underline_rows": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "merge": {"type": "array",
                          "description": "Merged cells, e.g. a heading spanning several lanes: "
                                         "[{\"row\": 1, \"col\": 2, \"colspan\": 4}]. row/col are 1-based "
                                         "(top-left cell of the block); its text is shown centered "
                                         "across the block.",
                          "items": {"type": "object",
                                    "properties": {"row": {"type": "integer", "minimum": 1},
                                                   "col": {"type": "integer", "minimum": 1},
                                                   "rowspan": {"type": "integer", "minimum": 1},
                                                   "colspan": {"type": "integer", "minimum": 1}},
                                    "required": ["row", "col"]}},
                "color": {"type": "string", "description": "Text color."},
                "align": {"type": "string", "enum": ["left", "center", "right"]},
                "outline": {"type": "boolean", "default": True},
                "inner_lines": {"type": "boolean", "default": True},
                "line_color": {"type": "string"},
                "thickness": {"type": "number", "minimum": 0.5, "description": "Line thickness in image px."},
                "column_widths": {"type": "array", "items": {"type": ["number", "null"]},
                                  "description": "Manual widths in image px, in order; null = automatic."},
                "row_heights": {"type": "array", "items": {"type": ["number", "null"]},
                                "description": "Manual heights in image px, in order; null = automatic."},
                "auto_pad": {"type": "boolean", "default": True,
                             "description": "Extend the top padding when the table does not fit above the labels."},
                "convert_to_items": {"type": "boolean", "default": False},
            },
            "required": ["cells"],
        },
    },
    {
        "name": "remove_custom_items",
        "description": "Delete custom text items and/or shapes by index (from get_annotations), "
                       "or all of them.",
        "inputSchema": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["text", "shape", "all"]},
                           "indices": {"type": "array", "items": {"type": "integer"}}},
            "required": ["kind"],
        },
    },
    # ------------------------------------------------------------------ analysis
    {
        "name": "define_lane_regions",
        "description": "Draw densitometry lane regions with the Analysis tab's region tool. Each "
                       "lane is {x,y,width,height} (rectangle) or {corners:[[x,y]x4]} (skewed "
                       "quadrilateral, clockwise from top-left). clear_existing removes old ones.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lanes": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                "clear_existing": {"type": "boolean", "default": False},
            },
            "required": ["lanes"],
        },
    },
    {
        "name": "analyze_lanes",
        "description": "Manual densitometry (prefer auto_analyze_gel). Run densitometry on the defined regions: as='sample' (Analyze as Sample(s)) "
                       "or as='standard' (Process Selected as Standard). This opens the interactive "
                       "densitometry dialog — check it with screenshot/list_controls, then press "
                       "its OK/Accept button with respond_to_dialog.",
        "inputSchema": {
            "type": "object",
            "properties": {"as": {"type": "string", "enum": ["sample", "standard"], "default": "sample"}},
        },
    },
    {
        "name": "wait",
        "description": "Wait for a still-running action (e.g. one that opened a dialog) to finish, "
                       "or simply pause. Returns finished background actions.",
        "inputSchema": {"type": "object",
                        "properties": {"ms": {"type": "integer", "default": 1000, "minimum": 0,
                                              "maximum": 120000}}},
        "readOnly": True,
    },
]

READ_ONLY_TOOLS = frozenset(t["name"] for t in TOOLS if t.get("readOnly"))


def public_tools():
    """Tool list as sent over MCP (internal keys stripped, annotations added)."""
    out = []
    for t in TOOLS:
        d = {k: v for k, v in t.items() if k != "readOnly"}
        d["annotations"] = {"readOnlyHint": bool(t.get("readOnly")),
                            "openWorldHint": False}
        out.append(d)
    return out
