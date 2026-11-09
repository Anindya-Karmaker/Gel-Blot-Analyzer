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


SERVER_INSTRUCTIONS = """\
You are operating Gel Blot Analyzer, a desktop app for annotating gel/western-blot images
(molecular-weight markers, labels, arrows) and band densitometry. Every action you take
happens live in the user's window, with a purple AI cursor and highlight so they can watch.

COORDINATES: all x/y values are IMAGE pixels of the current image (origin top-left,
see get_status -> image.width/height). Use screenshot(target="image", grid=true) to see
the image with a labelled pixel grid before clicking or placing anything. Coordinates may
lie outside the image (e.g. negative) to annotate the padding area.

TYPICAL WORKFLOW
 1. get_status -> open_image(path) if nothing is loaded.
 2. screenshot(target="image", grid=true) to look at the gel.
 3. Clean up: image_operation (invert, optimize_levels, flip...), crop, set_padding.
 4. Annotate: set_marker_preset + place_markers (MW ladder), add_text, add_shape.
 5. Quantify: define_lane_regions then analyze_lanes; a densitometry dialog opens —
    inspect it with list_controls(window_id=...) / screenshot(target="dialog") and
    finish with respond_to_dialog or click_control. Read numbers with get_analysis_results.
 6. save_image(path) / export_pptx(path).

ANYTHING ELSE: every button, slider, checkbox, combo box and text field in every window
is reachable with list_controls -> set_control / click_control (controls on other tabs are
switched to automatically). Menu commands: list_menu_actions -> trigger_menu_action.
Interactive tools are armed with click_control and then driven with viewer_click /
viewer_drag. When an action opens a dialog the result says status="dialog_open" with the
dialog's window_id, text and buttons; answer it before continuing. When a button would
open a file chooser, call answer_file_dialog(path) first (native file choosers can't be
operated otherwise). Every action returns the app's status-bar message — read it.
Prefer small verified steps: act, then check with get_status / screenshot.
"""

_XY = {"type": "number"}

TOOLS = [
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
        "description": "Place standard MW markers by clicking with the real marker tool. side "
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
        "description": "Add a custom text label (or arrow glyph like '←') at IMAGE coordinates "
                       "(x,y is the text anchor). Color as '#RRGGBB' or a name.",
        "inputSchema": {
            "type": "object",
            "properties": {"x": _XY, "y": _XY, "text": {"type": "string"},
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
        "description": "Run densitometry on the defined regions: as='sample' (Analyze as Sample(s)) "
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
