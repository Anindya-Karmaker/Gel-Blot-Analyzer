"""Gel Blot Analyzer — MCP tool catalogue (pure data, standard library only).

Shared by the two halves of the AI-control feature:

  * gba_ai_bridge.py   — runs INSIDE the app and executes the tools on the Qt GUI thread
                         (also serves MCP directly over Streamable HTTP at /mcp).
  * gba_mcp_server.py  — tiny stdio MCP server that AI clients launch; it forwards every
                         request to the running app.

Keeping the catalogue here (no Qt imports) lets the stdio server list every tool even
while the app is still starting, and guarantees both transports advertise the same tools.
"""

import difflib
import json
import math
import os
import re

SERVER_NAME = "gel-blot-analyzer"
SERVER_VERSION = "9.9.0"           # reported in serverInfo (keep in step with APP_VERSION)
SERVER_TITLE = "Gel Blot Analyzer"
DEFAULT_PORT = 47615
# Windows Store (MSIX) builds: App Execution Alias declared in the package manifest, so AI apps
# can launch the connector without knowing the versioned, protected WindowsApps install path.
APP_ALIAS = "gelblotanalyzer"
PORT_TRIES = 10
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
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

QUICKEST PATH ("analyze this gel"): auto_analyze_gel(path=<image file>) does everything; if it
returns status='needs_user_input', ask the user and call it again with the answer.

OTHERWISE START WITH: get_status -> (open_image) -> inspect_gel. inspect_gel tells you the band
polarity, lanes, ladder lane, real bands per lane, possible standards, the tilt of the gel, and
QUESTIONS to ask the user. If questions_for_user is not empty, ask the user before quantifying.
If inspect_gel reports a tilt, call image_operation(operation='straighten') first, then crop.

GOLDEN RULES
 1. Analysis needs DARK bands on a WHITE background. Do NOT judge this from the gel type (the
    same stain can be exported either way): inspect_gel MEASURES it, and auto_analyze_gel and
    label_mw_ladder fix it automatically. Never call image_operation(invert) or
    fix_band_polarity(mode='invert') unless the user asks to invert the image.
 2. MW labels: use label_mw_ladder (the app's Auto Lane analysis). It finds the ladder bands
    and writes the preset's values top->bottom. Choose the preset first (set_marker_preset or
    preset=...). Do NOT click markers by hand unless the user asks.
 3. "Analyze / quantify / measure this gel" = auto_analyze_gel (the app's Auto Analyze Gel
    wizard). It straightens a tilted gel, crops a raw imager file to the gel, finds lanes,
    labels the ladder, measures every band and removes noise peaks. If it returns
    status='needs_user_input', ask the user its question and call it again with the answer.
    A user's own ladder: pass ladder_values=[band sizes, top to bottom]. It removes noise peaks
    (only bands >= 5x the noise and >= 1% of the lane's strongest band are kept).
    Band fitting uses the dialog's IMPORTANT SETTINGS first: Min Prominence is tuned (one
    value for the whole gel, shown on its slider) until the detector itself finds the real
    bands; peaks are only added/removed one by one where no slider value fits. To adjust by
    hand, change Min Prominence (set_control) before touching individual peaks.
 4. Standards: if inspect_gel reports a standard-series candidate (same band position, similar
    size, different intensity), ASK the user whether those lanes are standards and their
    amounts. Then quantify_with_standards(standards=[{lane, quantity}, ...]).
 5. RESULTS COME FROM THE EXPORTED FILE. After densitometry call export_results(path,
    format='xlsx' or 'csv'): it writes the same workbook as the results window's "Export All
    Lanes to Excel" (Summary, one sheet per lane, Standard Curve Data, Band Comparisons) and
    returns its contents (bands, lane totals, comparisons). Quote numbers from that file, give
    the user the file path, and use read_results_file(path) to read an exported file again.
 6. Comparing bands (loading-control normalisation, treated vs control, fold change): use
    compare_bands, the app's Compare Bands tool. within_lane={target_band, reference_band}
    normalises each lane to its own loading control; reference={lane, band} compares every
    band with one band; pairs=[...] for specific pairs. Results give A/B, A-B, % change and
    log2(A/B) by Band Intensity, Band %, Lane % and Quantity, show in the Compare Bands window
    and are written to the "Band Comparisons" sheet by the next export_results.
 7. Reports: export_pdf_report(path) writes the app's protected PDF report (parameters,
    standard curve and fit, per-lane band tables and images, the annotated gel). Offer it
    after any quantification.
 8. Several images in one figure: combine_images(path, placement='below' | 'right' | ...) adds
    the padding and places a second image for you. For full control (copy the current image
    into a slot, place, move, resize, rotate, opacity, crop, flip, remove, rasterize) use
    overlay_image. Placed images stay separate layers: saving, exporting and densitometry work
    without rasterizing.
 9. Tables on the figure: add_table (any grid, e.g. +/- loading table above the lanes),
    add_results_table (densitometry numbers as a table), edit_table (change cells, formatting,
    position, or convert it to separate editable items).
10. Check your work with screenshot(target='image').

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
        "export_results(path='.../results.xlsx' or '.csv') and quote the numbers from the returned file contents.",
        "Ratios / loading-control normalisation -> compare_bands(...); a report -> export_pdf_report(path).",
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
        "auto_analyze_gel(densitometry=true) for band intensities; normalise to a loading control with "
        "compare_bands(within_lane={target_band: <target>, reference_band: <loading control>}) - "
        "a_over_b per lane is the normalised signal - then export_results so it is in the file.",
    ],
    "densitometry": [
        "auto_analyze_gel(densitometry=true, expected_lanes=<n if known>, ladder_lanes=[...]).",
        "Bands are fitted with the IMPORTANT SETTINGS first: the Min Prominence slider is tuned for the "
        "whole gel (see band_cleanup.important_settings: from/to) so the detector itself finds the "
        "bands; only leftovers are removed peak by peak.",
        "Noise is removed automatically: min_snr (default 5) and min_relative_intensity (default 0.01).",
        "Adjusting by hand: change Min Prominence (Important Settings) first; Min Distance, Band "
        "Position and the noise sliders are under Advanced Settings.",
        "Raise min_snr (e.g. 8) for very noisy blots; lower min_relative_intensity to 0.002 for "
        "purity/impurity analysis where faint bands matter.",
        "review=true leaves the densitometry dialog open so you can screenshot it before OK.",
        "export_results(path='<folder>/<name>.xlsx') (or .csv) -> the numbers come from the exported "
        "file the user can open; report band intensity, Band % and Lane % from it.",
        "Loading control / fold change: compare_bands(within_lane={target_band: 1, reference_band: 2}) "
        "or compare_bands(reference={lane: <control lane>, band: <n>}); then export_results again so "
        "the Band Comparisons sheet is in the file.",
        "export_pdf_report(path='<folder>/<name>.pdf') for a complete, protected report.",
        "get_analysis_results gives the raw arrays and saturation flags.",
        "Saturated bands (is_saturated) under-report - tell the user to re-image with shorter exposure.",
    ],
    "compare_bands": [
        "Run densitometry first (auto_analyze_gel). Bands are numbered from the TOP of each lane (1 = top).",
        "Loading control in every lane (e.g. target = band 1, actin/GAPDH = band 2): "
        "compare_bands(within_lane={target_band: 1, reference_band: 2}, measure='intensity') -> "
        "a_over_b is the normalised signal per lane.",
        "Treated vs control: compare_bands(pairs=[{lane_a: 3, band_a: 1, lane_b: 2, band_b: 1}]) -> "
        "a_over_b = fold change, log2_a_over_b, percent_change.",
        "Everything vs one reference band: compare_bands(reference={lane: 2, band: 1}, lanes=[3, 4, 5]).",
        "measure: intensity (band volume), band_pct (share of the lane's bands), lane_pct (share of the "
        "whole lane), quantity (needs standards), or all (default).",
        "clear=true starts a new list. export_results afterwards writes them to the Band Comparisons sheet.",
    ],
    "report": [
        "After densitometry (and quantify_with_standards if standards were used):",
        "export_results(path='.../results.xlsx') - workbook with Summary, per-lane sheets (with lane "
        "images), Standard Curve Data and Band Comparisons; the result returns every number.",
        "export_pdf_report(path='.../report.pdf', name='<analysis name>') - protected PDF with "
        "parameters, standard curve fit, band tables, lane images and the annotated gel.",
        "save_image(path) for the annotated figure, export_pptx(path) for editable slides.",
    ],
    "tables": [
        "add_table(cells=[[\"\", \"1\", \"2\"], [\"Drug\", \"-\", \"+\"]], first_lane=1) puts a lane-aligned "
        "table above the gel (columns snap to the top labels).",
        "add_results_table(measure='band_pct', decimals=1) draws the densitometry values under the gel.",
        "edit_table(index=<shape index from get_annotations>, set_cells=[{row, col, text}], bold_first_row=true, "
        "x=..., y=...) changes an existing table; convert_to_items=true splits it into text + lines.",
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
        "The placed image stays a separate layer: save_image, export_pptx and densitometry work without "
        "rasterizing (rasterize=true / overlay_image(action='rasterize') merges it if the user wants).",
        "Two gels from the app: open gel A -> overlay_image(action='copy_current', slot=1) -> open gel B -> "
        "overlay_image(action='place', slot=1, x=..., y=...) (add padding first with set_padding).",
        "Overlaying (superimposing) two images: overlay_image(action='load_file', path=...) then "
        "overlay_image(action='set', slot=2, opacity_percent=50, x=..., y=..., scale_percent=..., rotation_deg=...).",
        "overlay_image(action='crop', slot=2, x, y, width, height) crops the placed image in canvas "
        "coordinates (non-destructive: action='uncrop' restores it); 'status' shows every layer's outline.",
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
                "tune_settings": {"type": "boolean", "default": True,
                                  "description": "Fit the bands with the dialog's Important "
                                                 "Settings first (tune the Min Prominence slider "
                                                 "for the whole gel) before removing any peak by "
                                                 "hand. Leave on."},
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
                "tune_settings": {"type": "boolean", "default": True,
                                  "description": "Fit the bands with the dialog's Important "
                                                 "Settings first (tune the Min Prominence slider "
                                                 "for the whole gel) before removing any peak by "
                                                 "hand. Leave on."},
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
                       "densitometry, compare_bands, standards, report, combine, tables.",
        "inputSchema": {"type": "object", "properties": {"task": {"type": "string", "enum": [
            "overview", "protein_gel", "dna_gel", "western_blot", "densitometry", "compare_bands",
            "standards", "report", "combine", "tables"]}}},
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
                       "reset_image (discard ALL edits and annotations — asks first), "
                       "straighten (measure the tilt of the gel from its horizontal band and gel "
                       "edges and rotate it level with Transform > Rotation + Apply) and rotate "
                       "(by angle_deg; + = clockwise). Straighten or rotate BEFORE cropping and "
                       "labelling: rotation enlarges the canvas and clears ladder/lane labels.",
        "inputSchema": {
            "type": "object",
            "properties": {"operation": {"type": "string", "enum": [
                "invert", "flip_vertical", "flip_horizontal", "grayscale", "to_8bit",
                "optimize_levels", "reset_adjustments", "reset_image", "straighten", "rotate"]},
                "angle_deg": {"type": "number", "minimum": -180, "maximum": 180,
                              "description": "rotate only: degrees, + = clockwise, 0.05 deg steps."}},
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
    # ------------------------------------------------------------------ results files, comparisons, report
    {
        "name": "export_results",
        "description": "Export the densitometry results with the results window's own exporter and "
                       "RETURN THE FILE'S CONTENTS. format='xlsx' writes the 'Export All Lanes to Excel' "
                       "workbook (Summary, one sheet per lane with its lane image, Standard Curve Data, "
                       "Band Comparisons); format='csv' writes one row per band (Lane, Band, Band "
                       "Intensity, Band %, Lane %, Quantity) plus _band_comparisons.csv / "
                       "_standard_curve.csv when present. Returns bands, lane totals, standard curve and "
                       "comparisons parsed from the written file - use these numbers in your answer.",
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File to write (.xlsx or .csv)."},
            "format": {"type": "string", "enum": ["xlsx", "csv"],
                       "description": "Default: from the path's extension, else xlsx."},
            "name": {"type": "string", "description": "Analysis name shown in the files/report."},
            "include_comparisons": {"type": "boolean", "default": True},
            "overwrite": {"type": "boolean", "default": False}},
            "required": ["path"]},
    },
    {
        "name": "read_results_file",
        "description": "Read an Excel (.xlsx) or CSV results file exported by Gel Blot Analyzer (by "
                       "export_results or by the user) and return its bands, lane totals, standard "
                       "curve and band comparisons.",
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "include_raw_sheets": {"type": "boolean", "default": False,
                                   "description": "Also return every sheet's raw cell values."}},
            "required": ["path"]},
        "readOnly": True,
    },
    {
        "name": "compare_bands",
        "description": "The app's Compare Bands tool: compare any band of any lane with another band. "
                       "Returns A, B, A/B, A-B, % change and log2(A/B) for each comparison, opens the "
                       "Compare Bands window, and the list is written to the 'Band Comparisons' sheet by "
                       "export_results. Bands are numbered from the top of each lane (1 = top). "
                       "within_lane = loading-control normalisation in every lane; reference = every "
                       "band vs one band; pairs = specific comparisons.",
        "inputSchema": {"type": "object", "properties": {
            "pairs": {"type": "array", "items": {"type": "object", "properties": {
                "lane_a": {"type": "string"}, "band_a": {"type": "integer", "minimum": 1},
                "lane_b": {"type": "string"}, "band_b": {"type": "integer", "minimum": 1}},
                "required": ["lane_a", "band_a", "lane_b", "band_b"]}},
            "reference": {"type": "object", "properties": {
                "lane": {"type": "string"}, "band": {"type": "integer", "minimum": 1}},
                "description": "Compare every band (of `lanes`, default all) with this band."},
            "within_lane": {"type": "object", "properties": {
                "target_band": {"type": "integer", "minimum": 1},
                "reference_band": {"type": "integer", "minimum": 1}},
                "description": "In every lane (or `lanes`): target band vs reference band (e.g. loading control)."},
            "lanes": {"type": "array", "items": {"type": "string"},
                      "description": "Limit reference / within_lane to these lanes."},
            "measure": {"type": "string", "enum": ["all", "intensity", "band_pct", "lane_pct", "quantity"],
                        "default": "all"},
            "clear": {"type": "boolean", "default": False, "description": "Start a new comparison list."},
            "show": {"type": "boolean", "default": True, "description": "Open the Compare Bands window."}}},
    },
    {
        "name": "export_pdf_report",
        "description": "Export the app's PDF report of the current analysis (results window ▸ Export PDF "
                       "Report): analysis parameters, standard curve and fit, per-lane band tables with "
                       "lane images and saturation flags, and the annotated gel with densitometry boxes. "
                       "The PDF is protected against editing.",
        "inputSchema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "PDF file to write."},
            "name": {"type": "string", "description": "Analysis name printed in the report."},
            "standard_curve_model": {"type": "string",
                                     "description": "Linear, Power (Log-Log), Polynomial (Deg 2), "
                                                    "Polynomial (Deg 3) or 4-PL Sigmoidal."},
            "include_text": {"type": "boolean", "default": False,
                             "description": "Return the report's text as well."},
            "overwrite": {"type": "boolean", "default": False}},
            "required": ["path"]},
    },
    # ------------------------------------------------------------------ overlap images (layers)
    {
        "name": "overlay_image",
        "description": "Overlap Images tab: two image slots (Image 1 = base panel, Image 2 = overlay) "
                       "placed on the canvas as separate layers. actions: status (layers, positions, "
                       "outlines), copy_current (hold the current image in a slot; then open_image the OTHER "
                       "image before placing it), load_file (a file "
                       "into Image 2), place (x, y; include_labels = which labels of a copied image to "
                       "bring: custom, shapes, lr, top), set (x, y, scale_percent, rotation_deg, "
                       "opacity_percent, global_mixing_percent, invert), crop (x, y, width, height in "
                       "CANVAS image coordinates; non-destructive), uncrop, flip_horizontal, "
                       "flip_vertical, remove (take off the canvas, keep in the slot), reset (empty the "
                       "slot), rasterize (merge into one image - optional). For a simple gel + blot "
                       "figure prefer combine_images.",
        "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["status", "copy_current", "load_file", "place", "set",
                                                  "crop", "uncrop", "flip_horizontal", "flip_vertical",
                                                  "remove", "reset", "rasterize"]},
            "slot": {"type": "integer", "enum": [1, 2]},
            "path": {"type": "string"},
            "x": _XY, "y": _XY, "width": _XY, "height": _XY,
            "scale_percent": {"type": "number", "minimum": 1, "maximum": 1000},
            "rotation_deg": {"type": "number", "minimum": -180, "maximum": 180},
            "opacity_percent": {"type": "number", "minimum": 0, "maximum": 100},
            "global_mixing_percent": {"type": "number", "minimum": 0, "maximum": 100},
            "invert": {"type": "boolean"},
            "include_labels": {"type": "array", "items": {"type": "string",
                                                          "enum": ["custom", "shapes", "lr", "top"]}}},
            "required": ["action"]},
    },
    # ------------------------------------------------------------------ tables
    {
        "name": "edit_table",
        "description": "Change an on-image table (from add_table / add_results_table / the Table... "
                       "button): replace all cells, set single cells, formatting (bold/italic/underline "
                       "rows or columns, merge, colors, align, outline, inner_lines, font, font_size), "
                       "column/row sizes, position (x, y), or convert_to_items=true to split it into "
                       "separate text items and lines. index = the table's shape index (get_annotations). "
                       "A table above the gel grows upward (its bottom edge stays above the lane labels) "
                       "and top/left padding is added when it would leave the canvas.",
        "inputSchema": {"type": "object", "properties": {
            "index": {"type": "integer", "minimum": 0},
            "cells": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            "set_cells": {"type": "array", "items": {"type": "object", "properties": {
                "row": {"type": "integer", "minimum": 1}, "col": {"type": "integer", "minimum": 1},
                "text": {"type": "string"}}, "required": ["row", "col", "text"]}},
            "x": _XY, "y": _XY,
            "font": {"type": "string"}, "font_size": {"type": "integer", "minimum": 4, "maximum": 200},
            "bold": {"type": "boolean"}, "italic": {"type": "boolean"},
            "bold_first_row": {"type": "boolean"}, "bold_first_col": {"type": "boolean"},
            "bold_rows": {"type": "array", "items": {"type": "integer"}},
            "bold_columns": {"type": "array", "items": {"type": "integer"}},
            "italic_rows": {"type": "array", "items": {"type": "integer"}},
            "underline_rows": {"type": "array", "items": {"type": "integer"}},
            "merge": {"type": "array", "items": {"type": "object"}},
            "color": {"type": "string"}, "line_color": {"type": "string"},
            "align": {"type": "string", "enum": ["left", "center", "right"]},
            "outline": {"type": "boolean"}, "inner_lines": {"type": "boolean"},
            "first_column_is_labels": {"type": "boolean"},
            "column_widths_mode": {"type": "string", "enum": ["fit", "equal", "lanes"]},
            "column_widths": {"type": "array", "items": {"type": ["number", "null"]}},
            "row_heights": {"type": "array", "items": {"type": ["number", "null"]}},
            "thickness": {"type": "number"},
            "convert_to_items": {"type": "boolean", "default": False}}},
    },
    {
        "name": "add_results_table",
        "description": "Draw the densitometry results as a table on the figure (default: under the gel): "
                       "one column per lane, one row per band, values by measure (band_pct, lane_pct, "
                       "intensity or quantity). Accepts add_table's formatting options too.",
        "inputSchema": {"type": "object", "properties": {
            "measure": {"type": "string", "enum": ["band_pct", "lane_pct", "intensity", "quantity"],
                        "default": "band_pct"},
            "decimals": {"type": "integer", "minimum": 0, "maximum": 6},
            "lanes": {"type": "array", "items": {"type": "string"}},
            "bands": {"type": "array", "items": {"type": "integer", "minimum": 1}},
            "corner_label": {"type": "string"},
            "x": _XY, "y": _XY,
            "font_size": {"type": "integer", "minimum": 4, "maximum": 200}}},
    },

]

# ---------------------------------------------------------------------------------------------
# Small-model robustness: plain descriptions that point to ONE obvious tool per request, and the
# user's own ladder values on every ladder tool.
# ---------------------------------------------------------------------------------------------
_LADDER_VALUES = {"type": "array", "items": {"type": "number"},
                  "description": "The user's own ladder: band sizes from top to bottom (e.g. kDa or bp), "
                                 "e.g. [180, 130, 95, 72, 55, 43, 34, 26, 17, 10]. Overrides preset."}
_OVR = {
    "auto_analyze_gel": (
        "USE THIS for 'analyze / quantify / measure this gel'. Give path=<image file> (it opens the image "
        "itself) or call it after open_image. No other arguments unless the user gave values. It does "
        "everything in order: straightens a tilted gel, "
        "crops a raw imager file to the gel, makes bands dark on white, finds the lanes, labels the MW "
        "ladder and measures every band (noise peaks removed). If it returns status='needs_user_input', "
        "ask the user the question and call it again with the answer, e.g. expected_lanes=12. If the "
        "user names their own ladder sizes, pass ladder_values."),
    "label_mw_ladder": (
        "Only MW labels (no densitometry): finds the ladder bands in one lane and writes the ladder "
        "sizes beside them, top to bottom. Fixes polarity itself. Give lane (from inspect_gel) and "
        "either preset (a ladder name) or ladder_values (the user's own sizes, top to bottom)."),
    "set_marker_preset": (
        "Choose the MW ladder: preset (a ladder name, see get_annotations -> marker_presets) or "
        "ladder_values (the user's own ladder sizes, top to bottom, e.g. [180, 130, 95, 72, 55])."),
    "fix_band_polarity": (
        "Rarely needed: auto_analyze_gel and label_mw_ladder already do this. mode='auto' MEASURES the "
        "image and inverts only if the bands are light on a dark background, so the result is always "
        "dark bands on white; 'check' only reports. mode='invert' always flips the image: use it only "
        "when the user asks to invert."),
    "analyze_lanes": (
        "MANUAL densitometry for experts. NOT for 'analyze this gel' (use auto_analyze_gel). Runs "
        "densitometry on regions drawn with define_lane_regions and opens a dialog that must be "
        "answered with respond_to_dialog."),
    "define_lane_regions": (
        "MANUAL: draw densitometry lane boxes yourself. NOT needed for 'analyze this gel' "
        "(auto_analyze_gel finds the lanes)."),
}
for _t in TOOLS:
    if _t["name"] in _OVR:
        _t["description"] = _OVR[_t["name"]]
    if _t["name"] in ("auto_analyze_gel", "label_mw_ladder", "set_marker_preset"):
        _t["inputSchema"].setdefault("properties", {})["ladder_values"] = dict(_LADDER_VALUES)
    if _t["name"] == "auto_analyze_gel":
        _t["inputSchema"]["properties"]["path"] = {
            "type": "string", "description": "Image file to open first (optional if an image is already open)."}
        _t["inputSchema"]["properties"]["auto_prepare"] = {
            "type": "boolean", "description": "Straighten and crop to the gel automatically (default true)."}
    if _t["name"] == "set_marker_preset":
        _t["inputSchema"]["required"] = []

# ---------------------------------------------------------------------------------------------
# Tool profiles. 'essential' = the few tools a small local model needs, with short instructions
# (about a fifth of the prompt); every tool can still be CALLED in either profile.
# ---------------------------------------------------------------------------------------------
ESSENTIAL_TOOLS = ("get_status", "open_image", "inspect_gel", "auto_analyze_gel", "label_mw_ladder",
                   "set_marker_preset", "quantify_with_standards", "export_results", "export_pdf_report",
                   "image_operation", "crop", "screenshot", "save_image", "history")
ESSENTIAL_INSTRUCTIONS = """You operate Gel Blot Analyzer, a gel and western-blot analysis app, live in the user's window.
To analyze a gel, call ONE tool:
   auto_analyze_gel(path=<the image file>)
It opens the image and does everything: straightens and crops the gel, makes bands dark on
white, finds lanes, labels the ladder, measures every band. If it returns
status='needs_user_input', ask the user its question, then call auto_analyze_gel again with the
answer (e.g. expected_lanes=12). If the user gives their own ladder sizes, add
ladder_values=[sizes, top to bottom] to the call.
Then:
- Standards: ask the user which lanes are standards and their amounts. NEVER invent amounts.
  Then quantify_with_standards(standards=[{lane, quantity}, ...]).
- export_results(path) and report the numbers it returns; export_pdf_report(path) for a report.
Do not invert the image yourself: polarity is measured and fixed automatically.
Use history(action='undo') to undo a step. Coordinates are image pixels.
"""
TOOL_PROFILES = ("full", "essential")


def current_profile():
    """'full' (all tools) or 'essential' (small local models); chosen in the app's AI settings."""
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            v = (json.load(f) or {}).get("tool_profile", "full")
        return v if v in TOOL_PROFILES else "full"
    except Exception:
        return "full"


def instructions(profile=None):
    return ESSENTIAL_INSTRUCTIONS if (profile or current_profile()) == "essential" else SERVER_INSTRUCTIONS


READ_ONLY_TOOLS = frozenset(t["name"] for t in TOOLS if t.get("readOnly"))


def public_tools(profile=None):
    """Tool list as sent over MCP (internal keys stripped, annotations added)."""
    out = []
    prof = profile or current_profile()
    for t in TOOLS:
        if prof == "essential" and t["name"] not in ESSENTIAL_TOOLS:
            continue
        d = {k: v for k, v in t.items() if k != "readOnly"}
        d["annotations"] = {"readOnlyHint": bool(t.get("readOnly")),
                            "openWorldHint": False}
        out.append(d)
    return out


# =============================================================================================
# Argument normalisation (shared by the HTTP endpoint and the stdio relay)
# =============================================================================================
# AI clients — especially small local models — send arguments in many shapes: numbers as
# strings, booleans as "yes", arrays as JSON text, camelCase keys, the whole argument object
# as a JSON string, or nothing at all. Every tool call is normalised against the tool's
# inputSchema here, so a sloppy call either works or gets ONE clear message telling the model
# exactly what to send, never a Python traceback.
_TOOL_BY_NAME = {t["name"]: t for t in TOOLS}
_TRUE = {"1", "true", "yes", "y", "on", "t"}
_FALSE = {"0", "false", "no", "n", "off", "f", "none", "null", ""}


def _key_norm(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def tool_names():
    return list(_TOOL_BY_NAME)


def suggest_tools(name, n=3):
    """Closest tool names for an unknown/misspelt one."""
    name = str(name or "")
    names = tool_names()
    norm = {_key_norm(x): x for x in names}
    if _key_norm(name) in norm:
        return [norm[_key_norm(name)]]
    return difflib.get_close_matches(name, names, n=n, cutoff=0.5)


def resolve_tool_name(name):
    """Exact name, or an unambiguous variant (case, dashes, spaces, a client prefix such as
    'gel-blot-analyzer.get_status' / 'mcp__gel-blot-analyzer__get_status')."""
    if name in _TOOL_BY_NAME:
        return name
    s = str(name or "").strip()
    for sep in ("__", ".", "/", ":"):
        if sep in s:
            tail = s.rsplit(sep, 1)[-1]
            if tail in _TOOL_BY_NAME:
                return tail
            s = tail
    want = _key_norm(s)
    for t in _TOOL_BY_NAME:
        if _key_norm(t) == want:
            return t
    return None


def _parse_jsonish(v):
    if not isinstance(v, str):
        return v, False
    t = v.strip()
    if not t or t[0] not in "[{":
        return v, False
    try:
        return json.loads(t), True
    except ValueError:
        try:                                  # single-quoted python-ish literals
            return json.loads(t.replace("'", '"')), True
        except ValueError:
            return v, False


def _coerce(v, sch, path, problems):
    """Return v converted to the schema's type (best effort); record what could not be."""
    if not isinstance(sch, dict):
        return v
    typ = sch.get("type")
    if isinstance(typ, list):
        typ = next((x for x in typ if x != "null"), None)
    if typ in ("number", "integer"):
        x = v
        if isinstance(x, str):
            m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:px|%|ms|°|deg)?\s*", x)
            if not m:
                problems.append("'%s' must be a number (got %r)" % (path, v))
                return v
            x = float(m.group(1))
        elif isinstance(x, bool):
            x = int(x)
        elif isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
            x = x[0]
        if not isinstance(x, (int, float)):
            problems.append("'%s' must be a number (got %s)" % (path, type(v).__name__))
            return v
        if isinstance(x, float) and not math.isfinite(x):
            problems.append("'%s' must be a finite number" % path)
            return v
        lo, hi = sch.get("minimum"), sch.get("maximum")
        if lo is not None and x < lo:
            x = lo
        if hi is not None and x > hi:
            x = hi
        if typ == "integer":
            x = int(round(x))
        elif isinstance(x, float) and x.is_integer() and isinstance(v, int):
            x = int(x)
        return x
    if typ == "boolean":
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            t = v.strip().lower()
            if t in _TRUE:
                return True
            if t in _FALSE:
                return False
        problems.append("'%s' must be true or false (got %r)" % (path, v))
        return v
    if typ == "string":
        if isinstance(v, str):
            s = v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            s = ("%d" % v) if float(v).is_integer() else repr(v)
        elif isinstance(v, bool):
            s = "true" if v else "false"
        else:
            return v                          # leave structured values to the tool
        enum = sch.get("enum")
        if enum and s not in enum:
            low = {str(e).lower(): e for e in enum}
            key = s.strip().lower()
            if key in low:
                s = low[key]
            else:
                normed = {_key_norm(e): e for e in enum}
                if _key_norm(s) in normed:
                    s = normed[_key_norm(s)]
                else:
                    close = difflib.get_close_matches(key, list(low), n=1, cutoff=0.75)
                    if close:
                        s = low[close[0]]
                    # else: pass it through — several tools accept synonyms beyond the enum
                    # and give their own, more specific message.
        return s
    if typ == "array":
        v, _ = _parse_jsonish(v)
        if isinstance(v, tuple):
            v = list(v)
        if not isinstance(v, list):
            if v is None:
                return v
            if isinstance(v, str) and sch.get("items", {}).get("type") in ("number", "integer"):
                v = re.findall(r"[-+]?\d*\.?\d+", v)   # "1, 2, 3"
            elif isinstance(v, str) and sch.get("items", {}).get("type") == "string" and "," in v:
                v = [x.strip() for x in v.split(",") if x.strip()]
            else:
                v = [v]                        # a single item where a list was expected
        items = sch.get("items")
        if isinstance(items, dict):
            v = [_coerce(x, items, "%s[%d]" % (path, i), problems) for i, x in enumerate(v)]
        return v
    if typ == "object":
        v, _ = _parse_jsonish(v)
        if isinstance(v, dict) and isinstance(sch.get("properties"), dict):
            return _coerce_object(v, sch, path, problems)
        return v
    return v


def _coerce_object(args, sch, path, problems):
    props = sch.get("properties") or {}
    out = {}
    norm = {_key_norm(k): k for k in props}
    for k, v in args.items():
        key = k if k in props else norm.get(_key_norm(k), k)   # camelCase / dashes / case
        if key in out and key != k:
            continue                                            # exact key already given
        out[key] = _coerce(v, props.get(key), (path + "." + key) if path else key, problems) \
            if key in props else v
    return out


def normalize_arguments(name, args):
    """(clean_args, problem_text_or_None) for a tools/call.

    ``args`` may be None, a dict, or the dict as a JSON string. Unknown keys are kept (tools
    ignore them). The problem text names the tool's expected arguments so the model can retry
    correctly in one step.
    """
    tool = _TOOL_BY_NAME.get(name)
    if args is None or args == "" or args == []:
        args = {}
    if isinstance(args, str):
        parsed, ok = _parse_jsonish(args)
        req = ((tool or {}).get("inputSchema") or {}).get("required") or []
        if ok and isinstance(parsed, dict):
            args = parsed
        elif tool is not None and len(req) == 1 and args.strip():
            args = {req[0]: args.strip()}        # open_image("/path.png") style call
        elif tool is not None and not req:
            args = {}                            # nothing needed: ignore the stray text
        else:
            return {}, "'arguments' must be a JSON object, e.g. %s" % example_arguments(name)
    if not isinstance(args, dict):
        return {}, "'arguments' must be a JSON object, e.g. %s" % example_arguments(name)
    if tool is None:
        return args, None
    sch = tool.get("inputSchema") or {}
    problems = []
    out = _coerce_object(args, sch, "", problems)
    missing = [k for k in sch.get("required", []) if out.get(k) is None or out.get(k) == ""]
    if missing:
        problems.insert(0, "missing required argument%s: %s" % ("s" if len(missing) > 1 else "",
                                                                  ", ".join(missing)))
    if problems:
        return out, "%s: %s. Expected arguments: %s" % (name, "; ".join(problems),
                                                          describe_arguments(name))
    return out, None


def describe_arguments(name):
    tool = _TOOL_BY_NAME.get(name) or {}
    sch = tool.get("inputSchema") or {}
    req = set(sch.get("required", []))
    parts = []
    for k, p in (sch.get("properties") or {}).items():
        t = p.get("type") or "any"
        if p.get("enum"):
            t = "|".join(map(str, p["enum"]))
        parts.append("%s%s (%s)" % (k, "*" if k in req else "", t))
    if not parts:
        return "none"
    return ", ".join(parts) + ("  [* = required]" if req else "")


def example_arguments(name):
    tool = _TOOL_BY_NAME.get(name) or {}
    sch = tool.get("inputSchema") or {}
    ex = {}
    for k in sch.get("required", []):
        p = (sch.get("properties") or {}).get(k, {})
        t = p.get("type")
        ex[k] = (p.get("enum") or ["..."])[0] if t == "string" else \
            0 if t in ("number", "integer") else [] if t == "array" else \
            {} if t == "object" else True if t == "boolean" else "..."
    return json.dumps(ex)


def json_safe(v):
    """NaN/Infinity are not valid JSON (JavaScript clients reject the whole message): null them."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    return v
