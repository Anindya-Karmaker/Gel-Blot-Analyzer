# -*- mode: python ; coding: utf-8 -*-

# =============================================================================
# PyInstaller Spec File for Gel Blot Analyzer
#
# This file is optimized for space efficiency by:
# 1. Specifically listing required submodules instead of collecting all of them.
# 2. Excluding unnecessary large modules (e.g., QtPrintSupport).
# 3. Using UPX compression on binaries.
# =============================================================================

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Configuration ---
APP_NAME = "Gel Blot Analyzer"
SCRIPT_FILE = "Gel_blot_analyzer.py"  
ICON_FILE = "Icon.icns"      # For macOS builds
BUNDLE_ID = "com.anindyakarmaker.gelblotanalyzer"

# --- Platform-specific data files ---
datas = []
# Bundle the splash-screen / app icon at the bundle root so it can be loaded at
# runtime via sys._MEIPASS (see _resource_candidates in the script).
datas.append((os.path.join(SPECPATH, "Icon.png"), "."))
# macOS only: the .app bundle needs the Qt plugin tree laid out explicitly.
# The Windows spec relies on PyInstaller's own PySide6 hook instead, which is why
# this one line differs between the two specs.
datas.extend(collect_data_files('PySide6'))

# =============================================================================
# SHARED LIBRARY DEPENDENCIES
# Keep this block byte-identical with the matching block in the other build spec.
# Anything platform-specific belongs ABOVE or BELOW it, never inside.
# =============================================================================

# --- Build-time dependency guard -------------------------------------------------
# lmfit is REQUIRED for densitometry, and its absence is silent at runtime: the app
# does `try: import lmfit / except ImportError:` inside _fit_gaussians and quietly
# drops to a plain SciPy Gaussian fallback with no EMG refinement and no AIC shoulder
# detection. A build machine without lmfit therefore produces an app that looks fine,
# starts fine, and reports different band areas — which is exactly how the broken
# deconvolution shipped. Fail the BUILD instead, where it is obvious and cheap to fix.
try:
    import lmfit  # noqa: F401
except ImportError as exc:  # pragma: no cover - build-time only
    raise SystemExit(
        "\n[spec] FATAL: lmfit is not installed in the build environment.\n"
        "  Gaussian Deconvolution silently degrades to a SciPy fallback without it,\n"
        "  producing different band areas in the shipped app.\n"
        "  Fix:  pip install lmfit    (or: conda install -c conda-forge lmfit)\n"
    ) from exc

# --- Third-party data files ---
datas.extend(collect_data_files('matplotlib'))
# python-pptx ships its default .pptx template + XML part files as package data;
# they must be bundled or PowerPoint export (save_image_pptx) fails at runtime.
datas.extend(collect_data_files('pptx'))

# --- Hidden Imports ---
# This list is crucial for libraries that PyInstaller's static analysis might miss.
hiddenimports = [
    # PySide6 essentials
    'PySide6.QtSvg',  # For SVG icon support

    # Matplotlib backend for Qt
    'matplotlib.backends.backend_qtagg',

    # Specific submodules used from libraries
    'skimage.restoration',
    'scipy.signal',
    'scipy.ndimage',
    'scipy.interpolate',
    'scipy.optimize',   # For curve_fit
    'scipy.integrate',
    # scipy.sparse.linalg backs the AsLS baseline (spsolve in _baseline_als).
    'scipy.sparse.linalg',
    'scipy.sparse.csgraph._validation',

    # Core libraries
    'cv2',
    'openpyxl',
    'openpyxl.cell._writer',

    # These are often needed for SciPy/NumPy to function correctly when bundled.
    # It's safer to keep them to avoid runtime errors.
    'scipy.special._cdflib',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
]
# Collect whole trees for the libraries that resolve names dynamically, where a
# specific list cannot be trusted to be complete.
hiddenimports.extend(collect_submodules('skimage'))
hiddenimports.extend(collect_submodules('scipy'))
# python-pptx imports several oxml submodules dynamically; collect them all so the
# lazy `from pptx import ...` in save_image_pptx resolves in the frozen app.
hiddenimports.extend(collect_submodules('pptx'))
# lmfit and its dependency chain. `import lmfit` sits inside a function body in
# _fit_gaussians, and every one of these packages resolves names dynamically —
# asteval builds its symbol table from `ast` at runtime, uncertainties defers
# `uncertainties.unumpy`, and dill imports pickle targets on demand — so static
# analysis cannot be relied on to find them.
hiddenimports.extend(collect_submodules('lmfit'))
hiddenimports.extend(collect_submodules('asteval'))
hiddenimports.extend(collect_submodules('uncertainties'))
hiddenimports.extend(collect_submodules('dill'))

# --- Excluded Modules ---
# Other Qt bindings, plus two modules the app genuinely does not use:
# QtPrintSupport (no QPrinter/QPrintDialog anywhere, and matplotlib's qtagg backend
# does not reference it either) and tkinter.
excludes = ['PyQt5', 'PyQt6', 'PySide6.QtPrintSupport', 'tkinter']

# =============================================================================
# END SHARED LIBRARY DEPENDENCIES
# =============================================================================

# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,         # Use UPX for maximum binary compression.
    console=False,    # This creates a windowed GUI application, not a terminal one.
    icon=ICON_FILE,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME
)

# --- macOS App Bundle Configuration ---
# This section is only used when building on macOS.
app = BUNDLE(
    coll,
    name=f'{APP_NAME}.app',
    icon=ICON_FILE,
    bundle_identifier=BUNDLE_ID,
    info_plist={
        'NSHighResolutionCapable': 'True',                     # Enables support for Retina displays.
        'LSMinimumSystemVersion': '10.15',                     # Sets minimum supported OS to macOS Catalina.
        'CFBundleShortVersionString': '9.3',                   # Must track APP_VERSION in the script.
        'CFBundleVersion': '9.3',                              # Build number.
        'NSHumanReadableCopyright': 'Copyright © 2025 Anindya Karmaker. All rights reserved.',
    }
)