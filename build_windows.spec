# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import subprocess
from PySide6 import QtCore
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Configuration ---
APP_NAME = "Gel Blot Analyzer"
SCRIPT_FILE = "Gel_blot_analyzer.py"
ICON_FILE = "icon.ico"


# --- Code Signing Configuration (Optional - Edit for your needs) ---
# Set to True to enable code signing. If False, the signing step will be skipped.
ENABLE_SIGNING = False
# Full path to your .pfx certificate file.
CERT_PATH = "C:\\path\\to\\your\\certificate.pfx"
# The password for your certificate. It's safer to load this from an environment variable.
# In CMD: set CERT_PASS=your_password
# In PowerShell: $env:CERT_PASS="your_password"
CERT_PASS = os.environ.get("CERT_PASS")
# URL of the timestamp server. This is a common, free one.
TIMESTAMP_URL = "http://timestamp.sectigo.com"

# --- Helper function to find signtool.exe ---
def find_signtool():
    """Finds the path to signtool.exe from the Windows SDK."""
    base_paths = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Windows Kits", "10", "bin"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Windows Kits", "10", "bin"),
    ]
    for base in base_paths:
        if os.path.isdir(base):
            versions = sorted([d for d in os.listdir(base) if d.startswith("10.")], reverse=True)
            for v in versions:
                tool_path = os.path.join(base, v, "x64", "signtool.exe")
                if os.path.exists(tool_path):
                    print(f"Found signtool.exe at: {tool_path}")
                    return tool_path
    return None

# --- Data File Paths ---
# Get the directory where PySide6 stores its plugins
pyside_library = os.path.join(os.path.dirname(QtCore.__file__), "plugins")

# --- Platform-specific data files ---
datas = [
    # Bundle the splash-screen / app icon. Ship it both at the bundle root and in
    # _internal so it is found at runtime via sys._MEIPASS (see _resource_candidates).
    (os.path.join(SPECPATH, "Icon.png"), "."),
]
# NOTE: unlike the macOS spec there is no collect_data_files('PySide6') here — the
# Windows build relies on PyInstaller's own PySide6 hook to lay out the Qt plugins.
# This is the ONLY intentional dependency difference between the two spec files.

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
# Other Qt bindings, tkinter, and every Qt module the app does not import. The app uses
# only QtCore, QtGui, QtWidgets and QtSvg -- all of which live in PySide6-Essentials.
# Everything listed below is PySide6-Addons, which is 865 MB of the 1,191 MB Qt payload.
# None of it is reachable from the four modules above, so this list is insurance against a
# transitive import, not the primary defence: that is simply not calling
# collect_data_files('PySide6') (see the note near the top of the macOS spec).
# QtPrintSupport is here for the same reason as before -- no QPrinter/QPrintDialog anywhere,
# and matplotlib's qtagg backend does not reference it either.
excludes = [
    'PyQt5', 'PyQt6', 'tkinter',
    'PySide6.QtPrintSupport',
    # --- PySide6-Addons: never imported ---
    'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
    'PySide6.QtWebChannel', 'PySide6.QtWebSockets',
    'PySide6.QtQuick', 'PySide6.QtQuick3D', 'PySide6.QtQuickWidgets', 'PySide6.QtQuickControls2',
    'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
    'PySide6.QtCharts', 'PySide6.QtDataVisualization', 'PySide6.QtDesigner',
    'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DExtras', 'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic', 'PySide6.Qt3DAnimation',
    'PySide6.QtBluetooth', 'PySide6.QtNfc', 'PySide6.QtPositioning', 'PySide6.QtSerialPort',
    'PySide6.QtRemoteObjects', 'PySide6.QtScxml', 'PySide6.QtSensors', 'PySide6.QtSpatialAudio',
]

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
    upx=True,
    console=False,  # Creates a windowed app (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_FILE,
    version=None, # Embed version info from our file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# --- Post-Build Code Signing Step ---
if ENABLE_SIGNING:
    print("--- Starting Code Signing ---")
    signtool_path = find_signtool()
    if not signtool_path:
        raise FileNotFoundError("signtool.exe not found. Is the Windows SDK installed?")
    if not os.path.exists(CERT_PATH):
        raise FileNotFoundError(f"Certificate not found at: {CERT_PATH}")
    if not CERT_PASS:
        raise ValueError("Certificate password not set. Use 'set CERT_PASS=your_password'.")

    exe_path_to_sign = os.path.join(distpath, APP_NAME, f"{APP_NAME}.exe")
    
    command = [
        signtool_path,
        "sign",
        "/f", CERT_PATH,
        "/p", CERT_PASS,
        "/tr", TIMESTAMP_URL,
        "/td", "sha256",
        "/fd", "sha256",
        "/v", # Verbose output
        exe_path_to_sign,
    ]
    
    print(f"Signing command: {' '.join(command)}")
    try:
        subprocess.check_call(command)
        print("--- Code Signing Successful ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Code Signing FAILED: {e} ---")
        # Fail the build if signing fails
        sys.exit(1)