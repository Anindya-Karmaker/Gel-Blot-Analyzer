# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import subprocess
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Configuration ---
APP_NAME = "Gel Blot Analyzer"
SCRIPT_FILE = "Gel_blot_analyzer.py"
ICON_FILE = "Icon.ico"          # correct casing — icon.ico does not exist
VERSION_FILE = None             # version_file.txt is not present; set a path here if you create one

# --- Code Signing Configuration (Optional) ---
# Set ENABLE_SIGNING = True and fill in CERT_PATH / CERT_PASS to sign after build.
ENABLE_SIGNING = False
CERT_PATH = "C:\\path\\to\\your\\certificate.pfx"
CERT_PASS = os.environ.get("CERT_PASS", "")  # prefer env-var; fallback to empty string
TIMESTAMP_URL = "http://timestamp.sectigo.com"

# --- Helper: locate signtool.exe from the Windows SDK ---
def find_signtool():
    """Return path to signtool.exe from the highest-versioned Windows SDK found."""
    base_paths = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Windows Kits", "10", "bin"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Windows Kits", "10", "bin"),
    ]
    for base in base_paths:
        if not os.path.isdir(base):
            continue
        versions = sorted([d for d in os.listdir(base) if d.startswith("10.")], reverse=True)
        for v in versions:
            for arch in ("x64", "x86"):
                tool = os.path.join(base, v, arch, "signtool.exe")
                if os.path.isfile(tool):
                    print(f"Found signtool.exe at: {tool}")
                    return tool
    return None

# --- Data File Paths ---
# PySide6 plugins location differs between pip and conda-forge installations:
#   pip layout  : <site-packages>/PySide6/plugins/
#   conda layout: <env>/Library/lib/qt6/plugins/
#
# We probe both and use whichever actually contains qwindows.dll.

import PySide6 as _pyside6_pkg

_pip_plugins   = os.path.join(os.path.dirname(_pyside6_pkg.__file__), "plugins")
_conda_plugins = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_pyside6_pkg.__file__)))),
    "Library", "lib", "qt6", "plugins",
)

def _find_qt_plugins_dir():
    """Return the Qt plugins directory, searching pip and conda layouts."""
    for candidate in (_pip_plugins, _conda_plugins):
        if os.path.isfile(os.path.join(candidate, "platforms", "qwindows.dll")):
            print(f"INFO: Qt plugins found at: {candidate}")
            return candidate
    raise FileNotFoundError(
        "Could not locate Qt plugins (qwindows.dll) in either pip or conda layout.\n"
        f"  Tried:\n    {_pip_plugins}\n    {_conda_plugins}"
    )

_qt_plugins = _find_qt_plugins_dir()

def _add_plugin(rel_path, dest):
    """Add a plugin DLL only if it actually exists (warn otherwise)."""
    full = os.path.join(_qt_plugins, rel_path)
    if os.path.isfile(full):
        return [(full, dest)]
    print(f"WARNING: Optional plugin not found, skipping: {full}")
    return []

datas = []
# -- Essential platform plugin (required — build fails without it) --
datas += _add_plugin(os.path.join("platforms", "qwindows.dll"),
                     "_internal\\PySide6\\plugins\\platforms")

# -- Style plugins (optional — whichever is present) --
for _style_dll in ("qwindowsvistastyle.dll", "qmodernwindowsstyle.dll"):
    _added = _add_plugin(os.path.join("styles", _style_dll),
                         "_internal\\PySide6\\plugins\\styles")
    datas += _added
    if _added:
        break  # only need one style plugin

# -- App icon --
datas += [(os.path.join(SPECPATH, "Icon.png"), ".")]


# Automatically collect all necessary data files for matplotlib
datas.extend(collect_data_files('matplotlib'))

# --- Hidden Imports ---
# This list is crucial for libraries that PyInstaller's static analysis might miss.
hiddenimports = [
    'PySide6.QtSvg',
    'PySide6.QtPrintSupport',
    'matplotlib.backends.backend_qtagg',
    'skimage.restoration',
    'scipy.signal',
    'scipy.ndimage',
    'scipy.interpolate',
    'cv2',
    'openpyxl',
    'openpyxl.cell._writer',
    'scipy.special._cdflib',
    'scipy.integrate',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    'scipy.sparse.csgraph._validation',
]
# Automatically collect submodules; exclude scipy's optional torch bridge to
# prevent the spurious "No module named 'torch'" warning during analysis.
_scipy_mods = collect_submodules('scipy')
_scipy_mods = [m for m in _scipy_mods if 'torch' not in m]
hiddenimports.extend(collect_submodules('skimage'))
hiddenimports.extend(_scipy_mods)


# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'torch', 'scipy._lib.array_api_compat.torch'],
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
    **({'version': VERSION_FILE} if VERSION_FILE else {}),  # omit when None
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