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
VERSION_FILE = "version_file.txt" # You must create this file for version info

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

datas = [
    # Manually bundle essential PySide6 Qt platform plugins for Windows
    (os.path.join(pyside_library, "platforms", "qwindows.dll"), "_internal\\PySide6\\plugins\\platforms"),
    (os.path.join(pyside_library, "styles", "qwindowsvistastyle.dll"), "_internal\\PySide6\\plugins\\styles"),
]
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
# Automatically collect all submodules from key libraries to be safe
hiddenimports.extend(collect_submodules('skimage'))
hiddenimports.extend(collect_submodules('scipy'))


# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
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
    version=VERSION_FILE, # Embed version info from our file
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