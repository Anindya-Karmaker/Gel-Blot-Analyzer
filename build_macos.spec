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
from PyInstaller.utils.hooks import collect_data_files

# --- Configuration ---
APP_NAME = "Gel Blot Analyzer"
SCRIPT_FILE = "Gel_blot_analyzer.py"  
ICON_FILE = "Icon.icns"      # For macOS builds
BUNDLE_ID = "com.anindyakarmaker.gelblotanalyzer"

# --- Data Files ---
# Collects necessary data like Qt plugins, SSL certificates, and matplotlib fonts.
datas = []
datas.extend(collect_data_files('PySide6'))
datas.extend(collect_data_files('matplotlib'))

# --- Hidden Imports ---
# This list is crucial for modules that PyInstaller's static analysis might miss.
# We are being specific here to avoid including the entire scipy/skimage libraries.
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
    'scipy.optimize', # For curve_fit

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

# --- Excluded Modules ---
# Explicitly exclude other Qt bindings to prevent accidental bundling.
excludes = ['PyQt5', 'PyQt6', 'PySide6.QtPrintSupport', 'tkinter']

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
        'CFBundleShortVersionString': '4.0.0',                 # Your app's version number.
        'CFBundleVersion': '4.0',                              # Your app's build number.
        'NSHumanReadableCopyright': 'Copyright © 2025 Anindya Karmaker. All rights reserved.',
    }
)
