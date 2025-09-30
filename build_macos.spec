# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Configuration ---
APP_NAME = "Gel Blot Analyzer"
SCRIPT_FILE = "Gel_blot_analyzer.py"
ICON_FILE = "Icon.icns"  # Make sure this file is in the same directory as this .spec file
BUNDLE_ID = "com.anindyakarmaker.gelblotanalyzer"

# --- Data Files ---
# Use PyInstaller helpers to automatically find all necessary data files for libraries.
# This includes Qt plugins, matplotlib fonts, etc.
datas = []
datas.extend(collect_data_files('PySide6'))
datas.extend(collect_data_files('matplotlib'))
# No other manual data paths (like rembg models) seem necessary for this script.

# --- Hidden Imports ---
# This list is crucial for libraries that PyInstaller's static analysis might miss.
hiddenimports = [
    'PySide6.QtSvg',
    'PySide6.QtPrintSupport',
    'matplotlib.backends.backend_qtagg',
    'skimage',
    'skimage.restoration',
    'scipy',
    'scipy.signal',
    'scipy.ndimage',
    'scipy.interpolate',
    'cv2',
    'openpyxl',
    'openpyxl.cell._writer',
    # Often needed for scipy/numpy to work correctly when bundled
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
    # runtime_hooks=['runtime_hook.py'], # Only needed for special runtime configurations.
    runtime_hooks=[],                   # Keep it empty if you don't have a hook file.
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
    console=False,  # Set to False for a GUI application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_FILE
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

app = BUNDLE(
    coll,
    name=f'{APP_NAME}.app',
    icon=ICON_FILE,
    bundle_identifier=BUNDLE_ID,
    info_plist={
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.15', # Target macOS Catalina and newer
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1',
        'NSHumanReadableCopyright': 'Copyright © 2025 Anindya Karmaker. All rights reserved.',
    }
)