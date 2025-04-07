# PowerMeterApp.spec

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
import os
import glob

# Add hidden imports
hiddenimports = collect_submodules('skimage') + collect_submodules('nidaqmx')

# Include data files from skimage if needed (like lookup tables or other internal data)
datas = (
         collect_data_files('skimage') +
         copy_metadata("nidaqmx") +
         collect_data_files('customtkinter') +
         [(f, "ressources") for f in glob.glob("ressources/*")] +
         [(f, "packages") for f in glob.glob("packages/*")]
         )

block_cipher = None

a = Analysis(
    ['powermeter_ui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PowerMeterApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False if you want to hide the terminal window
    icon='ressources/QuebecWattAppLogo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PowerMeterApp'
)