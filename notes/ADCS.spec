# -*- mode: python ; coding: utf-8 -*-

import os, importlib

block_cipher = None


a = Analysis(['start_adcs.py'],
             pathex=['venv\\Lib\\site-packages\\tensorflow\\tensorflow'],
             binaries=[],
             datas=[(os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__),
                                  'lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so'),
                     'tensorflow/lite/experimental/microfrontend/python/ops/')],
             hiddenimports=['tensorflow._api.v2.compat', 'jinja2', 'pkg_resources.py2_warn', 'pkg_resources.markers'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='ADCS',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon='C:\\Users\\ZM\\Desktop\\ssmc-infrastructure\\ADC drive\\ADCS\\assets\\icon.ico')
