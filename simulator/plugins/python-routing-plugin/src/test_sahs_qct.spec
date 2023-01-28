# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['test_sahs_qct.py',
            'front_circuit.py',
            '/home/louhz/multi_objectives/cir_gen/interface.py',
            '/home/louhz/multi_objectives/init_mapping/get_init_map.py',
            '/home/louhz/multi_objectives/init_mapping/sa_mapping.py',
            '/home/louhz/multi_objectives/sahs/sahs_search.py',
            ],
             pathex=["/home/louhz/multi_objectives"],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=["scipy", "PIL", "matplotlib", "multiprocessing", "jedi", "jinja2", "notebook", "sqlite3", "pycparser", "nbconvert", "certifi", "lxml", "importlib_metadata", "difflib", "lib2to3", "pkg_resources"],
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
          name='route',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
