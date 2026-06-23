# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller portable build config for BubbleTomography.

This spec intentionally collects the complete imported OpenCV package
(`cv2`), including its compiled extension, bundled FFmpeg DLLs, data files,
and contrib subpackages when opencv-contrib-python is installed.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules


block_cipher = None
project_dir = os.path.abspath(SPECPATH)
entry_script = os.path.join(project_dir, "main.py")

cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")
cv2_hiddenimports += collect_submodules("cv2")


a = Analysis(
    [entry_script],
    pathex=[project_dir],
    binaries=cv2_binaries,
    datas=cv2_datas,
    hiddenimports=[
        # PyQt5
        "PyQt5.sip",
        "PyQt5.QtWidgets",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        # matplotlib Qt5 backend
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_agg",
        "matplotlib.pyplot",
        "matplotlib.patches",
        "matplotlib.colors",
        "matplotlib.figure",
        # scipy
        "scipy.ndimage",
        "scipy.signal",
        "scipy.sparse",
        "scipy.linalg",
        "scipy.optimize",
        "scipy.spatial",
        "scipy.interpolate",
        # numpy testing (scipy dependency)
        "numpy.testing",
        "numpy.core._multiarray_umath",
        # OpenCV: collect the full cv2 package explicitly above.
        "cv2",
        *cv2_hiddenimports,
        # scikit-image
        "skimage",
        "skimage.filters",
        "skimage.morphology",
        "skimage.measure",
        "skimage.transform",
        "skimage.segmentation",
        # tqdm
        "tqdm",
        # dateutil (matplotlib runtime dependency)
        "dateutil",
        "dateutil.parser",
        "dateutil.tz",
        "dateutil.relativedelta",
        # local modules
        "calibration",
        "calibration.camera_calibrator",
        "mart",
        "mart.mart_reconstructor",
        "particles",
        "particles.particle_reconstructor",
        "particles.velocity_field",
        "particles.piv2d",
        "utils",
        "utils.image_processor",
        "utils.image_editor",
        "visualization",
        "visualization.visualizer",
        "raytrace",
        "raytrace.raytrace_reconstructor",
        "gui",
        "gui.main_window",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Jupyter / IPython
        "IPython", "jupyter", "jupyter_client", "jupyter_core",
        "notebook", "nbformat", "nbconvert", "nbclient", "nbclassic",
        "ipykernel", "ipywidgets", "ipython_genutils",
        "traitlets", "comm", "debugpy",
        # documentation tools
        "sphinx", "docutils", "babel", "numpydoc",
        # network / async
        "tornado", "zmq", "pyzmq", "jinja2", "aiohttp", "asyncio",
        "urllib3", "requests", "httpx", "websocket", "websockets",
        # databases
        "sqlalchemy", "pymysql", "psycopg2", "sqlite3",
        # security / crypto
        "cryptography", "nacl", "bcrypt", "paramiko", "fabric", "pyOpenSSL",
        # test frameworks
        "pytest", "nose", "_pytest",
        # package managers
        "setuptools", "pip", "wheel", "pkg_resources",
        # Tk
        "tkinter", "_tkinter", "ttkthemes",
        # browser / crawler tooling
        "selenium", "playwright",
        # unrelated data analysis packages
        "pandas", "openpyxl", "xlrd", "xlwt", "xlsxwriter",
        "statsmodels", "patsy", "xarray", "pyarrow",
        # HDF5
        "h5py", "tables", "hdf5",
        # distributed computing
        "distributed", "cloudpickle", "dask", "fsspec",
        # machine learning stacks
        "sklearn", "scikit_learn", "joblib",
        "tensorflow", "torch", "keras", "transformers",
        # numerical / symbolic computing not needed by this app
        "numba", "llvmlite", "sympy", "networkx",
        # interactive visualization stacks not needed here
        "bokeh", "panel", "plotly", "altair", "holoviews", "pyviz_comms",
        "xyzservices", "param",
        # code formatters / linters
        "black", "yapf", "autopep8", "isort", "pylint", "flake8",
        # cloud / remote APIs
        "botocore", "boto3", "google", "azure",
        # Win32 COM
        "win32com", "pythoncom",
        # serialization extras
        "ruamel", "tomlkit",
        # other large unrelated packages
        "astropy", "astropy_iers_data",
        "intake", "lz4", "psutil",
        "py", "pygments",
        "numexpr", "bottleneck",
        "certifi", "chardet", "charset_normalizer",
        "jsonschema", "jsonschema_specifications",
        "markdown", "lxml",
        "pywt", "imageio", "tifffile",
        "platformdirs", "importlib_metadata", "importlib_resources",
        "zipp", "zoneinfo", "tzdata",
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BubbleTomography",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BubbleTomography",
)
