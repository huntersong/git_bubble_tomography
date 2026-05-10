# -*- mode: python ; coding: utf-8 -*-
# 三维多相流流场测量软件 - PyInstaller打包配置
# 主程序: main.py --gui

block_cipher = None

a = Analysis(
    [r'D:\Code\BubbleandPIV\bubble_tomography\main.py'],
    pathex=[r'D:\Code\BubbleandPIV\bubble_tomography'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PyQt5
        'PyQt5.sip',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        # matplotlib Qt5 backend
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.colors',
        'matplotlib.figure',
        # scipy
        'scipy.ndimage',
        'scipy.signal',
        'scipy.sparse',
        'scipy.linalg',
        'scipy.optimize',
        'scipy.spatial',
        'scipy.interpolate',
        # numpy testing (scipy 依赖)
        'numpy.testing',
        'numpy.core._multiarray_umath',
        # opencv
        'cv2',
        # scikit-image
        'skimage',
        'skimage.filters',
        'skimage.morphology',
        'skimage.measure',
        'skimage.transform',
        'skimage.segmentation',
        # tqdm
        'tqdm',
        # dateutil (matplotlib运行时依赖)
        'dateutil',
        'dateutil.parser',
        'dateutil.tz',
        'dateutil.relativedelta',
        # 本地模块
        'calibration',
        'calibration.camera_calibrator',
        'mart',
        'mart.mart_reconstructor',
        'particles',
        'particles.particle_reconstructor',
        'particles.velocity_field',
        'particles.piv2d',
        'utils',
        'utils.image_processor',
        'utils.image_editor',
        'visualization',
        'visualization.visualizer',
        'raytrace',
        'raytrace.raytrace_reconstructor',
        'gui',
        'gui.main_window',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Jupyter / IPython 生态
        'IPython', 'jupyter', 'jupyter_client', 'jupyter_core',
        'notebook', 'nbformat', 'nbconvert', 'nbclient', 'nbclassic',
        'ipykernel', 'ipywidgets', 'ipython_genutils',
        'traitlets', 'comm', 'debugpy',
        # 文档工具
        'sphinx', 'docutils', 'babel', 'numpydoc',
        # 网络 / 异步
        'tornado', 'zmq', 'pyzmq', 'jinja2', 'aiohttp', 'asyncio',
        'urllib3', 'requests', 'httpx', 'websocket', 'websockets',
        # 数据库
        'sqlalchemy', 'pymysql', 'psycopg2', 'sqlite3',
        # 安全/加密
        'cryptography', 'nacl', 'bcrypt', 'paramiko', 'fabric', 'pyOpenSSL',
        # 测试框架
        'pytest', 'nose', '_pytest',
        # 包管理
        'setuptools', 'pip', 'wheel', 'pkg_resources',
        # Tk
        'tkinter', '_tkinter', 'ttkthemes',
        # 浏览器/爬虫
        'selenium', 'playwright',
        # 数据分析（本项目不需要）
        'pandas', 'openpyxl', 'xlrd', 'xlwt', 'xlsxwriter',
        'statsmodels', 'patsy', 'xarray', 'pyarrow',
        # HDF5
        'h5py', 'tables', 'hdf5',
        # 分布式计算
        'distributed', 'cloudpickle', 'dask', 'fsspec',
        # 机器学习
        'sklearn', 'scikit_learn', 'joblib',
        'tensorflow', 'torch', 'keras', 'transformers',
        # 数值/符号计算
        'numba', 'llvmlite', 'sympy', 'networkx',
        # 可视化（不需要交互式）
        'bokeh', 'panel', 'plotly', 'altair', 'holoviews', 'pyviz_comms',
        'xyzservices', 'param',
        # 代码格式化
        'black', 'yapf', 'autopep8', 'isort', 'pylint', 'flake8',
        # 云/远程
        'botocore', 'boto3', 'google', 'azure',
        # Win32 COM
        'win32com', 'pythoncom',
        # 序列化
        'ruamel', 'tomlkit',
        # 其他大型无关包
        'astropy', 'astropy_iers_data',
        'intake', 'lz4', 'psutil',
        'py', 'pygments',
        'numexpr', 'bottleneck',
        'certifi', 'chardet', 'charset_normalizer',
        'jsonschema', 'jsonschema_specifications',
        'markdown', 'lxml',
        'pywt', 'imageio', 'tifffile',
        'platformdirs', 'importlib_metadata', 'importlib_resources',
        'zipp', 'zoneinfo', 'tzdata',
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
    name='BubbleTomography',
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
    name='BubbleTomography',
)
