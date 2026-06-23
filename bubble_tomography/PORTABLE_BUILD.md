# Windows 便携版打包说明

本项目已经配置为 PyInstaller 便携版发布方式。打包后，新电脑不需要安装
Python、OpenCV、PyQt5、NumPy、SciPy 等依赖，也可以直接运行软件。

## 当前可直接运行的位置

已经把打包后的启动程序放到了项目顶层，和 `main.py` 在同一个目录：

```text
BubbleTomography.exe
```

顶层的 `_internal` 文件夹是它的运行库目录，里面包含 Python 运行时、PyQt5、
OpenCV、NumPy、SciPy、Matplotlib 等依赖。请不要删除或单独移动 `_internal`。

如果要把软件拷贝到新电脑，推荐直接复制整个项目文件夹，或至少复制下面两个
顶层项目：

```text
BubbleTomography.exe
_internal
```

然后在新电脑上双击：

```text
BubbleTomography.exe
```

## 标准发布目录

打包脚本也会生成标准的 one-folder 发布目录：

```text
dist\BubbleTomography
```

这个目录中同样包含：

```text
dist\BubbleTomography\BubbleTomography.exe
dist\BubbleTomography\_internal
```

也可以只复制整个 `dist\BubbleTomography` 文件夹到新电脑，然后运行其中的
`BubbleTomography.exe`。

## 重新打包

在项目顶层运行：

```powershell
.\scripts\build_portable.ps1
```

或者双击：

```text
scripts\build_portable.bat
```

如果想重建干净的打包环境：

```powershell
.\scripts\build_portable.ps1 -RecreateVenv
```

脚本会创建或复用 `.venv_portable_build`，安装 `requirements.txt` 中的依赖，
运行 `bubble_tomo.spec`，并自动把最新的 `BubbleTomography.exe` 和 `_internal`
同步到项目顶层。

## OpenCV 说明

`requirements.txt` 使用 `opencv-contrib-python`，不是普通的 `opencv-python`。
这样打包时会包含更完整的 OpenCV 发行包，包括 contrib 模块。当前 spec 文件会
显式收集完整的 `cv2` 包，例如：

```text
cv2.pyd
opencv_videoio_ffmpeg*.dll
cv2\data
cv2\aruco
cv2\xfeatures2d
cv2\ximgproc
```

这就是为什么不能只复制单独一个 exe：OpenCV 和其他动态库都在 `_internal`
目录里。

## 常见问题

如果新电脑提示拦截未知程序，可以把软件文件夹复制到本地可信路径，例如：

```text
C:\Tools\BubbleTomography
```

然后再运行 `BubbleTomography.exe`。

如果双击没有反应，可以先从命令行进入软件目录运行：

```powershell
.\BubbleTomography.exe
```

这样更容易看到系统级错误提示。

---

# Portable Windows Build

Use this when you want to copy the software to a new Windows computer and run it
without installing Python or OpenCV on that computer.

## Current Project-Root Launcher

The packaged launcher has been mirrored to the project root, next to `main.py`:

```text
BubbleTomography.exe
```

The adjacent `_internal` directory contains Python, PyQt5, OpenCV, NumPy, SciPy,
Matplotlib, and other runtime libraries. Do not delete or move `_internal`
separately from the exe.

To move the project-root launcher to another computer, copy at least:

```text
BubbleTomography.exe
_internal
```

## Standard Portable Folder

The build also produces the standard one-folder package:

```text
dist\BubbleTomography
```

You can copy that whole folder to the new computer and run:

```text
dist\BubbleTomography\BubbleTomography.exe
```

## Build

From this project directory, run one of:

```powershell
.\scripts\build_portable.ps1
```

or double-click:

```text
scripts\build_portable.bat
```

For a completely fresh build environment:

```powershell
.\scripts\build_portable.ps1 -RecreateVenv
```

The script creates `.venv_portable_build`, installs the dependencies, runs
PyInstaller with `bubble_tomo.spec`, and mirrors the latest exe plus `_internal`
to the project root.

## OpenCV Notes

`requirements.txt` uses `opencv-contrib-python` instead of `opencv-python` so the
build includes the larger OpenCV distribution with contrib modules. The spec file
also explicitly collects the full imported `cv2` package.
