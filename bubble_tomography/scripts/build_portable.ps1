param(
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv_portable_build"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$DistDir = Join-Path $ProjectRoot "dist\BubbleTomography"
$ExePath = Join-Path $DistDir "BubbleTomography.exe"
$RootExePath = Join-Path $ProjectRoot "BubbleTomography.exe"
$RootInternalDir = Join-Path $ProjectRoot "_internal"

Set-Location $ProjectRoot

if ($RecreateVenv -and (Test-Path $VenvDir)) {
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating portable build virtual environment..."
    python -m venv $VenvDir
}

Write-Host "Installing build dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "OpenCV selected for this build:"
& $PythonExe -c "import cv2, pathlib; print('  version:', cv2.__version__); print('  path:', pathlib.Path(cv2.__file__).resolve())"

Write-Host "Cleaning previous build outputs..."
Remove-Item -Recurse -Force (Join-Path $ProjectRoot "build") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $DistDir -ErrorAction SilentlyContinue

Write-Host "Building one-folder portable package..."
& $PythonExe -m PyInstaller --clean --noconfirm (Join-Path $ProjectRoot "bubble_tomo.spec")

if (-not (Test-Path $ExePath)) {
    throw "Build failed: $ExePath was not created."
}

Write-Host "Mirroring portable runtime to project root..."
Copy-Item -Force $ExePath $RootExePath
Remove-Item -Recurse -Force $RootInternalDir -ErrorAction SilentlyContinue
Copy-Item -Recurse -Force (Join-Path $DistDir "_internal") $RootInternalDir

Write-Host ""
Write-Host "Portable package is ready:"
Write-Host "  $DistDir"
Write-Host "Project-root launcher is ready:"
Write-Host "  $RootExePath"
Write-Host ""
Write-Host "Copy the whole BubbleTomography folder, or run from the project root:"
Write-Host "  BubbleTomography.exe"
