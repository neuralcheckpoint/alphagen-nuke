@echo off
rem Usage: medratools.bat install ^| models ^| backend ^| check
setlocal
set "UV_BIN=uv"
where uv >nul 2>nul
if errorlevel 1 (
    set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
    if not exist "%USERPROFILE%\.local\bin\uv.exe" (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if errorlevel 1 exit /b 1
    )
)
"%UV_BIN%" run --no-project --python 3.12 "%~dp0medratools.py" %*
exit /b %errorlevel%
