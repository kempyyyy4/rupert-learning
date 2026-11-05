@echo off
title Rupert Sim - setup
cd /d "%~dp0"
echo.
echo === Rupert Sim setup (takes a few minutes, needs internet) ===
echo.
if exist ".venv\Scripts\python.exe" goto install
echo Creating a private Python workspace for Rupert...
py -3.14 -m venv .venv
if errorlevel 1 (
  echo.
  echo Could not find Python 3.14. Tell Claude what you see above.
  pause
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo The install hit a problem. Take a photo or screenshot of this window and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Setup finished. Now double-click check_install.bat
pause
