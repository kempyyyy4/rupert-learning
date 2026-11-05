@echo off
title Rupert Sim - install check
cd /d "%~dp0"
".venv\Scripts\python.exe" -m rupert.check_install
echo.
pause
