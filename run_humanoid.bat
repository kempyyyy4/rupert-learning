@echo off
title Rupert Sim - humanoid viewer
cd /d "%~dp0"
".venv\Scripts\python.exe" -m rupert.view_humanoid
echo.
pause
