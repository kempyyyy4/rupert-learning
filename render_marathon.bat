@echo off
title Rupert render - leave this window open
echo.
echo  Rendering everything for EP01 (about 3 days). Leave this window open and the PC on.
echo  Progress: renders\ep01_final\PROGRESS.txt
echo  If it stops (power cut, restart), double-click this file again: finished clips are skipped.
echo.
"E:\Program Files\Git\bin\bash.exe" -lc "cd '/e/AI Channel/Rupert Sim' && bash renders/ep01_final/marathon.sh"
echo.
echo  FINISHED. The clips are in renders\ep01_final. You can close this window.
pause
