@echo off
title Rupert weekend render - leave this window open
echo.
echo  Rendering the EP01 weekend batch: Olympic angles, the judge, no-counter versions.
echo  About 9 hours. Leave this window open and the PC on (sleep set to Never).
echo  If it gets interrupted, just double-click this file again: finished clips are skipped.
echo.
"E:\Program Files\Git\bin\bash.exe" -lc "cd '/e/AI Channel/Rupert Sim' && bash renders/ep01_final/weekend.sh"
echo.
echo  FINISHED. The clips are in renders\ep01_final. You can close this window.
pause
