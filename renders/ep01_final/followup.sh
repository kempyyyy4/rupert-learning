#!/usr/bin/env bash
# After marathon.sh: side-on timelapses with a camera that keeps pace with Rupert (side_pan lost
# him past ~60 m). Only chapters where he travels. Waits for the marathon to finish first.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
T=renders/ep01_final/timelapse
PROG=renders/ep01_final/PROGRESS.txt
until [ "$(grep -c 'ALL FINISHED' $PROG)" -ge 2 ]; do sleep 60; done
for ch in final_run cheat6_trex cheat7_limp cheat8_gallop cheat2_leaning_back cheat3_pogo cheat4_straight_legs; do
  [ -f "$T/tl_${ch}_ghost_side_follow_clean.mp4" ] && continue
  echo "=== timelapse $ch side_follow $(date '+%a %H:%M')" >> renders/ep01_final/log_marathon.txt
  if "$PY" -m rupert.make_sequence ghost --chapter "$ch" --seconds 10 --camera side_follow --out-dir "$T" --reuse \
       >> renders/ep01_final/log_marathon.txt 2>&1 < /dev/null; then
    echo "done    $(date '+%a %H:%M')  timelapse $ch side_follow" >> $PROG
  else
    echo "FAILED  $(date '+%a %H:%M')  timelapse $ch side_follow (see log_marathon.txt)" >> $PROG
  fi
done
echo "FOLLOW-UP FINISHED $(date)" >> $PROG
