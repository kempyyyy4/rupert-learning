#!/usr/bin/env bash
# After marathon.sh (Tom 2026-11-18: falls should land, not stop mid-air):
#   1. "the night he forgot": the attempt played out until he's flat on the track (3 angles)
#   2. timelapses v2 for all 9 chapters: after each fall the physics runs on 1.2 s so he lands
#      before the restart. Close camera, then the keep-pace camera (side_follow) where he travels.
# Waits for the marathon to finish first. Safe to restart: finished clips are skipped.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=renders/ep01_final
T=$OUT/timelapse_v2
PROG=$OUT/PROGRESS.txt
LOG=$OUT/log_marathon.txt
mkdir -p $T
until [ "$(grep -c 'ALL FINISHED' $PROG)" -ge 2 ]; do sleep 60; done

step() {   # label, then the command
  local label=$1; shift
  echo "=== $label $(date '+%a %H:%M')" >> "$LOG"
  if "$@" >> "$LOG" 2>&1 < /dev/null; then echo "done    $(date '+%a %H:%M')  $label" >> "$PROG"
  else echo "FAILED  $(date '+%a %H:%M')  $label (see log_marathon.txt)" >> "$PROG"; fi
}

[ -f $OUT/breakthroughs/bt5_463M_forgot_full_fall_three_quarter_clean.mp4 ] || \
  step "forgot: full fall" "$PY" -m rupert.make_sequence forgot --out-dir $OUT/breakthroughs

for ch in cheat1_stands_still cheat2_leaning_back cheat3_pogo cheat4_straight_legs cheat5_knees cheat6_trex cheat7_limp cheat8_gallop final_run; do
  [ -f "$T/tl_${ch}_ghost_side_close_clean.mp4" ] && continue
  step "timelapse v2 $ch side_close" "$PY" -m rupert.make_sequence ghost --chapter "$ch" --seconds 10 \
    --camera side_close --out-dir "$T"
done
for ch in final_run cheat6_trex cheat7_limp cheat8_gallop cheat2_leaning_back cheat3_pogo cheat4_straight_legs; do
  [ -f "$T/tl_${ch}_ghost_side_follow_clean.mp4" ] && continue
  step "timelapse v2 $ch side_follow" "$PY" -m rupert.make_sequence ghost --chapter "$ch" --seconds 10 \
    --camera side_follow --out-dir "$T" --reuse
done
echo "FOLLOW-UP FINISHED $(date)" >> $PROG
