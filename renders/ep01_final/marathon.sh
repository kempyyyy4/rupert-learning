#!/usr/bin/env bash
# EP01 four-day render (Tom away 2026-11-15 to 11-19: "render everything that you need").
# Most important first. Safe to restart: finished clips are skipped.
#   1. weekend.sh: Olympic angle for the 8 cheats, judge shots, no-counter cheat clips (~9 h)
#   2. The final 100 m (EXP-010, step 846M, attempt 4, 15.2 s) from 4 angles (~2 h)
#   3. Breakthroughs: each new personal best in EXP-010, whole run, Olympic angle,
#      plus the night he forgot how to run (463M) (~3 h)
#   4. Ghost timelapses for all 9 chapters, side-on close camera, then the panning one (~45 h)
# Progress: renders/ep01_final/PROGRESS.txt (short) and log_marathon.txt (everything)
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=renders/ep01_final
LOG=$OUT/log_marathon.txt
PROG=$OUT/PROGRESS.txt
echo "Started $(date)" >> "$PROG"

step() {   # label, then the command
  local label=$1; shift
  echo "=== $label $(date '+%a %H:%M')" >> "$LOG"
  if "$@" >> "$LOG" 2>&1 < /dev/null; then echo "done    $(date '+%a %H:%M')  $label" >> "$PROG"
  else echo "FAILED  $(date '+%a %H:%M')  $label (see log_marathon.txt)" >> "$PROG"; fi
}

run_clip() {   # out run step attempt start seconds camera chapter
  [ -f "${1%.mp4}_clean.mp4" ] && return
  mkdir -p "$(dirname "$1")"
  step "$(basename "$1")" "$PY" -m rupert.make_clip --run "outputs/$2" --step "$3" --replay "$4" \
    --start "$5" --seconds "$6" --camera "cam_$7" --chapter "$8" --out "$1"
}

# 1. weekend batch
step "weekend batch (Olympic cheats, judge, clean cheat clips)" bash $OUT/weekend.sh

# 2. the final 100 m
EXP=rupert_20261112_1312
for cam in olympic three_quarter side front_low; do
  run_clip $OUT/final/final_run_15s2_$cam.mp4 $EXP 846397440 4 0 17 $cam final_run
done

# 3. breakthroughs (whole runs, a second past the line), and the night he forgot
B=$OUT/breakthroughs
run_clip $B/bt1_079M_first_finish_22s9_olympic.mp4 $EXP 79626240 6 0 24 olympic final_run
run_clip $B/bt1_079M_first_finish_22s9_side.mp4    $EXP 79626240 6 0 24 side final_run
run_clip $B/bt2_198M_19s6_olympic.mp4              $EXP 197591040 6 0 21 olympic final_run
run_clip $B/bt3_286M_18s5_olympic.mp4              $EXP 286064640 1 0 20 olympic final_run
run_clip $B/bt4_375M_17s6_olympic.mp4              $EXP 374538240 4 0 19 olympic final_run
run_clip $B/bt5_463M_forgot_olympic.mp4            $EXP 463011840 6 0 5 olympic final_run
run_clip $B/bt5_463M_forgot_side.mp4               $EXP 463011840 6 0 5 side final_run
run_clip $B/bt6_699M_17s0_olympic.mp4              $EXP 698941440 3 0 18 olympic final_run

# 4. timelapses: one continuous side-on shot per chapter (close camera first, then panning)
T=$OUT/timelapse
mkdir -p $T
CHAPTERS="cheat1_stands_still cheat2_leaning_back cheat3_pogo cheat4_straight_legs cheat5_knees cheat6_trex cheat7_limp cheat8_gallop final_run"
for cam in side_close side_pan; do
  for ch in $CHAPTERS; do
    [ -f "$T/tl_${ch}_ghost_${cam}_clean.mp4" ] && continue
    reuse=""; [ -f "$T/tl_${ch}_ghost.blend" ] && reuse="--reuse"
    step "timelapse $ch $cam" "$PY" -m rupert.make_sequence ghost --chapter "$ch" --seconds 10 \
      --camera "$cam" --out-dir "$T" $reuse
  done
done
echo "ALL FINISHED $(date)" >> "$PROG"
