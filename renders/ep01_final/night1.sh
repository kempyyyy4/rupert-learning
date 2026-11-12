#!/usr/bin/env bash
# EP01 Night 1: the 8 cheats from 3 angles (10 s each, running-total counter), then the
# timelapses for chapters 1 and 2. Progress: renders/ep01_final/log_night1.txt
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
OUT=renders/ep01_final
LOG=$OUT/log_night1.txt
echo "Started $(date)" > "$LOG"

cheat() {   # chapter run step start label
  for cam in three_quarter side front_low; do
    case $cam in three_quarter) tag=a;; side) tag=b;; front_low) tag=c;; esac
    f="$OUT/$5_${tag}_${cam}.mp4"
    if [ -f "$f" ]; then echo "--- skip $f (exists)" >> "$LOG"; continue; fi
    echo "=== $f $(date +%H:%M)" >> "$LOG"
    if "$PY" -m rupert.make_clip --run "outputs/$2" --step "$3" --chapter "$1" --start "$4" --seconds 10 \
         --camera "cam_$cam" --out "$f" >> "$LOG" 2>&1; then
      echo "    done $(date +%H:%M)" >> "$LOG"
    else
      echo "    FAILED (see above)" >> "$LOG"
    fi
  done
}

timelapse() {   # chapter
  for shot in swarm montage; do
    echo "=== timelapse $1 $shot $(date +%H:%M)" >> "$LOG"
    if "$PY" -m rupert.make_sequence "$shot" --chapter "$1" --out-dir "$OUT" >> "$LOG" 2>&1; then
      echo "    done $(date +%H:%M)" >> "$LOG"
    else
      echo "    FAILED (see above)" >> "$LOG"
    fi
  done
}

cheat cheat1_stands_still  rupert_20261107_1159 23592960  1 cheat1_stands_still
cheat cheat2_leaning_back  colab_run3           117964800 4 cheat2_limbo_shuffle
cheat cheat3_pogo          rupert_20261106_2121 101253120 4 cheat3_pogo
cheat cheat4_straight_legs rupert_20261106_2220 345047040 8 cheat4_straight_legs
cheat cheat5_knees         rupert_20261107_1159 728432640 6 cheat5_knees
cheat cheat6_trex          rupert_20261107_1323 875888640 4 cheat6_trex
cheat cheat7_limp          rupert_20261107_2130 757923840 4 cheat7_superhero_gallop
cheat cheat8_gallop        rupert_20261108_0008 698941440 4 cheat8_skipping_gallop
timelapse cheat1_stands_still
timelapse cheat2_leaning_back
echo "Finished $(date)" >> "$LOG"
