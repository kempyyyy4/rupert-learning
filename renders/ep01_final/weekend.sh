#!/usr/bin/env bash
# EP01 weekend batch (Tom approved 2026-11-14): the Olympic angle for the 8 cheats, the judge
# shots at full size, then the 24 Night 1 cheat clips again so each also gets a no-counter
# (_clean) version. About 9 hours. Safe to restart: finished clips are skipped.
# Progress: renders/ep01_final/log_weekend.txt
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
R=$(pwd)
OUT=renders/ep01_final
LOG=$OUT/log_weekend.txt
echo "Started $(date)" >> "$LOG"

clip() {   # chapter run step start label cam tag
  f="$OUT/$5_$7_$6.mp4"
  if [ -f "${f%.mp4}_clean.mp4" ]; then echo "--- skip $f (done)" >> "$LOG"; return; fi
  echo "=== $f $(date +%H:%M)" >> "$LOG"
  if "$PY" -m rupert.make_clip --run "outputs/$2" --step "$3" --chapter "$1" --start "$4" --seconds 10 \
       --camera "cam_$6" --out "$f" >> "$LOG" 2>&1 < /dev/null; then
    echo "    done $(date +%H:%M)" >> "$LOG"
  else
    echo "    FAILED (see above)" >> "$LOG"
  fi
}

judge() {   # blend last_frame
  f="$OUT/$1.mp4"
  if [ -f "$f" ]; then echo "--- skip $f (done)" >> "$LOG"; return; fi
  echo "=== $f $(date +%H:%M)" >> "$LOG"
  mkdir -p "$R/$OUT/frames_$1"
  /e/Blender/blender.exe -b "renders/ep01_shots/$1.blend" --python-expr \
    "import bpy; s=bpy.context.scene; s.render.resolution_percentage=100; s.camera=s.objects['cam_face']" \
    -o "$R/$OUT/frames_$1/f_####" -s 1 -e "$2" -a >> "$LOG" 2>&1 < /dev/null
  "$PY" -c "import imageio_ffmpeg,subprocess; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-framerate','50','-i','$OUT/frames_$1/f_%04d.png','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','$f'],check=True)" \
    && echo "    done $(date +%H:%M)" >> "$LOG" || echo "    FAILED" >> "$LOG"
}

CHEATS="cheat1_stands_still:rupert_20261107_1159:23592960:1:cheat1_stands_still
cheat2_leaning_back:colab_run3:117964800:4:cheat2_limbo_shuffle
cheat3_pogo:rupert_20261106_2121:101253120:4:cheat3_pogo
cheat4_straight_legs:rupert_20261106_2220:345047040:8:cheat4_straight_legs
cheat5_knees:rupert_20261107_1159:728432640:6:cheat5_knees
cheat6_trex:rupert_20261107_1323:875888640:4:cheat6_trex
cheat7_limp:rupert_20261107_2130:757923840:4:cheat7_superhero_gallop
cheat8_gallop:rupert_20261108_0008:698941440:4:cheat8_skipping_gallop"

# 1. Olympic angle (new)
while IFS=: read ch run step start label; do clip "$ch" "$run" "$step" "$start" "$label" olympic d; done <<< "$CHEATS"
# 2. Judge at full size
judge judge_intro 225
judge judge_card 175
# 3. Night 1 angles again, now with no-counter versions
for cam in three_quarter:a side:b front_low:c; do
  while IFS=: read ch run step start label; do clip "$ch" "$run" "$step" "$start" "$label" "${cam%%:*}" "${cam##*:}"; done <<< "$CHEATS"
done
echo "Finished $(date)" >> "$LOG"
