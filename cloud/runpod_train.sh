#!/usr/bin/env bash
# Rupert Sim: training run on a rented RunPod GPU.
#
# Tom pastes ONE line into the pod's web terminal (see docs/runpod-guide.md):
#   cd /workspace && wget -qO runpod_train.sh https://raw.githubusercontent.com/kempyyyy4/rupert-learning/main/cloud/runpod_train.sh && bash runpod_train.sh 100000000
#
# Arguments (all optional):  [steps]  [max_training_hours]  [download_window_minutes]
#
# What it does:
#   1. restarts itself in the background, so closing the browser tab doesn't stop it
#   2. checks the GPU and driver are usable; quits within minutes if not (no wasted money)
#   3. installs the tested versions, trains (hard time limit), runs the test attempts
#   4. zips the results to /workspace/rupert_results_<time>.zip  -> Tom downloads it in Jupyter
#   5. waits for the download window, then DELETES THE POD. Always, even after an error.
set -uo pipefail

STEPS="${1:-100000000}"
MAX_HOURS="${2:-2}"
WINDOW_MIN="${3:-45}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG="/workspace/rupert_${STAMP}.log"

# ---- 1. run in the background, show the log ----------------------------------------
if [ -z "${RUPERT_BG:-}" ]; then
  RUPERT_BG=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
  echo "Started. It keeps running even if you close this tab."
  echo "Watching the log (press Ctrl+C to stop watching; the job carries on):"
  sleep 2
  exec tail -n +1 -f "$LOG"
fi

RUN="rupert_${STAMP}"
OUT="/workspace/${RUN}"
ZIP="/workspace/rupert_results_${STAMP}.zip"
STATUS="FAILED"

delete_pod() {
  for cmd in "runpodctl remove pod ${RUNPOD_POD_ID:-x}" "runpodctl pod delete ${RUNPOD_POD_ID:-x}" "runpodctl stop pod ${RUNPOD_POD_ID:-x}"; do
    if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ] && $cmd; then
      echo "Pod shut down with: $cmd"; return 0
    fi
  done
  echo "!!!!!! COULD NOT SHUT THE POD DOWN AUTOMATICALLY !!!!!!"
  echo "!!!!!! Go to runpod.io > Pods and press the bin icon NOW !!!!!!"
}

finish() {
  echo
  echo "=== Packing results ($STATUS) ==="
  cp "$LOG" "$OUT/" 2>/dev/null || true
  if [ -d "$OUT" ]; then
    python3 -m zipfile -c "$ZIP" "$OUT" && echo "Results zip: $ZIP ($(du -h "$ZIP" | cut -f1))"
  fi
  local wait_min=$WINDOW_MIN
  [ "$STATUS" = "OK" ] || wait_min=15
  echo
  echo "################################################################"
  echo "#  $STATUS. DOWNLOAD NOW:"
  echo "#  In the Jupyter tab, refresh the file list on the left, right-click"
  echo "#  $(basename "$ZIP")  >  Download."
  echo "#  Then delete the pod yourself (Pods page > bin icon)."
  echo "#  Otherwise it deletes itself in ${wait_min} minutes, at $(date -d "+${wait_min} min" +%H:%M) (pod clock)."
  echo "################################################################"
  sleep $((wait_min * 60))
  delete_pod
}
trap finish EXIT

echo "=== Rupert Sim cloud run ${RUN}: ${STEPS} steps, max ${MAX_HOURS}h ==="
date

# ---- 2. GPU + driver check -----------------------------------------------------------
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "No GPU found."; exit 1; }
CUDA_DRV=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 0)
if [ "$CUDA_DRV" -lt 13 ]; then
  echo "This machine's NVIDIA driver supports CUDA $CUDA_DRV, but the physics needs CUDA 13+."
  echo "It would run on the slow CPU and waste money. Stopping. Pick a pod with CUDA 13 (see guide)."
  exit 1
fi

# ---- 3. install, train, test ---------------------------------------------------------
cd /root
rm -rf rupert-learning
git clone -q https://github.com/kempyyyy4/rupert-learning.git && cd rupert-learning || exit 1
pip install -q uv
uv venv -q -p 3.12 /root/venv || exit 1
PY=/root/venv/bin/python
uv pip install -q -p "$PY" "jax[cuda12]==0.9.2" mujoco playground imageio-ffmpeg || exit 1
"$PY" -m rupert.check_install

"$PY" - <<'EOF' || { echo "GPU physics check failed. Stopping."; exit 1; }
import jax, warp as wp
wp.init()
assert any(d.platform == "gpu" for d in jax.devices()), f"JAX sees no GPU: {jax.devices()}"
assert wp.is_cuda_available(), "Warp (the physics engine) sees no GPU"
print("GPU check passed:", jax.devices())
EOF

timeout "${MAX_HOURS}h" "$PY" -m rupert.train --steps "$STEPS" --early-steps 30000000 --main-checkpoints 20 --out "$OUT"
TRAIN_EXIT=$?
[ $TRAIN_EXIT -eq 124 ] && echo "Hit the ${MAX_HOURS}h time limit; keeping what was saved."
timeout 1h "$PY" -m rupert.evaluate --run "$OUT" --every 3 --attempts 8
[ $TRAIN_EXIT -eq 0 ] && STATUS="OK"
date
