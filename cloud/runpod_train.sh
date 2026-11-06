#!/usr/bin/env bash
# Rupert Sim: full training run on a rented RunPod GPU. UNTESTED until our first paid run.
#
# What it does, in order:
#   1. downloads the code and installs the tested versions
#   2. trains (with a hard time limit, so a stuck job can't run up a bill)
#   3. runs the test attempts + replay files on every 2nd checkpoint
#   4. packs everything into /workspace/results_<date>.tar.gz (the network volume, which survives)
#   5. DELETES THIS POD, whatever happened above (success, crash or time-out)
#
# Usage on the pod:  bash runpod_train.sh  [steps]  [max_hours]
set -uo pipefail

STEPS="${1:-1000000000}"
MAX_HOURS="${2:-4}"
RUN="rupert_$(date +%Y%m%d_%H%M)"
OUT="/workspace/${RUN}"

shutdown_pod() {
  echo "=== Packing results and shutting the pod down ==="
  tar -czf "/workspace/${RUN}.tar.gz" -C /workspace "${RUN}" 2>/dev/null || true
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl remove pod "${RUNPOD_POD_ID}"
  else
    echo "!!! Could not auto-delete the pod. DELETE IT BY HAND in the RunPod website NOW. !!!"
  fi
}
trap shutdown_pod EXIT

cd /root
git clone https://github.com/kempyyyy4/rupert-learning.git
cd rupert-learning
pip install -q "jax[cuda12]==0.9.2" mujoco playground imageio-ffmpeg
python -m rupert.check_install

timeout "${MAX_HOURS}h" python -m rupert.train --steps "${STEPS}" --out "${OUT}" 2>&1 | tee "${OUT}.log"
timeout 1h python -m rupert.evaluate --run "${OUT}" --every 2 2>&1 | tee -a "${OUT}.log"
