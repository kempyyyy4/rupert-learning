# EXP-004 — First RunPod run, scoring v2 (tall posture + air-time strides)

**Date:** 2026-11-06 (pod clock, UTC)
**Episode:** EP01
**Goal:** Test scoring v2 and the RunPod pipeline.

## Setup
- Body: Rupert v1. Scoring v2. Brain 256-256-128.
- Parallel environments: 2048. Training steps: 101,253,120. From scratch.

## Run
- Where: RunPod, RTX 4090 (driver 595 / CUDA 13.2)
- Time taken: 8.5 min training (~200,000 steps/s, ~5x Colab T4); ~20 min pod total
- Cost: roughly £0.10–0.30 (plus ~£0.10–0.25 for the failed first attempt)
- Run folder: outputs/rupert_20261106_2121

## Result
- Pipeline: worked end to end (background run, zip, download).
- Learned: finishes 100 m in every test attempt from ~85M; best 21.8 s, top 4.9 m/s.
- Looks like: **upright (head 1.76 m) but stiff-legged**, tiny bouncy toe steps (7.5/s, 0.6 m), one arm held out in front. Not running yet.
- Cause: air-time stride reward is satisfied by small bounces; "tall" range pushed him to lock his legs.

## What to change next time (done for EXP-005)
- Stride reward = metres the foot travels per swing (alternating, swing ≥ 0.15 s), weight 2.0.
- Tall range relaxed to 1.40–1.62 m head height.
- Arm penalty for hands > 0.35 m in front of / behind the chest.
- Full-length run: 1B steps, 140 checkpoints.
