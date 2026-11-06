# EXP-005: First full-length run, scoring v3 (stride length, arm penalty)

**Date:** 2026-11-06 (pod clock, UTC)
**Episode:** EP01
**Goal:** Test scoring v3 over a full 1B-step run.

## Setup
- Body: Rupert v1. Scoring v3. Brain 256-256-128.
- Parallel environments: 2048. Training steps: ~1.0B. From scratch.
- PPO: Playground HumanoidRun settings (learning rate 1e-3, no gradient cap).

## Run
- Where: RunPod, RTX 4090
- Time taken: ~85 min training, ~1 h 45 min pod total
- Cost: roughly £0.50–£1.50
- Run folder: outputs/rupert_20261106_2220

## Result
- **Peak at 345M steps: every test attempt (8 of 8) finished 100 m in about 15.2 s. Best 15.15 s, top speed 7.4 m/s.** Ran the full 25 s without falling, lane drift ≤ 0.13 m.
- Looks like: **a real running stride.** He stays upright and alternates legs with clear flight phases (flight 0.49), about 0.9 m per step at about 7.6 steps/s. Arms are tucked high by the chest, not swinging naturally. Cadence is still high and strides are short for a sprinter.
- **Then training collapsed at ~350M.** The score dropped from ~1000 to ~20 and never recovered. From then on he falls after ~2.7 s. This is not reward hacking, because the score went down. One bad learning update wrecked the brain (a known PPO failure at high learning rates).
- Usable checkpoints: 286M–345M (best: step_0000345047040.pkl). Early-stage clips from 0–40M are all there.
- Earlier brief dip at 295M recovered, which was a warning sign.

## What to change next time (done for EXP-006)
- Learning rate 1e-3 → 3e-4, and gradient cap max_grad_norm = 1.0.
- Knees (Tom spotted it): measured at 345M, right knee bends at most 51°, left only 12°, and both push past straight (+17° / +7°). Added the `knees` reward (scoring v4).
- Next: decide whether the 345M gait is good enough for EP01, or rerun with the stable settings to get a smoother, faster gait.
