# EXP-003 — First Colab run of the rebuilt Rupert (scoring v1)

**Date:** 2026-11-07
**Episode:** EP01
**Goal:** Check the new body + scoring learns in the right direction; capture early stages.

## Setup
- Body: Rupert v1 (1.84 m, 70 kg, de Leva masses, force–velocity muscles)
- Environment: Rupert100m, scoring v1 (see decisions 2026-11-06)
- Parallel environments: 2048
- Training steps: 117,964,800 (asked 100M; rounding bug, now fixed)
- Brain: Brax default (policy 4×32)
- Started from: scratch

## Run
- Where: Colab, free T4
- Time taken: 51 min training (≈38,000 steps/s)
- Cost: £0

## Result
- Did it learn? **Yes.** Collapse → lunging falls (~18M) → stooped shuffle (~30M) → finishes 100 m every attempt from ~77M.
- Best 100 m: **19.2 s** at 112M steps; top speed 5.6 m/s. Still improving when it stopped.
- What it looks like: **not human running.** Hunched forward (head 1.52 m vs 1.75 standing), knees bent, arms hanging, 6.7 tiny steps a second (~0.8 m each). A fast stooped shuffle.
- Best moments: step 0 (collapse), ~18M (lunge and fall), ~30M (first shuffle beyond 10 m), ~77M (first 100 m finish).
- Colab filming crashed on the long final clip (likely memory). Fixed by a lighter renderer.

## What to change next time (done for EXP-004)
- Speed only counts fully when running tall (head ≥ 1.65 m); fall line raised to 1.2 m.
- Stride reward needs a proper foot swing (≥ 0.25 s) → longer strides.
- Upright checks chest and pelvis.
- Bigger brain (256-256-128).
