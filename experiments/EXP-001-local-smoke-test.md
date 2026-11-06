# EXP-001 — Local smoke test (code check, not a real training run)

**Date:** 2026-11-06
**Episode:** EP01
**Goal:** Prove `rupert/train.py` runs end to end (train, log reward, save checkpoints) on Tom's PC.

## Setup
- Body: Playground HumanoidRun (standard humanoid)
- Environment: flat floor, default rewards
- Rewards: Playground defaults
- Penalties: Playground defaults
- Parallel environments: 128
- Training steps: ~230,000 (tiny on purpose)
- Started from: scratch

## Run
- Where: Tom's PC, CPU only (JAX 0.9.2)
- Time taken: 17 minutes
- Cost: £0

## Result
- Did it learn? Reward went 1.52 -> 5.56. Too short to mean anything; it only proves the pipeline works.
- Checkpoints saved correctly (`step_*.pkl`).
- Found: JAX 0.12 breaks the trainer (`device_put_replicated` removed). JAX 0.9.2 works, so `requirements.txt` pins it.

## What to change next time
- Run the real thing on a GPU (EXP-002, Colab T4, 20M steps).
