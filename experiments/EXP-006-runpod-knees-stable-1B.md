# EXP-006: Full run, scoring v4 (knee rule) + stable settings

**Date:** 2026-11-07 (pod clock, UTC)
**Episode:** EP01
**Goal:** Get bent knees (EXP-005 ran straight-legged) and stop training from collapsing.

## Setup
- Body: Rupert v1. Scoring v4 (v3 + a knee rule: bonus for a bent knee while that foot is in the air, penalty for locked knees). Brain 256-256-128.
- PPO: learning rate 3e-4 (was 1e-3), gradient cap 1.0. 2048 parallel Ruperts, ~1.0B steps.

## Run
- Where: RunPod. ~1 h of training, ~1 h 5 min pod total.
- Cost: roughly £0.50–£1.50
- Run folder: outputs/rupert_20261107_1159

## Result
- **No permanent collapse.** He dipped at ~286M, ~550M and ~816M but recovered every time. The stable settings worked.
- **But he learned to run on his KNEES.** Both knees stay bent 110–168° the whole time, his shins skim the ground, and he shuffles forward kneeling.
  - **Why:** the kneeling head height (~1.3 m) was above the 1.2 m "fallen" line.
  - **And:** the knee bonus paid out whenever a foot was in the air with the knee bent. When kneeling both feet are in the air, so he collected the full bonus on every step.
- Best 100 m: 20.8 s (step 728M), top speed 5.3 m/s, 8 of 8 attempts finished at the best checkpoints. Progress was slower than EXP-005, as expected with the lower learning rate.
- Early stages are interesting: he learns to **stand still for the full 25 s** around 20–40M steps (the "just survive" phase) before moving forward.
- **Good footage:** "Rupert discovers knee-running" is a strong comedy beat. Clip: step_0000728432640.mp4.

## What to change next time (done for EXP-007, scoring v5)
- A knee joint or a hand within 0.22 m of the ground counts as a fall (no knee-running, no crawling). The "fallen" head height is raised to 1.3 m.
- The knee bonus only pays while that foot swings AND the other foot is planted (a real running stride).
- Same stable settings and length.
