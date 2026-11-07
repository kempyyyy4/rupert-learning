# EXP-008: Body v2 (ordinary young man) + arm-swing reward

**Date:** 2026-11-07 (pod clock, UTC)
**Episode:** EP01
**Goal:** Bring his speed down to a realistic level and get his arms swinging (EXP-007 was the "T-rex scuttle" at Bolt speed).

## Setup
- Body v2:
  - hip, knee and ankle strength from Harbo et al. 2012 (non-athletic men under 30), different each way
  - muscles take 10 ms to switch on and 40 ms to switch off
- Scoring v6: v5 plus `arm_swing` (weight 0.5). The reward is the left-minus-right hand position times the left-minus-right foot position, capped at ±1.
- PPO: learning rate 3e-4, gradient cap 1.0. 2048 Ruperts, ~1.0B steps, started from scratch.

## Run
- Where: RunPod. 21:31 to 22:34 UTC, about 1 h.
- Cost: roughly £0.50–£1.50
- Run folder: outputs/rupert_20261107_2130

## Result
- **Body v2 worked.** Best 100 m: **13.5 s** (step 758M), top speed 8.3 m/s. That's in the 12–14 s target, so the gap to Bolt is back.
  - Cadence is about 5 steps a second, which is human (EXP-007 was 6.2).
  - He's upright all 25 s, and 8 of 8 attempts finish.
  - Training was steady: one dip at 964M, recovered.
- Learning curve:
  - stands by ~50M
  - ~6.5 m/s and 16.7 s per 100 m from 227M to 490M, a long plateau
  - then improves to 13.5 s by 758M
- **New cheat (#7): the lopsided gallop with a frozen "superhero" arm.**
  - **Legs:** the left foot stays behind the right foot 97% of the time, so the same leg always leads. It's a gallop, not a run.
  - **Arms:** the right arm sticks out stiff behind him and the left hand hangs in front. Neither swings much.
  - **Why it pays:** the arm reward only checked whether the arms and legs were "opposite" at each moment. A frozen opposite pose, plus legs that are always in the same order, scores 0.80 out of a possible 1.0 without any swinging.
- Best clip: `step_0000757923840.mp4` (quick check render, side camera).

## What to change next time (done for EXP-009, scoring v7; Tom chose A)
- **Arm swing based on movement, not position.** Reward hand speed (forwards/backwards) moving opposite to foot speed. Frozen arms then score 0.
- **Even steps.** On a closer look, each foot does land in front of the other, but his steps alternate about 0.3 m / 0.75 m, a limp. Stride credit is now scaled by shorter step ÷ longer step (his limp: 0.47).
- Or move to the style judge (AMP). Tom to decide.
