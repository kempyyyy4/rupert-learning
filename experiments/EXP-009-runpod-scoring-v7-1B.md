# EXP-009: Scoring v7 (arm swing from movement, even steps)

**Date:** 2026-11-08 (pod clock, UTC)
**Episode:** EP01
**Goal:** Get him swinging his arms for real and stop the EXP-008 limp. This was the last reward-only attempt.

## Setup
- Body v2 (ordinary young man), the same as EXP-008.
- Scoring v7:
  - `arm_swing` uses hand and foot speeds, not positions (weight 0.5)
  - stride credit is scaled by shorter ÷ longer of the last two steps
  - a step only counts if the landing foot lands at least 0.1 m in front of the other foot
- PPO: learning rate 3e-4, gradient cap 1.0. 2048 Ruperts, ~1.0B steps, started from scratch.

## Run
- Where: RunPod. 00:09 to 01:28 UTC, about 1 h 20 min.
- Cost: roughly £0.50–£1.50
- Run folder: outputs/rupert_20261108_0008

## Result
- **The arms now swing properly.** Hands move opposite to the legs (correlation -0.98, arm-swing score 0.88 out of 1). This is the first Rupert with a real arm swing.
- **Slower and less steady than EXP-008.** Best 100 m: 15.4 s (step 699M), top speed 7.1 m/s, about 5.5 steps a second. Training dipped hard at about 433M, 787M and 964M and recovered each time.
- **New cheat (#8): the skipping gallop.**
  - **Legs:** his left foot always leads. Each left step lands about 0.95 m in front of the right foot. His right foot hops up to land about 0.3 m *behind* the left.
  - **Why it pays:** this is a loophole in the v7 even-steps rule. A landing behind the other foot didn't count as a step, so it never entered the evenness check. Each left step was compared with the previous left step, which always looks perfectly even.
  - He also leans back slightly (about 5–7°), and his head is a bit low (1.60 m against 1.75 m standing).
- Best clip: `step_0000698941440.mp4` (quick check render, side camera).

## What to change next time
- Plugging the loophole is a two-line fix: a landing behind the other foot resets the evenness check to 0.
- **As agreed before this run, cheat #8 means moving to "showing him real runners".** Tom to choose how.
