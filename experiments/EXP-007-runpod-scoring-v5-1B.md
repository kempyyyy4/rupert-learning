# EXP-007: Full run, scoring v5 (no knee-running)

**Date:** 2026-11-07 (pod clock, UTC)
**Episode:** EP01
**Goal:** Stop the knee-running from EXP-006 and get a real running stride.

## Setup
- Body: Rupert v1 (strengths still TO VERIFY). Brain 256-256-128.
- Scoring v5:
  - a knee or a hand within 0.22 m of the ground counts as a fall
  - the "fallen" head height is now 1.3 m
  - the knee bonus only pays during a real stride (one foot swinging while the other is planted)
- PPO: learning rate 3e-4, gradient cap 1.0. 2048 parallel Ruperts, ~1.0B steps. Started from scratch.

## Run
- Where: RunPod. 13:23 to 14:25 UTC, about 1 h of pod time.
- Cost: roughly £0.50–£1.50
- Run folder: outputs/rupert_20261107_1323

## Result
- **The knee fix worked.** He never goes below 0.41 m at the knee, and he stays upright for all 25 s.
- **He got very fast, too fast.** Best 100 m: **10.0 s** at step 875M, top speed 12.1 m/s, 8 of 8 attempts finished. He was already at 11.3 s by 168M steps.
- **New cheat: the "T-rex scuttle".**
  - **Arms:** one arm is clamped across his chest and the other hangs dead. Neither swings. This comes from our own arms rule, which only penalises hands far in front of or behind the chest, so the cheapest move is to lock them to his body.
  - **Legs:** the front foot always lands under his hips (never more than 0.15 m in front), and the back leg flicks up high behind him. His knees never straighten (always bent at least 31°).
  - **Speed:** about 6.2 steps a second of 1.8 m each. Humans top out around 4.5–5 steps a second, so this cadence isn't human. Likely causes: muscles that switch on instantly, no fatigue, and fast muscle-speed limits (all TO VERIFY).
- Training still dips every so often (at 256M, 315–345M, 463M, 551M and 699M) but always recovers.
- **Possible early clips, not yet checked:**
  - 0–25M: collapse and lunging.
  - ~32M: upright for 25 s but covering only ~36 m, so a slow walk or shuffle. Possibly his "first walk".
- Best clip: `step_0000875888640.mp4` (quick check render, side camera).

## What to change next time (needs Tom's agreement)
- **Arms:** reward arms swinging opposite to the legs, instead of only penalising far-out hands.
- **Body:** the body is likely too strong and too quick, because a beginner shouldn't reach Bolt's speed. Check strengths and muscle speeds against sources and add a short muscle switch-on delay. Aim for an "ordinary fit bloke" ceiling (100 m in roughly 12–14 s) so the gap to Bolt survives for EP02.
- **Fallback:** if rewards still can't produce a human-looking run, use a style judge that has watched real runners (motion capture).
