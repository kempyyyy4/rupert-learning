# EXP-010: Style judge (scoring v8, AMP)

**Date:** 2026-11-12 (pod clock, UTC)
**Episode:** EP01 (the ending)
**Goal:** Stop writing anti-cheat rules. Show him real runners instead: a second AI (the style judge) learns to tell real joggers from Rupert, and Rupert earns style points for fooling it.

## Setup
- **Body:** v2 (ordinary young man), the same as EXP-008/009.
- **Scoring v8:**
  - All the special cheat fixes are off: run tall, strides, knees, arm swing, hand reach.
  - Only the basics remain: forward speed 0.25, alive 0.2, heading 0.1, lane -0.3, energy, jerk.
  - Style points from the judge, weight 1.5.
- **Judge:**
  - 1,640 real-runner moments from 26 CMU jogging clips (3 people, 2.6–4.5 m/s), fitted onto Rupert's body.
  - It sees form, not forward speed.
  - Learning rate 5e-5, gradient penalty 5.
- **Practice starts:** 30% of practice attempts start mid-stride in a real runner's pose. Test runs and 100 m trials always start standing.
- **PPO:** learning rate 3e-4, gradient cap 1.0. 2048 Ruperts, ~1.0B steps, started from scratch.

## Run
- **Where:** RunPod. Training 13:13 to 14:13 UTC, 60 min. The pod ran about 1 h 10 min in total.
- **Cost:** roughly £0.50–£1.00.
- **Run folder:** `outputs/rupert_20261112_1312`

## Result
**He runs like a person. No new cheat found.** For the first time, every gait measure lands inside the real joggers' range (new tool: `python -m rupert.gait <replay.npz> --ref`).

| | Rupert (step 846M) | Real joggers (17 clips) | EXP-009 (cheat #8) |
|---|---|---|---|
| Feet take turns | always | always | yes, but the right foot lands *behind* |
| Landing foot ahead of the other | L 0.47 m, R 0.61 m | 0.57–0.94 m | L +1.02, R **-0.34** |
| Forward lean | 17° | 14–28° | -4° (leaning back) |
| Head height while running | 1.69 m | 1.54–1.71 m | 1.60 m |
| Arms swing with the opposite leg (1 = perfect) | 0.83 | 0.68–0.89 | 0.94 |
| Knee bend (swing leg, peak) | 108–118° | 87–117° | 125–152° (high-stepping) |
| Steps a second | 3.7 | 2.0–3.0 (they jog at 3–4 m/s) | 5.5 |

- **Speed:** best 100 m 15.2 s (step 846M), top speed 7.2 m/s, and the same in all 8 test runs (15.25–15.37 s). For comparison, EXP-009 ran 15.4 s.
- **Faster than the joggers he copied.** He runs about twice their speed, so he takes quicker, shorter steps. That's what a person speeding up does, too.
- **Start: a crouch-and-drive.**
  - His head dips 28 cm for about half a second, then he drives forward, reaching 6.7 m after 2 s.
  - His knees stay at least 34 cm off the ground.
  - The judge never saw a standing start (all the recordings are mid-jog), so this part is his own invention. It reads like a sprinter's drive phase; not a cheat.
- **Training wobbles:**
  - At 463M he briefly forgot how to run (average 12 m), recovering by the next check-in.
  - At 757M a couple of test runs fell.
  - Otherwise every check-in from 80M on finished 100 m.
- **The judge's view:**
  - The judge never got fooled completely: style points stayed around 0.2–0.5 out of 1 all run, and its score for Rupert stayed below its score for the real runners.
  - That's normal for this method: the judge keeps learning too, so it always finds *some* difference.
  - The gait numbers above are the real test.
- **Best clip:** `outputs/rupert_20261112_1312/step_0000846397440.mp4` (quick check render, side camera, 10 s).

## What to change next time
- Nothing for EP01. **Use step 846M for the cinematic 100 m.**
- **For EP02 (sprinting):**
  - The judge needs sprint recordings, not jogging.
  - Standing or crouch starts in the data would let it judge the start too.
