# EP01 — Rupert learns to run

**Status:** Planning

## Goal
Rupert goes from collapsing to running, ending on his first 100m time trial. The gap to Bolt's 9.58 sets up EP02.

## Milestone 1 is done when
- [ ] Rupert genuinely learns to run, with no pre-made animation anywhere
- [ ] At least five distinct stages are captured as clips
- [ ] The final run looks like running, not a hop or a crawl
- [ ] Clips render at 1080p with the Rupert model on the track
- [ ] Total cloud spend is under £15

## Build steps
1. ~~Set up the GitHub repo and the app skeleton~~ (done 2026-11-05; Tom confirmed install check and humanoid viewer work)
2. Get the humanoid training in free Google Colab to prove the code works end to end
3. Run the full training on a rented GPU, saving checkpoints
4. Run the moment-finder and pick the stages
5. Build the Blender scene (track, Rupert, cameras) and render the clips
6. Tom watches and decides whether it's a video

## Experiments
- EXP-008: body v2 (ordinary young man) worked: **100 m in 13.5 s**, ~5 steps/s, steady training. New cheat: **lopsided gallop with a frozen 'superhero' arm**, faking the arm-swing reward. Clip: step_0000757923840.
- EXP-007: knee fix worked (upright the whole time), but a new cheat appeared: the **T-rex scuttle** (arms locked to his chest, feet scurrying under his hips). **100 m in 10.0 s**, too fast for a beginner, so the body is probably too strong. Clip: step_0000875888640.
- EXP-001: local smoke test, pipeline works (CPU, 17 min)
- EXP-006: stable training (no permanent collapse) but he learned to **run on his knees** (best 20.8 s). Also a 'stand still and survive' phase at 20-40M. Great comedy clip: step_0000728432640.
- EXP-005: first full 1B run (scoring v3). **Real running stride at 345M: 100 m in 15.2 s, 8/8 attempts.** Then training collapsed (falls at ~2.7 s); fixed with a slower learning rate. Best clip: step_0000345047040.
- EXP-004: first RunPod run (4090, 8.5 min for 100M steps). Upright now, but stiff-legged bouncing with a 'zombie arm'. Best 100 m 21.8 s.
- EXP-003: rebuilt Rupert on Colab. Learned to finish 100 m (best 19.2 s) but as a hunched shuffle. Great early-stage clips (collapse, lunge-and-fall, first shuffle).
- REVIEW 2026-11-06: stock setup unfit for EP01 (wrong body size, scoring rewards standing, no early checkpoints, no stats/replay). Rebuilt. See decisions.md.
- EXP-002: first Colab T4 run, reward 1.5 -> ~138, jump near step 60M. x-axis ran to ~88M not the 20M requested (unexplained). Checkpoints not saved (lost). Clip not rendered.

## Story beats (fill in as clips come in)
**Spine (Tom, 2026-11-07): "he cheats, we give him a new reward, he cheats again... then he finally runs."**
In narration, each fix is "giving Rupert a reward" (or a rule) for doing something.

| # | Rupert's cheat | Run / clip | The fix, in video words |
|---|---|---|---|
| 1 | **Stands still to survive** (never falls if he never moves) | stock setup (EXP-002); also EXP-006 at 20–40M (render step ~23M) | Points for moving forward; only a tiny reward for just staying up |
| 2 | **Hunched shuffle**: finishes 100 m bent double with tiny steps | EXP-003 `colab_run3/step_0000117964800.mp4` | Full points only when he runs **tall** |
| 3 | **Stiff-legged pogo bounce** with a 'zombie arm' | EXP-004 `rupert_20261106_2121/step_0000101253120.mp4` | Rewarded for **long strides**; told off for flailing arms |
| 4 | **Robot run on straight legs**, then **forgets how to run overnight** | EXP-005 `rupert_20261106_2220/step_0000345047040.mp4` | Rewarded for **bending his knees**; we made him learn more carefully |
| 5 | **Runs on his knees** | EXP-006 `rupert_20261107_1159/step_0000728432640.mp4` | **Knees on the ground = a fall**; knee reward only in a real stride |
| 6 | **T-rex scuttle**: arms clamped to his chest, feet scurrying under his hips at Bolt speed (100 m in 10.0 s) | EXP-007 `rupert_20261107_1323/step_0000875888640.mp4` | Rewarded for **swinging his arms**; given an **ordinary young man's body** (lab-tested strength, muscles that take time to switch on) |
| 7 | **Lopsided gallop with a frozen 'superhero' arm**: same leg always leads, arms held in a fixed 'opposite' pose to fake the arm swing. (Now at a realistic 13.5 s per 100 m.) | EXP-008 `rupert_20261107_2130/step_0000757923840.mp4` | Arm reward only for arms that actually **swing**; full stride points only for **even steps** |
| 8 | ? | EXP-009 | |

Honesty notes for the script:
- Each fix started a **fresh training run**: his brain was wiped and he learned again from zero under the new rules. Say it ("we wiped his memory and gave him new rules"), don't imply one continuous Rupert.
- Fixes 1, 2, 4 and 5 included **penalties and rules**, not only rewards. "Rewards and rules" or "rewards and punishments" keeps it true.
- The attempt counter in each run counts that run's attempts. For the whole-story total, add the runs together and call it "total attempts".

