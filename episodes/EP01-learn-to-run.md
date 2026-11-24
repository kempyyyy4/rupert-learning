# EP01 — Rupert learns to run

**Status:** Editing (all shots rendered at full quality, 2026-11-24; Tom on script, voiceover and rough cut)

## Goal
Rupert goes from collapsing to running, ending on his first 100m time trial. The gap to Bolt's 9.58 sets up EP02.

## Milestone 1 is done when
- [x] Rupert genuinely learns to run, with no pre-made animation anywhere
- [x] At least five distinct stages are captured as clips
- [x] The final run looks like running, not a hop or a crawl
- [x] Clips render at 1080p with the Rupert model on the track
- [x] Total cloud spend is under £15 (logged estimates EXP-004 to EXP-010: about £3–£9)

## Build steps
1. ~~Set up the GitHub repo and the app skeleton~~ (done 2026-11-05; Tom confirmed install check and humanoid viewer work)
2. Get the humanoid training in free Google Colab to prove the code works end to end
3. Run the full training on a rented GPU, saving checkpoints
4. Run the moment-finder and pick the stages
5. Build the Blender scene (track, Rupert, cameras) and render the clips
6. Tom watches and decides whether it's a video

## Experiments
- EXP-009: **arms finally swing** (first real arm swing), but cheat #8 is a **skipping gallop**: the same leg always leads, exploiting a loophole in the even-steps rule. Best 100 m 15.4 s; training less steady. Clip: step_0000698941440.
- EXP-008: body v2 (ordinary young man) worked: **100 m in 13.5 s**, ~5 steps/s, steady training. New cheat: **lopsided gallop with a frozen 'superhero' arm**, faking the arm-swing reward. Clip: step_0000757923840.
- EXP-007: knee fix worked (upright the whole time), but a new cheat appeared: the **T-rex scuttle** (arms locked to his chest, feet scurrying under his hips). **100 m in 10.0 s**, too fast for a beginner, so the body is probably too strong. Clip: step_0000875888640.
- EXP-001: local smoke test, pipeline works (CPU, 17 min)
- EXP-006: stable training (no permanent collapse) but he learned to **run on his knees** (best 20.8 s). Also a 'stand still and survive' phase at 20-40M. Great comedy clip: step_0000728432640.
- EXP-005: first full 1B run (scoring v3). **Real running stride at 345M: 100 m in 15.2 s, 8/8 attempts.** Then training collapsed (falls at ~2.7 s); fixed with a slower learning rate. Best clip: step_0000345047040.
- EXP-004: first RunPod run (4090, 8.5 min for 100M steps). Upright now, but stiff-legged bouncing with a 'zombie arm'. Best 100 m 21.8 s.
- EXP-003: rebuilt Rupert on Colab. Learned to finish 100 m (best 19.2 s) but as a leaning-back shuffle (bent over backwards; first noted as "hunched", corrected 2026-11-08). Great early-stage clips (collapse, lunge-and-fall, first shuffle).
- REVIEW 2026-11-06: stock setup unfit for EP01 (wrong body size, scoring rewards standing, no early checkpoints, no stats/replay). Rebuilt. See decisions.md.
- EXP-002: first Colab T4 run, reward 1.5 -> ~138, jump near step 60M. x-axis ran to ~88M not the 20M requested (unexplained). Checkpoints not saved (lost). Clip not rendered.

## Story beats (fill in as clips come in)
**Spine (Tom, 2026-11-07): "he cheats, we give him a new reward, he cheats again... then he finally runs."**
In narration, each fix is "giving Rupert a reward" (or a rule) for doing something.

| # | Rupert's cheat | Run / clip | The fix, in video words |
|---|---|---|---|
| 1 | **Stands still to survive** (never falls if he never moves) | stock setup (EXP-002); also EXP-006 at 20–40M (render step ~23M) | Points for moving forward; only a tiny reward for just staying up |
| 2 | **Leaning-back shuffle**: finishes 100 m bent over backwards (head 38 cm behind his hips, gazing at the sky) with tiny steps | EXP-003 `colab_run3/step_0000117964800.mp4` | Full points only when he runs **tall** |
| 3 | **Stiff-legged pogo bounce** with a 'zombie arm' | EXP-004 `rupert_20261106_2121/step_0000101253120.mp4` | Rewarded for **long strides**; told off for flailing arms |
| 4 | **Robot run on straight legs**, then **forgets how to run overnight** | EXP-005 `rupert_20261106_2220/step_0000345047040.mp4` | Rewarded for **bending his knees**; we made him learn more carefully |
| 5 | **Runs on his knees** | EXP-006 `rupert_20261107_1159/step_0000728432640.mp4` | **Knees on the ground = a fall**; knee reward only in a real stride |
| 6 | **T-rex scuttle**: arms clamped to his chest, feet scurrying under his hips at Bolt speed (100 m in 10.0 s) | EXP-007 `rupert_20261107_1323/step_0000875888640.mp4` | Rewarded for **swinging his arms**; given an **ordinary young man's body** (lab-tested strength, muscles that take time to switch on) |
| 7 | **Lopsided gallop with a frozen 'superhero' arm**: same leg always leads, arms held in a fixed 'opposite' pose to fake the arm swing. (Now at a realistic 13.5 s per 100 m.) | EXP-008 `rupert_20261107_2130/step_0000757923840.mp4` | Arm reward only for arms that actually **swing**; full stride points only for **even steps** |
| 8 | **Skipping gallop**: arms finally swing, but the same leg always leads and the other hops in behind it (a loophole in the even-steps rule). 15.4 s per 100 m. | EXP-009 `rupert_20261108_0008/step_0000698941440.mp4` | TBC: show him real runners |
| 9 | **No cheat: he runs like a person.** Every gait measure inside the real joggers' range; 15.2 s per 100 m; crouch-and-drive start | EXP-010 `rupert_20261112_1312/step_0000846397440.mp4` | All the special rules switched off; a second AI (the judge, which watched 26 real runs) scores whether he looks human |

**Finished-look clips of cheats 1–8** (1080p, stadium, 70s kit, attempts counter, 6 s each): `renders/cheat_reel/cheat1_…mp4` to `cheat8_…mp4`. Rebuild with `bash renders/cheat_reel/render_all.sh` (about 1.5 hours). Cheats 2 and 3 were re-tested for these clips (their replays weren't saved) and matched the notes: 18.5 s vs 19.2 s, and 21.6 s vs 21.8 s.

**The ending (Tom, 2026-11-08): show him real runners.** After eight cheats we stop writing rules. A second AI, the style judge, studies recordings of real people running and tries to tell them apart from Rupert. Rupert gets points for fooling it. Footage: `mocap/run_reference.mp4` shows the real runs on Rupert's body ("this is what we showed him").

Honesty notes for the script:
- From EXP-010 on: 30% of practice attempts start mid-stride in a real runner's pose. Test runs and the 100 m trials always start from standing. The judge never sees his speed, only his form.
- Each fix started a **fresh training run**: his brain was wiped and he learned again from zero under the new rules. Say it ("we wiped his memory and gave him new rules"), don't imply one continuous Rupert.
- Fixes 1, 2, 4 and 5 included **penalties and rules**, not only rewards. "Rewards and rules" or "rewards and punishments" keeps it true.
- The attempt counter in each run counts that run's attempts. For the whole-story total, add the runs together and call it "total attempts".


## Running order v2 (Tom, 2026-11-12): replaces the 2026-11-08 order below
Tom's brief: "a cinematic masterpiece". Rewards must be properly explained.
1. **Cold open.** Zoom out from Rupert standing on the track, timed to "This is Rupert". He falls over: "he's useless".
2. **Series promise.** A fake 'dream' montage: Rupert dunking a basketball, trading, flirting. Then "but Rupert needs to walk before he can run" (or similar).
3. **First run plays.**
4. **How he learns.** Narration over runs and timelapses, with interjections. The main device: **Teacher Rupert** (teacher costume, glasses, pointing stick) at a whiteboard that drops down. He explains rewards, trial and error and so on. Possibly overlays, graphs and studies too (research what).
5. **The cheats.** Show each cheat, then the fix, with Teacher Rupert where an explanation is needed. Comedy from the commentary and the shots, plus silences for effect.
6. **The judge.** Introduced like Rupert, and why we need him. Then scenes of the judge watching the real runner and scoring Rupert, holding up score cards at Rupert's bad runs.
7. **The finish.** Build-up to Rupert running properly. Then future videos, thank the viewer, ask them to subscribe.

## Running order v1 (Tom, 2026-11-08)
1. **Intro.** Rupert stands at the start line and waves ("This is Rupert"), then a face close-up. Attempt #1: he falls flat on his face. "He's useless." Series promise ("over this series I'll teach him everything from X to Y"). Today: 100 m.
2. **How he learns.** A few early runs with pop-ups (points, attempts, trial and error).
3. **For every cheat, a chapter:** training timelapse (swarm + fast-cut montage, counter spinning), then a cinematic breakthrough, then the reveal that he's cheating, then the fix ("we gave him a new reward"), then the next timelapse.
4. **Frustration beat** after cheat 8 (optional rapid recap).
5. **The judge.** Introduced like Rupert: a close-up push-in on the stern Olympic panel judge at his desk. Then the real runner he studied (a jogger, looped), then score cards.
6. **Final timelapse, then the cinematic 100 m** (EXP-010, step 846M: 15.2 s).
7. **Outro.** His time against an ordinary person and against Bolt, then the EP02 tease.

## Characters
- **Rupert:** 70s kit (`costumes.py`).
- **The judge** (`blender/characters.py`, look "judge"): bald with grey professor tufts, glasses, walrus moustache, stern face, navy suit, red tie. Sits at a "JUDGE" desk beside lane 1 at the finish, and holds up score cards. Tom chose the Olympic panel judge style.
- **The real runner** (look "runner"): darker skin, short black hair, blue kit. Moves with a real CMU jogger's stride (09_01, 3.9 m/s, one stride looped). Show this one as "what the judge studied". Not a sprinter, and no real-person lookalike (Tom agreed: the recordings are joggers; sprinting is for EP02).
- All three share Rupert's skeleton and cartoon head style. The wave and the judge's poses are hand animation (`rupert/poses.py`), not physics.

## The counter
A running total over all training, in story order (`rupert/ep01_story.py`), ending at **about 26 million attempts**. Cheat #1's footage comes from the start of EXP-006, standing in for the stock-setup run that wasn't saved. The cheat clips in `renders/ep01_final/` show the running total (re-rendered in the weekend batch, with no-counter `_clean` versions too). The old per-run `renders/cheat_reel/` clips are superseded.

## Timelapse shots (per chapter, `python -m rupert.make_sequence swarm|montage --chapter <name>`)
- **swarm:** 24 Ruperts practising at once, like real training: practice behaviour (randomness on), and each one restarts at the start line the moment his attempt ends. Three check-ins through the chapter, with cuts between them.
- **montage:** about 16 fast cuts (0.6 s each) between different test attempts across the chapter, switching between three cameras. The last check-in (the cheat itself) is held back for the reveal.

## Finished shots (renders/, not in git)
- `renders/cheat_reel/`: cheats 1 to 8 (see the table above).
- `renders/ep01_shots/` (made by `python -m rupert.make_sequence intro|timelapse|judge`):
  - `intro_1_face_closeup.mp4`: 4.5 s slow push-in on his face at the start line (no counter).
  - `intro_2_first_attempt_collapse.mp4`: attempt #1, real speed. He falls flat on his face.
  - `intro_3_collapse_slowmo_4x.mp4`: the same fall in 4x slow motion, side view.
  - `timelapse_early_training.mp4`: fixed wide shot of the start line. 16 attempts from successive check-ins (EXP-009, 0 to 80M steps) at double speed, with the board and counter climbing from 1 to about 850,000. He goes from collapsing instantly to walking off.
  - `judge_real_runners.mp4`: the real runners the judge studied, fitted onto Rupert's body. Four play in full, then the other 22 flash by while a "REAL RUNNER" counter races to 26.
- Hand-animated set pieces (2026-11-24, `python -m rupert.make_sequence <shot> [--draft]`; quick tests in `renders/ep01_shots/tests/`, full quality in `renders/ep01_shots/final/`, rendered 2026-11-24):
  - `judge_walk`: the judge walks in towards a low long lens (a real person's walk, CMU 35_01), stops, stares at us, takes the whistle off his chest and blows it.
  - `judge_scores`: two-shot, the judge at his desk across the track while Rupert does a real cheat run past in the foreground; head follows him, card up, head shake. Pogo 2, T-rex 1, superhero gallop 0 (`--cards cheat:card,...`).
  - `final100`: Rupert's real 15.2 s run from five cameras: low in front off the start, rail cam, long-lens head-on, 4x slow-motion finish with the race clock stopping at 15.2, and the judge's slow card raise (halfway... pause... 10).
  - `race`: Rupert's real final run against the world-record sprinter in lane 4, at Bolt's real 10 m splits, 41 steps. Freeze at frame 480 (9.58 s): Rupert is at 60.0 m. Two cameras: high wide side, low wide at the line.
  - `zap`: the electrocution gag (lightning, X-ray skeleton flashes, hair on end, soot, smoke, topples over stiff).
  - `preview`: series preview, bike / candlelit date (night) / trading desk.
  - `outro`: wide to close on Rupert past the finish waving, then he falls flat on his face (cut before the fall to end on the wave). `subscribe`: the judge turns his 10 round: SUBSCRIBE.
- Honesty notes:
  - The attempt #1 clips keep the physics going after the attempt officially ended (on his knees), so he finishes falling. The brain and the moves are unchanged.
  - The sprinter is our animation moving at Bolt's published split times. Not a recording of Bolt. Don't show the splits on screen as official.
  - The judge's walk is a real person's walk (CMU motion capture), on the judge's body.
  - The judge clips are real people's movements on Rupert's body, not Rupert. Say so on screen.
