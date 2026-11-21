# Decisions log

Newest first. Each entry: what we decided, and why.

## 2026-11-21
- **Teacher Rupert boards 2–4 built** (brain, score graph, boat; quick tests in `renders/ep01_shots/tests/`). The score graph shows every real check-in as a thin line plus a bold "typical score" line, and labels the dips "bad days!". The 463M "forgot" dip is one of 12 similar drops, so singling it out would mislead.
- **Teacher Rupert's whole body helps him point** (Tom: "his arm is a bit buggy... his body should lean"). The arm now stays inside its real joint limits; his waist twists and leans and he turns on his feet to reach, with his hips and feet planted. He stands at -115° (was -100°) so every spot is in reach (worst miss 1.3 cm). His head glances at the board as each point starts, then turns back to the camera.

## 2026-11-20
- **EP01 timelapses use the close side-on camera** (`renders/ep01_final/timelapse_v2/tl_*_ghost_side_close*.mp4`, all 9 chapters, falls land). Tom preferred it over the follow-cam versions.
- **No Olympic-angle timelapses.** It would have been about 9–10 h of overnight rendering, plus the camera would have to keep pace with Rupert instead of trailing like a real TV camera; Tom: "we wont bother".

## 2026-11-14
- **Timelapse from behind the start is out; side-on instead** (Tom: "side on one would be better either wide shot or closer up"). Four side-on test cameras made (`side_close`, `side_mid`, `side_wide`, `side_pan`, in `rupert/make_sequence.py`); Tom picks from quick test clips.
- **Quick test clips (`--draft`) before any long render.** Tom asked for fast rough versions to judge shots before committing hours. Stick figures were considered and rejected: they can't show the camera angle, the stadium or the light, which is what Tom is judging. Instead the real scene renders small (640x360) at low quality: about 0.15 s a frame against several seconds, and it looks nearly the same. `--reuse` skips the physics when only the camera changes.
- **Four-day render while Tom is away (2026-11-15 to 11-19)** (Tom: "render everything that you need"). `renders/ep01_final/marathon.sh`, most important first: the weekend batch; the final 100 m (846M, attempt 4, 15.2 s) from 4 angles; every EXP-010 personal best as a whole run from the Olympic angle, plus the 463M night he forgot; then side-on ghost timelapses for all 9 chapters (close camera first, then panning), so Tom can pick per chapter. EXP-010 added to `ep01_story.py` as chapter `final_run`.
- **2026-11-18 check of the four-day render:** all finished by Mon 11:29, but two faults. (1) A script bug ran only cheat 1 in the weekend batch: the render program swallowed the rest of the cheat list. Fixed (`< /dev/null`), the 28 missing clips re-queued. (2) The `side_pan` timelapse camera follows at 85% of his speed, so on long runs he outruns it (lost in up to 31% of frames). New `side_follow` camera keeps pace; queued for the 7 chapters where he travels (`followup.sh`, task "RupertRenderFollow").
- **Animated shots, round 1 (Tom said yes to starting): the opening and Teacher Rupert**, as quick tests first (`renders/ep01_shots/drafts/`, `python -m rupert.make_sequence opening|teacher --draft`).
  - Opening: one continuous shot. Face close-up from in front, pull back to wide while he gives a big wave (hand above his head), then attempt #1 for real (the untrained brain, physics run on so he lands, as in the old intro shot). Big screen reads RUPERT.
  - Teacher Rupert: new `teacher` costume (light-blue shirt, maroon tie, brown cords, glasses, his own head and hair); a whiteboard drops out of the sky on cables and bounces; he taps the drawings with a pointer stick. The stick tip is aimed by solving his arm every frame for where the tip should be (`poses.teacher`), so it lands on the drawing to within millimetres and moves the short way round.
  - Board drawings are our own, drawn by code in a marker style (`rupert/board_art.py`, Ink Free font). First board: "How Rupert scores" (+ points / attempt over). More boards are just more functions there.
  - New `blender/props.py`: keyframed cameras, the whiteboard and the pointer, from a JSON file (`build_scene.py --props`).
- **Falls must land** (Tom: "far funnier to see him fall rather than just end before he hits the ground"). Test replays freeze the moment an attempt ends, mid-fall. Now: (1) timelapse practice keeps the physics going 1.2 s after an attempt ends before the restart; (2) the 463M "forgot" clip is re-simulated with the same brain (test-run mode) until he's flat on the track, choosing from 6 attempts the one that gets furthest before going down (about 25 m: don't quote a distance on screen). The brain keeps acting while he lands, as it would. Re-rendered as `timelapse_v2/` and `breakthroughs/bt5_463M_forgot_full_fall_*` (`followup.sh`). The 8 cheat clips never fall in shot, so they're unaffected.
- **Long renders run from Windows Task Scheduler (task "RupertRender"), no time limit.** The first weekend launch, started from Claude's session, was killed when the session closed. Task Scheduler keeps it independent; its default 3-day stop is switched off. Restart by hand: double-click `render_marathon.bat` (finished clips are skipped).
- **Timelapse temp frames go beside the clip on E:, not in Windows temp on C:.** A full-size panning timelapse needs 10–20 GB of frames and C: had 29 GB free.
- **Judge shot and Olympic angle approved.** Weekend batch (`render_weekend.bat`, about 9 h): the Olympic angle for all 8 cheats, the two judge shots at full size, and the 24 Night 1 cheat clips again so each has a no-counter version.

## 2026-11-13
- **Tom's review of Night 1 renders.**
  - **Timelapses rejected.** The grid of Ruperts and the quick cuts don't read as a timelapse. New style:
    - one continuous locked-off wide shot (Olympic broadcast angle)
    - the counter racing
    - one ghostly Rupert, his real attempts sped up so much they blur into streaks
    - the sun and shadows sweeping across the stadium
  - **Swarm (many Ruperts) shots dropped.** For "2,048 Ruperts practise at once": a joke shot of a track with 2,048 lanes, from a wide angle. (Tom thought it was 26; it's 2,048, the real number of parallel attempts.)
  - **Cheat clips are good.** Add an Olympic-style high wide angle to each.
  - **Every shot also comes without the on-screen counter**, so Tom can choose. The stadium board stays.
  - **The judge looks good, but the shirt and tie sit in the wrong place.** Fix them.
  - **More emphasis on breakthroughs.** Document Rupert's progress through the final (judged) run better.
  - **More cinematic shots: Tom wants all 14 of Claude's list:**
    1. bullet time
    2. rail cam
    3. foot-strike slow-mo
    4. finish crane
    5. telephoto head-on
    6. photo-finish image
    7. golden hour
    8. night race
    9. shallow focus
    10. judge cutaways
    11. scoreboard reveal
    12. Mexican wave
    13. split screen
    14. shaky reveal zoom
  - **Plus a celebration:** close-up shaky-cam shots as Rupert runs off celebrating (crowd and Rupert), and a gold-medal podium with only him on it. Celebration poses are hand animation.
  - **The judge keeps the suit** (Tom didn't ask for the coach again). Fix his shirt and tie; add a whistle for the walk-in.

## 2026-11-12
- **EXP-010 is EP01's ending.** The style judge worked: he runs like a person (every gait measure is inside the real joggers' range) in 15.2 s. Use step 846M for the cinematic 100 m. No more training for EP01.
- **EP01 running order v2 (Tom).**
  - Cold open: a zoom out on standing Rupert, then the fall.
  - A fake 'dream' montage (basketball dunk, trading, flirting) as the series promise.
  - **Teacher Rupert** at a drop-down whiteboard explains the learning.
  - The cheats with fixes, then the judge scoring Rupert's bad runs with cards, then the real run and the outro.
  - Full details are in `episodes/EP01-learn-to-run.md`.
- **EP01 animation and script calls (Tom).**
  - Rupert smiles and waves in the intro.
  - Hand-animated shots are fine wherever the story needs them (preview montage, Teacher Rupert, the judge). The preview montage is a 'coming in this series' preview, not a dream.
  - Board drawings are in our own style, not screenshots.
  - The judge is mainly an animated character: his score cards don't need to be real scores (so the narration must not call them real). He sits on the sideline in runs after his introduction.
  - The script is a guideline for Tom's own words. Only the facts are fixed. Claude drafts, Tom edits: `episodes/EP01-script.md`.
- **Script draft 2 calls (Tom).**
  - Keep the Nvidia CEO joke (Claude advised swapping it).
  - Say the memory wipes.
  - Leave the mid-stride practice starts out of the narration. Claude had suggested one line; no line may claim he never saw a running pose.
  - The judge becomes a coach-style character with the moustache.
  - The ending gets a race: Rupert's real final run against a world-record runner moving at Bolt's real 2009 split times.
  - **The runner's look (Tom decided, after Claude advised against a likeness):** a Bolt lookalike.
    - A tall Black sprinter in a yellow vest and green shorts, drawn in our cartoon head style. He moves in Bolt's style: tall and upright, with long strides.
    - To keep it a caricature rather than a copy: no name, logo or sponsor on the kit, and not the trademarked lightning-bolt pose.

## 2026-11-08
- **Style judge (AMP) for EP01's ending** (Tom chose B2 over 'copy one runner': better story). Rupert found cheat #8 (skipping gallop, EXP-009), so as agreed we stop adding rules and show him real runners.
  - A second small AI (the judge) learns to tell real runners from Rupert. Rupert earns style points when it can't tell. Method: Adversarial Motion Priors, Peng et al. 2021.
  - **Data:** 26 running/jogging clips (2.6–4.5 m/s) of 3 people from the CMU Motion Capture Database. The licence allows use in commercial products but not reselling the data. Credit CMU in the description (wording in `mocap/README.md`). Raw files stay out of git.
  - **Fitted onto Rupert** by matching each body segment's direction (his proportions differ; position matching left errors of 10–19 cm). The result is within about 4–7° per segment.
  - **Scoring v8:** all the special cheat fixes (run tall, strides, knees, arm swing, hand reach) are switched off. Only the basics remain (move forward, stay up, face the track, stay in lane, don't waste energy), plus style points (weight 1.5).
  - The judge sees body shape and limb movement but **not** forward speed, so it judges form, not pace.
  - **30% of practice attempts start mid-stride** in a real runner's pose ('reference state initialisation', standard for this method). Test runs and the 100 m trials always start standing. Say so honestly in the video.
- **Body v3:** hip sideways range widened to human norms (out 33°, in 14°; was 25° / 5°). Source in `docs/body-sources.md`.
- **Fixed: bookkeeping carried over between attempts.** The trainer's auto-restart only resets the body, so muscle state, foot timers etc. leaked from a fallen attempt into the next. They're now cleared on the first step of each attempt.

## 2026-11-07
- **Scoring v7 after EXP-008 (lopsided gallop, frozen arm)** (Tom chose another reward fix over the style judge, as the last reward-only attempt). (1) Arm swing now uses hand and foot *speeds*, not positions, so a frozen pose scores 0. (2) Stride credit is scaled by how even the last two steps are (shorter ÷ longer), and a step only counts if the landing foot lands at least 0.1 m in front of the other foot. Checked by replaying old brains: EXP-008's limp scores 0.47 evenness, EXP-007 0.85, EXP-005 0.92. Arm swing is ~0 for all of them, since none actually swung. If EXP-009 finds cheat #8, the style judge (AMP) becomes the EP01 ending.
- **Body v2: Rupert gets an ordinary young man's body** (Tom agreed). EXP-007 ran 100 m in 10.0 s at 6+ steps a second, near Bolt, because body v1 was a 'strong trained adult' with instant muscles. A beginner reaching world-record pace would kill the EP02 'chase Bolt' gap and wouldn't be honest. Hip, knee and ankle strengths now come from a lab study of non-athletic men under 30 (Harbo et al. 2012), with different strength each way (e.g. knee straighten 265 / bend 122 N·m, toes down 147 / up 44). Muscles now take 10 ms to switch on and 40 ms to switch off (Zajac/Winters). Muscle speed limits and side/twist strengths are unchanged (still TO VERIFY). Old runs replay with body v1 automatically (`env.load_for_run`). Sources are in `docs/body-sources.md`.
- **Scoring v6: reward for swinging the arms** (Tom chose this over a style judge). EXP-007 clamped an arm to his chest (the 'T-rex scuttle'), because the arms rule only punished hands held far out. New `arm_swing` reward (weight 0.5): + when each hand swings forward with the opposite leg. Frozen arms score about 0 (checked on the EXP-007 replay: -0.04). A style judge (AMP) stays as the fallback if he cheats again.
- **Fixed: environment settings were shared between environments.** Loading one with an override (e.g. a shorter attempt) changed the defaults for every one made after it. Each environment now gets fresh settings.
- **Stabler training after EXP-005.** Scoring v3 worked: at 345M steps Rupert ran with a real stride (100 m in 15.2 s). Then the learning collapsed for good. Learning rate is now 3e-4 (was 1e-3), with a cap on how big each update can be (gradient clipping, 1.0). Slower but safer; the collapse cost us 650M wasted steps.
- **Blender scene started (build step 5).** Claude runs Blender in the background (`E:\Blender\blender.exe`, 5.4 Alpha; switch to the stable version if it crashes), so Tom doesn't link anything; he opens the saved `.blend`. Rupert is "a touch cartoony": head 1.45× bigger, mitten hands, chunky orange shoes and an orange bobble beanie, no face. Limb lengths and joints exactly match the physics body, so it's still honestly the body that learned. The look is bright and eye-catching: blue track, green grass, purple stands, rainbow boards, a distance board every 10 m and a yellow FINISH board. Standard colour (AgX greyed it out).
- **EP01 story spine: Rupert's cheats** (Tom). Every exploit he finds is kept as a story beat: 'he cheats, we give him a reward or rule, he cheats again... then he runs'. Every run's clips and checkpoints are kept, and nothing is deleted. The cheat table lives in `episodes/EP01-learn-to-run.md`, with honesty notes: each fix was a fresh run (memory wiped), and fixes include penalties as well as rewards.
- **Scoring v5 after EXP-006 (knee-running).** The knee rule plus a low fall line let Rupert shuffle along on his knees. Now a knee joint or hand within 0.22 m of the ground counts as a fall, the head fall line is 1.3 m, and the knee bonus only pays while one foot swings and the other is planted. The stable training settings (lr 3e-4, gradient cap) worked: no permanent collapse in EXP-006. Keep EXP-005 (straight-legged 15 s run) and EXP-006 (knee-running) clips as comedy/failure beats for the video.
- **Rupert v7 (Tom's notes on v6).** **No black outlines** (`OUTLINE_M = 0`; the code is kept in case we want them back). **Cheekbones are fuller, higher and wider-set.** He has the **happy profile-picture face** again: big oval eyes, arched brows and an open D smile. The 'determined' face from the drawings is kept as an option. **Hair is layered and flicky, not a helmet**: three layers of locks, each shorter and higher than the one below, with tips flicking out, layered sides, and back locks of alternating lengths. Long locks are routed over the top of the head; before this, some went the short way round, across his face. The headband now only covers the forehead.
- **Rupert v6 (Tom's drawings + Roblox hair reference).** His head is **bigger and boxier**, with a strong square jaw, defined jaw corners and a flat chin. His face is the 'determined' look from the drawings: narrow eyes with pupils, straight brows angled down to the nose, and a straight, slightly smirking mouth. The profile picture's D smile is kept as the 'happy' expression. His **hair is chunky swept-back locks** with pointed tips: a smooth cap underneath, locks sweeping back over the top, the sides swept past the ears, longer at the back with tips flicking out, and two locks framing the face. The headband only covers the front, with its ends tucked under the hair. **Shorts are slimmer and shorter** (they had looked like boxing shorts). **Vest is plain white.** **Stadium:** the coloured barrier boards are gone and an extra front row of seats takes their place; the distance signs and attempts board stand on the grass verge.
- **Rupert v5 (Tom's notes on v4).** His **head is sculpted to the character sheet**: a round dome, wide cheekbones with a visible corner, and straight jawlines down to a narrow chin. It's a separate shape, not a blob, so the angles stay crisp. **Hair is a medium-length wavy 'flow'** (Tom's reference photo): lifted off the forehead, swept back and out over the ears, down to the collar with flicked-out ends. It stays caramel to match the drawings. **Shorts are now one smooth piece of fabric** (waist and legs melted together, then cut open cleanly at the hems), which fixes the ridge and bunching at the hips. Tom approved the skin, vest, socks, shoes and bands. **Stadium:** the 'RUPERT'S AI DIARIES' branding is gone from the barrier boards, which are now plain colours. Big raised distance signs every 10 m stand above the barrier so they read in wide shots. The attempts board sits just behind the start line.
- **Rupert v4 (Tom's notes on v3).** His **head now matches the profile pictures**: a big round head with fuller cheeks, wide-set oval eyes, little brows and the open "D" smile from the watermark, with a black ink outline round him like the drawings. His **skin is warm off-white/cream**, so white kit stands out. His **hair is a 70s swept-back caramel mop**, flowing back from the forehead to the nape. **EP01 kit:** loose orange satin running shorts as separate fabric, with white piping and side splits like Tom's reference photo; a white vest with orange trim at the neck and armholes; white tube socks with two orange stripes; an orange towelling headband (snug on the forehead, tucked under the hair) and wristbands; and orange running shoes.
- **Attempt counters are LED track boards** (amber seven-segment digits, dot-matrix 'ATTEMPTS' label): one in the corner of close shots (`rupert/hud.py`), and a **physical board in the stadium** by the start (`blender/attempts_board.py`) that can change number frame by frame for the sped-up training montages. A fixed wide camera `cam_wide_start` frames the start line and the board.
- **Rupert's character and costumes (scene v3).** Rupert has a face and hair from the channel's character sheet: white skin, dot eyes, a small smile and a caramel-brown lumpy mop. The skin plays the part the morph suit did, so he has no visible skeleton. **He wears a different costume in every video** (`blender/costumes.py`, chosen with `--costume`). EP01 is a 70s track star: orange short shorts, a white vest with two orange stripes, an orange towelling headband and wristbands, and orange running shoes with white soles and stripes. The beanie is now just the "classic" costume.
- **Attempt counter** in the top-left corner (`rupert/hud.py`, added by `rupert/make_clip.py`, also saved as a PNG for Premiere). Training doesn't count attempts, so it's **estimated**: steps taken ÷ how long each attempt lasted at that point (`rupert/attempts.py`). It probably undercounts a bit. Narration should say "over" or "about", never present it as exact. EXP-005: about 45,000 attempts by 1M steps, about 2.4 million when he first ran 15 s.
- **Scene v2 after Tom's feedback** (v1 "looks ass" next to polished rivals). Rupert now wears a **one-piece royal-blue morph suit**: a smooth skin built around the physics limbs that bends at the joints, with no visible skeleton. He has white-soled orange sneakers, a bigger head and the orange bobble beanie, still no face. Traditional bright red track with texture, white lines and lane numbers, mown-stripe grass, branded "RUPERT'S AI DIARIES" boards, a stand with a colourful crowd, a roof and a big screen. Soft sun, ray-traced lighting, motion blur, background blur, and a 3/4 tracking camera. Colour: Khronos PBR Neutral (keeps colours bright without blowing out). This supersedes v1's blue track and capsule mannequin.
- **Knee rule (scoring v4).** Tom spotted that Rupert runs on straight legs (measured: left knee bends only 12°, both lock past straight). Added a reward for bending the swinging leg's knee (up to ~90°) and a penalty for locking past straight. If this still looks robotic, the next step is the style judge trained on real human running.
- **Scoring v3 after EXP-004** (stiff-legged bouncing): strides scored by distance each foot travels, relaxed 'tall' range, arm penalty. Next run is full length (1B steps) because a 4090 does it in ~85 min for well under £2.
- **Cloud installs use exact versions** (`requirements-lock.txt`, frozen from Tom's working PC, Python 3.14, installed without re-solving). First RunPod attempt installed MuJoCo 4.0 and a broken Playground when asked for "latest". Cost: a few pence.
- **Moved test runs to RunPod** after Colab's free GPU limit blocked us (Tom's choice over waiting a day). $10 prepaid credit, auto-pay OFF, Community Cloud RTX 4090 (~$0.34/h). Guide: `docs/runpod-guide.md`.
- **Cloud script safety:** runs in the background; refuses machines without a CUDA 13 driver (the physics engine would silently fall back to CPU and burn money); hard time limit; zips results for download in Jupyter; then deletes the pod after a 45-min download window (15 min after a failure). Tom always checks the Pods page is empty.
- **Scoring v2 after EXP-003.** v1 learned to finish 100 m (19.2 s) but as a leaning-back (first noted as "hunched"), tiny-step shuffle. Speed now only counts fully when running tall, the fall line is higher (1.2 m), strides need a real swing (≥0.25 s), and upright checks the pelvis too.
- **Bigger brain:** policy 256-256-128, value 256×3 (was 4×32). Stored in each run's config.json so old runs still replay.
- **Throughput measured:** T4 ≈ 38,000 steps/s on the Rupert setup.

## 2026-11-06
- **Bolt race deprioritised.** Tom feels the Bolt angle is a bit played out. Focus on EP01 (learn to run); decide EP02's angle after the first video. Fair-start/timing issues vs Bolt are parked.
- **Reviewed the stock setup and rebuilt it before any paid training** (Tom agreed all fixes). The stock score (~138) matched "standing upright, barely moving", not running.
- **Rupert's body:** stock humanoid rescaled to 1.84 m / 70 kg, masses by body part from de Leva (1996), joint strengths in N·m, plus a muscle speed limit (faster joint = weaker). Sources and what is still unverified: `docs/body-sources.md`. Nothing marked TO VERIFY goes on screen.
- **New scoring (`rupert/env.py`):** forward speed down the track (capped at 12.5 m/s, above Bolt's peak), small alive bonus, upright, facing forward, alternating strides; penalties for energy, jitter and leaving the lane. **The attempt ends when he falls.** Weights are first guesses and will need tuning.
- **No "style judge" trained on human running for now.** Try without it first; revisit if the gait looks wrong. Keeps "no pre-made animation" fully true.
- **Physics accuracy raised:** 10 solver iterations (4 hit the limit on ~37% of steps), line search 50, 0.004 s steps, slightly stiffer contacts.
- **Solved the 88M-step mystery:** Playground's `num_resets_per_eval=10` made Brax round each check-in up to ~9.8M steps. Set to 1.
- **Checkpoints in two phases:** dense early (default every 1M steps for the first 40M), then ~100 spread over the rest. Only the brain is saved per checkpoint; the final one is full so training can resume.
- **Per-attempt stats + Blender replay:** `rupert/evaluate.py` saves distance, top speed, time upright, 100 m time, steps, flight time and "firsts" per attempt, plus every body part's position/rotation per frame (`.npz`). `blender/import_replay.py` turns that into animated empties.
- **Cloud runs delete their own pod** when they finish, crash or hit a time limit (`cloud/runpod_train.sh`, untested until first paid run). Results go to the network volume.
- **JAX is pinned to 0.9.2.** The newest JAX (0.12) breaks Brax's PPO trainer. Revisit when Brax is updated.

## 2026-11-05
- **GitHub repo is `kempyyyy4/rupert-learning`.** Notes and code live together; `.obsidian/` is ignored so Tom's viewer settings stay private.
- **App runs in a private Python workspace (`.venv`) on Python 3.14.** Keeps the project's packages separate from the rest of the PC. Python 3.15 also works but is newer and riskier.
- **Humanoid base is Playground's `HumanoidRun`** (standard MuJoCo humanoid, 21 motors, in `dm_control_suite`). The PyPI package is called `playground`.
- **Confirmed: training can't happen on Tom's PC.** JAX has no Windows GPU version, so MJX runs on the CPU at about 320 steps/sec (64 humanoids). Cloud training stays the plan. Local PC is fine for viewing, checking and Blender.
- **Project notes live in `E:\AI Channel\Rupert Sim`, opened as an Obsidian vault.** Claude reads the same markdown files; Obsidian is Tom's view of them. No Obsidian–Claude plugins for now.
- **Rupert is the trainee, not Bot.** Tom and the audience are the overseer and commentator. Rupert has no face: stick-figure style, tall and thin, orange beanie and orange shoes.
- **Rupert's 3D model is built around the standard MuJoCo humanoid**, scaled gently towards his build. The physics body stays fixed across the series so later episodes can build on earlier training.
- **No cheap first episode.** Rupert's model and a proper athletics track are built for Ep 1 and reused in every episode.
- **The series is one long goal with Rupert learning in stages** (working spine: Ep 1 learn to run; Ep 2 open, Bolt race parked 2026-11-06). Ideas beyond that are deliberately open.
- **Long-form first**, because it shows the learning journey.
- **Build our own app on proven free tools** (MuJoCo, Blender, Python), adding features only when an episode needs them.
- **Train in the cloud, render on Tom's PC.** The 1660 Ti can't run Isaac Lab and is too slow for big runs; cloud RTX 4090 time is roughly 25–55p an hour.
- **Claude scripts the cameras.** Shots are written as a shot-list file and rendered in Blender, so Tom doesn't have to operate a camera.
- **Ruled out:** NVIDIA Isaac Lab (needs RTX, 16GB+ VRAM, Linux), Unity ML-Agents (slow CPU training), Unreal (weak RL tooling), Stable-Baselines3 alone (CPU-bound).
- **Budget:** £50 max for the first video, aiming for £5–15. Stay on the £20 Claude plan until limits actually block progress.

## Still to decide
- [ ] Tom signs off the plan and Milestone 1
- [x] Final check of Rupert's look before modelling (face, hair, per-video costumes; 2026-11-07)
- [x] GitHub repo name (rupert-learning)
- [ ] RunPod account + $10 credit, auto-pay off (Tom, 2026-11-07, in progress)
- [x] Blender installed on the PC (E:\Blender, 5.4 Alpha)
