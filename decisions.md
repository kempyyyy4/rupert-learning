# Decisions log

Newest first. Each entry: what we decided, and why.

## 2026-11-07
- **Stabler training after EXP-005.** Scoring v3 worked: at 345M steps Rupert ran with a real stride (100 m in 15.2 s). Then the learning collapsed for good. Learning rate is now 3e-4 (was 1e-3), with a cap on how big each update can be (gradient clipping, 1.0). Slower but safer; the collapse cost us 650M wasted steps.
- **Blender scene started (build step 5).** Claude runs Blender in the background (`E:\Blender\blender.exe`, 5.4 Alpha; switch to the stable version if it crashes), so Tom doesn't link anything; he opens the saved `.blend`. Rupert is "a touch cartoony": head 1.45× bigger, mitten hands, chunky orange shoes and an orange bobble beanie, no face. Limb lengths and joints exactly match the physics body, so it's still honestly the body that learned. The look is bright and eye-catching: blue track, green grass, purple stands, rainbow boards, a distance board every 10 m and a yellow FINISH board. Standard colour (AgX greyed it out).
- **Knee rule (scoring v4).** Tom spotted that Rupert runs on straight legs (measured: left knee bends only 12°, both lock past straight). Added a reward for bending the swinging leg's knee (up to ~90°) and a penalty for locking past straight. If this still looks robotic, the next step is the style judge trained on real human running.
- **Scoring v3 after EXP-004** (stiff-legged bouncing): strides scored by distance each foot travels, relaxed 'tall' range, arm penalty. Next run is full length (1B steps) because a 4090 does it in ~85 min for well under £2.
- **Cloud installs use exact versions** (`requirements-lock.txt`, frozen from Tom's working PC, Python 3.14, installed without re-solving). First RunPod attempt installed MuJoCo 4.0 and a broken Playground when asked for "latest". Cost: a few pence.
- **Moved test runs to RunPod** after Colab's free GPU limit blocked us (Tom's choice over waiting a day). $10 prepaid credit, auto-pay OFF, Community Cloud RTX 4090 (~$0.34/h). Guide: `docs/runpod-guide.md`.
- **Cloud script safety:** runs in the background; refuses machines without a CUDA 13 driver (the physics engine would silently fall back to CPU and burn money); hard time limit; zips results for download in Jupyter; then deletes the pod after a 45-min download window (15 min after a failure). Tom always checks the Pods page is empty.
- **Scoring v2 after EXP-003.** v1 learned to finish 100 m (19.2 s) but as a hunched, tiny-step shuffle. Speed now only counts fully when running tall, the fall line is higher (1.2 m), strides need a real swing (≥0.25 s), and upright checks the pelvis too.
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
- [ ] Final check of Rupert's look before modelling
- [x] GitHub repo name (rupert-learning)
- [ ] RunPod account + $10 credit, auto-pay off (Tom, 2026-11-07, in progress)
- [x] Blender installed on the PC (E:\Blender, 5.4 Alpha)
