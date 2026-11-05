# Decisions log

Newest first. Each entry: what we decided, and why.

## 2026-11-05
- **GitHub repo is `kempyyyy4/rupert-learning`.** Notes and code live together; `.obsidian/` is ignored so Tom's viewer settings stay private.
- **App runs in a private Python workspace (`.venv`) on Python 3.14.** Keeps the project's packages separate from the rest of the PC. Python 3.15 also works but is newer and riskier.
- **Humanoid base is Playground's `HumanoidRun`** (standard MuJoCo humanoid, 21 motors, in `dm_control_suite`). The PyPI package is called `playground`.
- **Confirmed: training can't happen on Tom's PC.** JAX has no Windows GPU version, so MJX runs on the CPU at about 320 steps/sec (64 humanoids). Cloud training stays the plan. Local PC is fine for viewing, checking and Blender.
- **Project notes live in `E:\AI Channel\Rupert Sim`, opened as an Obsidian vault.** Claude reads the same markdown files; Obsidian is Tom's view of them. No Obsidian–Claude plugins for now.
- **Rupert is the trainee, not Bot.** Tom and the audience are the overseer and commentator. Rupert has no face: stick-figure style, tall and thin, orange beanie and orange shoes.
- **Rupert's 3D model is built around the standard MuJoCo humanoid**, scaled gently towards his build. The physics body stays fixed across the series so later episodes can build on earlier training.
- **No cheap first episode.** Rupert's model and a proper athletics track are built for Ep 1 and reused in every episode.
- **The series is one long goal with Rupert learning in stages** (working spine: Ep 1 learn to run, Ep 2 vs Usain Bolt's 9.58). Ideas beyond that are deliberately open.
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
- [ ] RunPod account + £10 credit (Tom, when we reach the training step)
- [ ] Blender installed on the PC
