# Rupert Sim — project rules for Claude

Read this first, every session. Then read `decisions.md` and the latest note in `experiments/`.

## What this project is
A custom simulation app for "watch AI learn" long-form videos on **Rupert's AI Diaries**.
An AI controls a physics body, learns tasks by reinforcement learning (trial and error), and we turn the training into cinematic footage. Tom narrates; Rupert is the trainee on screen.

Full plan: [Rupert's AI Simulation Platform: Plan](https://claude.ai/artifact/Ti4Gd6z9AryamifxsWiZW8)

## Who's who
- **Tom** makes the creative and strategic decisions, narrates, and edits in Premiere Pro.
- **Claude** is the technical co-founder: picks tools, writes and tests all code, explains decisions, keeps these notes up to date.

## How to work with Tom
- Tom is a complete beginner with code and the command line. Every step he runs must be plain, numbered instructions, ideally a double-click `.bat` file. Never assume he can read a traceback.
- Be direct, give opinions, push back when an idea won't work.
- Don't start major development without Tom agreeing the milestone.
- Log every decision in `decisions.md` with the reason. Log every training run in `experiments/`.
- Add new jargon to `glossary.md` in plain English.

## Stack (skeleton built; training and Blender not yet)
- Physics + training: MuJoCo (MJX) with MuJoCo Playground's PPO, run on a rented Linux GPU (RunPod) or Google Colab.
- Rendering + cameras: Blender (Eevee) on Tom's PC, cameras scripted from a shot-list file Claude writes.
- Our app: Python, GitHub repo, `.bat` launchers.
- Tom's PC: GTX 1660 Ti (6GB, no RTX), 16GB RAM, Windows. Too small for big training runs; fine for Blender.

## Folder layout
- `CLAUDE.md` — this file
- `decisions.md` — decisions log
- `ideas.md` — series and episode ideas inbox
- `glossary.md` — plain-English terms
- `experiments/` — one note per training run (use `_template.md`)
- `episodes/` — one note per video
- `rupert/` — the Python app (`check_install.py`, `view_humanoid.py`)
- `setup.bat`, `check_install.bat`, `run_humanoid.bat` — double-click launchers
- `requirements.txt` — Python packages; `.venv/` is the private workspace (not in git)

## Money rules
- First video budget: about £50 max, aim for much less.
- Cloud GPUs bill while running: always shut pods down when a job ends.
- Tom enters any card details himself.
