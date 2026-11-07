# Glossary

Plain-English meanings. Claude adds to this whenever a new term comes up.

**Reinforcement learning (RL)** — Learning by trial and error. The AI tries something, gets points (reward) or loses points (penalty), and slowly does more of what scores well.

**Agent** — The AI doing the learning. Here, Rupert's "brain".

**Environment** — The world the agent lives in: the track, gravity, the floor.

**Reward function** — The scoring rules. "Points for moving forward, points off for falling." The AI does exactly what this rewards, which is why it sometimes finds weird shortcuts.

**Policy** — The agent's brain: a neural network that looks at the body's position and decides how to move each joint.

**Neural network** — A big adjustable maths function. Training nudges its numbers until it makes good decisions.

**PPO** — The training method we'll use (Proximal Policy Optimisation). The standard, reliable choice for teaching bodies to move.

**Episode / attempt** — One go, from starting position until Rupert falls or time runs out.

**Checkpoint** — A saved copy of the brain at a point in training. We replay checkpoints to film each stage.

**Parallel environments** — Thousands of copies of Rupert training at once on one GPU. This is how millions of attempts happen in hours.

**MuJoCo** — The free physics engine that simulates the body, gravity and collisions.

**MJX** — The GPU version of MuJoCo, which runs the thousands of copies.

**Blender** — Free 3D software. We use it to put Rupert's look on the physics body and film it.

**GPU** — The graphics card. Good at doing lots of maths at once, which is what training needs.

**Cloud GPU / RunPod** — Renting someone else's powerful GPU by the hour.

**Gait** — The style of a walk or run.

**Repo (repository)** — The project folder tracked by Git and stored on GitHub, with a full history of changes.

**Commit** — A saved snapshot of the project at a moment in time, with a short note about what changed.

**Virtual environment (.venv)** — A private Python workspace inside the project, so its packages don't mix with anything else on the PC.

**JAX** — The maths library that lets physics and training run in huge batches on a GPU. On Windows it only runs on the CPU, which is too slow for training.

**MJX** — MuJoCo built on JAX, so thousands of Ruperts can be simulated at once on a GPU.

**MuJoCo Playground** — A ready-made set of robot tasks and training code built on MJX. Our starting point.

**Checkpoint** — A saved copy of Rupert's brain at one moment in training. We replay old ones to show how he improved.

**Reward / score** — The points Rupert gets for what he does. He learns whatever earns the most points, so the scoring rules decide what he learns.

**Termination** — Ending an attempt early, here when Rupert falls over.

**Force–velocity limit** — Real muscles are weaker the faster they're already moving. We copy that so Rupert can't swing his legs impossibly fast.

**Replay file (.npz)** — The position and rotation of every body part on every frame of an attempt. Blender uses it to animate Rupert's 3D model.

**Style judge (AMP)** — An optional extra AI that scores how human a movement looks, trained on real running footage. Not used for now.

- **Learning rate**: how big a change the AI makes to its brain after each batch of practice. Too big and it can 'forget' everything in one bad update (what happened in EXP-005); too small and learning is slow.
- **Training collapse**: when the score suddenly crashes and never comes back, because one bad update broke the brain. Different from reward hacking, where the score goes *up* for the wrong behaviour.
- **Gradient clipping**: a safety cap on how much one update can change the brain.
- **HUD / overlay**: graphics laid over the video, like the attempt counter in the corner.
- **Metaballs**: blobby 3D shapes that melt into each other when close. We use them to make Rupert's smooth skin, clothes and hair.
- **Costume**: Rupert's outfit for one video. Each video gets its own (`blender/costumes.py`).
