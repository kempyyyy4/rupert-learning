"""Makes a short mp4 of Rupert from a saved checkpoint (plain MuJoCo look, for checking only).

  python -m rupert.render --run outputs/run1                          (newest checkpoint)
  python -m rupert.render --run outputs/run1 --checkpoint <file.pkl> --seconds 10

Keeps only body positions in memory (not the full physics state), so long clips
don't run out of memory on Colab.
"""
import argparse
import pickle
from pathlib import Path

import jax
import mediapy
import mujoco
import numpy as np

from rupert import env as rupert_env
from rupert import networks

try:  # on a PC without ffmpeg installed, use the bundled copy
    import imageio_ffmpeg
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    pass

VIDEO_FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run")
    ap.add_argument("--task", default="Rupert100m")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--camera", default="side")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else sorted((Path(args.run) / "checkpoints").glob("step_*.pkl"))[-1]
    out = Path(args.out) if args.out else Path(args.run) / f"{ckpt.stem}.mp4"
    print(f"Rendering {ckpt.name} -> {out}")
    params = pickle.load(open(ckpt, "rb"))

    env = rupert_env.load_for_run(args.run, args.task)
    policy = networks.make_policy_fn(env, args.run)((params[0], params[1]), deterministic=True)
    policy, reset, step = jax.jit(policy), jax.jit(env.reset), jax.jit(env.step)

    state, rng = reset(jax.random.PRNGKey(0)), jax.random.PRNGKey(1)
    every = max(1, int(round(1 / env.dt / VIDEO_FPS)))
    qpos, total = [np.asarray(state.data.qpos)], 0.0
    for i in range(int(args.seconds / env.dt)):
        rng, k = jax.random.split(rng)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        total += float(state.reward)
        if i % every == 0:
            qpos.append(np.asarray(state.data.qpos))
        if float(state.done):
            break

    m = env.mj_model
    d = mujoco.MjData(m)
    frames = []
    with mujoco.Renderer(m, height=480, width=640) as r:
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=args.camera)
            frames.append(r.render())
    mediapy.write_video(str(out), frames, fps=VIDEO_FPS)
    print(f"Saved {out}. Score over clip: {total:.1f}")


if __name__ == "__main__":
    main()
