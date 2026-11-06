"""Makes a short mp4 of a trained humanoid from a saved checkpoint.

  python -m rupert.render --run outputs/colab_test --task HumanoidRun
Uses the newest checkpoint in <run>/checkpoints unless --checkpoint is given.
"""
import argparse
import pickle
from pathlib import Path

import jax
import mediapy
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from rupert import env as rupert_env

try:  # on a PC without ffmpeg installed, use the bundled copy
    import imageio_ffmpeg
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/run")
    ap.add_argument("--task", default="Rupert100m")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else sorted((Path(args.run) / "checkpoints").glob("step_*.pkl"))[-1]
    out = Path(args.out) if args.out else Path(args.run) / f"{ckpt.stem}.mp4"
    print(f"Rendering {ckpt.name} -> {out}")
    params = pickle.load(open(ckpt, "rb"))

    env = rupert_env.load(args.task)
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    policy = ppo_networks.make_inference_fn(nets)((params[0], params[1]), deterministic=True)
    policy, reset, step = jax.jit(policy), jax.jit(env.reset), jax.jit(env.step)

    state, rng, frames = reset(jax.random.PRNGKey(0)), jax.random.PRNGKey(1), []
    n_steps = int(args.seconds / env.dt)
    traj = []
    for _ in range(n_steps):
        rng, k = jax.random.split(rng)
        action, _ = policy(state.obs, k)
        state = step(state, action)
        traj.append(state)
        if float(state.done):
            break
    fps = int(round(1 / env.dt))
    frames = env.render(traj[:: max(1, fps // 25)], height=480, width=640, camera="side")
    mediapy.write_video(str(out), frames, fps=25)
    print(f"Saved {out}. Reward over clip: {sum(float(t.reward) for t in traj):.1f}")


if __name__ == "__main__":
    main()
