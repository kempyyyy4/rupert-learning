"""Trains the humanoid with PPO (MuJoCo Playground + Brax) and saves checkpoints.

Used by the Colab notebook now and by the rented GPU later.
  python -m rupert.train --task HumanoidRun --minutes-hint 10 --out outputs/run1
  python -m rupert.train --smoke      (tiny 1-minute test to check the code works)
"""
import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="HumanoidRun")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--timesteps", type=int, default=None, help="override total training steps")
    ap.add_argument("--smoke", action="store_true", help="tiny test run, only proves the code works")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = dm_control_suite_params.brax_ppo_config(args.task)
    if args.smoke:
        cfg.num_timesteps, cfg.num_envs, cfg.batch_size = 200_000, 128, 32
        cfg.num_minibatches, cfg.num_updates_per_batch, cfg.num_evals = 4, 2, 2
    if args.timesteps:
        cfg.num_timesteps = args.timesteps

    print(f"Devices: {jax.devices()}")
    print(f"Task {args.task}, {cfg.num_timesteps:,} steps, {cfg.num_envs} parallel humanoids")

    env = registry.load(args.task)
    eval_env = registry.load(args.task)
    params_dict = dict(cfg)
    params_dict.pop("network_factory", None)

    log, start = [], time.time()

    def progress(step, metrics):
        reward = float(metrics["eval/episode_reward"])
        log.append({"step": int(step), "reward": reward, "seconds": round(time.time() - start)})
        print(f"  step {int(step):>12,}  reward {reward:8.2f}  ({log[-1]['seconds']}s)", flush=True)
        (out / "progress.json").write_text(json.dumps(log, indent=1))

    def save_checkpoint(step, make_policy, params):
        with open(out / "checkpoints" / f"step_{int(step):012d}.pkl", "wb") as f:
            pickle.dump(params, f)

    train_fn = functools.partial(
        ppo.train,
        **params_dict,
        network_factory=ppo_networks.make_ppo_networks,
        progress_fn=progress,
        policy_params_fn=save_checkpoint,
        seed=1,
    )
    train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    print(f"Done in {round(time.time() - start)}s. Checkpoints in {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
