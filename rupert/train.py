"""Trains Rupert with PPO (Brax) and saves checkpoints, dense early and sparser later.

  python -m rupert.train --steps 1000000000 --out outputs/run1     (real run, on a GPU)
  python -m rupert.train --smoke --out outputs/smoke                (tiny code check, any PC)

Training happens in two phases so we capture the funny early stages:
  phase 1 "early":  the first --early-steps, with a checkpoint about every --early-every steps
  phase 2 "main":   the rest, with about --main-checkpoints checkpoints spread evenly
Phase 2 continues from phase 1's brain (only the optimiser's short-term memory restarts).

Style judge (default, scoring v8): a second network that has watched real runners scores how human
Rupert looks (rupert/amp.py, trainer rupert/amp_ppo.py). --no-judge trains with the v7 rules instead.
With the judge, --ref-start of attempts start mid-stride in a real runner's pose (test runs and
the 100 m time trials always start standing).

Outputs in --out:
  config.json            every setting used, plus the real step totals Brax will run
  progress.json          one line per check-in: step, time, score, distance, time upright...
  checkpoints/*.pkl      policy (the brain) at each check-in; the final one also has the value net
"""
import argparse
import functools
import json
import math
import pickle
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground import wrapper

from rupert import amp as rupert_amp
from rupert import amp_ppo
from rupert import env as rupert_env
from rupert import networks


def ppo_config(task):
    """Starting point: Playground's tuned settings for the humanoid tasks."""
    from mujoco_playground.config import dm_control_suite_params
    cfg = dm_control_suite_params.brax_ppo_config("HumanoidRun")
    cfg.episode_length = rupert_env.default_config().episode_length if task == "Rupert100m" else cfg.episode_length
    # Playground's value (10) makes Brax round each check-in up to ~9.8M steps,
    # which is why the first Colab run did 88M steps instead of 20M.
    cfg.num_resets_per_eval = 1
    # Stability for long runs: at Playground's 1e-3 with no cap, run rupert_20261106_2220
    # peaked at 345M steps (15.1 s 100 m) then one bad update wrecked it for good.
    cfg.learning_rate = 3e-4
    cfg.max_grad_norm = 1.0
    return cfg


def steps_per_checkin(cfg):
    return cfg.batch_size * cfg.unroll_length * cfg.num_minibatches * cfg.action_repeat * max(cfg.num_resets_per_eval, 1)


def plan_phase(cfg, steps, n_checkins):
    """Brax rounds up to whole training chunks; pick numbers it won't round up.

    (colab_run3 asked for 100M and got 118M because of this rounding.)
    """
    chunk = steps_per_checkin(cfg)
    chunks_per = max(1, round(steps / max(n_checkins, 1) / chunk))
    n_checkins = max(1, round(steps / (chunks_per * chunk)))
    per = chunks_per * chunk
    return {"num_evals": n_checkins + 1, "num_timesteps": per * n_checkins, "real_steps": per * n_checkins, "steps_per_checkin": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Rupert100m")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--steps", type=int, default=1_000_000_000)
    ap.add_argument("--early-steps", type=int, default=40_000_000)
    ap.add_argument("--early-every", type=int, default=1_000_000)
    ap.add_argument("--main-checkpoints", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="tiny test run, only proves the code works")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-judge", action="store_true", help="old scoring v7, no style judge")
    ap.add_argument("--ref-start", type=float, default=0.3, help="share of practice attempts starting mid-stride")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = ppo_config(args.task)
    if args.smoke:
        cfg.num_envs, cfg.batch_size, cfg.num_minibatches, cfg.num_updates_per_batch = 32, 16, 4, 2
        cfg.unroll_length, cfg.episode_length = 10, 100
        args.steps, args.early_steps, args.early_every, args.main_checkpoints = 20_000, 8_000, 2_000, 2

    early = plan_phase(cfg, min(args.early_steps, args.steps), max(1, args.early_steps // args.early_every))
    phases = [("early", early)]
    if args.steps > early["real_steps"]:
        phases.append(("main", plan_phase(cfg, args.steps - early["real_steps"], args.main_checkpoints)))
    total = sum(p["real_steps"] for _, p in phases)

    print(f"Devices: {jax.devices()}")
    print(f"Task {args.task}: {total:,} training steps in total, {cfg.num_envs} Ruperts at once")
    for name, p in phases:
        print(f"  phase {name}: {p['real_steps']:,} steps, checkpoint every {p['steps_per_checkin']:,}")
    use_judge = args.task == "Rupert100m" and not args.no_judge
    env_overrides = {"episode_length": cfg.episode_length, "style_judge": use_judge} if args.task == "Rupert100m" else None
    train_overrides = dict(env_overrides or {}, ref_start_prob=args.ref_start) if use_judge else env_overrides
    judge = None
    if use_judge:
        judge = rupert_amp.Judge(rupert_env.load(args.task, env_overrides).mj_model)
        print(f"Style judge: {judge.ref.shape[0]:,} real-runner moments to learn from")
    env_cfg = None
    if args.task == "Rupert100m":
        env_cfg = rupert_env.default_config().to_dict()
        env_cfg.update(train_overrides)
    (out / "config.json").write_text(json.dumps({
        "task": args.task, "seed": args.seed, "requested_steps": args.steps, "real_total_steps": total,
        "phases": dict(phases), "ppo": cfg.to_dict(), "network": networks.DEFAULT_SIZES,
        "env": env_cfg,
        "reward_weights": (rupert_env.REWARD_WEIGHTS_JUDGE if use_judge else rupert_env.REWARD_WEIGHTS)
        if args.task == "Rupert100m" else None,
        "judge": {k: getattr(judge, k) for k in ("style_weight", "task_weight", "learning_rate", "grad_penalty",
                                                 "batch", "updates_per_step")} if judge else None,
    }, indent=1, default=str))
    log, start, offset, restore = [], time.time(), 0, None
    final_params = {}

    def progress(step, metrics, phase):
        if phase == "main" and step == 0:
            return   # same moment as the end of phase 1
        row = {"step": int(step) + offset, "phase": phase, "seconds": round(time.time() - start),
               "score": float(metrics.get("eval/episode_reward", float("nan")))}
        for k in ("distance_m", "time_upright_s"):
            if f"eval/episode_{k}" in metrics:
                row[k] = round(float(metrics[f"eval/episode_{k}"]), 2)
        log.append(row)
        for k in ("real", "rupert", "style_reward"):
            if f"training/judge/{k}" in metrics:
                row[f"judge_{k}"] = round(float(metrics[f"training/judge/{k}"]), 3)
        extra = "".join(f"  {k} {row[k]:7.2f}" for k in ("distance_m", "time_upright_s") if k in row)
        if "judge_style_reward" in row:
            extra += f"  style {row['judge_style_reward']:.2f} (judge: real {row['judge_real']:+.2f}, Rupert {row['judge_rupert']:+.2f})"
        print(f"  step {row['step']:>13,}  score {row['score']:9.2f}{extra}  ({row['seconds']}s)", flush=True)
        (out / "progress.json").write_text(json.dumps(log, indent=1))

    def save_checkpoint(step, make_policy, params, phase):
        if phase == "main" and step == 0:
            return
        final_params["p"] = params
        brain = (params[0], params[1])   # normaliser + policy: all we need to replay or render
        with open(out / "checkpoints" / f"step_{int(step) + offset:013d}.pkl", "wb") as f:
            pickle.dump(brain, f)

    judge_params = None
    for name, p in phases:
        train_cfg = dict(cfg)
        train_cfg.pop("network_factory", None)
        train_cfg.update(num_timesteps=p["num_timesteps"], num_evals=p["num_evals"])
        trainer, extra = (amp_ppo.train, {"judge": judge, "restore_judge": judge_params}) if judge else (ppo.train, {})
        _, _, end_metrics = trainer(
            environment=rupert_env.load(args.task, train_overrides),
            eval_env=rupert_env.load(args.task, env_overrides),
            wrap_env_fn=wrapper.wrap_for_brax_training,
            network_factory=networks.factory(),
            progress_fn=functools.partial(progress, phase=name),
            policy_params_fn=functools.partial(save_checkpoint, phase=name),
            restore_params=restore,
            seed=args.seed,
            **extra,
            **train_cfg,
        )
        restore = final_params["p"]
        offset += p["real_steps"]
        if judge:
            judge_params = end_metrics["judge_params"]
            with open(out / "checkpoints" / "judge.pkl", "wb") as f:
                pickle.dump(jax.device_get(judge_params), f)   # the judge's brain, for continuing or inspecting

    with open(out / "checkpoints" / "final_full.pkl", "wb") as f:
        pickle.dump(final_params["p"], f)   # includes the value net, so training can be resumed
    print(f"Done in {round(time.time() - start)}s. {total:,} steps. Checkpoints in {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
