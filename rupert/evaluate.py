"""Runs test attempts for checkpoints and records per-attempt stats + Blender replay data.

  python -m rupert.evaluate --run outputs/run1                 (every checkpoint)
  python -m rupert.evaluate --run outputs/run1 --every 5       (every 5th checkpoint)
  python -m rupert.evaluate --run outputs/run1 --attempts 16 --seconds 25

For each checkpoint it writes into <run>/eval/:
  step_XXXX.json   one row per attempt: distance, top speed, time upright, fell?, 100m time,
                   steps taken, plus "firsts" flags the moment-finder can use
  step_XXXX.npz    replay of every attempt: each body part's position + rotation per frame
                   (positions in metres, rotations as quaternions w,x,y,z; 50 frames a second)
It also writes <run>/eval/summary.json: one line per checkpoint, best and average.
"""
import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from rupert import env as rupert_env
from rupert import networks

TRACK_LENGTH_M = 100.0
TOP_SPEED_WINDOW_S = 0.5   # top speed = best average speed over half a second (ignores twitches)


def make_rollout(env, n_attempts, n_steps, run_dir):
    make_policy = networks.make_policy_fn(env, run_dir)

    @jax.jit
    def rollout(brain, rng):
        policy = make_policy(brain, deterministic=True)
        keys = jax.random.split(rng, n_attempts)
        state = jax.vmap(env.reset)(keys)

        def one_step(carry, _):
            state, alive = carry
            action, _ = jax.vmap(policy)(state.obs, keys)
            nxt = jax.vmap(env.step)(state, action)
            # freeze attempts that have already ended so the replay shows them lying still
            def keep(a, b):
                if a.ndim == 0 or a.shape[0] != n_attempts:
                    return a
                return jp.where(alive.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
            nxt = jax.tree.map(keep, nxt, state)
            alive = alive & (nxt.done == 0)
            d = nxt.data
            frame = dict(
                xpos=d.xpos, xquat=d.xquat, com_x=d.subtree_com[:, 1, 0], com_y=d.subtree_com[:, 1, 1],
                vx=jax.vmap(env._com_vel)(d)[:, 0],
                head_z=d.xpos[:, env._head_id, 2],
                feet=jax.vmap(env._feet_on_ground)(d), alive=alive)
            return (nxt, alive), frame

        _, frames = jax.lax.scan(one_step, (state, jp.ones(n_attempts, bool)), None, length=n_steps)
        return frames

    return rollout


def attempt_stats(f, i, dt):
    alive = f["alive"][:, i]
    n_alive = int(alive.sum())
    x = f["com_x"][:, i]
    vx = f["vx"][:, i]
    w = max(1, int(round(TOP_SPEED_WINDOW_S / dt)))
    smooth = np.convolve(vx[:max(n_alive, 1)], np.ones(w) / w, mode="valid") if n_alive >= w else vx[:1]
    finish = np.nonzero(x >= TRACK_LENGTH_M)[0]
    t100 = None
    if len(finish):
        k = finish[0]
        x0 = x[k - 1] if k > 0 else 0.0
        t100 = round(float((k + (TRACK_LENGTH_M - x0) / max(x[k] - x0, 1e-6)) * dt), 3)
    feet = f["feet"][:n_alive, i]          # (steps, 2)
    touchdowns = int(((feet[1:] & ~feet[:-1]).sum())) if n_alive > 1 else 0
    both_air = float((~feet.any(axis=1)).mean()) if n_alive else 0.0
    return {
        "attempt": i,
        "time_upright_s": round(n_alive * dt, 2),
        "fell": bool(n_alive < len(alive)),
        "distance_m": round(float(x[n_alive - 1] if n_alive else x[0]), 2),
        "top_speed_ms": round(float(smooth.max()), 2),
        "time_100m_s": t100,
        "steps_taken": touchdowns,
        "flight_fraction": round(both_air, 3),   # share of time with both feet off the ground (running has flight)
        "max_lane_drift_m": round(float(np.abs(f["com_y"][:max(n_alive, 1), i]).max()), 2),
        # "firsts" for the moment-finder
        "stood_2s": n_alive * dt >= 2,
        "took_steps": touchdowns >= 4,
        "reached_10m": bool((x >= 10).any()),
        "ran": bool(smooth.max() >= 4.0 and both_air > 0.1),
        "finished_100m": t100 is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="Rupert100m")
    ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--checkpoint", default=None, help="evaluate just this one file")
    args = ap.parse_args()

    run = Path(args.run)
    out = run / "eval"
    out.mkdir(exist_ok=True)
    env = rupert_env.load(args.task, {"episode_length": int(args.seconds / 0.02)})
    n_steps = int(args.seconds / env.dt)
    rollout = make_rollout(env, args.attempts, n_steps, run)
    body_names = [env.mj_model.body(b).name for b in range(env.mj_model.nbody)]

    ckpts = [Path(args.checkpoint)] if args.checkpoint else \
        sorted(p for p in (run / "checkpoints").glob("step_*.pkl"))[:: args.every]
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for ck in ckpts:
        brain = pickle.load(open(ck, "rb"))
        brain = (brain[0], brain[1])
        frames = jax.device_get(rollout(brain, jax.random.PRNGKey(0)))
        f = {k: np.asarray(v) for k, v in frames.items()}
        rows = [attempt_stats(f, i, env.dt) for i in range(args.attempts)]
        (out / f"{ck.stem}.json").write_text(json.dumps(rows, indent=1))
        np.savez_compressed(out / f"{ck.stem}.npz",
                            xpos=f["xpos"].astype(np.float32), xquat=f["xquat"].astype(np.float32),
                            alive=f["alive"], body_names=np.array(body_names), fps=np.array(1 / env.dt))
        best = max(rows, key=lambda r: r["distance_m"])
        summary[ck.stem] = {
            "step": int(ck.stem.split("_")[1]),
            "avg_distance_m": round(float(np.mean([r["distance_m"] for r in rows])), 2),
            "avg_time_upright_s": round(float(np.mean([r["time_upright_s"] for r in rows])), 2),
            "best_attempt": best["attempt"], "best_distance_m": best["distance_m"],
            "best_top_speed_ms": max(r["top_speed_ms"] for r in rows),
            "best_100m_s": min((r["time_100m_s"] for r in rows if r["time_100m_s"]), default=None),
            "firsts": {k: sum(r[k] for r in rows) for k in ("stood_2s", "took_steps", "reached_10m", "ran", "finished_100m")},
        }
        s = summary[ck.stem]
        print(f"{ck.stem}: avg {s['avg_distance_m']:6.1f} m, upright {s['avg_time_upright_s']:5.1f} s, "
              f"top speed {s['best_top_speed_ms']:5.2f} m/s, best 100m {s['best_100m_s']}", flush=True)
        summary_path.write_text(json.dumps(dict(sorted(summary.items())), indent=1))


if __name__ == "__main__":
    main()
