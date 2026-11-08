"""Renders EP01's special shots, the ones that aren't a single test attempt:

  python -m rupert.make_sequence intro      face close-up on the start line, then attempt #1: he collapses
  python -m rupert.make_sequence timelapse  early training sped up: attempt after attempt, counter spinning
  python -m rupert.make_sequence judge      the real runners the style judge studied, on Rupert's body

Add --scale 25 for a fast low-quality test. Output: renders/ep01_shots/.
Each shot joins several pieces into one replay file (with hard cuts between them), builds one
Blender scene around it, renders, and lays the attempt counter over the top.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from rupert.attempts import attempts_by_step
from rupert.hud import make_hud
from rupert.make_clip import BLENDER, FPS, ROOT, run

OUT = ROOT / "renders" / "ep01_shots"
EXP009 = ROOT / "outputs" / "rupert_20261108_0008"   # same body family as the ending; full set of early check-ins
CHEAT8_STEP = 698941440
TIMELAPSE_STEPS = [0, 2949120, 5898240, 8847360, 11796480, 14745600, 17694720, 20643840, 23592960,
                   26542080, 29491200, 32440320, 35389440, 38338560, 50135040, 79626240]


def attempt(run_dir, step, i=0, after_fall=1.0, max_s=None):
    """One test attempt: positions and rotations of every body part, until shortly after he falls."""
    z = np.load(Path(run_dir) / "eval" / f"step_{step:013d}.npz")
    alive = z["alive"][:, i]
    n = len(alive) if alive.all() else min(len(alive), int(np.argmin(alive)) + int(after_fall * FPS))
    if max_s:
        n = min(n, int(max_s * FPS))
    return z["xpos"][:n, i], z["xquat"][:n, i], [str(b) for b in z["body_names"]]


def first_attempt_full_fall(run_dir, seconds=2.2):
    """Attempt #1 (the untrained brain), but the physics keeps going after the attempt ends,
    so he finishes falling instead of freezing on his knees like the test replays do."""
    import pickle
    import jax
    from rupert import env as rupert_env, networks
    env = rupert_env.load_for_run(run_dir, "Rupert100m", {"episode_length": 1000})
    brain = pickle.load(open(Path(run_dir) / "checkpoints" / f"step_{0:013d}.pkl", "rb"))
    policy = networks.make_policy_fn(env, run_dir)(brain, deterministic=True)
    key = jax.random.PRNGKey(0)
    step, act = jax.jit(env.step), jax.jit(lambda o: policy(o, key)[0])
    s = jax.jit(env.reset)(key)
    xp, xq = [np.asarray(s.data.xpos)], [np.asarray(s.data.xquat)]
    for _ in range(int(seconds * FPS) - 1):
        s = step(s, act(s.obs))
        xp.append(np.asarray(s.data.xpos)); xq.append(np.asarray(s.data.xquat))
    m = env.mj_model
    return np.array(xp), np.array(xq), [m.body(i).name for i in range(m.nbody)]


LANE_W = 1.22


def practice(run_dir, step, n=24, seconds=4.0, seed=0):
    """Practice attempts exactly like training: n Ruperts at once with the brain from this check-in,
    using his practice behaviour (trial-and-error randomness on), and each one restarts at the start
    line the moment his attempt ends. Returns xpos, xquat (frames, n, bodies, ...), resets
    (frames, n: True on the last frame of an attempt), body names, and how many attempts ended."""
    import pickle
    import jax
    import jax.numpy as jp
    from rupert import env as rupert_env, networks
    env = rupert_env.load_for_run(run_dir, "Rupert100m", {"episode_length": 100000})
    brain = pickle.load(open(Path(run_dir) / "checkpoints" / f"step_{step:013d}.pkl", "rb"))
    policy = networks.make_policy_fn(env, run_dir)((brain[0], brain[1]), deterministic=False)

    def pick(done, a, b):
        if a.ndim == 0 or a.shape[0] != n:
            return b
        return jp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)

    @jax.jit
    def go(key):
        k0, key = jax.random.split(key)
        state = jax.vmap(env.reset)(jax.random.split(k0, n))

        def one(carry, _):
            state, key = carry
            key, ka, kr = jax.random.split(key, 3)
            act, _ = jax.vmap(policy)(state.obs, jax.random.split(ka, n))
            nxt = jax.vmap(env.step)(state, act)
            done = nxt.done > 0
            fresh = jax.vmap(env.reset)(jax.random.split(kr, n))
            frame = dict(xpos=nxt.data.xpos, xquat=nxt.data.xquat, reset=done)
            return (jax.tree.map(lambda a, b: pick(done, a, b), fresh, nxt), key), frame

        _, frames = jax.lax.scan(one, (state, key), None, length=int(seconds * FPS))
        return frames

    f = jax.device_get(go(jax.random.PRNGKey(seed)))
    names = [env.mj_model.body(i).name for i in range(env.mj_model.nbody)]
    return np.asarray(f["xpos"]), np.asarray(f["xquat"]), np.asarray(f["reset"]), names, int(f["reset"].sum())


def swarm_grid(xpos, rows=3, spacing=5.0):
    """Spread the practising Ruperts over the track: one per lane, rows one behind the other."""
    xpos = xpos.copy()
    n = xpos.shape[1]
    for i in range(n):
        lane, row = i % 8, i // 8
        xpos[:, i, :, 0] -= row * spacing
        xpos[:, i, :, 1] += (lane + 1 - 3) * LANE_W     # lanes 1..8, Rupert's lane 3 at y = 0
    return xpos


def save_sequence(path, pieces, names):
    """Join pieces [(xpos, xquat), ...] into one replay file, remembering where the cuts are."""
    xpos = np.concatenate([p[0] for p in pieces])
    xquat = np.concatenate([p[1] for p in pieces])
    starts = np.cumsum([0] + [len(p[0]) for p in pieces])
    np.savez(path, xpos=xpos[:, None], xquat=xquat[:, None], alive=np.ones((len(xpos), 1), bool),
             body_names=np.array(names), fps=np.array(FPS), cuts=starts[1:-1])
    return len(xpos), starts


def build(npz, blend, board, screen, costume="ep01_70s", face_frames=0, board_frames=None, camera_cuts=None,
          extra=()):
    cmd = [BLENDER, "-b", "--python", ROOT / "blender" / "build_scene.py", "--", npz, "--out", blend,
           "--costume", costume, "--screen", screen, "--attempts", board, *extra]
    if camera_cuts:
        cf = blend.with_suffix(".cameras.json")
        cf.write_text(json.dumps(camera_cuts))
        cmd += ["--camera-cuts", cf]
    if face_frames:
        cmd += ["--face-frames", face_frames]
    if board_frames:
        bf = blend.with_suffix(".board.json")
        bf.write_text(json.dumps(board_frames))
        cmd += ["--board", bf]
    run(cmd)


def render(blend, camera, first, last, out, hud=None, label="ATTEMPTS", slowmo=1, scale=100):
    """hud: None (no counter), a number, or {frame: number} for a changing counter."""
    w, h = 1920 * scale // 100, 1080 * scale // 100
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        settings = (f"import bpy; s=bpy.context.scene; s.render.resolution_percentage={scale}; "
                    f"s.camera=s.objects['{camera}']")
        if slowmo > 1:   # stretch time: Blender fills in the in-between poses
            settings += f"; s.render.frame_map_old=1; s.render.frame_map_new={slowmo}; s.frame_end={last * slowmo}"
        f0, f1 = first * slowmo, last * slowmo
        run([BLENDER, "-b", blend, "--python-expr", settings, "-o", tmp / "f_####", "-s", f0, "-e", f1, "-a"])
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", FPS,
               "-start_number", f0, "-i", tmp / "f_%04d.png"]
        if isinstance(hud, dict):
            cache = {}
            for f in range(f0, f1 + 1):
                n = hud[(f - f0) // slowmo + first]
                if n not in cache:
                    cache[n] = make_hud(tmp / f"hud_n{len(cache)}.png", w, h, n, label)
                shutil.copyfile(cache[n], tmp / f"h_{f:04d}.png")
            cmd += ["-framerate", FPS, "-start_number", f0, "-i", tmp / "h_%04d.png", "-filter_complex", "overlay"]
        elif hud is not None:
            cmd += ["-i", make_hud(out.with_suffix(".hud.png"), w, h, hud, label), "-filter_complex", "overlay"]
        run(cmd + ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out])
    print(f"Saved {out}", flush=True)


def intro(scale):
    """Rupert standing still (face close-up), then attempt #1 from his very first check-in: he collapses."""
    xp, xq, names = first_attempt_full_fall(EXP009)
    assert names == attempt(EXP009, 0)[2]
    hold = int(4.5 * FPS)
    still = (np.repeat(xp[:1], hold, 0), np.repeat(xq[:1], hold, 0))
    n, _ = save_sequence(OUT / "intro.npz", [still, (xp, xq)], names)
    blend = OUT / "intro.blend"
    build(OUT / "intro.npz", blend, 1, "ATTEMPT 1", face_frames=hold)
    fall = hold + 1 + int(np.argmax(xp[:, names.index("head"), 2] < 1.2))   # frame his head drops below 1.2 m
    render(blend, "cam_face", 1, hold, OUT / "intro_1_face_closeup.mp4", scale=scale)
    render(blend, "cam_three_quarter", hold - 50, n, OUT / "intro_2_first_attempt_collapse.mp4", hud=1, scale=scale)
    render(blend, "cam_side", hold - 5, min(n, fall + 30), OUT / "intro_3_collapse_slowmo_4x.mp4", hud=1,
           slowmo=4, scale=scale)


def timelapse(scale):
    """Fixed wide shot of the start line. Attempt after attempt from successive check-ins, at double speed,
    while the counter spins up. Each new number rolls in over 0.3 s."""
    pieces, counts, names = [], [], None
    for k, step in enumerate(TIMELAPSE_STEPS):
        xp, xq, names = attempt(EXP009, step, i=k % 8, after_fall=0.6, max_s=3.0)
        pieces.append((xp[::2], xq[::2]))
        counts.append(attempts_by_step(EXP009, step))
    n, starts = save_sequence(OUT / "timelapse.npz", pieces, names)
    numbers, prev, roll = {}, 1, int(0.3 * FPS)
    for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        for f in range(a, b):
            t = min(1.0, (f - a) / roll)
            t = 1 - (1 - t) ** 3   # fast then settling
            numbers[f + 1] = int(round(prev + (counts[k] - prev) * t))
        prev = counts[k]
    blend = OUT / "timelapse.blend"
    changes = [[f, v] for f, v in sorted(numbers.items()) if numbers.get(f - 1) != v]
    build(OUT / "timelapse.npz", blend, 1, "TRAINING", board_frames=changes)
    render(blend, "cam_wide_start", 1, n, OUT / "timelapse_early_training.mp4", hud=numbers, scale=scale)
    print("Counter by piece:", ", ".join(f"{c:,}" for c in counts))


def judge(scale):
    """The 26 real runners (motion capture fitted to Rupert's body). The first four play in full,
    then the rest flash by, so the 'real runner' counter races to 26."""
    import mujoco
    from rupert import env as rupert_env
    m = rupert_env.load().mj_model
    d = mujoco.MjData(m)
    names = [m.body(i).name for i in range(m.nbody)]
    ref_names = attempt(EXP009, 0)[2]
    assert names == ref_names, (names, ref_names)
    z = np.load(ROOT / "mocap" / "run_reference.npz")
    q, starts = z["qpos"].astype(np.float64), z["starts"]
    pieces, numbers = [], {}
    for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if k >= 4:   # quick flashes from mid-stride
            mid = (a + b) // 2
            a, b = mid - 11, mid + 11
        xp, xq = [], []
        for i in range(a, b):
            d.qpos[:] = q[i]
            mujoco.mj_kinematics(m, d)
            xp.append(d.xpos.copy()); xq.append(d.xquat.copy())
        xp = np.array(xp)
        xp[..., 0] += 30.0   # 30 m down the track, away from the attempts board
        pieces.append((xp, np.array(xq)))
    n, st = save_sequence(OUT / "judge.npz", pieces, names)
    for k, (a, b) in enumerate(zip(st[:-1], st[1:])):
        for f in range(a, b):
            numbers[f + 1] = k + 1
    blend = OUT / "judge.blend"
    build(OUT / "judge.npz", blend, attempts_by_step(EXP009, CHEAT8_STEP), "REAL RUNNERS")
    render(blend, "cam_side", 1, n, OUT / "judge_real_runners.mp4", hud=numbers, label="REAL RUNNER", scale=scale)


# ---- training timelapses: one per story chapter (see rupert/ep01_story.py) ---------------
def _spin(starts, counts, end_count, n):
    """Counter per frame (1-based): from counts[k] at the start of piece k, rising steadily to the next."""
    targets = list(counts[1:]) + [end_count]
    numbers = {}
    for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        for f in range(a, b):
            numbers[f + 1] = int(counts[k] + (targets[k] - counts[k]) * (f - a) / max(1, b - a))
    return numbers


def swarm(name, scale=100, stages=3, seconds=2.0, n=24):
    """24 Ruperts practising at once (like training), at a few check-ins through the chapter, with
    hard cuts between them. The counter spins from one check-in's total to the next."""
    from rupert import ep01_story as story
    _, (_, run, *_r) = story.chapter(name)
    run_dir = ROOT / "outputs" / run
    steps = story.eval_steps(name)
    picks = [steps[int(round(k * (len(steps) - 1) * 0.7 / max(1, stages - 1)))] for k in range(stages)]  # stop short of the cheat
    xs, qs, rs = [], [], []
    for k, step in enumerate(picks):
        xp, xq, rs_k, names, ended = practice(run_dir, step, n=n, seconds=seconds, seed=k)
        rs_k = rs_k.copy(); rs_k[-1, :] = True
        xs.append(swarm_grid(xp)); qs.append(xq); rs.append(rs_k)
        print(f"  check-in {step:,}: {ended} attempts ended in {seconds} s", flush=True)
    xp, xq, rs = np.concatenate(xs), np.concatenate(qs), np.concatenate(rs)
    starts = [k * int(seconds * FPS) for k in range(stages + 1)]
    counts = [story.counter(name, s) for s in picks]
    numbers = _spin(starts, counts, story.counter(name, steps[-1]), len(xp))
    npz = OUT / f"tl_{name}_swarm.npz"
    np.savez(npz, xpos=xp, xquat=xq, alive=np.ones(rs.shape, bool), resets=rs, body_names=np.array(names),
             fps=np.array(FPS), cuts=np.array(starts[1:-1]))
    blend = npz.with_suffix(".blend")
    board = [[f, v] for f, v in sorted(numbers.items()) if numbers.get(f - 1) != v]
    build(npz, blend, counts[0], "TRAINING", board_frames=board, extra=["--swarm"])
    render(blend, "cam_swarm", 1, len(xp), OUT / f"tl_{name}_1_swarm.mp4", hud=numbers, scale=scale)


def montage(name, scale=100, pieces=16, seconds=0.6):
    """Fast cuts between many different test attempts through the chapter, close up, switching
    cameras, counter spinning. Falls are cut to show the moment he goes down."""
    import random
    from rupert import ep01_story as story
    _, (_, run, *_r) = story.chapter(name)
    run_dir = ROOT / "outputs" / run
    steps = story.eval_steps(name)
    steps = steps[:-1] if len(steps) > 2 else steps     # the last one is the cheat: save it for the reveal
    rnd = random.Random(1)
    L = int(seconds * FPS)
    out, used, names = [], [], None
    for k in range(pieces):
        step = steps[int(round(k * (len(steps) - 1) / max(1, pieces - 1)))]
        z = np.load(run_dir / "eval" / f"step_{step:013d}.npz")
        names = [str(b) for b in z["body_names"]]
        i = rnd.randrange(z["alive"].shape[1])
        alive = z["alive"][:, i]
        if not alive.all() and int(np.argmin(alive)) > 0:
            fall = int(np.argmin(alive))
            a = max(0, fall - L + 12)
        else:
            a = rnd.randrange(25, max(26, len(alive) - L))
        xp, xq = z["xpos"][a:a + L, i], z["xquat"][a:a + L, i]
        if len(xp) < L:                                   # pad by holding the last pose
            xp = np.concatenate([xp, np.repeat(xp[-1:], L - len(xp), 0)])
            xq = np.concatenate([xq, np.repeat(xq[-1:], L - len(xq), 0)])
        out.append((xp, xq)); used.append(step)
    n, starts = save_sequence(OUT / f"tl_{name}_montage.npz", out, names)
    counts = [story.counter(name, s) for s in used]
    numbers = _spin(starts, counts, story.counter(name, story.eval_steps(name)[-1]), n)
    cams = ["cam_three_quarter", "cam_side", "cam_front_low"]
    cuts = [[int(st) + 1, cams[k % len(cams)]] for k, st in enumerate(starts[:-1])]
    blend = OUT / f"tl_{name}_montage.blend"
    board = [[f, v] for f, v in sorted(numbers.items()) if numbers.get(f - 1) != v]
    build(OUT / f"tl_{name}_montage.npz", blend, counts[0], "TRAINING", board_frames=board, camera_cuts=cuts)
    render(blend, cams[0], 1, n, OUT / f"tl_{name}_2_montage.mp4", hud=numbers, scale=scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot", choices=["intro", "timelapse", "judge", "swarm", "montage"])
    ap.add_argument("--chapter", default="cheat1_stands_still", help="for swarm/montage: see rupert/ep01_story.py")
    ap.add_argument("--scale", type=int, default=100, help="resolution percent of 1920x1080 (25 = quick test)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.shot in ("swarm", "montage"):
        {"swarm": swarm, "montage": montage}[a.shot](a.chapter, a.scale)
    else:
        {"intro": intro, "timelapse": timelapse, "judge": judge}[a.shot](a.scale)


if __name__ == "__main__":
    sys.exit(main())
