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


def first_attempt_full_fall(run_dir, seconds=2.2, step=0, seed=0):
    """A test attempt (deterministic brain, like the test runs) with the physics carrying on after
    the attempt ends, so he finishes falling instead of freezing mid-fall like the test replays do.
    Default: attempt #1, the untrained brain."""
    import pickle
    import jax
    from rupert import env as rupert_env, networks
    env = rupert_env.load_for_run(run_dir, "Rupert100m", {"episode_length": 1000})
    brain = pickle.load(open(Path(run_dir) / "checkpoints" / f"step_{step:013d}.pkl", "rb"))
    policy = networks.make_policy_fn(env, run_dir)(brain, deterministic=True)
    key = jax.random.PRNGKey(seed)
    step, act = jax.jit(env.step), jax.jit(lambda o: policy(o, key)[0])
    s = jax.jit(env.reset)(key)
    xp, xq = [np.asarray(s.data.xpos)], [np.asarray(s.data.xquat)]
    for _ in range(int(seconds * FPS) - 1):
        s = step(s, act(s.obs))
        xp.append(np.asarray(s.data.xpos)); xq.append(np.asarray(s.data.xquat))
    m = env.mj_model
    return np.array(xp), np.array(xq), [m.body(i).name for i in range(m.nbody)]


LANE_W = 1.22


_PRACTICE = {}


def _practice_fn(run_dir, n, frames, episode_length, linger=0):
    """Compiled once per (run, size, length); the brain is passed in, so check-ins reuse it.
    linger: frames the physics keeps going after an attempt ends (so a fall lands) before the reset."""
    key = (str(run_dir), n, frames, episode_length, linger)
    if key in _PRACTICE:
        return _PRACTICE[key]
    import jax
    import jax.numpy as jp
    from rupert import env as rupert_env, networks
    env = rupert_env.load_for_run(run_dir, "Rupert100m", {"episode_length": episode_length})
    make = networks.make_policy_fn(env, run_dir)

    def pick(done, a, b):
        if a.ndim == 0 or a.shape[0] != n:
            return b
        return jp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)

    @jax.jit
    def go(params, key):
        policy = make(params, deterministic=False)
        k0, key = jax.random.split(key)
        state = jax.vmap(env.reset)(jax.random.split(k0, n))

        def one(carry, _):
            state, timer, key = carry
            key, ka, kr = jax.random.split(key, 3)
            act, _ = jax.vmap(policy)(state.obs, jax.random.split(ka, n))
            nxt = jax.vmap(env.step)(state, act)
            ended = (nxt.done > 0) & (timer == 0)          # the attempt just ended
            timer = jp.where(ended, linger + 1, timer)     # ... let him finish falling first
            timer = jp.maximum(timer - 1, 0)
            done = ended & (linger == 0) | (timer == 1)
            nxt = nxt.replace(done=jp.zeros_like(nxt.done))  # no early stop while he lands
            fresh = jax.vmap(env.reset)(jax.random.split(kr, n))
            frame = dict(xpos=nxt.data.xpos, xquat=nxt.data.xquat, reset=done)
            timer = jp.where(done, 0, timer)
            return (jax.tree.map(lambda a, b: pick(done, a, b), fresh, nxt), timer, key), frame

        _, out = jax.lax.scan(one, (state, jp.zeros(n, jp.int32), key), None, length=frames)
        return out

    names = [env.mj_model.body(i).name for i in range(env.mj_model.nbody)]
    _PRACTICE[key] = (go, names)
    return _PRACTICE[key]


def practice(run_dir, step, n=24, seconds=4.0, seed=0, episode_length=100000, linger=0):
    """Practice attempts exactly like training: n Ruperts at once with the brain from this check-in,
    using his practice behaviour (trial-and-error randomness on), and each one restarts at the start
    line the moment his attempt ends. Returns xpos, xquat (frames, n, bodies, ...), resets
    (frames, n: True on the last frame of an attempt), body names, and how many attempts ended."""
    import pickle
    import jax
    go, names = _practice_fn(run_dir, n, int(seconds * FPS), episode_length, linger)
    brain = pickle.load(open(Path(run_dir) / "checkpoints" / f"step_{step:013d}.pkl", "rb"))
    f = jax.device_get(go((brain[0], brain[1]), jax.random.PRNGKey(seed)))
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
        enc = ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
        run(cmd + enc + [out])
        if hud is not None:   # the same shot without the on-screen counter (the stadium board stays)
            run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", FPS,
                 "-start_number", f0, "-i", tmp / "f_%04d.png"] + enc + [clean_path(out)])
    print(f"Saved {out}", flush=True)


def clean_path(out):
    out = Path(out)
    return out.with_name(out.stem + "_clean" + out.suffix)


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


def forgot(scale):
    """EXP-010 at 463M, the check-in where he briefly forgot how to run: a test attempt played out
    until he is flat on the track (the test replay freezes mid-fall), from the side and Olympic angles."""
    from rupert import ep01_story as story
    run_dir, step = ROOT / "outputs" / "rupert_20261112_1312", 463011840
    best = None
    for seed in range(6):              # the attempt that gets furthest before going down: the biggest fall
        xp, xq, names = first_attempt_full_fall(run_dir, seconds=8.0, step=step, seed=seed)
        head = xp[:, names.index("head"), 2]
        down = int(np.argmax(head < 0.5)) if (head < 0.5).any() else None
        if down and (best is None or down > best[0]):
            best = (down, xp, xq, names)
    down, xp, xq, names = best
    n = min(len(xp), down + int(1.8 * FPS))           # cut 1.8 s after his head hits the deck
    still = (np.repeat(xp[:1], 25, 0), np.repeat(xq[:1], 25, 0))
    npz = OUT / "forgot_463M.npz"
    save_sequence(npz, [still, (xp[:n], xq[:n])], names)
    blend = OUT / "forgot_463M.blend"
    count = story.counter("final_run", step)
    build(npz, blend, count, f"ATTEMPT {count:,}")
    for cam, tag in (("cam_side", "side"), ("cam_olympic", "olympic"), ("cam_three_quarter", "three_quarter")):
        render(blend, cam, 1, 25 + n, OUT / f"bt5_463M_forgot_full_fall_{tag}.mp4", hud=count, scale=scale)


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


# Side-on timelapse cameras, made at render time so old scenes can use them too:
# (x it centres on, distance from Rupert's lane, height, lens mm, pan). Tilted so his lane sits
# two-thirds of the way down the frame. pan = share of his forward movement the camera follows.
TL_CAMS = {
    "side_close": (3.0, 10.0, 2.5, 28, 0.0),     # start line, about 13 m of track
    "side_mid": (12.0, 28.0, 5.0, 40, 0.0),      # start to 25 m
    "side_wide": (27.0, 60.0, 12.0, 35, 0.0),    # start to 60 m
    "side_pan": (4.0, 18.0, 3.0, 50, 0.85),      # drifts after him (loses him past ~60 m)
    "side_follow": (2.0, 18.0, 3.0, 50, 1.0),    # keeps pace: him just left of centre, the signs fly past
}
DRAFT = ("; s.render.resolution_x=640; s.render.resolution_y=360; s.render.resolution_percentage=100; "
         "s.eevee.taa_render_samples=4; s.eevee.use_raytracing=False")


def _cam_expr(name):
    """Python (for Blender --python-expr, with s = the scene) that makes `name` the camera."""
    if name not in TL_CAMS:
        return f"s.camera=s.objects['{name}']"
    x, d, h, lens, pan = TL_CAMS[name]
    code = [
        "import math",
        f"c=bpy.data.objects.new('tl_cam',bpy.data.cameras.new('tl_cam')); c.data.lens={lens}",
        f"c.location=({x},{-d},{h}); s.collection.objects.link(c); s.camera=c",
        f"down=math.atan(({h}-0.6)/{d})-math.atan(0.3*0.5*36*9/16/{lens})",
        "c.rotation_euler=(math.pi/2-down,0,0)",
    ]
    if pan:
        code += ["k=c.constraints.new('COPY_LOCATION'); k.target=s.objects['rupert_torso']",
                 f"k.use_y=k.use_z=False; k.use_offset=True; k.influence={pan}"]
    return "\n".join(code)


def ghost(name, scale=100, seconds=10.0, out_fps=25, window=40, sub=10, segments=10,
          camera="cam_timelapse", tag="", draft=False, reuse=False):
    """A proper timelapse: ONE continuous locked-off shot. One Rupert practises exactly like in
    training (randomness on, restarting each time he falls), his brain upgraded through the
    chapter's check-ins. Each finished frame covers `window` physics frames and is the average of
    `sub` moments in it: the newest moment solid, the older ones fading behind it as a ghost
    trail (each moment is cut out against an empty-stadium plate rendered at the same instant,
    so the sun and shadows match). The counter races through the chapter's real attempt totals."""
    from rupert import ep01_story as story
    stem = f"tl_{name}{tag}"
    npz = OUT / f"{stem}_ghost.npz"
    blend = npz.with_suffix(".blend")
    if reuse and blend.exists():
        # same practice footage as last time: skip the physics and the scene build
        T = len(np.load(npz)["xpos"])
        K = T // window
        board = json.loads(npz.with_suffix(".board.json").read_text())
        numbers, v = {}, board[0][1]
        for k in range(K):
            v = max([b for f, b in board if f <= 1 + k * window] or [v])
            numbers[k] = v
    else:
        T, K, numbers = _ghost_physics(story, name, npz, blend, seconds, out_fps, window, segments)
    cam = "" if camera == "cam_timelapse" else "_" + camera
    _ghost_render(blend, T, K, numbers, window, sub, camera, scale, out_fps, draft,
                  OUT / f"{stem}_ghost{cam}{'_draft' if draft else ''}.mp4")


def _ghost_physics(story, name, npz, blend, seconds, out_fps, window, segments):
    _, (_, run_name, *_r) = story.chapter(name)
    run_dir = ROOT / "outputs" / run_name
    steps = [st for st in story.eval_steps(name)
             if (run_dir / "checkpoints" / f"step_{st:013d}.pkl").exists()]
    if len(steps) > 2:
        steps = steps[:-1]                                # the cheat itself is saved for the reveal
    picks = [steps[int(round(i * (len(steps) - 1) / max(1, segments - 1)))] for i in range(segments)]
    K = int(seconds * out_fps) // segments * segments   # equal segments: the physics compiles once
    seg_frames = [K // segments] * segments
    xs, qs, rs, names = [], [], [], None
    for i, (st, kf) in enumerate(zip(picks, seg_frames)):
        xp, xq, r, names, ended = practice(run_dir, st, n=1, seconds=kf * window / FPS, seed=i,
                                           episode_length=1000,   # 20 s, as in training
                                           linger=int(1.2 * FPS))  # he lands before the restart
        r = r.copy(); r[-1, :] = True                    # a new brain: he restarts at the line
        xs.append(xp); qs.append(xq); rs.append(r)
        print(f"  check-in {st:,}: {ended} attempts in {kf} frames", flush=True)
    xp, xq, rs = np.concatenate(xs), np.concatenate(qs), np.concatenate(rs)
    T = len(xp)
    # counter: real totals at each check-in, racing smoothly between them
    ends = [story.counter(name, st) for st in picks] + [story.counter(name, story.eval_steps(name)[-1])]
    numbers, k = {}, 0
    for i, kf in enumerate(seg_frames):
        for j in range(kf):
            numbers[k] = int(ends[i] + (ends[i + 1] - ends[i]) * j / kf); k += 1
    np.savez(npz, xpos=xp, xquat=xq, alive=np.ones(rs.shape, bool), resets=rs, body_names=np.array(names),
             fps=np.array(FPS))
    board = [[1 + k * window, v] for k, v in sorted(numbers.items()) if numbers.get(k - 1) != v]
    build(npz, blend, ends[0], "TRAINING", board_frames=board, extra=["--timelapse"])
    return T, K, numbers


def _ghost_render(blend, T, K, numbers, window, sub, camera, scale, out_fps, draft, out):
    from PIL import Image
    step_sub = window // sub
    moving = TL_CAMS.get(camera, (0,) * 5)[4] > 0      # a panning camera needs a plate per moment
    w, h = (640, 360) if draft else (1920 * scale // 100, 1080 * scale // 100)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:     # beside the clip: full-size frames are big
        tmp = Path(tmp)
        settings = (f"import bpy; s=bpy.context.scene; s.render.resolution_percentage={scale}"
                    + (DRAFT if draft else "") + "\n" + _cam_expr(camera))
        run([BLENDER, "-b", blend, "--python-expr", settings, "-o", tmp / "f_#####",
             "-s", 1, "-e", T, "-j", step_sub, "-a"])
        run([BLENDER, "-b", blend, "--python-expr", settings + "\nbpy.data.collections['rupert'].hide_render=True",
             "-o", tmp / "b_#####", "-s", 1, "-e", T, "-j", step_sub if moving else window, "-a"])  # empty stadium
        load = lambda f: np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0
        cache = {}
        for k in range(K):
            frames = [1 + k * window + j * step_sub for j in range(sub)]
            plate = load(tmp / f"b_{(frames[-1] if moving else frames[0]):05d}.png")
            img = plate.copy()
            for j, f in enumerate(frames):                       # oldest first, newest on top
                fr = load(tmp / f"f_{f:05d}.png")
                own = load(tmp / f"b_{f:05d}.png") if moving else plate
                mask = np.clip((np.abs(fr - own).max(2, keepdims=True) - 0.03) * 12.0, 0.0, 1.0)
                a = (0.12 + 0.88 * (j / max(1, sub - 1)) ** 2) * mask
                img = img * (1 - a) + fr * a
            Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(tmp / f"g_{k:04d}.png")
            n = numbers[k]
            if n not in cache:
                cache[n] = make_hud(tmp / f"hud_n{len(cache)}.png", w, h, n, "ATTEMPTS")
            shutil.copyfile(cache[n], tmp / f"h_{k:04d}.png")
        ff, enc = imageio_ffmpeg.get_ffmpeg_exe(), ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
        base = [ff, "-y", "-loglevel", "error", "-framerate", out_fps, "-i", tmp / "g_%04d.png"]
        run(base + ["-framerate", out_fps, "-i", tmp / "h_%04d.png", "-filter_complex", "overlay"] + enc + [out])
        run(base + enc + [clean_path(out)])
    print(f"Saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot", choices=["intro", "timelapse", "judge", "swarm", "montage", "ghost", "forgot"])
    ap.add_argument("--seconds", type=float, default=10.0, help="ghost: length of the shot")
    ap.add_argument("--chapter", default="cheat1_stands_still", help="for swarm/montage: see rupert/ep01_story.py")
    ap.add_argument("--scale", type=int, default=100, help="resolution percent of 1920x1080 (25 = quick test)")
    ap.add_argument("--out-dir", default=None, help="folder for the clips (default renders/ep01_shots)")
    ap.add_argument("--camera", default="cam_timelapse",
                    help="ghost: camera(s), comma separated, e.g. " + ",".join(TL_CAMS))
    ap.add_argument("--draft", action="store_true", help="ghost: quick 640x360 test version (minutes, not hours)")
    ap.add_argument("--reuse", action="store_true", help="ghost: reuse the last practice footage (skip the physics)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = Path(a.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.shot == "ghost":
        for cam in a.camera.split(","):
            ghost(a.chapter, a.scale, seconds=a.seconds, camera=cam, draft=a.draft, reuse=a.reuse)
            a.reuse = True                                  # later cameras reuse the same footage
    elif a.shot in ("swarm", "montage"):
        {"swarm": swarm, "montage": montage}[a.shot](a.chapter, a.scale)
    else:
        {"intro": intro, "timelapse": timelapse, "judge": judge, "forgot": forgot}[a.shot](a.scale)


if __name__ == "__main__":
    sys.exit(main())
