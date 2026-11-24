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


def render(blend, camera, first, last, out, hud=None, label="ATTEMPTS", slowmo=1, scale=100, draft=False):
    """hud: None (no counter), a number, or {frame: number} for a changing counter.
    draft: quick 640x360 low-quality test version."""
    w, h = (640, 360) if draft else (1920 * scale // 100, 1080 * scale // 100)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        settings = (f"import bpy; s=bpy.context.scene; s.render.resolution_percentage={scale}; "
                    f"s.camera=s.objects['{camera}']" + (DRAFT if draft else ""))
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


def _blend_poses(a, b, frames):
    """Ease from pose a (xpos, xquat) to pose b over `frames` frames."""
    (pa, qa), (pb, qb) = a, b
    qb = np.where((qa * qb).sum(-1, keepdims=True) < 0, -qb, qb)
    xs, qs = [], []
    for i in range(1, frames + 1):
        t = i / (frames + 1)
        t = t * t * (3 - 2 * t)
        xs.append(pa + t * (pb - pa))
        q = qa + t * (qb - qa)
        qs.append(q / np.linalg.norm(q, axis=-1, keepdims=True))
    return np.array(xs), np.array(qs)


def opening(scale, draft=False):
    """Cold open, one continuous shot: tight on his face, pull back to the wide stadium while he
    waves at us ("This is Rupert"), then attempt #1 for real: the untrained brain, and he collapses."""
    from rupert import poses
    xw, qw, names = poses.opening()                          # 5.5 s: stand, wave
    xf, qf, names_f = first_attempt_full_fall(EXP009, seconds=3.4)
    assert names == names_f
    xb, qb = _blend_poses((xw[-1], qw[-1]), (xf[0], qf[0]), 20)
    hold = int(1.2 * FPS)
    xp = np.concatenate([xw, xb, xf, np.repeat(xf[-1:], hold, 0)])
    xq = np.concatenate([qw, qb, qf, np.repeat(qf[-1:], hold, 0)])
    npz = OUT / "opening.npz"
    np.savez(npz, xpos=xp[:, None], xquat=xq[:, None], alive=np.ones((len(xp), 1), bool),
             body_names=np.array(names), fps=np.array(FPS))
    hx, hy, hz = (float(v) for v in xw[0, names.index("head")])
    off = lambda dx, dy, dz: [hx + dx, hy + dy, hz + dz]
    push, wide = int(1.4 * FPS), int(4.4 * FPS)
    cam = {"name": "cam_opening", "fstop": 2.8, "keys": [
        [1, off(1.95, -0.45, 0.12), off(0, 0, 0.13), 85],          # in front of him: face to camera
        [push, off(1.85, -0.43, 0.12), off(0, 0, 0.13), 85],       # a slow creep while we hold on the face
        [wide, off(7.6, -2.3, 0.2), off(0.4, 0, -0.6), 32],        # wide: him, the track and the stadium
        [len(xp), off(7.8, -2.36, 0.2), off(0.4, 0, -0.6), 32]]}
    props = OUT / "opening.props.json"
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_opening"}))
    blend = OUT / "opening.blend"
    build(npz, blend, 1, "RUPERT", extra=["--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_opening", 1, len(xp), OUT / f"opening_this_is_rupert{tag}.mp4", hud=1, scale=scale, draft=draft)


def teacher(scale, draft=False, board="points"):
    """Teacher Rupert: a whiteboard drops out of the sky beside him (lands with a bounce), and he
    taps or traces the drawings with his pointer stick. Boards: rupert/board_art.py (BOARDS)."""
    from rupert import board_art, poses
    spec = board_art.BOARDS[board]
    X0 = 20.0                                                 # on the track, 20 m from the start
    W, H, centre = 2.2, 1.4, (X0 - 1.55, 0.5, 1.5)
    secs = spec["seconds"]
    n_frames = int(secs * FPS)
    if spec.get("dynamic"):                                   # moving drawing: one picture per frame
        folder = OUT / f"board_{board}"
        spots = board_art.draw_frames(board, folder, n_frames, FPS)
        img = folder / "f_0001.png"
    else:
        img = OUT / f"board_{board}.png"
        spots = board_art.draw(board, img)
    x0, ztop, y_face = centre[0] - W / 2, centre[2] + H / 2, centre[1] - 0.03
    world = lambda uv: (x0 + uv[0] * W, y_face, ztop - uv[1] * H)
    points = []
    for name, t0, t1 in spec["taps"]:
        spot = spots[name]
        tip = [world(uv) for uv in spot] if isinstance(spot, list) else world(spot)
        points.append((t0, t1, tip))
    xp, xq, names = poses.teacher(at=(X0, 0.0), yaw_deg=-115.0, seconds=secs, points=points,
                                  look_at=(X0 - 1.2, -4.0, 1.5))          # the camera
    for (t0, t1, tip) in points:
        tip = np.atleast_2d(tip)
        f = int((t0 + t1) / 2 * FPS)
        want = poses._along(tip, 0.5)
        err = np.linalg.norm(poses.stick_tip(xp, names, f) - want)
        print(f"  stick tip {err * 100:.1f} cm from its target", flush=True)
    npz = OUT / f"teacher_{board}.npz"
    np.savez(npz, xpos=xp[:, None], xquat=xq[:, None], alive=np.ones((len(xp), 1), bool),
             body_names=np.array(names), fps=np.array(FPS))
    n = len(xp)
    cam = {"name": "cam_teacher", "fstop": 5.6, "keys": [
        [1, [X0 - 1.2, -4.1, 1.5], [X0 - 1.2, 0.2, 1.42], 35],
        [n, [X0 - 1.2, -3.7, 1.5], [X0 - 1.2, 0.2, 1.42], 35]]}
    wb = {"image": str(img.resolve()), "centre": centre, "size": [W, H], "land": int(0.9 * FPS)}
    if spec.get("dynamic"):
        wb["sequence"] = n_frames
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_teacher", "pointer": True, "whiteboard": wb}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, spec["screen"], costume="teacher", extra=["--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_teacher", 1, n, OUT / f"teacher_{board}{tag}.mp4", scale=scale, draft=draft)


def _save(npz, xp, xq, names):
    np.savez(npz, xpos=xp[:, None], xquat=xq[:, None], alive=np.ones((len(xp), 1), bool),
             body_names=np.array(names), fps=np.array(FPS))


def _follow_keys(xp, names, body, every, off, aim_off, lens, ease_aim=None):
    """Camera keys every `every` frames: camera at body + off, aimed at body + aim_off."""
    b = names.index(body)
    keys = []
    for f in list(range(0, len(xp), every)) + [len(xp) - 1]:
        p = xp[f, b]
        a = aim_off(f) if callable(aim_off) else aim_off
        o = off(f) if callable(off) else off
        keys.append([f + 1, [float(p[0] + o[0]), float(p[1] + o[1]), float(p[2] + o[2])],
                     [float(p[0] + a[0]), float(p[1] + a[1]), float(p[2] + a[2])], lens])
    return keys


def judge_walk(scale, draft=False):
    """The judge walks in down the track towards a low long-lens camera, stops, stares into the
    lens and blows his whistle (a real person's walk from CMU, hand animation for the whistle)."""
    from rupert import poses
    xp, xq, names, beats = poses.judge_walk_in()
    npz = OUT / "judge_walk_in.npz"
    _save(npz, xp, xq, names)
    n = len(xp)
    h = names.index("head")
    stop_x = float(xp[beats["stop"] - 1, h, 0])
    cam_x = stop_x - 3.9
    keys = []
    for f in list(range(0, n, 10)) + [n - 1]:
        u = poses._ease(f / beats["stop"])
        hx, hy, hz = (float(v) for v in xp[f, h])
        aim_z = 1.05 + u * (hz - 0.12 - 1.05)            # whole body at first, then his face
        keys.append([f + 1, [cam_x - 0.5 * (1 - u) - 0.25 * f / n, -0.25, 1.38], [hx, hy, aim_z], 85])
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({
        "cameras": [{"name": "cam_walk", "fstop": 2.0, "keys": keys}], "camera": "cam_walk",
        "whistle": {"grab": beats["grab"] + 1, "mouth": beats["blow"] + 1}}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "THE JUDGE", extra=["--character", "judge", "--no-desk", "--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_walk", 1, n, OUT / f"judge_walk_in{tag}.mp4", scale=scale, draft=draft)
    print("beats (frame):", beats)


CHEATS = {   # name: (run, check-in step, seconds into test attempt 0 that the cheat clips use)
    "pogo": ("rupert_20261106_2121", 101253120, 4.0),
    "trex": ("rupert_20261107_1323", 875888640, 4.0),
    "gallop": ("rupert_20261107_2130", 757923840, 4.0),
    "straight_legs": ("rupert_20261106_2220", 345047040, 8.0),
    "knees": ("rupert_20261107_1159", 728432640, 6.0),
}
JUDGE_SIDE_Y = 7.55          # judge's chair on the far verge, in front of the stand, facing the track (-y)


def _cast_file(npz, cast):
    f = npz.with_suffix(".cast.json")
    f.write_text(json.dumps(cast))
    return f


def judge_scores(scale, draft=False, cheat="pogo", card="2", seconds=5.0, pass_at=1.4):
    """A two-shot: the judge at his desk across the track (crowd behind him) while Rupert does his
    real cheat run past the camera in the foreground (out of focus). The judge's head follows him,
    then up goes the card, and a slow head shake. The cards are for fun, not real scores."""
    from rupert import poses
    run, step, start = CHEATS[cheat]
    z = np.load(ROOT / "outputs" / run / "eval" / f"step_{step:013d}.npz")
    names = [str(b) for b in z["body_names"]]
    xp_all, xq_all = z["xpos"][:, 0], z["xquat"][:, 0]
    torso = names.index("torso")
    a0 = int(start * FPS)
    # the desk: between two distance signs, where Rupert is a little after the shot starts
    xj = float(np.floor(xp_all[a0 + int(pass_at * FPS), torso, 0] / 10) * 10 + 5)
    k = int(np.argmax(xp_all[:, torso, 0] >= xj))       # frame he passes the desk
    a = max(0, k - int(pass_at * FPS))
    n = int(seconds * FPS)
    xp, xq = xp_all[a:a + n], xq_all[a:a + n]
    if len(xp) < n:
        xp = np.concatenate([xp, np.repeat(xp[-1:], n - len(xp), 0)]); xq = np.concatenate([xq, np.repeat(xq[-1:], n - len(xq), 0)])
    stem = OUT / f"judge_scores_{cheat}"
    npz = stem.with_suffix(".npz")
    _save(npz, xp, xq, names)
    card_up = pass_at + 1.0
    jp, jq, jn = poses.judge_watching((xj, JUDGE_SIDE_Y), -90.0, xp[:, names.index("head")],
                                      card_up_at=card_up, shake_at=card_up + 0.9)
    jnpz = OUT / f"judge_scores_{cheat}_judge.npz"
    _save(jnpz, jp, jq, jn)
    cast = _cast_file(npz, [{"replay": str(jnpz.resolve()), "character": "judge", "prefix": "judge", "desk": True,
                             "yaw": -90.0, "cards": [{"text": card, "from": int(card_up * FPS) + 1, "face_yaw": -90.0}]}])
    cam = {"name": "cam_two_shot", "fstop": 2.0, "keys": [     # long lens: judge waist-up, Rupert a blur
        [1, [xj - 0.3, -6.5, 1.3], [xj, JUDGE_SIDE_Y - 0.3, 1.22], 135],
        [n, [xj + 0.3, -6.2, 1.3], [xj, JUDGE_SIDE_Y - 0.3, 1.22], 135]]}
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_two_shot"}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "THE JUDGE", extra=["--cast", cast, "--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_two_shot", 1, n, OUT / f"judge_scores_{cheat}_{card}{tag}.mp4", scale=scale, draft=draft)


FINAL = ("rupert_20261112_1312", 846397440, 4)      # EXP-010, the 15.2 s run (test attempt 4)


def final_run(seconds=None):
    run, step, i = FINAL
    z = np.load(ROOT / "outputs" / run / "eval" / f"step_{step:013d}.npz")
    n = len(z["xpos"]) if seconds is None else int(seconds * FPS)
    return z["xpos"][:n, i], z["xquat"][:n, i], [str(b) for b in z["body_names"]]


def race(scale, draft=False, seconds=10.6, cams="race_wide,race_finish"):
    """Rupert's real final run against a world-record sprinter in the next lane, moving at Bolt's
    real Berlin 2009 split times (rupert/sprint.py). Freeze-frame point: when the sprinter
    crosses the line (9.58 s), printed below."""
    from rupert import sprint
    xp, xq, names = final_run(seconds)
    npz = OUT / "race.npz"
    _save(npz, xp, xq, names)
    sp, sq, sn = sprint.sprinter(seconds=seconds)
    snpz = OUT / "race_sprinter.npz"
    _save(snpz, sp, sq, sn)
    n = len(xp)
    pel = names.index("pelvis")
    line = int(round(sprint.SPLITS[-1] * FPS))
    print(f"  sprinter crosses the line at frame {line + 1} (9.58 s): Rupert is at {xp[line, pel, 0]:.1f} m")
    cast = _cast_file(npz, [{"replay": str(snpz.resolve()), "character": "sprinter", "prefix": "sprinter"}])
    mid = lambda f: (xp[f, pel, 0] + sp[f, pel, 0]) / 2
    keys_wide = []
    for f in list(range(0, n, 10)) + [n - 1]:
        gap = sp[f, pel, 0] - xp[f, pel, 0]
        c = float(mid(f))
        dist = 14 + 0.75 * gap                            # pulls back as the gap opens
        keys_wide.append([f + 1, [c - 0.1 * dist, -dist, 2.0 + 0.18 * dist], [c, 0.6, 0.9], 35])
    # low, 12 m past the line, looking back up the track: the sprinter big at the line, Rupert small but
    # clearly there 40 m behind (a long lens squashed the gap; a wide one lost Rupert altogether)
    finish = [[1, [112.0, 2.6, 0.9], [80.0, 0.4, 0.95], 50], [n, [112.0, 2.6, 0.9], [80.0, 0.4, 0.95], 50]]
    cams_spec = [{"name": "cam_race_wide", "fstop": 8.0, "keys": keys_wide},
                 {"name": "cam_race_finish", "fstop": 11.0, "keys": finish}]
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": cams_spec, "camera": "cam_race_wide"}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "THE RACE", extra=["--cast", cast, "--props", props])
    tag = "_draft" if draft else ""
    for cam in cams.split(","):
        render(blend, f"cam_{cam}", 1, n, OUT / f"{cam}{tag}.mp4", scale=scale, draft=draft)


JUDGE_FINISH_X = 106.5       # the judge's desk on the far verge, just past the line


def final100(scale, draft=False, seconds=19.0, cams="start_low,rail,head_on,finish_slowmo,judge_ten"):
    """The cinematic 100 m: Rupert's real final run (EXP-010, 15.2 s) from five cameras, in one
    scene with the trackside race clock and the judge at his desk past the finish line. He follows
    Rupert with his head, then slowly raises his card... 10. Each camera renders its own clip."""
    from rupert import poses
    xp, xq, names = final_run(seconds)
    n = len(xp)
    npz = OUT / "final100.npz"
    _save(npz, xp, xq, names)
    torso = names.index("torso")
    cross = int(np.argmax(xp[:, torso, 0] >= 100.0))         # frame index he reaches the line
    print(f"  Rupert reaches the line at {cross / FPS:.2f} s (frame {cross + 1}); the clock stops at 15.2")
    card_up = cross / FPS + 1.6
    jp, jq, jn = poses.judge_watching((JUDGE_FINISH_X, JUDGE_SIDE_Y), -90.0, xp[:, names.index("head")],
                                      card_up_at=card_up, slow=True)
    jnpz = OUT / "final100_judge.npz"
    _save(jnpz, jp, jq, jn)
    cast = _cast_file(npz, [{"replay": str(jnpz.resolve()), "character": "judge", "prefix": "judge", "desk": True,
                             "yaw": -90.0, "cards": [{"text": "10", "from": int(card_up * FPS) + 1, "face_yaw": -90.0}]}])
    clock = [[f + 1, min(int((f / FPS) * 10), 151)] for f in range(cross)] + [[cross + 1, 152]]
    clock = [c for k, c in enumerate(clock) if k == 0 or c[1] != clock[k - 1][1]]
    T = lambda f: xp[min(f, n - 1), torso]
    cams_spec = []
    # 1. low and in front off the start: he drives up at the lens
    keys = []
    for f in list(range(0, 200, 5)):
        p = T(f); u = f / 200
        keys.append([f + 1, [float(p[0] + 3.2 - 1.4 * u), -0.7, 0.32], [float(p[0]), 0.0, float(p[2] - 0.55)], 24])
    cams_spec.append({"name": "cam_start_low", "fstop": 2.8, "keys": keys})
    # 2. rail cam beside him at full speed
    keys = []
    for f in range(150, 460, 5):
        p = T(f)
        keys.append([f + 1, [float(p[0] + 0.4), -4.3, 0.95], [float(p[0] + 0.4), 0.0, float(p[2] - 0.45)], 40])
    cams_spec.append({"name": "cam_rail", "fstop": 2.8, "keys": keys})
    # 3. long lens head-on from way past the finish: the lens zooms so he stays the same size
    keys = []
    for f in range(400, min(n, cross + 20), 5):
        p = T(f); dist = 160.0 - p[0]
        keys.append([f + 1, [160.0, 0.2, 1.35], [float(p[0]), 0.0, float(p[2] - 0.4)], float(np.clip(dist * 8.0, 100, 1200))])
    cams_spec.append({"name": "cam_head_on", "fstop": 5.6, "keys": keys})
    # 4. side on at the line, for the slow-motion finish (clock, FINISH sign and judge behind)
    cams_spec.append({"name": "cam_finish_slowmo", "fstop": 4.0, "keys": [
        [1, [100.4, -7.5, 0.85], [100.0, 2.0, 1.05], 38], [n, [100.4, -7.5, 0.85], [100.0, 2.0, 1.05], 38]]})
    # 5. the judge: Rupert crosses in the foreground, then... the card
    cams_spec.append({"name": "cam_judge_ten", "fstop": 2.0, "keys": [
        [1, [JUDGE_FINISH_X - 0.3, -6.5, 1.3], [JUDGE_FINISH_X, JUDGE_SIDE_Y - 0.3, 1.22], 135],
        [n, [JUDGE_FINISH_X + 0.3, -6.1, 1.3], [JUDGE_FINISH_X, JUDGE_SIDE_Y - 0.3, 1.22], 135]]})
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": cams_spec, "camera": "cam_rail",
                                 "clock": {"loc": [95.8, JUDGE_SIDE_Y, 0.25], "digit_h": 0.62, "frames": clock}}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "FINAL RUN", extra=["--cast", cast, "--props", props])
    tag = "_draft" if draft else ""
    ranges = {"start_low": (1, 200, 1), "rail": (150, 460, 1), "head_on": (400, min(n, cross + 20), 1),
              "finish_slowmo": (cross - 45, cross + 35, 4), "judge_ten": (cross - 60, min(n, int((card_up + 3.2) * FPS)), 1)}
    for cam in cams.split(","):
        f0, f1, slow = ranges[cam]
        render(blend, f"cam_{cam}", f0, f1, OUT / f"final100_{cam}{tag}.mp4", slowmo=slow, scale=scale, draft=draft)


XRAY_PASS = ("\nbpy.data.collections['rupert'].hide_render=True; bpy.data.collections['stadium'].hide_render=True"
             "\nbpy.data.collections['xray'].hide_render=False"
             "\nw=bpy.data.worlds.new('xray'); w.use_nodes=True"
             "\nw.node_tree.nodes['Background'].inputs[0].default_value=(0.01,0.03,0.14,1); s.world=w"
             "\nfor o in s.objects:\n    o.hide_render = o.hide_render or o.type=='LIGHT'")


def zap(scale, draft=False):
    """'He gets electrocuted when he does something bad': Rupert on the start line gets zapped.
    Lightning, X-ray skeleton flashes, then hair on end, soot, smoke, and he topples over stiff."""
    from PIL import Image  # noqa: F401  (imageio below needs the frames as files)
    from rupert import poses
    xp, xq, names, beats = poses.zapped()
    npz = OUT / "zap.npz"
    _save(npz, xp, xq, names)
    n = len(xp)
    f0, f1 = beats["zap"] + 1, beats["zap_end"] + 1
    cam = {"name": "cam_zap", "fstop": 4.0, "keys": [
        [1, [3.4, -2.5, 1.35], [0.0, 0.0, 1.05], 32],
        [f0, [3.3, -2.45, 1.35], [0.0, 0.0, 1.05], 32],
        [f0 + 3, [3.05, -2.25, 1.4], [0.0, 0.0, 1.2], 32],        # a jolt in as it hits (room for his arms)
        [beats["fall"], [3.0, -2.2, 1.4], [0.0, 0.0, 1.2], 32],
        [n, [3.6, -2.6, 1.5], [-0.6, 0.0, 0.5], 32]]}             # follows him down
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_zap", "zap": {"start": f0, "end": f1}}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "OUCH", extra=["--props", props])
    tag = "_draft" if draft else ""
    out = OUT / f"zap{tag}.mp4"
    w, h = (640, 360) if draft else (1920 * scale // 100, 1080 * scale // 100)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = (f"import bpy; s=bpy.context.scene; s.render.resolution_percentage={scale}; "
                f"s.camera=s.objects['cam_zap']" + (DRAFT if draft else ""))
        run([BLENDER, "-b", blend, "--python-expr", base, "-o", tmp / "f_####", "-s", 1, "-e", n, "-a"])
        run([BLENDER, "-b", blend, "--python-expr", base + XRAY_PASS, "-o", tmp / "x_####", "-s", f0, "-e", f1, "-a"])
        for f in range(f0, f1):
            if ((f - f0) // 2) % 2 == 1:                      # every other pair of frames: the X-ray
                shutil.copyfile(tmp / f"x_{f:04d}.png", tmp / f"f_{f:04d}.png")
        run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", FPS, "-start_number", 1,
             "-i", tmp / "f_%04d.png", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out])
    print(f"Saved {out}  (zap frames {f0}-{f1}, falls at {beats['fall'] + 1})", flush=True)


def outro(scale, draft=False):
    """Bookends the opening: wide on Rupert just past the finish, push in to his face while he gives
    a big wave... then he falls flat on his face (cut before it if you'd rather end on the wave)."""
    from rupert import poses
    xp, xq, names, beats = poses.outro()
    npz = OUT / "outro.npz"
    _save(npz, xp, xq, names)
    n = len(xp)
    hx, hy, hz = (float(v) for v in xp[0, names.index("head")])
    off = lambda dx, dy, dz: [hx + dx, hy + dy, hz + dz]
    push = int(3.6 * FPS)
    cam = {"name": "cam_outro", "fstop": 2.8, "keys": [            # he faces -x: camera on the -x side
        [1, off(-7.8, -2.36, 0.2), off(-0.4, 0, -0.6), 32],
        [push, off(-2.5, -0.65, 0.0), off(0, 0, -0.05), 40],
        [beats["fall"], off(-2.4, -0.62, 0.0), off(0, 0, -0.05), 40],
        [beats["fall"] + 22, off(-3.0, -0.9, -0.5), off(-1.0, 0, -1.45), 35],   # tilts down as he comes over
        [n, off(-3.1, -0.95, -0.55), off(-1.0, 0, -1.5), 35]]}                # his face in the track
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_outro"}))
    blend = npz.with_suffix(".blend")
    build(npz, blend, 1, "THANKS FOR WATCHING", extra=["--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_outro", 1, n, OUT / f"outro_wave{tag}.mp4", scale=scale, draft=draft)
    print(f"  he starts to fall at {beats['fall'] / FPS:.1f} s (frame {beats['fall'] + 1}): cut there to end on the wave")


def subscribe(scale, draft=False):
    """The judge gets the last word: he holds up his 10... and turns it round. SUBSCRIBE."""
    from rupert import poses
    xj = JUDGE_FINISH_X
    jp, jq, jn = poses.judge_subscribe((xj, JUDGE_SIDE_Y))
    npz = OUT / "judge_subscribe.npz"
    _save(npz, jp, jq, jn)
    n = len(jp)
    flip = int(2.2 * FPS)
    cast = _cast_file(npz, [{"replay": str(npz.resolve()), "character": "judge", "prefix": "judge", "desk": True,
                             "yaw": -90.0, "cards": [{"text": "10", "from": 1, "face_yaw": -90.0,
                                                      "back": "SUBSCRIBE", "flip": flip}]}])
    # the scene needs a main character: Rupert, out of shot, waving at the finish
    xp, xq, names, _ = poses.outro()
    rnpz = OUT / "judge_subscribe_rupert.npz"
    _save(rnpz, xp[:n], xq[:n], names)
    cam = {"name": "cam_subscribe", "fstop": 2.8, "keys": [
        [1, [xj, JUDGE_SIDE_Y - 3.4, 1.3], [xj, JUDGE_SIDE_Y, 1.35], 50],
        [n, [xj, JUDGE_SIDE_Y - 2.9, 1.3], [xj, JUDGE_SIDE_Y, 1.38], 50]]}
    props = npz.with_suffix(".props.json")
    props.write_text(json.dumps({"cameras": [cam], "camera": "cam_subscribe"}))
    blend = npz.with_suffix(".blend")
    build(rnpz, blend, 1, "SUBSCRIBE", extra=["--cast", cast, "--props", props])
    tag = "_draft" if draft else ""
    render(blend, "cam_subscribe", 1, n, OUT / f"judge_subscribe{tag}.mp4", scale=scale, draft=draft)


def preview(scale, draft=False, which="bike,date,trading"):
    """'Coming up in this series': Rupert on a bike, on a date, and at a trading desk (2-3 s each)."""
    from rupert import previews as P
    tag = "_draft" if draft else ""
    for kind in which.split(","):
        npz = OUT / f"preview_{kind}.npz"
        spec = {}
        if kind == "bike":
            xp, xq, names, keys = P.bike()
            spec["bike"] = {"keys": keys, "wheel_r": P.WHEEL_R, "crank": list(P.CRANK), "crank_l": P.CRANK_L,
                            "saddle": list(P.SADDLE), "bars": list(P.BARS), "wheelbase": list(P.WHEELBASE)}
            n = len(xp)
            keys_c = []
            for f in list(range(0, n, 10)) + [n - 1]:             # tracking alongside, 3/4 from the front
                x = keys[f][0][0]
                keys_c.append([f + 1, [x + 3.3, -2.9, 1.25], [x + 0.2, 0.0, 0.95], 35])
            cam = {"name": "cam_preview", "fstop": 2.8, "keys": keys_c}
            screen_text = "COMING SOON"
        elif kind == "date":
            xp, xq, names = P.date()
            n = len(xp)
            x, y = P.TABLE
            spec["date"] = {"table": list(P.TABLE), "table_h": P.TABLE_H, "frames": n}
            spec["night"] = True
            cam = {"name": "cam_preview", "fstop": 2.0, "keys": [            # the date's point of view
                [1, [x + 1.55, y - 0.45, 1.35], [x - 0.45, y, 0.98], 30],
                [n, [x + 1.35, y - 0.4, 1.33], [x - 0.45, y, 1.0], 30]]}
            screen_text = "TABLE FOR TWO"
        else:
            xp, xq, names = P.trading()
            n = len(xp)
            x, y = P.DESK
            images = [P.chart_frames(OUT / f"preview_trading_screen{k}", n, kind=kd, seed=k)
                      for k, kd in enumerate(("candles", "line", "tickers"))]
            spec["trading"] = {"desk": list(P.DESK), "frames": n, "images": [str(i.resolve()) for i in images]}
            cam = {"name": "cam_preview", "fstop": 2.8, "keys": [            # over his shoulder at the screens
                [1, [x - 1.9, y - 1.5, 1.6], [x + 0.2, y + 0.05, 1.05], 32],
                [n, [x - 1.6, y - 1.25, 1.55], [x + 0.2, y + 0.05, 1.08], 32]]}
            screen_text = "TO THE MOON"
        _save(npz, xp, xq, names)
        spec.update({"cameras": [cam], "camera": "cam_preview"})
        props = npz.with_suffix(".props.json")
        props.write_text(json.dumps(spec))
        blend = npz.with_suffix(".blend")
        build(npz, blend, 1, screen_text, extra=["--props", props])
        render(blend, "cam_preview", 1, n, OUT / f"preview_{kind}{tag}.mp4", scale=scale, draft=draft)


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
    ap.add_argument("shot", choices=["intro", "timelapse", "judge", "swarm", "montage", "ghost", "forgot",
                                     "opening", "teacher", "judge_walk", "judge_scores", "race", "final100",
                                     "zap", "outro", "subscribe", "preview"])
    ap.add_argument("--which", default="bike,date,trading", help="preview: bike, date, trading (comma separated)")
    ap.add_argument("--cams", default=None, help="race: which cameras (race_wide, race_finish)")
    ap.add_argument("--cards", default="pogo:2,trex:1,gallop:0", help="judge_scores: cheat:card, comma separated")
    ap.add_argument("--seconds", type=float, default=10.0, help="ghost: length of the shot")
    ap.add_argument("--chapter", default="cheat1_stands_still", help="for swarm/montage: see rupert/ep01_story.py")
    ap.add_argument("--scale", type=int, default=100, help="resolution percent of 1920x1080 (25 = quick test)")
    ap.add_argument("--out-dir", default=None, help="folder for the clips (default renders/ep01_shots)")
    ap.add_argument("--camera", default="cam_timelapse",
                    help="ghost: camera(s), comma separated, e.g. " + ",".join(TL_CAMS))
    ap.add_argument("--draft", action="store_true", help="ghost: quick 640x360 test version (minutes, not hours)")
    ap.add_argument("--reuse", action="store_true", help="ghost: reuse the last practice footage (skip the physics)")
    ap.add_argument("--board", default="points", help="teacher: board(s), comma separated: points, brain, graph, boat")
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
    elif a.shot == "teacher":
        for board in a.board.split(","):
            teacher(a.scale, draft=a.draft, board=board)
    elif a.shot == "opening":
        opening(a.scale, draft=a.draft)
    elif a.shot == "judge_walk":
        judge_walk(a.scale, draft=a.draft)
    elif a.shot == "race":
        race(a.scale, draft=a.draft, cams=a.cams or "race_wide,race_finish")
    elif a.shot == "final100":
        final100(a.scale, draft=a.draft)
    elif a.shot == "zap":
        zap(a.scale, draft=a.draft)
    elif a.shot == "outro":
        outro(a.scale, draft=a.draft)
    elif a.shot == "preview":
        preview(a.scale, draft=a.draft, which=a.which)
    elif a.shot == "subscribe":
        subscribe(a.scale, draft=a.draft)
    elif a.shot == "judge_scores":
        for spec in a.cards.split(","):
            cheat, card = spec.split(":")
            judge_scores(a.scale, draft=a.draft, cheat=cheat, card=card)
    else:
        {"intro": intro, "timelapse": timelapse, "judge": judge, "forgot": forgot}[a.shot](a.scale)


if __name__ == "__main__":
    sys.exit(main())
