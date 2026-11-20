"""Hand-made animations (not physics, not learned): character moments like Rupert waving
hello in the intro, or the judge holding up a score card.

Poses are set by saying where the elbows and hands should be (relative to the chest);
a small solver finds the joint angles. Joint limits are ignored: it's animation, not the
body that trains. Output is the same replay format as evaluate.py, so the Blender
pipeline renders it like any other clip.
"""
import mujoco
import numpy as np

from rupert import env as rupert_env

FPS = 50
ARM_Q = {"right": slice(22, 25), "left": slice(25, 28)}
LEG_Q = {"right": slice(10, 16), "left": slice(16, 22)}


def model():
    return rupert_env.load().mj_model


def _solve_arm(m, d, q, side, elbow, hand, iters=60):
    """Joint angles of one arm so the elbow and hand sit at these points (chest frame, metres)."""
    t = m.body("torso").id
    targets = [(m.body(f"{side}_lower_arm").id, np.array(elbow)), (m.body(f"{side}_hand").id, np.array(hand))]
    sl = ARM_Q[side]
    dof = np.arange(sl.start - 1, sl.stop - 1)   # qpos index - 1 = dof index for hinge joints after the free joint
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        R, p0 = d.xmat[t].reshape(3, 3), d.xpos[t]
        res, J = [], []
        for b, target in targets:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jac, None, b)
            res.append(d.xpos[b] - (p0 + R @ target)); J.append(jac[:, dof])
        r, J = np.concatenate(res), np.concatenate(J)
        step = -np.linalg.solve(J.T @ J + 1e-4 * np.eye(len(dof)), J.T @ r)
        q[sl] += step
        if np.abs(step).max() < 1e-6:
            break
    return q


def kinematics(m, qs):
    """qpos per frame -> replay arrays (xpos, xquat) and body names."""
    d = mujoco.MjData(m)
    xp, xq = [], []
    for q in qs:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        xp.append(d.xpos.copy()); xq.append(d.xquat.copy())
    return np.array(xp), np.array(xq), [m.body(i).name for i in range(m.nbody)]


def _ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# Chest frame: x = forward, y = his left, z = up. Shoulders are at about y = +-0.17, z = +0.06.
ARM_DOWN = {"right": ((0.02, -0.23, -0.29), (0.07, -0.24, -0.65)),
            "left": ((0.02, 0.23, -0.29), (0.07, 0.24, -0.65))}
WAVE_UP = ((0.06, -0.52, 0.0), (0.12, -0.50, 0.36))   # right elbow out to the side, hand up above his head


def standing(m, d, x=0.0, y=0.0, yaw_deg=0.0):
    """Standing still, arms by his sides, at (x, y) facing yaw_deg (0 = down the track, +x)."""
    q = m.qpos0.copy()
    q[0], q[1], q[2] = x, y, 1.50
    a = np.radians(yaw_deg) / 2
    q[3:7] = (np.cos(a), 0.0, 0.0, np.sin(a))
    for side in ("right", "left"):
        q = _solve_arm(m, d, q, side, *ARM_DOWN[side])
    return q


WAVE_BIG = ((0.10, -0.40, 0.12), (0.22, -0.38, 0.50))   # hand high above his head: a big hello


def wave_qs(m, d, q0, seconds=4.0, waves=4, up_pose=WAVE_UP):
    """Joint angles per frame: raising his right hand, waving (side to side), lowering it."""
    n = int(seconds * FPS)
    up, down = 0.6, seconds - 0.6
    qs, q = [], q0.copy()
    for i in range(n):
        t = i / FPS
        lift = _ease(t / up) * (1 - _ease((t - down) / 0.6))
        swing = np.sin(2 * np.pi * waves * (t - up) / (down - up)) if up < t < down else 0.0
        (e0, h0), (e1, h1) = ARM_DOWN["right"], up_pose
        elbow = np.array(e0) + lift * (np.array(e1) - np.array(e0))
        hand = np.array(h0) + lift * (np.array(h1) - np.array(h0)) + lift * np.array([0.0, 0.13 * swing, -0.03 * abs(swing)])
        q = _solve_arm(m, d, q.copy(), "right", elbow, hand)
        q[2] = q0[2] + 0.004 * np.sin(2 * np.pi * t / 1.6)   # tiny breathing bob
        qs.append(q.copy())
    return qs


def wave(seconds=4.0, waves=4):
    """Rupert standing, raising his right hand and waving (hand swings side to side), then lowering it."""
    m = model()
    d = mujoco.MjData(m)
    return kinematics(m, wave_qs(m, d, standing(m, d), seconds, waves))


def opening(hold=1.9, wave_s=3.4, after=0.2):
    """Cold open: standing on the start line (the camera is on his face), then a big wave."""
    m = model()
    d = mujoco.MjData(m)
    q0 = standing(m, d)
    qs = [q0.copy() for _ in range(int(hold * FPS))]
    for i, q in enumerate(qs):
        q[2] = q0[2] + 0.004 * np.sin(2 * np.pi * (i / FPS) / 1.6)
    qs += wave_qs(m, d, q0, wave_s, waves=3, up_pose=WAVE_BIG)
    qs += [qs[-1].copy() for _ in range(int(after * FPS))]
    return kinematics(m, qs)


# ---- Teacher Rupert: pointing at a whiteboard with a stick --------------------------------
STICK = 0.75     # pointer stick length beyond the wrist (blender/props.py uses the same)


def _frame(m, d, q, body):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.xmat[m.body(body).id].reshape(3, 3).copy(), d.xpos[m.body(body).id].copy()


def _tip(m, d, q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    e, h = d.xpos[m.body("right_lower_arm").id], d.xpos[m.body("right_hand").id]
    return h + STICK * (h - e) / np.linalg.norm(h - e)


def _solve_tip(m, d, q, target, iters=40):
    """Right-arm joint angles that put the pointer stick's tip on `target` (world). Three joints,
    three numbers to hit, so it's solvable when in reach."""
    q = q.copy()
    sl = ARM_Q["right"]
    target = np.asarray(target, float)
    for _ in range(iters):
        r = _tip(m, d, q) - target
        if np.abs(r).max() < 1e-4:
            break
        J = np.zeros((3, 3))
        for j, k in enumerate(range(sl.start, sl.stop)):
            qq = q.copy(); qq[k] += 1e-4
            J[:, j] = (_tip(m, d, qq) - _tip(m, d, q)) / 1e-4
        q[sl] -= np.linalg.solve(J.T @ J + 1e-3 * np.eye(3), J.T @ r)
    return q


def teacher(at=(0.0, 0.0), yaw_deg=-115.0, seconds=6.0, points=(), taps=2, tap_dir=(0.0, -1.0, 0.0), rest_tip=None):
    """Teacher Rupert standing beside the board. points: [(start_s, end_s, target)], where a target is
    one spot (world xyz: he taps it `taps` times and holds) or a path of spots (traced from start to end).
    The arm swings up to the board in 0.5 s and back down to rest afterwards, joints moving together like
    a real arm; between points less than 0.6 s apart the tip glides across instead. While pointing, the
    arm is solved every frame from where the tip should be. tap_dir: away from the board."""
    m = model()
    d = mujoco.MjData(m)
    q0 = standing(m, d, at[0], at[1], yaw_deg)
    sl = ARM_Q["right"]
    f = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
    right = np.array([f[1], -f[0]])
    if rest_tip is None:          # holding the stick angled down in front of him, tip off the ground
        rest_tip = (*(np.array(at) + 0.45 * f + 0.2 * right), 0.35)
    rest_tip = np.asarray(rest_tip, float)
    q = q0.copy()
    for k in range(1, 21):        # ease the arm into the rest grip from hanging straight down
        q = _solve_tip(m, d, q, _tip(m, d, q0) + k / 20 * (rest_tip - _tip(m, d, q0)), iters=30)
    q_rest = q.copy()
    pts = [(t0, t1, np.atleast_2d(np.asarray(tip, float))) for t0, t1, tip in points]
    gap = [pts[k + 1][0] - t1 if k + 1 < len(pts) else 99.0 for k, (_, t1, _) in enumerate(pts)]
    glide = [g < 0.6 for g in gap]                      # tip slides across the board to the next spot
    direct = [0.6 <= g < 1.0 for g in gap]              # arm swings straight to the next spot
    # the arm pose at the first spot of each target, reached from rest (the swing up aims for this)
    q_at = [_solve_tip(m, d, q_rest, path[0], iters=200) for _, _, path in pts]
    for qa in q_at:
        qa[sl] = q_rest[sl] + _wrap(qa[sl] - q_rest[sl])
    qs, worst, worst_t, q_left = [], 0.0, 0.0, None
    for i in range(int(seconds * FPS)):
        t = i / FPS
        want, swing = None, None           # tip target while pointing, or (from, to, amount) while swinging
        for k, (t0, t1, path) in enumerate(pts):
            glide_in = k and (glide[k - 1] or direct[k - 1])
            if t0 - 0.5 <= t < t0 and not glide_in:                         # swinging up to the board
                swing = (q_rest, q_at[k], _ease((t - (t0 - 0.5)) / 0.5))
            elif t0 <= t <= t1:                                               # pointing / tracing
                want = _along(path, (t - t0) / max(t1 - t0, 1e-6))
                if len(path) == 1:
                    tap_len = max(0.4, (t1 - t0) * 0.5)
                    if t <= t0 + tap_len:
                        want = want + 0.07 * np.sin(np.pi * taps * (t - t0) / tap_len) ** 2 * np.asarray(tap_dir)
            elif t1 < t < pts[k + 1][0] if glide[k] else False:               # gliding to the next one
                want = path[-1] + _ease((t - t1) / (pts[k + 1][0] - t1)) * (pts[k + 1][2][0] - path[-1])
            elif t1 < t < pts[k + 1][0] if direct[k] else False:              # swinging to the next one
                swing = (q_left, q_at[k + 1], _ease((t - t1) / gap[k]))
            elif t1 < t <= t1 + 0.5 and not (glide[k] or direct[k]):          # swinging back down
                swing = (q_left, q_rest, _ease((t - t1) / 0.5))
        if want is not None:
            q = _solve_tip(m, d, q, want, iters=15)
            miss = float(np.linalg.norm(_tip(m, d, q) - want))
            if miss > worst:
                worst, worst_t = miss, t
            q_left = q.copy()
        elif swing is not None:
            a_, b_, u = swing
            q = q.copy()
            q[sl] = a_[sl] + u * _wrap(b_[sl] - a_[sl])                   # the short way round
        else:
            q = q_rest.copy()
        q[2] = q0[2] + 0.004 * np.sin(2 * np.pi * t / 1.6)
        qs.append(q.copy())
    print(f"  stick tip: worst miss while pointing {worst * 100:.1f} cm (at {worst_t:.2f} s)")
    return kinematics(m, qs)


def _wrap(a):
    """Angles brought into -pi..pi (the solver can wind a joint round several full turns)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _along(path, s):
    """The point a fraction s (0..1) of the way along a path of points, by distance."""
    if len(path) == 1:
        return path[0]
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    d = np.clip(s, 0, 1) * cum[-1]
    j = min(np.searchsorted(cum, d, side="right") - 1, len(seg) - 1)
    return path[j] + (d - cum[j]) / max(seg[j], 1e-9) * (path[j + 1] - path[j])


def stick_tip(xp, names, frame):
    """Where the stick tip is at this frame (world), for checking the aim."""
    e, h = xp[frame, names.index("right_lower_arm")], xp[frame, names.index("right_hand")]
    return h + STICK * (h - e) / np.linalg.norm(h - e)


if __name__ == "__main__":
    xp, xq, names = wave()
    print(xp.shape)


# ---- the style judge: seated behind his desk ------------------------------------------
JUDGE_AT = (104.0, -5.8)      # just past the finish line, beside lane 1, facing down the track (+x)
JUDGE_ARMS_DESK = {"right": ((0.10, -0.27, -0.33), (0.43, -0.15, -0.30)),
                   "left": ((0.10, 0.27, -0.33), (0.43, 0.15, -0.30))}
JUDGE_CARD_UP = ((0.10, -0.47, -0.22), (0.16, -0.60, 0.08))   # right hand holds a score card up by his head


def _seated(m, d):
    q = m.qpos0.copy()
    q[0], q[1] = JUDGE_AT
    for hip, knee in ((12, 13), (18, 19)):
        q[hip], q[knee] = np.radians(-90), np.radians(-90)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    feet = [d.xpos[m.body(f"{s}_foot").id, 2] for s in ("right", "left")]
    q[2] -= min(feet) - 0.008          # feet on the floor
    for side in ("right", "left"):
        q = _solve_arm(m, d, q, side, *JUDGE_ARMS_DESK[side])
    return q


def judge(seconds=4.5, card_up_at=None):
    """The judge sitting at his desk (tiny breathing). card_up_at: second at which he raises a score card."""
    m = model()
    d = mujoco.MjData(m)
    q0 = _seated(m, d)
    z0 = q0[2]
    qs, q = [], q0.copy()
    for i in range(int(seconds * FPS)):
        t = i / FPS
        lift = _ease((t - card_up_at) / 0.5) if card_up_at is not None else 0.0
        (e0, h0), (e1, h1) = JUDGE_ARMS_DESK["right"], JUDGE_CARD_UP
        q = _solve_arm(m, d, q.copy(), "right", np.array(e0) + lift * (np.array(e1) - np.array(e0)),
                       np.array(h0) + lift * (np.array(h1) - np.array(h0)))
        q[2] = z0 + 0.003 * np.sin(2 * np.pi * t / 2.2)
        qs.append(q.copy())
    return kinematics(m, qs)


# ---- the real runner: one looped stride from a motion-capture clip ----------------------

def looped_run(trial="09_01", seconds=6.0, x0=20.0):
    """A real person's jog (CMU motion capture fitted onto the body), one stride repeated so it
    can run for as long as the shot needs. Returns replay arrays."""
    from pathlib import Path
    z = np.load(Path(__file__).parent.parent / "mocap" / "run_reference.npz")
    trials = [str(t) for t in z["trials"]]
    k = trials.index(trial)
    q = z["qpos"][z["starts"][k]:z["starts"][k + 1]].astype(np.float64)
    # find the stride: the later frame whose pose best matches an early frame (a full cycle later)
    best = None
    for i in range(0, 10):
        for j in range(i + 25, len(q)):
            err = np.linalg.norm(q[j, 7:] - q[i, 7:]) + 2 * abs(q[j, 2] - q[i, 2])
            if best is None or err < best[0]:
                best = (err, i, j)
    _, i, j = best
    cycle = q[i:j]
    dx = q[j, 0] - q[i, 0]
    m = model()
    qs = []
    for f in range(int(seconds * FPS)):
        c, r = divmod(f, len(cycle))
        qq = cycle[r].copy()
        qq[0] += c * dx - q[i, 0] + x0
        qs.append(qq)
    print(f"{trial}: stride {len(cycle)} frames ({len(cycle) / FPS:.2f} s), {dx:.2f} m, {dx / (len(cycle) / FPS):.1f} m/s")
    return kinematics(m, qs)
