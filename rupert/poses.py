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


def wave(seconds=4.0, waves=4):
    """Rupert standing, raising his right hand and waving (hand swings side to side), then lowering it."""
    m = model()
    d = mujoco.MjData(m)
    q0 = m.qpos0.copy()
    q0[2] = 1.50
    for side in ("right", "left"):
        q0 = _solve_arm(m, d, q0, side, *ARM_DOWN[side])
    n = int(seconds * FPS)
    up, down = 0.6, seconds - 0.6
    qs, q = [], q0.copy()
    for i in range(n):
        t = i / FPS
        lift = _ease(t / up) * (1 - _ease((t - down) / 0.6))
        swing = np.sin(2 * np.pi * waves * (t - up) / (down - up)) if up < t < down else 0.0
        (e0, h0), (e1, h1) = ARM_DOWN["right"], WAVE_UP
        elbow = np.array(e0) + lift * (np.array(e1) - np.array(e0))
        hand = np.array(h0) + lift * (np.array(h1) - np.array(h0)) + lift * np.array([0.0, 0.13 * swing, -0.03 * abs(swing)])
        q = _solve_arm(m, d, q.copy(), "right", elbow, hand)
        q[2] = 1.50 + 0.004 * np.sin(2 * np.pi * t / 1.6)   # tiny breathing bob
        qs.append(q.copy())
    return kinematics(m, qs)


if __name__ == "__main__":
    xp, xq, names = wave()
    print(xp.shape)


# ---- the style judge: seated behind his desk ------------------------------------------
JUDGE_AT = (104.0, -5.8)      # just past the finish line, beside lane 1, facing down the track (+x)
JUDGE_ARMS_DESK = {"right": ((0.10, -0.27, -0.33), (0.43, -0.15, -0.30)),
                   "left": ((0.10, 0.27, -0.33), (0.43, 0.15, -0.30))}
JUDGE_CARD_UP = ((0.12, -0.42, -0.22), (0.22, -0.46, 0.14))   # right hand holds a score card up by his head


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
