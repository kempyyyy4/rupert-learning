"""The world-record sprinter in EP01's race: hand-made animation (not physics, not learned).

His position along the track follows Usain Bolt's published 10 m split times from Berlin 2009
(9.58 s, including his 0.146 s reaction), and he takes 41 steps to the line, like Bolt. The
body is ours: each foot is planted on the track while it's down (no sliding), the legs are
solved to reach each footstep, the heel kicks up in the swing, and the arms pump against the legs.

Splits (cumulative from the gun): IAAF Berlin 2009 biomechanics analysis, as quoted on The
Science of Sport. Sources differ by about 0.02 s; fine for animation, not for on-screen numbers.
"""
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from rupert import poses

FPS = 50
REACTION = 0.146
SPLITS = [1.89, 2.88, 3.78, 4.64, 5.47, 6.29, 7.10, 7.92, 8.75, 9.58]
STEPS = 41
LANE_Y = 1.22            # lane 4, next to Rupert's lane 3
FOOT_Y = 0.10            # feet either side of the middle of his lane
ANKLE_Z = 0.103          # ankle height standing, foot flat
LAND, PUSH = 0.27, 0.40  # a foot is down while his hips are from LAND behind it to PUSH past it
LEGS = {"right": 10, "left": 16}     # first qpos index of each leg: hip x, hip z, hip y, knee, ankle y, ankle x


def distance(t):
    """Metres from the start line at time t (s after the gun)."""
    t = np.asarray(t, float)
    f = PchipInterpolator([0.0, REACTION] + SPLITS, [0.0, 0.0] + [10.0 * (i + 1) for i in range(10)])
    v_end = float(f.derivative()(SPLITS[-1]))
    after = t - SPLITS[-1]
    slow = 100 + v_end * after - 1.6 * after ** 2          # easing up after the line
    return np.where(t <= SPLITS[-1], f(np.clip(t, 0, None)), np.where(after < v_end / 3.2, slow,
                    100 + v_end ** 2 / 6.4))


def footfalls():
    """Where each foot lands: right at even k, left at odd k (k = -2, -1 are the standing start)."""
    first, tau = 0.95, 5.0
    # longest step so the 41st footfall lands just past the line
    lo, hi = 1.5, 4.0
    for _ in range(60):
        mx = (lo + hi) / 2
        L = mx - (mx - first) * np.exp(-np.arange(STEPS) / tau)
        if 0.2 + L.sum() < 100.4:
            lo = mx
        else:
            hi = mx
    F = {-2: -0.22, -1: 0.2}
    for k in range(STEPS + 6):
        F[k] = F[k - 1] + (mx - (mx - first) * np.exp(-k / tau))
    return F, mx


def _foot(F, side, b, speed):
    """Ankle target (x, z), foot pitch (deg, + = toes up) and whether it's down, for hips at b."""
    ks = sorted(k for k in F if (k % 2 == 0) == (side == "right"))
    for k in ks:                                         # planted?
        if F[k] - LAND <= b <= F[k] + PUSH:
            u = (b - (F[k] - LAND)) / (LAND + PUSH)
            heel = 0.11 * max(0.0, (u - 0.45) / 0.55) ** 2 * min(1.0, speed / 4)   # heel lifts as he pushes off
            return F[k] - 0.02 * heel / 0.11, ANKLE_Z + heel, -6 - 40 * max(0.0, u - 0.45) * min(1.0, speed / 4), True
    for a, c in zip(ks[:-1], ks[1:]):                    # swinging from a to c
        lift_off, touch = F[a] + PUSH, F[c] - LAND
        if lift_off < b < touch:
            s = (b - lift_off) / (touch - lift_off)
            x = F[a] + (F[c] - F[a]) * (s * s * (3 - 2 * s)) ** 0.9
            high = 0.08 + 0.5 * min(1.0, speed / 11.0)
            z = ANKLE_Z + 0.11 * min(1.0, speed / 4) * (1 - s) + high * np.sin(np.pi * s ** 0.8)
            pitch = -40 * (1 - s) ** 2 + 12 * np.sin(np.pi * s)
            return x, z, pitch, False
    return F[ks[0]], ANKLE_Z, 0.0, True


THIGH, SHIN = 0.455, 0.391      # hip to knee, knee to ankle (m)
HIP_BACK_MAX = np.radians(40)   # animation: a sprinter's leg drives further back than the training body allows


def _leg_ik(m, d, q, side, target):
    """Hip and knee angles that put this ankle at `target` (world): a two-bone leg solved exactly
    (law of cosines) in the plane he runs in. Thigh pointing straight ahead (no splay or twist)."""
    i0 = LEGS[side]
    q[i0:i0 + 4] = 0.0
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    hip = d.xanchor[m.joint(f"{side}_hip_y").id]
    R = d.xmat[m.body("pelvis").id].reshape(3, 3)
    r = R.T @ (np.asarray(target, float) - hip)            # in the hips' frame: x forward, z up
    rx, rz = r[0], r[2]
    dist = np.clip(np.hypot(rx, rz), abs(THIGH - SHIN) + 1e-3, THIGH + SHIN - 1e-4)
    bend = np.pi - np.arccos(np.clip((THIGH ** 2 + SHIN ** 2 - dist ** 2) / (2 * THIGH * SHIN), -1, 1))
    back = np.arctan2(-rx, -rz)                            # hip-to-ankle line, from straight down (+ = behind)
    lean = np.arcsin(np.clip(SHIN * np.sin(bend) / dist, -1, 1))
    q[i0 + 2] = np.clip(back - lean, np.radians(-110), HIP_BACK_MAX)   # hip y: + swings the thigh back
    q[i0 + 3] = -bend                                                    # knee: - bends the shin back
    return q


def _foot_pitch(m, d, q, side):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    fx = d.xmat[m.body(f"{side}_foot").id].reshape(3, 3)[:, 0]
    return np.degrees(np.arctan2(fx[2], np.hypot(fx[0], fx[1])))


def _set_pitch(m, d, q, side, want):
    k = LEGS[side] + 4
    lo, hi = m.jnt_range[m.joint(f"{side}_ankle_y").id]
    for _ in range(4):
        p0 = _foot_pitch(m, d, q, side)
        q2 = q.copy(); q2[k] += 0.01
        slope = (_foot_pitch(m, d, q2, side) - p0) / 0.01
        if abs(slope) < 1e-6:
            break
        q[k] = np.clip(q[k] + (want - p0) / slope, lo, hi)
    return q


ARM_FWD = {"right": ((0.16, -0.24, -0.24), (0.36, -0.13, 0.04)), "left": ((0.16, 0.24, -0.24), (0.36, 0.13, 0.04))}
ARM_BACK = {"right": ((-0.30, -0.25, -0.10), (-0.30, -0.26, -0.45)), "left": ((-0.30, 0.25, -0.10), (-0.30, 0.26, -0.45))}


def sprinter(seconds=11.5, x0=0.0, y=LANE_Y, hold=0.0):
    """Replay arrays for the sprinter: `hold` seconds standing on the line, then the race."""
    m = poses.model()
    d = mujoco.MjData(m)
    F, longest = footfalls()
    pel = m.body("pelvis").id
    q = poses.standing(m, d, x0, y)
    q_stand = q.copy()
    qs = [q_stand.copy() for _ in range(int(hold * FPS))]
    n = int(seconds * FPS)
    ts = np.arange(n) / FPS
    xs = distance(ts)
    speed = np.gradient(xs, 1 / FPS)
    for i, t in enumerate(ts):
        b, v = float(xs[i]), float(speed[i])
        lean = 9 + 26 * np.exp(-t / 0.9) * (t > REACTION * 0.5)
        feet = {s: _foot(F, s, b, v) for s in ("right", "left")}
        down = [s for s in feet if feet[s][3]]
        # hips: lowest mid-stance, highest in the air
        if down:
            u = np.clip((b - (feet[down[0]][0] - LAND)) / (LAND + PUSH), 0, 1)
            hip_z = 0.955 - 0.025 * np.sin(np.pi * u) * min(1.0, v / 4)
        else:
            hip_z = 0.955 + 0.02 * min(1.0, v / 6)
        hip_z -= 0.03 * min(1.0, v / 5)                       # runs a little lower than he stands
        a = np.radians(lean) / 2
        q = q.copy()
        q[3:7] = (np.cos(a), 0.0, np.sin(a), 0.0)            # lean forward from the ankles
        q[7:10] = 0.0
        for _ in range(2):                                    # root so the hips sit at (b, y, hip_z)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            q[0:3] += np.array([x0 + b, y, hip_z]) - d.xpos[pel]
        for s in ("right", "left"):
            fx, fz, pitch, _ = feet[s]
            target = np.array([x0 + fx, y + (FOOT_Y if s == "left" else -FOOT_Y), fz])
            q = _leg_ik(m, d, q, s, target)
            q = _set_pitch(m, d, q, s, pitch)
        # arms against the legs: right arm forward while the right foot is behind him
        for s, other in (("right", "left"), ("left", "right")):
            rel = np.clip((feet[s][0] - b) / 0.9, -1, 1)
            amt = (1 - rel) / 2 * min(1.0, 0.3 + v / 6) + (1 - min(1.0, 0.3 + v / 6)) * 0.5
            (ef, hf), (eb, hb) = ARM_FWD[s], ARM_BACK[s]
            el = np.array(eb) + amt * (np.array(ef) - np.array(eb))
            ha = np.array(hb) + amt * (np.array(hf) - np.array(hb))
            q = poses._solve_arm(m, d, q, s, el, ha, iters=20)
        qs.append(q.copy())
    print(f"  sprinter: {STEPS} steps to the line, longest {longest:.2f} m, top speed {speed.max():.1f} m/s")
    return poses.kinematics(m, qs)


if __name__ == "__main__":
    F, mx = footfalls()
    print("longest step", round(mx, 2), "41st footfall", round(F[STEPS - 1], 2))
    for t in (0.5, 1.0, 2.0, 4.0, 6.0, 9.58, 10.5):
        print(t, float(distance(t)))
