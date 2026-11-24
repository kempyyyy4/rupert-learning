"""EP01's 'coming up in this series' preview clips (about 2-3 s each), all hand animation:

  bike     Rupert wobbling along the track on a bicycle, pedalling
  date     Rupert at a candlelit table for two, at night, smiling and giving a little wave
  trading  Rupert at a desk of screens, typing like mad, then both arms up as the numbers go green

Poses come from rupert/poses.py and the exact two-bone leg solver in rupert/sprint.py. The sets
(bike, table, desk, screens) are built in blender/sets.py from the numbers here.
"""
import mujoco
import numpy as np

from rupert import poses, sprint

FPS = 50

# ---- bike (bike frame: x forward, z up, origin on the ground under the crank) ----------
WHEEL_R = 0.34
CRANK = np.array([0.0, 0.0, 0.30])
CRANK_L = 0.17
SADDLE = np.array([-0.22, 0.0, 0.95])
BARS = np.array([0.40, 0.0, 1.04])
GRIP_Y = 0.25
WHEELBASE = (-0.52, 0.56)            # rear and front axle x


def _rigid(xp, xq, R, p, qrot):
    """Move a whole posed body (xpos, xquat for one frame) by rotation R / quaternion qrot and offset p."""
    xp = xp @ R.T + p
    xq = np.array([poses._quat_mul(qrot, q) for q in xq])
    return xp, xq


def _rot(roll, pitch, yaw):
    q = poses._quat_mul([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                        poses._quat_mul([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0],
                                        [np.cos(roll / 2), np.sin(roll / 2), 0, 0]))
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3), q


def bike(seconds=3.0, x0=6.0, speed=2.6):
    """Rupert wobbling along on a bike. Returns replay arrays and the bike's own keys per frame:
    [location, quaternion, wheel angle, crank angle]."""
    m = poses.model()
    d = mujoco.MjData(m)
    pel = m.body("pelvis").id
    lean = np.radians(22)
    rng = np.random.default_rng(2)
    qs_local, bike_keys = [], []
    q = poses.standing(m, d)
    for i in range(int(seconds * FPS)):
        t = i / FPS
        dist = speed * t
        crank = dist * 2 * np.pi / 4.2                       # one turn of the pedals every 4.2 m
        # body on the bike (bike frame): hips on the saddle, leaning forward
        q = q.copy()
        q[3:7] = (np.cos(lean / 2), 0.0, np.sin(lean / 2), 0.0)
        q[7:] = 0.0
        q[8] = np.radians(-4) * np.sin(crank * 2)            # a little rock at the waist as he pushes
        for _ in range(2):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            q[0:3] += SADDLE + np.array([0, 0, 0.09]) - d.xpos[pel]
        for side, ph in (("right", 0.0), ("left", np.pi)):
            a = crank + ph
            pedal = CRANK + CRANK_L * np.array([np.sin(a), 0.0, np.cos(a)])
            pedal[1] = -0.12 if side == "right" else 0.12
            q = sprint._leg_ik(m, d, q, side, pedal + np.array([-0.04, 0.0, 0.09]))
            q = sprint._set_pitch(m, d, q, side, -8 + 14 * np.sin(a))
        for side, sy in (("right", -1), ("left", 1)):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            T = d.xmat[m.body("torso").id].reshape(3, 3)
            p0 = d.xpos[m.body("torso").id]
            hand = T.T @ (BARS + np.array([0, sy * GRIP_Y, 0]) - p0)
            elbow = poses.arm_for_fist(hand, (-0.3, sy * 0.6, -0.7), side, reach=0.0)[0]
            q = poses._solve_arm(m, d, q, side, elbow, hand, iters=30)
        qs_local.append(q.copy())
        # the wobble: the whole bike rolls side to side, steering a little to catch itself
        roll = np.radians(7) * np.sin(2 * np.pi * 0.55 * t) + np.radians(3) * np.sin(2 * np.pi * 1.7 * t + 1.0)
        yaw = -0.6 * roll + np.radians(2) * rng.normal() * 0
        y = 0.18 * np.sin(2 * np.pi * 0.55 * t - 0.8)
        R, qr = _rot(roll, 0.0, yaw)
        bike_keys.append([[x0 + dist, y, 0.0], [float(v) for v in qr], dist / WHEEL_R, crank])   # + about y rolls forward
    xp, xq, names = poses.kinematics(m, qs_local)
    for f, (loc, qr, _, _) in enumerate(bike_keys):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.array(qr)); R = R.reshape(3, 3)
        xp[f], xq[f] = _rigid(xp[f], xq[f], R, np.array(loc), qr)
    return xp, xq, names, bike_keys


# ---- the date: seated at a table for two --------------------------------------------------
TABLE = np.array([60.0, -0.2])        # table centre on the track (x, y); Rupert sits on its -x side
TABLE_H = 0.76
HANDS_ON_TABLE = {"right": ((0.12, -0.27, -0.32), (0.45, -0.17, -0.27)),
                  "left": ((0.12, 0.27, -0.32), (0.45, 0.17, -0.27))}
SMALL_WAVE = ((0.22, -0.34, -0.16), (0.34, -0.22, 0.20))


def date(seconds=3.4):
    """Seated at the table facing his date (the camera), swaying dreamily, then a shy little wave."""
    m = poses.model()
    d = mujoco.MjData(m)
    at = (TABLE[0] - 0.62, TABLE[1])
    q0 = poses._seated(m, d, at, 0.0)
    qs, q = [], q0.copy()
    for side in ("right", "left"):
        q = poses._solve_arm(m, d, q, side, *HANDS_ON_TABLE[side], iters=120)
    rest = q.copy()
    for i in range(int(seconds * FPS)):
        t = i / FPS
        up = poses._ease((t - 1.5) / 0.4) * (1 - poses._ease((t - 2.9) / 0.4))
        wig = np.sin(2 * np.pi * 2.6 * (t - 1.9)) if 1.9 < t < 2.9 else 0.0
        (e0, h0), (e1, h1) = HANDS_ON_TABLE["right"], SMALL_WAVE
        el = np.array(e0) + up * (np.array(e1) - np.array(e0))
        ha = np.array(h0) + up * (np.array(h1) - np.array(h0)) + up * np.array([0, 0.06 * wig, 0])
        q = poses._solve_arm(m, d, rest.copy(), "right", el, ha, iters=40)
        q[2] = rest[2] + 0.003 * np.sin(2 * np.pi * t / 2.0)
        qs.append(q)
    xp, xq, names = poses.kinematics(m, qs)
    # dreamy head tilt side to side (a roll about his facing direction)
    h = names.index("head")
    for f in range(len(xp)):
        t = f / FPS
        a = np.radians(9) * np.sin(2 * np.pi * t / 2.4)
        xq[f, h] = poses._quat_mul([np.cos(a / 2), np.sin(a / 2), 0, 0], xq[f, h])
    return xp, xq, names


# ---- the trading desk ------------------------------------------------------------------------
DESK = np.array([40.0, -0.3])         # desk centre (x, y); Rupert sits on its -x side facing +x
KEYS_HANDS = {"right": ((0.14, -0.26, -0.33), (0.47, -0.10, -0.29)),
              "left": ((0.14, 0.26, -0.33), (0.47, 0.10, -0.29))}
ARMS_UP = {"right": ((0.05, -0.42, 0.33), (0.10, -0.40, 0.70)), "left": ((0.05, 0.42, 0.33), (0.10, 0.40, 0.70))}


def trading(seconds=3.2, cheer_at=1.9):
    """Typing like mad at the desk, then both arms up in the air as the numbers go green."""
    m = poses.model()
    d = mujoco.MjData(m)
    at = (DESK[0] - 0.62, DESK[1])
    q = poses._seated(m, d, at, 0.0)
    for side in ("right", "left"):
        q = poses._solve_arm(m, d, q, side, *KEYS_HANDS[side], iters=120)
    rest = q.copy()
    qs = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        up = poses._ease((t - cheer_at) / 0.3)
        q = rest.copy()
        for side, ph in (("right", 0.0), ("left", np.pi)):
            (e0, h0), (e1, h1) = KEYS_HANDS[side], ARMS_UP[side]
            tap = 0.025 * max(0.0, np.sin(2 * np.pi * 7 * t + ph)) * (1 - up)
            el = np.array(e0) + up * (np.array(e1) - np.array(e0))
            ha = np.array(h0) + up * (np.array(h1) - np.array(h0)) + np.array([0, 0, tap])
            if up > 0:
                ha = ha + up * np.array([0, 0, 0.05 * np.sin(2 * np.pi * 3 * (t - cheer_at))])
            q = poses._solve_arm(m, d, q, side, el, ha, iters=30)
        q[2] = rest[2] + 0.02 * up * abs(np.sin(2 * np.pi * 1.5 * (t - cheer_at)))   # bounces in his seat
        qs.append(q)
    return poses.kinematics(m, qs)


def chart_frames(folder, n, kind="line", seed=0, w=960, h=540):
    """Green-numbers-going-up screens, one picture per frame (f_0001.png ...)."""
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pts = np.cumsum(rng.normal(0.25, 1.0, 400)) + np.linspace(0, 60, 400) ** 1.5 / 8
    try:
        big = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", int(h * 0.16))
        small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", int(h * 0.07))
    except OSError:
        big = small = ImageFont.load_default()
    green, dark, grid = (40, 230, 90), (8, 12, 18), (30, 45, 40)
    tickers = ["RUPRT", "TRACK", "SHOES", "PIZZA", "MOON", "BANANA"]
    for f in range(n):
        u = f / max(1, n - 1)
        img = Image.new("RGB", (w, h), dark)
        dr = ImageDraw.Draw(img)
        for k in range(1, 6):
            dr.line([(0, k * h / 6), (w, k * h / 6)], fill=grid)
            dr.line([(k * w / 6, 0), (k * w / 6, h)], fill=grid)
        if kind == "line":
            m_ = int(60 + 340 * u)
            ys = pts[:m_]
            lo, hi = ys.min(), ys.max() + 1e-6
            xy = [(w * 0.04 + w * 0.92 * i / 400, h * 0.92 - (y - lo) / (hi - lo) * h * 0.55) for i, y in enumerate(ys)]
            dr.line(xy, fill=green, width=max(3, w // 200))
            money = int(1000 + 1203553 * u ** 3)
            dr.text((w * 0.05, h * 0.05), f"${money:,}", font=big, fill=green)
            dr.text((w * 0.05, h * 0.24), f"+{int(12 + 900 * u ** 2)}%", font=small, fill=green)
        elif kind == "candles":
            k = int(8 + 22 * u)
            base = 0.8
            for i in range(k):
                up_ = rng.random() < 0.8
                y0 = h * (base - 0.018 * i)
                y1 = y0 - h * rng.uniform(0.02, 0.07) * (1 if up_ else -1)
                x = w * 0.05 + i * w * 0.03
                col = green if up_ else (220, 50, 50)
                dr.line([(x + w * 0.009, min(y0, y1) - 10), (x + w * 0.009, max(y0, y1) + 10)], fill=col, width=2)
                dr.rectangle([x, min(y0, y1), x + w * 0.018, max(y0, y1)], fill=col)
            rng = np.random.default_rng(seed)
        else:                                            # tickers, all going up
            for i, t in enumerate(tickers):
                val = 10 + 7 * i + 150 * u * (1 + i * 0.3)
                dr.text((w * 0.06, h * (0.08 + 0.15 * i)), f"{t:<7}{val:8.2f}  +{int(40 * u * (i + 1))}%", font=small, fill=green)
        img.save(folder / f"f_{f + 1:04d}.png")
    return folder / "f_0001.png"
