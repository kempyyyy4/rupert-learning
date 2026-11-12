"""Gait check: how a replay's running actually looks, in numbers (so cheats can be spotted
without watching every clip). Also measures the real-runner recordings for comparison.

    python -m rupert.gait outputs/<run>/eval/step_XXXX.npz [more.npz ...] [--ref]
"""
import sys

import numpy as np


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def analyse(xpos, xquat, names, fps=50, skip_s=3.0):
    """xpos (T, bodies, 3) for one attempt. Returns a dict of gait numbers."""
    b = {n: i for i, n in enumerate(names)}
    P = lambda n: xpos[:, b[n]]
    s = int(skip_s * fps)
    xpos, xquat = xpos[s:], xquat[s:]
    out = {}
    feet = {side: xpos[:, b[f"{side}_foot"]] for side in ("left", "right")}
    lands = []
    for side, other in (("left", "right"), ("right", "left")):
        z = feet[side][:, 2]
        down = z < z.min() + 0.04
        for t in np.flatnonzero(down[1:] & ~down[:-1]) + 1:
            lands.append((t, side, feet[side][t, 0] - feet[other][t, 0]))
    lands.sort()
    for side in ("left", "right"):
        d = [o for _, sd, o in lands if sd == side]
        out[f"{side}_step_m"] = float(np.mean(d)) if d else float("nan")
        out[f"{side}_steps"] = len(d)
    secs = len(xpos) / fps
    out["steps_per_s"] = len(lands) / secs
    alternate = sum(1 for a, c in zip(lands, lands[1:]) if a[1] != c[1])
    out["alternating"] = alternate / max(1, len(lands) - 1)
    up = np.array([_rot(q)[:, 2] for q in xquat[:, b["torso"]]])
    out["lean_deg"] = float(np.degrees(np.mean(np.arctan2(up[:, 0], up[:, 2]))))   # + = forward
    out["head_m"] = float(xpos[:, b["head"], 2].mean())
    pel = xpos[:, b["pelvis"], 0]
    corr = []
    for hand, foot in (("right", "left"), ("left", "right")):
        h = xpos[:, b[f"{hand}_hand"], 0] - pel
        f = xpos[:, b[f"{foot}_foot"], 0] - pel
        corr.append(np.corrcoef(h, f)[0, 1])
    out["arm_leg_corr"] = float(np.mean(corr))   # +1 = hand swings with the OPPOSITE foot (human)
    out["hand_swing_m"] = float(np.mean([np.ptp(xpos[:, b[f"{s}_hand"], 0] - pel) for s in ("left", "right")]))
    for side in ("left", "right"):
        hip, knee, ank = (xpos[:, b[f"{side}_{n}"]] for n in ("thigh", "shin", "foot"))
        u, v = knee - hip, ank - knee
        ang = np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
        out[f"{side}_knee_max_deg"] = float(np.percentile(ang, 95))
    out["speed_ms"] = float((pel[-1] - pel[0]) / secs)
    return out


def from_replay(path, attempt=0):
    z = np.load(path, allow_pickle=True)
    alive = z["alive"][:, attempt].astype(bool)
    end = int(np.argmin(alive)) if not alive.all() else len(alive)
    return analyse(z["xpos"][:end, attempt], z["xquat"][:end, attempt], [str(n) for n in z["body_names"]])


def reference():
    """Each real-runner recording, measured the same way."""
    from pathlib import Path
    from rupert import poses
    z = np.load(Path(__file__).parent.parent / "mocap" / "run_reference.npz")
    m = poses.model()
    res = []
    for k, trial in enumerate(z["trials"]):
        q = z["qpos"][z["starts"][k]:z["starts"][k + 1]].astype(np.float64)
        if len(q) < 60:
            continue
        xp, xq, names = poses.kinematics(m, q)
        res.append((str(trial), analyse(xp, xq, names, skip_s=0.0)))
    return res


def show(label, g):
    print(f"{label:28s} speed {g['speed_ms']:4.1f}  steps/s {g['steps_per_s']:3.1f}  alt {g['alternating']:.2f}  "
          f"step L {g['left_step_m']:+.2f} R {g['right_step_m']:+.2f}  lean {g['lean_deg']:+5.1f}  head {g['head_m']:.2f}  "
          f"arms {g['arm_leg_corr']:+.2f} ({g['hand_swing_m']:.2f} m)  knee L {g['left_knee_max_deg']:3.0f} R {g['right_knee_max_deg']:3.0f}")


if __name__ == "__main__":
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        for a in range(2):
            show(f"{p.split('/')[-3][-13:]} {p[-12:-4]} #{a}", from_replay(p, a))
    if "--ref" in sys.argv:
        rows = reference()
        for t, g in rows:
            show(f"real {t}", g)
