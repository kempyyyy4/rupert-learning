"""Fit the CMU running recordings onto Rupert's body ('retargeting').

For every frame, find the pose of Rupert (root position, orientation and all 21 joint
angles) whose joints sit as close as possible to the real runner's joints, after scaling
the runner to Rupert's leg length. Joint limits are respected.

Run:  python -m rupert.retarget          -> mocap/run_reference.npz (+ a check video)
Output: qpos per frame for each clip (50 frames/s), used by the style judge.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from rupert import env as rupert_env
from rupert import mocap

OUT = Path(__file__).parent.parent / "mocap" / "run_reference.npz"
FPS = 50.0
# Fit by SEGMENT DIRECTIONS (Rupert's proportions differ from the runners'), plus pelvis position.
# Each entry: (Rupert start, Rupert end, CMU start, CMU end, weight). "toe" = front of that foot.
SEGMENTS = [
    ("left_thigh", "right_thigh", "lhipjoint", "rhipjoint", 2.0),          # pelvis orientation
    ("pelvis", "torso", "root", "thorax", 2.0),                            # spine lean
    ("torso", "head", "thorax", "upperneck", 0.5),
    ("left_upper_arm", "right_upper_arm", "lclavicle", "rclavicle", 1.0),  # shoulder line (twist)
    *[(f"{s}_thigh", f"{s}_shin", f"{c}hipjoint", f"{c}femur", 3.0) for s, c in (("right", "r"), ("left", "l"))],
    *[(f"{s}_shin", f"{s}_foot", f"{c}femur", f"{c}tibia", 3.0) for s, c in (("right", "r"), ("left", "l"))],
    *[(f"{s}_foot", f"{s}_toe", f"{c}tibia", f"{c}foot", 2.0) for s, c in (("right", "r"), ("left", "l"))],
    *[(f"{s}_upper_arm", f"{s}_lower_arm", f"{c}clavicle", f"{c}humerus", 1.0) for s, c in (("right", "r"), ("left", "l"))],
    *[(f"{s}_lower_arm", f"{s}_hand", f"{c}humerus", f"{c}radius", 1.0) for s, c in (("right", "r"), ("left", "l"))],
]
PELVIS_WEIGHT = 3.0
POSE_PULL = 0.05   # tiny pull towards a neutral pose, so joints the data can't see don't drift


def _toe_offsets(m):
    """Point at the front of each foot (ball of the foot), in the foot body's own frame."""
    out = {}
    for side in ("right", "left"):
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body(f"{side}_foot").id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                q = np.zeros(9); mujoco.mju_quat2Mat(q, m.geom_quat[g]); axis = q.reshape(3, 3)[:, 2]
                ends = [m.geom_pos[g] + axis * m.geom_size[g, 1], m.geom_pos[g] - axis * m.geom_size[g, 1]]
                pts.append(max(ends, key=lambda p: p[0]))
        out[side] = np.mean(pts, axis=0)
    return out


def _rupert_leg_length(m, d):
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    return np.linalg.norm(d.xpos[m.body("left_thigh").id] - d.xpos[m.body("left_foot").id])


def _point(m, d, name, toe_off):
    """Position of a Rupert point and its Jacobian (3 x nv)."""
    jacp = np.zeros((3, m.nv))
    if name.endswith("_toe"):
        b = m.body(name.replace("_toe", "_foot")).id
        p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ toe_off[name.split("_")[0]]
        mujoco.mj_jac(m, d, jacp, None, p, b)
    else:
        b = m.body(name).id
        p = d.xpos[b].copy()
        mujoco.mj_jacBody(m, d, jacp, None, b)
    return p, jacp


def _ground_gap(m, d, qs):
    """How high the feet float above the floor in this clip (10th percentile of the lowest foot point)."""
    low = []
    for q in qs:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        low.append(min(d.geom_xpos[m.geom(g).id][2] - m.geom_size[m.geom(g).id][0]
                       for pair in rupert_env.FOOT_GEOMS for g in pair))
    return float(np.percentile(low, 10))


def rest_lengths(m, d, toe_off):
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    return [np.linalg.norm(_point(m, d, b, toe_off)[0] - _point(m, d, a, toe_off)[0]) for a, b, *_ in SEGMENTS]


def fit_clip(m, d, mk, pelvis_target, toe_off, lengths, q_init, iters=50):
    """Per frame: pelvis at the target, every segment pointing the same way as the runner's."""
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    q = q_init.copy()
    out, errs = [], []
    for f in range(len(pelvis_target)):
        for it in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p, Jp = _point(m, d, "pelvis", toe_off)
            res, J = [PELVIS_WEIGHT * (p - pelvis_target[f])], [PELVIS_WEIGHT * Jp]
            seg_err = []
            for (a, b, ca, cb, w), L in zip(SEGMENTS, lengths):
                pa, Ja = _point(m, d, a, toe_off); pb, Jb = _point(m, d, b, toe_off)
                v = mk[cb][f] - mk[ca][f]
                r = (pb - pa) - L * v / np.linalg.norm(v)
                res.append(w * r); J.append(w * (Jb - Ja)); seg_err.append(np.linalg.norm(r) / L)
            pose = np.zeros(m.nv); pose[6:] = q[7:] - m.qpos0[7:]
            Jpose = np.zeros((m.nv, m.nv)); Jpose[6:, 6:] = np.eye(m.nv - 6)
            res.append(POSE_PULL * pose); J.append(POSE_PULL * Jpose)
            r = np.concatenate(res); J = np.concatenate(J)
            dq = -np.linalg.solve(J.T @ J + 1e-3 * np.eye(m.nv), J.T @ r)
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[7:] = np.clip(q[7:], lo, hi)
            if np.abs(dq).max() < 1e-5:
                break
        out.append(q.copy())
        errs.append(np.degrees(np.arcsin(np.clip(np.mean(seg_err), 0, 1))))   # rough mean angle error
    return np.array(out), np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="also render a check video")
    args = ap.parse_args()

    env = rupert_env.load()
    m = env.mj_model
    d = mujoco.MjData(m)
    toe_off = _toe_offsets(m)
    rupert_leg = _rupert_leg_length(m, d)
    mujoco.mj_resetData(m, d)
    q0 = m.qpos0.copy()
    lengths = rest_lengths(m, d, toe_off)

    clips, starts, report = [], [0], []
    for trial in mocap.RUN_TRIALS:
        mk, skel = mocap.load_trial(trial, FPS)
        subj_leg = skel.bones["lfemur"]["length"] + skel.bones["ltibia"]["length"]
        s = rupert_leg / subj_leg
        pelvis = mk["root"] * s
        qi = q0.copy(); qi[:3] = pelvis[0]
        qs, err = fit_clip(m, d, mk, pelvis, toe_off, lengths, qi)
        qs[:, 2] -= _ground_gap(m, d, qs)   # stand on the floor (proportions differ, so the height doesn't scale exactly)
        clips.append(qs); starts.append(starts[-1] + len(qs))
        speed = (qs[-1, 0] - qs[0, 0]) / (len(qs) / FPS)
        report.append((trial, s, speed, err.mean(), err.max()))
        print(f"{trial}: scale {s:.2f}  speed {speed:.1f} m/s  direction error mean {err.mean():.1f} deg, worst {err.max():.1f} deg", flush=True)

    qpos = np.concatenate(clips)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, qpos=qpos.astype(np.float32), starts=np.array(starts), fps=np.array(FPS),
                        trials=np.array(mocap.RUN_TRIALS))
    print(f"Saved {OUT}: {len(clips)} clips, {len(qpos)} frames ({len(qpos)/FPS:.0f} s of running)")

    if args.video:
        import mediapy, imageio_ffmpeg
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        frames = []
        with mujoco.Renderer(m, height=480, width=640) as r:
            for q in qpos:
                d.qpos[:] = q; mujoco.mj_forward(m, d)
                r.update_scene(d, camera="side"); frames.append(r.render())
        mediapy.write_video(str(OUT.with_suffix(".mp4")), frames, fps=FPS)
        print(f"Check video: {OUT.with_suffix('.mp4')}")


if __name__ == "__main__":
    main()
