"""Real-runner recordings (CMU motion capture) -> joint positions in metres.

The CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu) stores each person's
skeleton in an .asf file and each recording in an .amc file (joint angles, 120 frames/s).
This module reads them and works out where every joint is, frame by frame.

Licence: CMU data may be used in commercial products but not resold as data.
Credit: "The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217."
"""
from pathlib import Path

import numpy as np

CMU_DIR = Path(__file__).parent.parent / "mocap" / "cmu"
INCH = 0.0254
CMU_LENGTH_SCALE = 0.45   # .asf lengths are inches x 0.45

# Straight running trials we use (from the CMU 'run' search, 2026-11-08).
RUN_TRIALS = (
    [f"09_{i:02d}" for i in range(1, 12) if i != 10]   # subject 9: run (09_10 has a glitch: ankle 0.9 m underground)
    + [f"16_{i}" for i in (35, 36, 45, 46, 55, 56)]   # subject 16: run/jog
    + [f"35_{i}" for i in range(17, 27)]               # subject 35: run/jog
)


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_xyz(x, y, z):
    """ASF/AMC convention: rotate about x, then y, then z (fixed axes) = Rz @ Ry @ Rx."""
    return _rot("z", z) @ _rot("y", y) @ _rot("x", x)


class Skeleton:
    def __init__(self, asf_path):
        self.bones = {}
        self.children = {}
        self.root_order = []
        self._parse(Path(asf_path).read_text().splitlines())

    def _parse(self, lines):
        section, bone = None, None
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(":"):
                section = line.split()[0][1:]
                if section == "root":
                    pass
                continue
            parts = line.split()
            if section == "root":
                if parts[0] == "order":
                    self.root_order = [p.lower() for p in parts[1:]]
                elif parts[0] == "axis":
                    self.root_axis = parts[1]
            elif section == "bonedata":
                if parts[0] == "begin":
                    bone = {"dof": [], "axis": np.eye(3)}
                elif parts[0] == "end":
                    self.bones[bone["name"]] = bone
                elif parts[0] == "name":
                    bone["name"] = parts[1]
                elif parts[0] == "direction":
                    bone["direction"] = np.array([float(v) for v in parts[1:4]])
                elif parts[0] == "length":
                    bone["length"] = float(parts[1]) / CMU_LENGTH_SCALE * INCH
                elif parts[0] == "axis":
                    x, y, z = (float(v) for v in parts[1:4])
                    bone["axis"] = _euler_xyz(x, y, z)
                elif parts[0] == "dof":
                    bone["dof"] = [p.lower() for p in parts[1:]]
            elif section == "hierarchy":
                if parts[0] in ("begin", "end"):
                    continue
                self.children[parts[0]] = parts[1:]

    def joint_positions(self, frame):
        """frame: {bone_name: [values]} from one AMC frame -> {bone_name: end position (m)}, CMU axes (y up)."""
        root = dict(zip(self.root_order, frame["root"]))
        root_pos = np.array([root["tx"], root["ty"], root["tz"]]) / CMU_LENGTH_SCALE * INCH
        root_rot = _euler_xyz(root.get("rx", 0.0), root.get("ry", 0.0), root.get("rz", 0.0))
        pos, rot = {"root": root_pos}, {"root": root_rot}
        stack = ["root"]
        while stack:
            parent = stack.pop()
            for child in self.children.get(parent, []):
                b = self.bones[child]
                ang = dict(zip(b["dof"], frame.get(child, [])))
                local = _euler_xyz(ang.get("rx", 0.0), ang.get("ry", 0.0), ang.get("rz", 0.0))
                C = b["axis"]
                rot[child] = rot[parent] @ C @ local @ C.T
                pos[child] = pos[parent] + rot[child] @ (b["direction"] * b["length"])
                stack.append(child)
        return pos


def read_amc(path):
    frames, cur = [], None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) == 1:
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[parts[0]] = [float(v) for v in parts[1:]]
    return frames


# CMU (y up) -> MuJoCo (z up): a proper rotation, x stays x.
CMU_TO_MJ = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])

# Rupert body (whose origin is that joint) <- CMU bone whose END is that joint.
MARKERS = {
    "pelvis": "root",
    "right_thigh": "rhipjoint", "left_thigh": "lhipjoint",
    "right_shin": "rfemur", "left_shin": "lfemur",
    "right_foot": "rtibia", "left_foot": "ltibia",
    "right_upper_arm": "rclavicle", "left_upper_arm": "lclavicle",
    "right_lower_arm": "rhumerus", "left_lower_arm": "lhumerus",
    "right_hand": "rradius", "left_hand": "lradius",
    "head": "upperneck",
    "torso": "thorax",
}
TOES = {"right": "rfoot", "left": "lfoot"}   # end of CMU 'foot' bone = ball of the foot


def load_trial(trial, fps_out=50.0):
    """One recording -> dict of marker positions (frames, 3) in MuJoCo axes at fps_out,
    turned so the runner runs along +x, starting at x = 0."""
    subject = trial.split("_")[0]
    skel = Skeleton(CMU_DIR / f"{subject}.asf")
    frames = read_amc(CMU_DIR / f"{trial}.amc")
    names = list(MARKERS.values()) + list(TOES.values())
    raw = {n: [] for n in names}
    for f in frames:
        p = skel.joint_positions(f)
        for n in names:
            raw[n].append(CMU_TO_MJ @ p[n])
    raw = {n: np.array(v) for n, v in raw.items()}
    # resample 120 fps -> fps_out
    t_in = np.arange(len(frames)) / 120.0
    t_out = np.arange(0, t_in[-1], 1.0 / fps_out)
    res = {n: np.stack([np.interp(t_out, t_in, v[:, k]) for k in range(3)], 1) for n, v in raw.items()}
    # turn to run along +x
    root = res["root"]
    d = root[-1, :2] - root[0, :2]
    yaw = np.arctan2(d[1], d[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    start = root[0] * np.array([1, 1, 0])
    out = {n: (v - start) @ R.T for n, v in res.items()}
    return out, skel
