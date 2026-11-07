"""Saves Rupert's body shapes (from the physics model) to blender/rupert_shapes.json.

  python -m rupert.export_shapes

Blender can't load MuJoCo, so it reads this file to build the visual model with
limbs exactly where the physics limbs are. Re-run it whenever rupert/body.py changes.
"""
import json
from pathlib import Path

import mujoco

from rupert import env as rupert_env

OUT = Path(__file__).parent.parent / "blender" / "rupert_shapes.json"
TYPES = {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}


def main():
    m = rupert_env.load().mj_model
    shapes = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if body == "world" or m.geom_type[g] not in TYPES:
            continue
        shapes.append({
            "name": m.geom(g).name,
            "body": body,
            "type": TYPES[m.geom_type[g]],
            "radius": float(m.geom_size[g][0]),
            "half_length": float(m.geom_size[g][1]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0,
            "pos": [float(v) for v in m.geom_pos[g]],
            "quat": [float(v) for v in m.geom_quat[g]],   # w, x, y, z; capsule runs along local z
        })
    OUT.write_text(json.dumps(shapes, indent=1))
    print(f"Saved {len(shapes)} shapes to {OUT}")


if __name__ == "__main__":
    main()
