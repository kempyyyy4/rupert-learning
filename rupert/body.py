"""Builds Rupert's physics body: the standard MuJoCo humanoid, rescaled to human size.

Run `python -m rupert.body` to regenerate rupert/assets/rupert.xml.
Every number below is a claim we may say on camera, so each has a source in
docs/body-sources.md. Anything marked TO VERIFY must be checked before Ep 2.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from mujoco_playground._src import mjx_env

SOURCE_XML = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
OUT_XML = Path(__file__).parent / "assets" / "rupert.xml"

# --- Size -----------------------------------------------------------------
STOCK_HEIGHT_M = 1.56   # measured from the stock model (feet to top of head)
HEIGHT_M = 1.85         # Rupert: tall and thin
MASS_KG = 70.0          # lean adult male
SCALE = HEIGHT_M / STOCK_HEIGHT_M   # lengths scale by this; limb thickness does not (thin build)

# --- Mass split: de Leva (1996), adult male segment mass as % of body mass ---
# Trunk 43.46% is split into upper / middle / lower trunk (15.96 / 16.33 / 11.17).
SEGMENT_MASS_PCT = {
    "head": 6.94,
    "torso": 15.96, "lower_waist": 16.33, "pelvis": 11.17,
    "thigh": 14.16, "shin": 4.33, "foot": 1.37,
    "upper_arm": 2.71, "lower_arm": 1.62, "hand": 0.61,
}

# --- Peak joint torques (N·m), strong trained adult. TO VERIFY against sources ---
# These are the most each joint motor can ever push. Force–velocity limits are
# applied on top in rupert/env.py (faster joint = less torque, like real muscle).
JOINT_TORQUE_NM = {
    "abdomen_y": 200, "abdomen_z": 100, "abdomen_x": 150,
    "hip_x": 150, "hip_z": 80, "hip_y": 300,
    "knee": 250,
    "ankle_y": 250, "ankle_x": 60,
    "shoulder1": 80, "shoulder2": 80, "elbow": 60,
}


def _scale_attr(el, name, k):
    if name in el.attrib:
        el.set(name, " ".join(f"{float(v) * k:.4g}" for v in el.attrib[name].split()))


def _segment(body_name):
    for side in ("left_", "right_"):
        if body_name.startswith(side):
            return body_name[len(side):]
    return body_name


def build():
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    root.set("model", "rupert")
    torso = root.find("./worldbody/body[@name='torso']")

    # 1. Scale lengths of everything on the body (positions and capsule end points).
    for el in torso.iter():
        if el.tag in ("body", "geom", "joint", "site", "camera", "light"):
            _scale_attr(el, "pos", SCALE)
            _scale_attr(el, "fromto", SCALE)
        if el.tag == "geom" and el.get("type") == "sphere":   # head and hands grow a little
            _scale_attr(el, "size", SCALE ** 0.5)
        if el.tag == "joint":   # heavier body needs a bit more joint inertia to stay stable
            el.set("armature", "0.02")

    # 2. Masses: put each segment's de Leva share on its geoms.
    for body in torso.iter("body"):
        geoms = body.findall("geom")
        seg = _segment(body.get("name"))
        if seg not in SEGMENT_MASS_PCT:
            raise ValueError(f"no mass for body {body.get('name')}")
        mass = MASS_KG * SEGMENT_MASS_PCT[seg] / 100
        for g in geoms:
            g.set("mass", f"{mass / len(geoms):.4g}")

    # 3. Motor strengths: gear = peak torque, since control is -1..1.
    for m in root.find("actuator"):
        key = _segment(m.get("joint"))
        m.set("gear", str(JOINT_TORQUE_NM[key]))

    # 4. Physics accuracy (tested 2026-11-06): 4 solver iterations hit the limit on ~37%
    #    of steps; 10 never did. Slightly stiffer contacts reduce sinking into the floor.
    opt = root.find("option")
    opt.set("timestep", "0.004")
    opt.set("iterations", "10")
    opt.set("ls_iterations", "50")   # 20 still hit the limit during falls; 50 never did
    for el in root.iter():
        if el.get("solref") == ".015 1":
            el.set("solref", ".008 1")

    # 5. Start pose: lower the body so the feet just touch the floor
    #    (the stock model starts ~0.26 m in the air and drops).
    torso.set("pos", f"0 0 {1.5 * SCALE - 0.255:.4g}")
    root.find("statistic").set("extent", "3")

    OUT_XML.parent.mkdir(exist_ok=True)
    ET.indent(tree)
    tree.write(OUT_XML, encoding="unicode")
    return OUT_XML


if __name__ == "__main__":
    print(f"Wrote {build()}")
