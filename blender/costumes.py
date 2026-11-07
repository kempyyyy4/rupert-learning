"""Rupert's outfits: a new one for every video. Pick with --costume <name> in build_scene.py.

Each costume is a function that adds clothes to a rupert_model.Body. Clothes that bend
(vest, shorts) are made like his skin: smooth blobs around the limbs, glued to the skeleton.
Rigid bits (shoes, bands) are simply attached to one body part.
"""
import math

from mathutils import Matrix, Quaternion, Vector

import bl_helpers as H

ORANGE = (1.0, 0.3, 0.0)
WHITE = (0.93, 0.93, 0.92)


# ---- reusable pieces ---------------------------------------------------------------
def running_shoes(body, upper_rgb, sole_rgb, stripe_rgb, name="shoe"):
    """Chunky running shoes: tapered toe, thick sole, slanted side stripes."""
    sole_m = H.material(f"{name}_sole", sole_rgb, rough=0.55)

    def stripes(nodes, links, bsdf):
        m = H.Maths(nodes, links)
        sep = nodes.new("ShaderNodeSeparateXYZ")
        links.new(nodes.new("ShaderNodeTexCoord").outputs["Object"], sep.inputs[0])
        x, y, z = sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]
        slant = m.op("FRACT", m.op("DIVIDE", m.op("MULTIPLY_ADD", z, -0.7, x), 0.032))
        on = m.op("LESS_THAN", slant, 0.38)
        on = m.op("MULTIPLY", on, m.op("GREATER_THAN", m.op("ABSOLUTE", y), 0.03))
        on = m.op("MULTIPLY", on, m.op("MULTIPLY", m.op("GREATER_THAN", x, -0.08), m.op("LESS_THAN", x, 0.05)))
        on = m.op("MULTIPLY", on, m.op("MULTIPLY", m.op("GREATER_THAN", z, -0.03), m.op("LESS_THAN", z, 0.028)))
        H.two_colour(nodes, links, on, upper_rgb, stripe_rgb, bsdf)

    upper_m = H.node_material(f"{name}_upper", stripes, rough=0.6, sheen=0.3)
    for side in ("right", "left"):
        foot = [s for s in body.shapes if s["body"] == f"{side}_foot"]
        cy = sum(s["pos"][1] for s in foot) / len(foot)
        cx, r, h = foot[0]["pos"][0], foot[0]["radius"], foot[0]["half_length"]
        length = 2 * (h + r) + 0.03
        parent = body.empties[f"{side}_foot"]
        sole = H.mesh_object(f"rupert_{name}_{side}_sole", H.rounded_box_bm((length, 0.115, 0.032), 0.013), sole_m, body.coll)
        H.attach(sole, parent, (cx + 0.005, cy, -r + 0.013))
        bm = H.rounded_box_bm((length - 0.02, 0.1, 0.085), 0.035, 5)
        half = (length - 0.02) / 2
        for v in bm.verts:                       # lower, narrower toe
            t = max(0.0, (v.co.x / half - 0.1) / 0.9)
            v.co.z = -0.0425 + (v.co.z + 0.0425) * (1 - 0.45 * t)
            v.co.y *= 1 - 0.18 * t
        upper = H.mesh_object(f"rupert_{name}_{side}_upper", bm, upper_m, body.coll)
        H.attach(upper, parent, (cx, cy, -r + 0.06))


def wristbands(body, rgb, name="wristband"):
    m = H.material(name, rgb, rough=0.95, sheen=1.0)       # towelling
    for side in ("right", "left"):
        p = body.limb_point(f"{side}_lower_arm", f"{side}_hand", 0.78)
        band = H.mesh_object(f"rupert_{name}_{side}", H.tube_bm(0.047, 0.075), m, body.coll)
        band.modifiers.new("thick", "SOLIDIFY").thickness = 0.01
        H.attach(band, body.empties[f"{side}_lower_arm"], p, H.look_along(p, axis=(0, 0, 1)))


def headband(body, rgb, elev=22, tilt_deg=0, front_gap=0.006, hair_gap=0.026, name="headband"):
    """Towelling headband: snug on the forehead, sitting on top of the hair at the sides and back."""
    from rupert_model import HEAD_LIFT, HEAD_R
    m = H.material(name, rgb, rough=0.95, sheen=1.0)
    e = math.radians(elev)
    rc = HEAD_R * math.cos(e)
    a = rc + (front_gap + hair_gap) / 2          # front-back half width (oval, shifted back)
    shift = (hair_gap - front_gap) / 2
    band = H.mesh_object(f"rupert_{name}", H.tube_bm(1.0, 0.04, 56), m, body.coll)
    band.scale = (a, rc + hair_gap, 1.0)
    band.modifiers.new("thick", "SOLIDIFY").thickness = 0.016
    q = Quaternion((0, 1, 0), math.radians(tilt_deg))
    H.attach(band, body.empties["head"], Vector((0, 0, HEAD_LIFT + HEAD_R * math.sin(e))) + q @ Vector((-shift, 0, 0)), q)


def beanie(body, rgb):
    from rupert_model import HEAD_LIFT, HEAD_R
    m = H.material("beanie", rgb, rough=0.7, sheen=0.5)
    head = body.empties["head"]
    dome = H.mesh_object("rupert_beanie", H.sphere_bm(HEAD_R * 1.1, top_half_only=True), m, body.coll)
    H.attach(dome, head, (0, 0, HEAD_LIFT + 0.1 * HEAD_R))
    cuff = H.mesh_object("rupert_beanie_cuff", H.tube_bm(HEAD_R * 1.13, HEAD_R * 0.38), m, body.coll)
    cuff.modifiers.new("thick", "SOLIDIFY").thickness = 0.012
    H.attach(cuff, head, (0, 0, HEAD_LIFT + 0.22 * HEAD_R))


def striped_vest(body, rgb, stripe_rgb, stripes=(-0.02, -0.065), width=0.013, name="vest"):
    """Running vest / singlet, with optional horizontal chest stripes."""
    def build(nodes, links, bsdf):
        m = H.Maths(nodes, links)
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "rest_z"
        on = 0.0
        for c in stripes:
            band = m.op("COMPARE", attr.outputs["Fac"], c, width)
            on = band if on == 0.0 else m.op("MAXIMUM", on, band)
        H.two_colour(nodes, links, on, rgb, stripe_rgb, bsdf)

    mat = H.node_material(name, build, rough=0.7, sheen=0.4) if stripes else H.material(name, rgb, rough=0.7, sheen=0.4)
    lw = body.shape("lower_waist")
    els = [
        ("capsule", body.chest_m(), 0.124, 0.1),
        ("capsule", body.shape_m("torso"), 0.078, body.shape("torso")["half_length"] * 0.45),  # straps
        ("capsule", body.shape_m("lower_waist"), 0.098, lw["half_length"]),
    ]
    vest = H.metaball_mesh(f"rupert_{name}", els, body.coll)
    H.store_rest_height(vest, body.rest["torso"]["pos"][2])
    vest.data.materials.append(mat)
    H.skin_to_skeleton(vest, ["torso", "lower_waist"], body.segs, body.skeleton)


def short_shorts(body, rgb, leg_frac=0.38, name="shorts"):
    """70s short shorts: waist, seat and the top of each thigh, melted together."""
    els = [
        ("capsule", body.shape_m("lower_waist"), 0.102, body.shape("lower_waist")["half_length"]),
        ("capsule", body.shape_m("butt"), 0.104, body.shape("butt")["half_length"]),
    ]
    for side in ("right", "left"):
        s = body.shape(f"{side}_thigh")
        m = body.m(f"{side}_thigh") @ Matrix.Translation(Vector(s["pos"]) * leg_frac) @ Quaternion(s["quat"]).to_matrix().to_4x4()
        els.append(("capsule", m, 0.086, s["half_length"] * leg_frac))
    body.skinned(f"rupert_{name}", els, ["lower_waist", "pelvis", "right_thigh", "left_thigh"],
                 H.material(name, rgb, rough=0.6, sheen=0.3), blend=0.02)


# ---- the outfits -------------------------------------------------------------------
def ep01_70s(body):
    """EP01 'Learn to run': 70s track star. Orange short shorts, white vest with orange
    stripes, orange towelling headband and wristbands, orange running shoes."""
    short_shorts(body, ORANGE)
    striped_vest(body, WHITE, ORANGE)
    headband(body, ORANGE)
    wristbands(body, ORANGE)
    running_shoes(body, ORANGE, WHITE, WHITE)


def classic(body):
    """The channel look: orange beanie and orange shoes."""
    beanie(body, ORANGE)
    running_shoes(body, ORANGE, WHITE, ORANGE)


COSTUMES = {
    "ep01_70s": {"build": ep01_70s, "hair": {"fringe_down_to": 30}},
    "classic": {"build": classic, "hair": {"fringe_down_to": 20}},
}
