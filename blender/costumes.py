"""Rupert's outfits: a new one for every video. Pick with --costume <name> in build_scene.py.

Each costume is a function that adds clothes to a rupert_model.Body.
- Clothes that bend (vest, shorts) are glued to the skeleton like his skin.
- Loose clothes (shorts) are built as separate fabric with room between it and the skin.
- Rigid bits (shoes, socks, bands) are simply attached to one body part.
"""
import math

import bmesh
import bpy
from mathutils import Vector

import bl_helpers as H

ORANGE = (1.0, 0.3, 0.0)
WHITE = (0.95, 0.95, 0.94)


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


def tube_socks(body, rgb, stripe_rgb, top=-0.12, bottom=-0.42, stripes=((0.088, 0.1), (0.11, 0.122)), name="sock"):
    """Classic 70s tube socks up the calf, with stripes near the top (heights from the top)."""
    def build(nodes, links, bsdf):
        m = H.Maths(nodes, links)
        sep = nodes.new("ShaderNodeSeparateXYZ")
        links.new(nodes.new("ShaderNodeTexCoord").outputs["Object"], sep.inputs[0])
        down = m.op("SUBTRACT", (top - bottom) / 2, sep.outputs["Z"])      # distance below the sock top
        on = 0.0
        for a, b in stripes:
            band = m.op("MULTIPLY", m.op("GREATER_THAN", down, a), m.op("LESS_THAN", down, b))
            on = band if on == 0.0 else m.op("MAXIMUM", on, band)
        H.two_colour(nodes, links, on, rgb, stripe_rgb, bsdf)

    mat = H.node_material(name, build, rough=0.9, sheen=0.8)
    for side in ("right", "left"):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=40, radius1=0.055, radius2=0.061, depth=top - bottom)
        sock = H.mesh_object(f"rupert_{name}_{side}", bm, mat, body.coll)
        sock.modifiers.new("thick", "SOLIDIFY").thickness = 0.004
        H.attach(sock, body.empties[f"{side}_shin"], (0, 0, (top + bottom) / 2))


def wristbands(body, rgb, name="wristband"):
    m = H.material(name, rgb, rough=0.95, sheen=1.0)       # towelling
    for side in ("right", "left"):
        p = body.limb_point(f"{side}_lower_arm", f"{side}_hand", 0.78)
        band = H.mesh_object(f"rupert_{name}_{side}", H.tube_bm(0.047, 0.075), m, body.coll)
        band.modifiers.new("thick", "SOLIDIFY").thickness = 0.01
        H.attach(band, body.empties[f"{side}_lower_arm"], p, H.look_along(p, axis=(0, 0, 1)))


def headband(body, rgb, elev=23, back_elev=30, height_deg=12, gap=0.004, name="headband"):
    """Towelling headband: snug on the forehead, disappearing under the big hair at the sides and back."""
    from rupert_model import direction
    m = H.material(name, rgb, rough=0.95, sheen=1.0)
    n = 64
    bm = bmesh.new()
    rows = []
    for dz in (-height_deg / 2, height_deg / 2):
        row = []
        for i in range(n):
            az = 360 * i / n
            e = elev + (back_elev - elev) * (1 - math.cos(math.radians(az))) / 2 + dz
            row.append(bm.verts.new(body.head_surface(direction(e, az), gap)))   # under the hair
        rows.append(row)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((rows[0][i], rows[0][j], rows[1][j], rows[1][i]))     # normals face outward
    band = H.mesh_object(f"rupert_{name}", bm, m, body.coll)
    sol = band.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = 0.014, 1.0
    band.parent = body.empties["head"]


def beanie(body, rgb):
    from rupert_model import HEAD_LIFT, HEAD_R
    m = H.material("beanie", rgb, rough=0.7, sheen=0.5)
    head = body.empties["head"]
    dome = H.mesh_object("rupert_beanie", H.sphere_bm(HEAD_R * 1.18, top_half_only=True), m, body.coll)
    H.attach(dome, head, (0, 0, HEAD_LIFT + 0.1 * HEAD_R))
    cuff = H.mesh_object("rupert_beanie_cuff", H.tube_bm(HEAD_R * 1.2, HEAD_R * 0.38), m, body.coll)
    cuff.modifiers.new("thick", "SOLIDIFY").thickness = 0.012
    H.attach(cuff, head, (0, 0, HEAD_LIFT + 0.22 * HEAD_R))


def trimmed_vest(body, rgb, trim_rgb, name="vest"):
    """Running vest / singlet with contrasting trim round the neck and armholes."""
    def build(nodes, links, bsdf):
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "trim"
        on = H.Maths(nodes, links).op("GREATER_THAN", attr.outputs["Fac"], 0.5)
        H.two_colour(nodes, links, on, rgb, trim_rgb, bsdf)

    lw = body.shape("lower_waist")
    els = [
        ("capsule", body.chest_m(), 0.124, 0.1),
        ("capsule", body.shape_m("torso"), 0.078, body.shape("torso")["half_length"] * 0.45),  # straps
        ("capsule", body.shape_m("lower_waist"), 0.098, lw["half_length"]),
    ]
    vest = H.metaball_mesh(f"rupert_{name}", els, body.coll)
    # trim: the band of vest nearest the neck and the shoulders (where neck/arms come out)
    neck = body.neck_m()
    na, nb = neck @ Vector((0, 0, -0.07)), neck @ Vector((0, 0, 0.07))
    shoulders = [Vector(body.rest[f"{s}_upper_arm"]["pos"]) for s in ("right", "left")]
    vals = []
    for v in vest.data.vertices:
        near_neck = H._dist(v.co, na, nb) < 0.083
        near_arm = min((v.co - p).length for p in shoulders) < 0.082
        vals.append(1.0 if (near_neck or near_arm) else 0.0)
    attr = vest.data.attributes.new("trim", "FLOAT", "POINT")
    attr.data.foreach_set("value", vals)
    vest.data.materials.append(H.node_material(name, build, rough=0.75, sheen=0.4))
    H.skin_to_skeleton(vest, ["torso", "lower_waist"], body.segs, body.skeleton)


def _slice_radius(points, centre, axis_u, axis_v, theta, tol_deg=9.0):
    """Largest distance from centre, among points roughly in direction theta (in the u-v plane)."""
    best = 0.0
    for p in points:
        d = p - centre
        u, v = d.dot(axis_u), d.dot(axis_v)
        diff = (math.degrees(math.atan2(v, u) - theta) + 180) % 360 - 180
        if abs(diff) < tol_deg:
            best = max(best, math.hypot(u, v))
    return best


def _smooth(vals, passes=2):
    n = len(vals)
    for _ in range(passes):
        vals = [(vals[i - 1] + 2 * vals[i] + vals[(i + 1) % n]) / 4 for i in range(n)]
    return vals


def loose_shorts(body, rgb, trim_rgb, hem_below_hip=0.13, split=0.04, name="shorts"):
    """70s satin running shorts: separate fabric hanging loose off the hips, short flared legs
    with a split up the outer side, white piping on the hems and waistband."""
    skin_pts = {k: [v.co.copy() for v in bpy.data.objects[f"rupert_skin_{k}"].data.vertices]
                for k in ("core", "leg_r", "leg_l")}
    segs = 40
    thetas = [2 * math.pi * j / segs for j in range(segs)]
    X, Y, Z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    bm = bmesh.new()
    weights, trims = [], []          # per vertex: {body: weight}, trim 0/1

    def ring(centre, au, av, radii, w, trim):
        out = []
        for th, r in zip(thetas, radii):
            out.append(bm.verts.new(centre + au * (r * math.cos(th)) + av * (r * math.sin(th))))
            weights.append(w)
            trims.append(trim)
        return out

    def bridge(r0, r1):
        for j in range(segs):
            bm.faces.new((r0[j], r0[(j + 1) % segs], r1[(j + 1) % segs], r1[j]))

    # trunk: waist down to the crotch, wrapped round hips and both thighs
    centre_xy = Vector((-0.015, 0, 0))
    pts = skin_pts["core"] + skin_pts["leg_r"] + skin_pts["leg_l"]
    heights = [1.245, 1.236, 1.2, 1.15, 1.1, 1.05, 1.0, 0.965]
    trunk = []
    for z in heights:
        sl = [p for p in pts if abs(p.z - z) < 0.015]
        c = centre_xy + Z * z
        clear = 0.02 + 0.016 * (1.245 - z) / (1.245 - 0.965)
        radii = _smooth([_slice_radius(sl, c, X, Y, th) + clear for th in thetas])
        w_lw = min(1.0, max(0.0, (z - 1.06) / 0.14))
        trunk.append(ring(c, X, Y, radii, {"lower_waist": w_lw, "pelvis": 1 - w_lw}, 1.0 if z > 1.23 else 0.0))
    for a, b in zip(trunk, trunk[1:]):
        bridge(a, b)
    crotch = bm.verts.new(centre_xy + Z * 0.95)
    weights.append({"pelvis": 1.0})
    trims.append(0.0)
    last = trunk[-1]
    for j in range(segs):
        bm.faces.new((last[(j + 1) % segs], last[j], crotch))

    # legs: flared tubes from inside the trunk down past the hem, outer side split
    for side, key, bone in ((-1, "leg_r", "right_thigh"), (1, "leg_l", "left_thigh")):
        hip = Vector(body.rest[bone]["pos"])
        th_out = math.atan2(side, 0)                    # outer side of this leg (+/- y)
        hem = [hem_below_hip - split * max(0.0, math.cos(th - th_out)) ** 4 for th in thetas]
        rings = []
        for f in (-0.25, 0.15, 0.45, 0.75, 0.9, 1.0):
            depth_c = hem_below_hip * f
            sl = [p for p in skin_pts[key] if abs(p.z - (hip.z - depth_c)) < 0.015]
            c = hip - Z * depth_c
            base = _smooth([max(_slice_radius(sl, c, X, Y, th), 0.07) for th in thetas])
            clear = 0.016 + 0.026 * max(0.0, f)
            cs = []
            radii = []
            for th, r, h in zip(thetas, base, hem):
                radii.append(r + clear)
                cs.append(h * f)
            verts = []
            for th, r, d in zip(thetas, radii, cs):
                verts.append(bm.verts.new(hip - Z * d + X * (r * math.cos(th)) + Y * (r * math.sin(th))))
                w_thigh = 0.85 * min(1.0, max(0.0, f))
                weights.append({bone: w_thigh, "pelvis": 1 - w_thigh})
                trims.append(1.0 if f >= 0.9 else 0.0)
            rings.append(verts)
        for a, b in zip(rings, rings[1:]):
            bridge(a, b)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    def build(nodes, links, bsdf):
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "trim"
        on = H.Maths(nodes, links).op("GREATER_THAN", attr.outputs["Fac"], 0.5)
        H.two_colour(nodes, links, on, rgb, trim_rgb, bsdf)

    obj = H.mesh_object(f"rupert_{name}", bm, H.node_material(name, build, rough=0.4, sheen=0.2), body.coll)
    attr = obj.data.attributes.new("trim", "FLOAT", "POINT")
    attr.data.foreach_set("value", trims)
    groups = {}
    for i, w in enumerate(weights):
        for b, x in w.items():
            if x > 0:
                g = groups.get(b) or obj.vertex_groups.new(name=b)
                groups[b] = g
                g.add([i], x, "REPLACE")
    obj.modifiers.new("skeleton", "ARMATURE").object = body.skeleton
    obj.modifiers.new("thick", "SOLIDIFY").thickness = 0.004
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    return obj


# ---- the outfits -------------------------------------------------------------------
def ep01_70s(body):
    """EP01 'Learn to run': 70s track star. Orange satin running shorts with white piping,
    white vest with orange trim, orange towelling headband and wristbands, white tube socks
    with orange stripes, orange running shoes."""
    loose_shorts(body, ORANGE, WHITE)
    trimmed_vest(body, WHITE, ORANGE)
    headband(body, ORANGE)
    wristbands(body, ORANGE)
    tube_socks(body, WHITE, ORANGE)
    running_shoes(body, ORANGE, WHITE, WHITE)


def classic(body):
    """The channel look: orange beanie and orange shoes."""
    beanie(body, ORANGE)
    running_shoes(body, ORANGE, WHITE, ORANGE)


COSTUMES = {
    "ep01_70s": {"build": ep01_70s, "expression": "happy"},
    "classic": {"build": classic, "expression": "happy"},
}
