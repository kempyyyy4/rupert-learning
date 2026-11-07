"""Rupert's outfits: a new one for every video. Pick with --costume <name> in build_scene.py.

Each costume is a function that adds clothes to a rupert_model.Body.
- Clothes that bend (vest, shorts) are glued to the skeleton like his skin.
- Loose clothes (shorts) are built as separate fabric with room between it and the skin.
- Rigid bits (shoes, socks, bands) are simply attached to one body part.
"""
import math

import bmesh
from mathutils import Matrix, Vector

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
    """Towelling headband: snug on the forehead, its ends tucked under the hair at the sides."""
    from rupert_model import direction
    m = H.material(name, rgb, rough=0.95, sheen=1.0)
    n = 64
    bm = bmesh.new()
    rows = []
    for dz in (-height_deg / 2, height_deg / 2):
        row = []
        for i in range(n + 1):
            az = -105 + 210 * i / n                   # front half only; the ends tuck under the side hair
            e = elev + (back_elev - elev) * (1 - math.cos(math.radians(az))) / 2 + dz
            row.append(bm.verts.new(body.head_surface(direction(e, az), gap)))   # under the hair
        rows.append(row)
    for i in range(n):
        j = i + 1
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
        on = H.Maths(nodes, links).op("GREATER_THAN", attr.outputs["Fac"], 0.0)
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
        neck_in = 0.083 - H._dist(v.co, na, nb)
        arm_in = 0.082 - min((v.co - p).length for p in shoulders)
        vals.append(max(neck_in, arm_in))                  # > 0 inside the trim band
    attr = vest.data.attributes.new("trim", "FLOAT", "POINT")
    attr.data.foreach_set("value", vals)
    vest.data.materials.append(H.node_material(name, build, rough=0.75, sheen=0.4))
    H.skin_to_skeleton(vest, ["torso", "lower_waist"], body.segs, body.skeleton)


def loose_shorts(body, rgb, trim_rgb, hem_below_hip=0.1, split=0.03, waist_z=1.245, name="shorts"):
    """70s satin running shorts: ONE smooth piece of fabric hanging loose off the hips (waist and
    both legs melted together, so no seams or ledges), legs cut open at the hem, hem angled up on
    the outer side like a side split, white piping on the hems and waistband."""
    els = [
        ("capsule", body.shape_m("lower_waist"), 0.106, body.shape("lower_waist")["half_length"]),
        ("capsule", body.shape_m("butt"), 0.11, body.shape("butt")["half_length"]),
    ]
    hips = {}
    for side, bone in ((-1, "right_thigh"), (1, "left_thigh")):
        hip = Vector(body.rest[bone]["pos"])
        hips[side] = hip
        n = 6
        for i in range(n):                          # flared tube down the thigh, past the hem
            d0, d1 = -0.04 + 0.22 * i / n, -0.04 + 0.22 * (i + 1) / n
            r = 0.086 + 0.01 * max(0.0, (d0 + d1) / 2) / hem_below_hip
            a, b = hip - Vector((0, 0, d0)), hip - Vector((0, 0, d1))
            m = Matrix.Translation((a + b) / 2)
            els.append(("capsule", m, r, (a - b).length / 2))
    obj = H.metaball_mesh(f"rupert_{name}", els, body.coll, resolution=0.009)

    # cut open the waist and the leg hems (clean straight cuts), with a second cut for the piping line
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    piping = 0.013

    def cut(plane_co, plane_no, keep_test, delete=True):
        geom = [g for g in list(bm.verts) + list(bm.edges) + list(bm.faces) if keep_test(g)]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=plane_co, plane_no=plane_no, dist=1e-5)
        if delete:
            dead = [v for v in bm.verts if keep_test(v) and (v.co - plane_co).dot(plane_no) < -1e-5]
            bmesh.ops.delete(bm, geom=dead, context="VERTS")

    def centre(g):
        return g.co if isinstance(g, bmesh.types.BMVert) else (
            sum((v.co for v in g.verts), Vector()) / len(g.verts))

    up = Vector((0, 0, 1))
    cut(Vector((0, 0, waist_z)), -up, lambda g: True)
    cut(Vector((0, 0, waist_z - piping)), -up, lambda g: True, delete=False)
    for side, hip in hips.items():
        # hem plane: hem_below_hip under the hip, tilted up toward the outer side by `split`
        no = Vector((0, -side * split / 0.12, 1)).normalized()
        co = hip - Vector((0, 0, hem_below_hip))
        on_leg = (lambda sd: lambda g: centre(g).y * sd > 0.0 and centre(g).z < 1.0)(side)
        cut(co, no, on_leg)
        cut(co + no * piping, no, on_leg, delete=False)
    bm.to_mesh(obj.data)
    bm.free()
    for poly in obj.data.polygons:
        poly.use_smooth = True

    # piping (white) next to the cuts; weights: waist follows the hips, legs follow the thighs
    trims, weights = [], []
    for v in obj.data.vertices:
        c = v.co
        t = c.z - (waist_z - piping)                       # > 0 inside the waistband piping
        side = 1 if c.y > 0 else -1
        hip = hips[side]
        depth = hip.z - c.z
        no = Vector((0, -side * split / 0.12, 1)).normalized()
        if c.z < 1.0:
            t = max(t, piping - (c - (hip - Vector((0, 0, hem_below_hip)))).dot(no))   # > 0 in the hem piping
        trims.append(t)
        sideways = min(1.0, max(0.0, (abs(c.y) - 0.02) / 0.05))        # crotch centre stays with the pelvis
        w_thigh = 0.85 * min(1.0, max(0.0, depth / hem_below_hip)) * sideways
        w_lw = min(1.0, max(0.0, (c.z - 1.08) / 0.12))
        weights.append({("left_thigh" if side > 0 else "right_thigh"): w_thigh,
                        "lower_waist": (1 - w_thigh) * w_lw, "pelvis": (1 - w_thigh) * (1 - w_lw)})
    attr = obj.data.attributes.new("trim", "FLOAT", "POINT")
    attr.data.foreach_set("value", trims)
    groups = {}
    for i, w in enumerate(weights):
        for b, x in w.items():
            if x > 1e-4:
                g = groups.get(b) or obj.vertex_groups.new(name=b)
                groups[b] = g
                g.add([i], x, "REPLACE")

    def build(nodes, links, bsdf):
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "trim"
        on = H.Maths(nodes, links).op("GREATER_THAN", attr.outputs["Fac"], 0.0)
        H.two_colour(nodes, links, on, rgb, trim_rgb, bsdf)

    obj.data.materials.append(H.node_material(name, build, rough=0.4, sheen=0.2))
    obj.modifiers.new("skeleton", "ARMATURE").object = body.skeleton
    cs = obj.modifiers.new("no_creases", "CORRECTIVE_SMOOTH")   # irons out pinching when the legs swing
    cs.factor, cs.iterations, cs.rest_source = 0.6, 6, "ORCO"
    obj.modifiers.new("thick", "SOLIDIFY").thickness = 0.004
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    return obj


# ---- the outfits -------------------------------------------------------------------
def ep01_70s(body):
    """EP01 'Learn to run': 70s track star. Orange satin running shorts with white piping,
    plain white vest, orange towelling headband and wristbands, white tube socks
    with orange stripes, orange running shoes."""
    loose_shorts(body, ORANGE, WHITE)
    trimmed_vest(body, WHITE, WHITE)          # plain white vest
    headband(body, ORANGE)
    wristbands(body, ORANGE)
    tube_socks(body, WHITE, ORANGE)
    running_shoes(body, ORANGE, WHITE, WHITE)


def classic(body):
    """The channel look: orange beanie and orange shoes."""
    beanie(body, ORANGE)
    running_shoes(body, ORANGE, WHITE, ORANGE)


COSTUMES = {
    "ep01_70s": {"build": ep01_70s, "expression": "determined"},
    "classic": {"build": classic, "expression": "happy"},
}
