"""Blender: builds the EP01 scene (stadium + Rupert in a morph suit) around one replay.

Claude runs this; Tom just opens the .blend it saves.
  blender -b --python blender/build_scene.py -- <replay .npz> <attempt> <out .blend> ["SCREEN TEXT"]

Rupert (decisions.md): no face, orange bobble beanie, orange shoes, now in a one-piece
blue morph suit. The suit is ONE smooth skin built around the physics body's limbs
(blender/rupert_shapes.json) and bent at the joints by a skeleton that copies the replay,
so limb lengths and joints are exactly the body that learned. Only thickness and the
head size are exaggerated, for looks.

Coordinates: metres, Z up, track runs along +X from the start line at X = 0,
Rupert's lane is centred on Y = 0. Cameras sit on the -Y side.
"""
import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.geometry import intersect_point_line

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from import_replay import import_replay  # noqa: E402

random.seed(7)

# ---- look --------------------------------------------------------------------------
COL = {
    "suit": (0.02, 0.22, 0.95),       # bright royal blue: pops on the red track
    "orange": (1.0, 0.32, 0.0),
    "sole": (0.95, 0.95, 0.95),
    "track": (0.75, 0.13, 0.05),      # traditional red tartan, a bit brighter
    "track2": (0.62, 0.09, 0.03),
    "line": (1.0, 1.0, 1.0),
    "grass": (0.16, 0.55, 0.06),
    "grass2": (0.11, 0.45, 0.04),
    "stand": (0.55, 0.6, 0.7),
    "seat": (0.0, 0.35, 0.75),
    "screen": (0.02, 0.02, 0.03),
}
BOARD_COLOURS = [(1.0, 0.1, 0.4), (1.0, 0.72, 0.0), (0.0, 0.7, 0.95), (0.5, 0.15, 0.9), (0.2, 0.75, 0.1)]
CROWD_COLOURS = [(0.9, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.9), (1, 0.8, 0.1), (0.1, 0.6, 0.2),
                 (0.95, 0.45, 0.1), (0.6, 0.2, 0.7), (0.15, 0.15, 0.18), (0.9, 0.5, 0.6)]
SKIN_TONES = [(0.96, 0.8, 0.69), (0.87, 0.65, 0.5), (0.63, 0.42, 0.29), (0.42, 0.27, 0.18)]

LANE_W = 1.22
N_LANES = 8
RUPERT_LANE = 3                      # lanes 1 (nearest camera) .. 8
TRACK_X = (-40.0, 240.0)
FINISH_X = 100.0
CHANNEL = "RUPERT'S AI DIARIES"

# ---- the suit: surface radius (m) of each limb, by physics shape ------------------
SUIT_RADIUS = {
    "thigh": 0.072, "shin": 0.052, "upper_arm": 0.045, "lower_arm": 0.037, "hand": 0.05,
    "lower_waist": 0.09, "butt": 0.095, "torso": 0.07,
}
HEAD_R = 0.14           # physics head is 0.098; cartoon head
METABALL_SCALE = 1 / 0.57   # Blender metaball: surface sits at ~0.57 x element radius
# each group melts into one smooth piece; groups meet at hip and shoulder like suit seams
SUIT_GROUPS = {
    "suitcore": ["torso", "head", "lower_waist", "pelvis"],
    "suitlegr": ["right_thigh", "right_shin"],
    "suitlegl": ["left_thigh", "left_shin"],
    "suitarmr": ["right_upper_arm", "right_lower_arm", "right_hand"],
    "suitarml": ["left_upper_arm", "left_lower_arm", "left_hand"],
}


def lane_edge(i):
    """Y of the line on the near side of lane i (lane RUPERT_LANE is centred on 0)."""
    return (i - RUPERT_LANE - 0.5) * LANE_W


# ---- helpers -----------------------------------------------------------------------
def material(name, rgb, rough=0.5, emit=0.0, sheen=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def node_material(name, build):
    """Material whose node tree is built by build(nodes, links, bsdf)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    build(nt.nodes, nt.links, nt.nodes.get("Principled BSDF"))
    return m


def mesh_object(name, bm, mat, coll, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    if mat:
        me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    return obj


def sphere_bm(radius, top_half_only=False):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=18, radius=radius)
    if top_half_only:
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-4], context="VERTS")
    return bm


def rounded_box_bm(size, bevel, segments=4):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts), offset=bevel, segments=segments,
                    affect="EDGES", profile=0.5)
    return bm


def box(name, size, loc, mat, coll, bevel=0.0):
    if bevel:
        obj = mesh_object(name, rounded_box_bm(size, bevel, 2), mat, coll, smooth=True)
    else:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=size, verts=bm.verts)
        obj = mesh_object(name, bm, mat, coll, smooth=False)
    obj.location = loc
    return obj


def text(name, body, loc, rot, size, mat, coll, extrude=0.005):
    c = bpy.data.curves.new(name, "FONT")
    c.body, c.size, c.extrude = body, size, extrude
    c.align_x, c.align_y = "CENTER", "CENTER"
    try:
        c.font = bpy.data.fonts.load("C:/Windows/Fonts/arialbd.ttf", check_existing=True)
    except (RuntimeError, OSError):
        pass
    obj = bpy.data.objects.new(name, c)
    obj.data.materials.append(mat)
    obj.location, obj.rotation_euler = loc, [math.radians(a) for a in rot]
    coll.objects.link(obj)
    return obj


def attach(obj, parent, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    obj.location, obj.rotation_quaternion = pos, quat


def rest_matrix(rest, body):
    r = rest[body]
    return Matrix.Translation(r["pos"]) @ Quaternion(r["quat"]).to_matrix().to_4x4()


def shape_matrix(s):
    return Matrix.Translation(s["pos"]) @ Quaternion(s["quat"]).to_matrix().to_4x4()


# ---- Rupert ------------------------------------------------------------------------
def suit_pieces(shapes, rest):
    """Every blob of the suit, in the standing rest pose (world coordinates).

    Returns list of (group, body, kind, world matrix, surface radius, half length).
    kind 'capsule' runs along the matrix's local Z.
    """
    pieces = []
    for s in shapes:
        body, name = s["body"], s["name"]
        if body.endswith("foot") or name == "upper_waist":
            continue                       # shoes are separate; chest replaces upper_waist
        group = next(g for g, bodies in SUIT_GROUPS.items() if body in bodies)
        m = rest_matrix(rest, body) @ shape_matrix(s)
        if name == "head":
            pieces.append((group, body, "ball", rest_matrix(rest, body) @ Matrix.Translation((0, 0, 0.02)), HEAD_R, 0))
            continue
        key = next((k for k in SUIT_RADIUS if name.endswith(k) or name == k), None)
        r = SUIT_RADIUS.get(key, s["radius"])
        pieces.append((group, body, "ball" if s["type"] == "sphere" else "capsule", m, r, s["half_length"]))
    torso = rest_matrix(rest, "torso")
    # chest (one smooth barrel instead of the physics model's stacked bars) and neck
    pieces.append(("suitcore", "torso", "capsule", torso @ Matrix.Translation((0, 0, -0.09)), 0.115, 0.1))
    head_z = (torso.inverted() @ Vector(rest["head"]["pos"])).z
    pieces.append(("suitcore", "torso", "capsule", torso @ Matrix.Translation((0, 0, head_z * 0.55)), 0.045, head_z * 0.3))
    return pieces


def build_suit_mesh(pieces, coll):
    """Metaballs -> one smooth mesh per group (still in the rest pose)."""
    objs = []
    to_z = Quaternion((0, 1, 0), -math.pi / 2)        # metaball capsules run along X; ours along Z
    for group in SUIT_GROUPS:
        mb = bpy.data.metaballs.new(group)
        mb.resolution = mb.render_resolution = 0.012
        mb.threshold = 0.6
        for g, _, kind, m, r, h in pieces:
            if g != group:
                continue
            e = mb.elements.new(type="CAPSULE" if kind == "capsule" else "BALL")
            e.co = m.to_translation()
            e.rotation = m.to_quaternion() @ to_z
            e.radius = r * METABALL_SCALE
            e.stiffness = 2.0
            if kind == "capsule":
                e.size_x = h
        obj = bpy.data.objects.new(group, mb)
        coll.objects.link(obj)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        coll.objects.unlink(obj)
        bpy.data.objects.remove(obj)
        mesh_obj = bpy.data.objects.new(f"rupert_{group}", me)
        coll.objects.link(mesh_obj)
        for p in me.polygons:
            p.use_smooth = True
        objs.append(mesh_obj)
    return objs


def body_segments(pieces):
    """Each body's 'bone line' (start, end) segments in the rest pose, for skin weights."""
    segs = {}
    for _, body, kind, m, r, h in pieces:
        a = m @ Vector((0, 0, -h)) if kind == "capsule" else m.to_translation()
        b = m @ Vector((0, 0, h)) if kind == "capsule" else m.to_translation()
        segs.setdefault(body, []).append((a, b))
    return segs


def _dist(p, a, b):
    if (b - a).length < 1e-6:
        return (p - a).length
    _, t = intersect_point_line(p, a, b)
    t = min(max(t, 0.0), 1.0)
    return (p - (a + (b - a) * t)).length


def skin_weights(obj, group, segs):
    """Each skin point follows the body part it's closest to, blended over ~2 cm at joints."""
    bodies = SUIT_GROUPS[group]
    vgs = {b: obj.vertex_groups.new(name=b) for b in bodies}
    for v in obj.data.vertices:
        d = {b: min(_dist(v.co, a, c) for a, c in segs[b]) for b in bodies if b in segs}
        dmin = min(d.values())
        w = {b: math.exp(-(x - dmin) / 0.012) for b, x in d.items()}
        total = sum(w.values())
        for b, x in w.items():
            if x / total > 0.02:
                vgs[b].add([v.index], x / total, "REPLACE")


def build_skeleton(empties, rest, coll):
    """One bone per body part, each glued to that part's replay empty."""
    arm_data = bpy.data.armatures.new("rupert_skeleton")
    arm = bpy.data.objects.new("rupert_skeleton", arm_data)
    coll.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for body, r in rest.items():
        if body not in empties:
            continue
        b = arm_data.edit_bones.new(body)
        b.head = Vector(r["pos"])
        b.tail = b.head + Vector((0, 0, 0.08))
    bpy.ops.object.mode_set(mode="OBJECT")
    for body in rest:
        if body not in empties:
            continue
        c = arm.pose.bones[body].constraints.new("CHILD_OF")
        c.target = empties[body]
        c.inverse_matrix = rest_matrix(rest, body).inverted()
    arm.hide_render = True
    return arm


def build_shoe(name, foot_shapes, parent, coll, upper_m, sole_m):
    ys = [s["pos"][1] for s in foot_shapes]
    cy = sum(ys) / len(ys)
    cx = foot_shapes[0]["pos"][0]
    r = foot_shapes[0]["radius"]
    length = 2 * (foot_shapes[0]["half_length"] + r) + 0.03
    sole = mesh_object(f"{name}_sole", rounded_box_bm((length, 0.115, 0.03), 0.012), sole_m, coll)
    attach(sole, parent, (cx + 0.005, cy, -r + 0.012))
    upper = mesh_object(f"{name}_upper", rounded_box_bm((length - 0.02, 0.1, 0.085), 0.035, 5), upper_m, coll)
    attach(upper, parent, (cx, cy, -r + 0.06))


def build_rupert(empties, coll):
    data = json.loads((HERE / "rupert_shapes.json").read_text())
    shapes, rest = data["shapes"], data["rest"]
    suit_m = material("rupert_suit", COL["suit"], rough=0.42, sheen=0.3)
    orange = material("rupert_orange", COL["orange"], rough=0.7, sheen=0.5)
    sole_m = material("rupert_sole", COL["sole"], rough=0.5)

    pieces = suit_pieces(shapes, rest)
    segs = body_segments(pieces)
    arm = build_skeleton(empties, rest, coll)
    for obj in build_suit_mesh(pieces, coll):
        group = obj.name.replace("rupert_", "")
        obj.data.materials.append(suit_m)
        skin_weights(obj, group, segs)
        mod = obj.modifiers.new("skeleton", "ARMATURE")
        mod.object = arm

    # beanie on the head (dome + turned-up cuff + bobble)
    hz = 0.02
    dome = mesh_object("rupert_beanie", sphere_bm(HEAD_R * 1.07, top_half_only=True), orange, coll)
    attach(dome, empties["head"], (0, 0, hz + 0.1 * HEAD_R))
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=40, radius1=HEAD_R * 1.1, radius2=HEAD_R * 1.12, depth=HEAD_R * 0.38)
    cuff = mesh_object("rupert_beanie_cuff", bm, orange, coll)
    cuff.modifiers.new("thick", "SOLIDIFY").thickness = 0.012
    attach(cuff, empties["head"], (0, 0, hz + 0.22 * HEAD_R))
    bob = mesh_object("rupert_bobble", sphere_bm(HEAD_R * 0.33), orange, coll)
    attach(bob, empties["head"], (0, 0, hz + 1.25 * HEAD_R))

    for side in ("right", "left"):
        foot = [s for s in shapes if s["body"] == f"{side}_foot"]
        build_shoe(f"rupert_shoe_{side}", foot, empties[f"{side}_foot"], coll, orange, sole_m)
    for e in empties.values():
        e.hide_render = True
        e.hide_viewport = True


# ---- stadium -----------------------------------------------------------------------
def track_material():
    def build(nodes, links, bsdf):
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 600.0
        noise.inputs["Detail"].default_value = 4.0
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*COL["track2"], 1)
        ramp.color_ramp.elements[1].color = (*COL["track"], 1)
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.35, 0.65
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.25
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.85
    return node_material("track", build)


def grass_material():
    """Mown stripes every 5 m along the track (also shows speed in tracking shots)."""
    def build(nodes, links, bsdf):
        coord = nodes.new("ShaderNodeTexCoord")
        sep = nodes.new("ShaderNodeSeparateXYZ")
        div = nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 5.0
        flo = nodes.new("ShaderNodeMath"); flo.operation = "FLOOR"
        mod = nodes.new("ShaderNodeMath"); mod.operation = "FLOORED_MODULO"; mod.inputs[1].default_value = 2.0
        mix = nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
        mix.inputs["A"].default_value = (*COL["grass"], 1)
        mix.inputs["B"].default_value = (*COL["grass2"], 1)
        links.new(coord.outputs["Object"], sep.inputs[0])
        links.new(sep.outputs["X"], div.inputs[0])
        links.new(div.outputs[0], flo.inputs[0])
        links.new(flo.outputs[0], mod.inputs[0])
        links.new(mod.outputs[0], mix.inputs["Factor"])
        links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.95
    return node_material("grass", build)


def crowd_material():
    def build(nodes, links, bsdf):
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "col"
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.8
    return node_material("crowd", build)


def build_crowd(coll, x0, x1, tiers):
    """Rows of simple spectators (body + head), random shirt colours, some empty seats."""
    bm = bmesh.new()
    col = bm.loops.layers.color.new("col")
    for y, z in tiers:
        x = x0
        while x < x1:
            x += random.uniform(0.5, 0.75)
            if random.random() < 0.15:
                continue
            h = random.uniform(0.38, 0.5)
            shirt, skin = random.choice(CROWD_COLOURS), random.choice(SKIN_TONES)
            px, py = x, y + random.uniform(-0.12, 0.12)
            before = set(bm.faces)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.2, radius2=0.15, depth=h,
                                  matrix=Matrix.Translation((px, py, z + h / 2)))
            body_faces = set(bm.faces) - before
            before = set(bm.faces)
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.11,
                                       matrix=Matrix.Translation((px, py, z + h + 0.1)))
            head_faces = set(bm.faces) - before
            for faces, c in ((body_faces, shirt), (head_faces, skin)):
                for f in faces:
                    for loop in f.loops:
                        loop[col] = (*c, 1.0)
    return mesh_object("crowd", bm, crowd_material(), coll, smooth=False)


def build_stadium(coll, screen_text):
    track_m, grass_m = track_material(), grass_material()
    line_m = material("line", COL["line"], rough=0.6)
    length = TRACK_X[1] - TRACK_X[0]
    mid_x = sum(TRACK_X) / 2
    y0, y1 = lane_edge(1), lane_edge(N_LANES + 1)

    box("grass", (length + 600, 600, 0.02), (mid_x, 0, -0.012), grass_m, coll)
    box("track", (length, y1 - y0 + 1.2, 0.01), (mid_x, (y0 + y1) / 2, -0.004), track_m, coll)
    for i in range(1, N_LANES + 2):
        box(f"lane_line_{i}", (length, 0.05, 0.004), (mid_x, lane_edge(i), 0.002), line_m, coll)
    box("kerb", (length, 0.08, 0.06), (mid_x, y0 - 0.65, 0.03), line_m, coll, bevel=0.01)
    box("start_line", (0.05, y1 - y0, 0.004), (0, (y0 + y1) / 2, 0.003), line_m, coll)
    box("finish_line", (0.05, y1 - y0, 0.004), (FINISH_X, (y0 + y1) / 2, 0.003), line_m, coll)
    for i in range(1, N_LANES + 1):    # painted lane numbers behind the start line
        text(f"lane_no_{i}", str(i), (-1.2, lane_edge(i) + LANE_W / 2, 0.003), (0, 0, -90), 0.7, line_m, coll, 0.001)

    # advertising boards along the far side; a distance board every 10 m, FINISH at 100 m
    by = y1 + 1.6
    white_m = material("board_white", (1, 1, 1), rough=0.4, emit=0.1)
    num_m = material("board_text", (0.03, 0.05, 0.2), rough=0.5)
    fin_m = material("finish_board", (1.0, 0.8, 0.0), rough=0.4, emit=0.25)
    txt_m = material("board_text_white", (1, 1, 1), rough=0.4, emit=0.5)
    board_ms = [material(f"board_{i}", c, rough=0.35, emit=0.35) for i, c in enumerate(BOARD_COLOURS)]
    x, k = TRACK_X[0], 0
    while x < TRACK_X[1] - 0.01:
        near10 = round(x / 10) * 10
        if abs(x - (near10 - 1.25)) < 0.01 and 0 < near10 <= 200:
            fin = near10 == FINISH_X
            box(f"board_{near10}m", (2.4, 0.15, 0.9), (near10, by, 0.45), fin_m if fin else white_m, coll, bevel=0.03)
            text(f"board_{near10}m_txt", "FINISH" if fin else f"{near10}m", (near10, by - 0.085, 0.43), (90, 0, 0),
                 0.42 if fin else 0.55, num_m, coll)
            x += 2.5
            continue
        nxt = (math.floor(x / 10) + 1) * 10 - 1.25          # where the next distance board starts
        seg = nxt - x if 0 < nxt + 1.25 <= 200 else 5.0
        if seg < 0.05:
            seg = 5.0
        box(f"board_{k}", (seg - 0.1, 0.15, 0.9), (x + seg / 2, by, 0.45), board_ms[k % len(board_ms)], coll, bevel=0.03)
        if seg > 5:
            text(f"board_{k}_txt", CHANNEL, (x + seg / 2, by - 0.085, 0.43), (90, 0, 0), 0.36, txt_m, coll)
        x += seg
        k += 1

    # stands: stepped concrete with blue seats and a crowd
    stand_m = material("stand", COL["stand"], rough=0.8)
    seat_m = material("seat", COL["seat"], rough=0.6)
    tiers = []
    for t in range(8):
        y, z = y1 + 4.0 + 1.6 * t, 0.9 + 0.55 * t
        box(f"stand_{t}", (length, 1.6, z), (mid_x, y + 0.8, z / 2), stand_m, coll)
        box(f"seats_{t}", (length, 0.5, 0.12), (mid_x, y + 1.1, z + 0.06), seat_m, coll)
        tiers.append((y + 0.75, z))
    box("stand_wall", (length, 0.4, 1.0), (mid_x, y1 + 3.6, 0.5), stand_m, coll)
    build_crowd(coll, TRACK_X[0], TRACK_X[1], tiers)
    back_y = y1 + 4.0 + 1.6 * 8 + 0.5
    box("stand_back", (length, 1.0, 9.0), (mid_x, back_y, 4.5), stand_m, coll)
    box("roof", (length, 16, 0.4), (mid_x, back_y - 6, 9.2), material("roof", (0.92, 0.93, 0.95), rough=0.5), coll)

    # big screen on the roof, near the finish
    box("screen", (14, 0.3, 4.2), (FINISH_X - 10, back_y - 1.0, 11.8), material("screen", COL["screen"], rough=0.3), coll, bevel=0.05)
    text("screen_title", "RUPERT  100m", (FINISH_X - 10, back_y - 1.2, 12.8), (90, 0, 0), 1.1,
         material("screen_text", (1.0, 0.85, 0.2), emit=4.0), coll)
    text("screen_sub", screen_text, (FINISH_X - 10, back_y - 1.2, 11.2), (90, 0, 0), 0.9,
         material("screen_text2", (1, 1, 1), emit=4.0), coll)


def build_world_and_light():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.75, 0.9, 1.0, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.5, (0.12, 0.42, 0.95, 1)
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.5
    sun.angle = math.radians(4)          # soft-edged shadows
    obj = bpy.data.objects.new("sun", sun)
    obj.rotation_euler = (math.radians(42), 0, math.radians(35))
    scene.collection.objects.link(obj)


def build_cameras(empties):
    """Main: a 3/4 tracking camera ahead of Rupert. Also a straight side camera."""
    scene = bpy.context.scene
    target = bpy.data.objects.new("cam_target", None)
    target.location = (0, 0, 1.0)
    c = target.constraints.new("COPY_LOCATION")
    c.target, c.use_y, c.use_z = empties["torso"], False, False
    scene.collection.objects.link(target)

    def camera(name, loc, lens):
        cam = bpy.data.cameras.new(name)
        cam.lens = lens
        cam.dof.use_dof = True
        cam.dof.focus_object = target
        cam.dof.aperture_fstop = 1.4   # blurs the crowd so Rupert stands out
        obj = bpy.data.objects.new(name, cam)
        obj.location = loc
        c = obj.constraints.new("COPY_LOCATION")
        c.target, c.use_y, c.use_z, c.use_offset = empties["torso"], False, False, True
        t = obj.constraints.new("TRACK_TO")
        t.target, t.track_axis, t.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
        scene.collection.objects.link(obj)
        return obj

    main = camera("cam_three_quarter", (4.0, -6.5, 1.7), 50)
    camera("cam_side", (0.0, -7.5, 1.2), 50)
    scene.camera = main


def render_settings():
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            pass
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    for attr, val in (("taa_render_samples", 64), ("use_shadows", True), ("use_raytracing", True),
                      ("use_fast_gi", True), ("shadow_resolution_scale", 1.0)):
        try:
            setattr(scene.eevee, attr, val)
        except (AttributeError, TypeError):
            pass
    scene.view_settings.view_transform = "Khronos PBR Neutral"   # keeps bright colours bright
    scene.view_settings.look = "None"


def main():
    extra = sys.argv[sys.argv.index("--") + 1:]
    replay, attempt, out = extra[0], int(extra[1]), extra[2]
    screen_text = extra[3] if len(extra) > 3 else "TRAINING"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    empties = import_replay(replay, attempt)
    rupert = bpy.data.collections.new("rupert")
    stadium = bpy.data.collections.new("stadium")
    bpy.context.scene.collection.children.link(rupert)
    bpy.context.scene.collection.children.link(stadium)
    build_rupert(empties, rupert)
    build_stadium(stadium, screen_text)
    build_world_and_light()
    build_cameras(empties)
    render_settings()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out).resolve()))
    print(f"Saved scene {out}")


main()
