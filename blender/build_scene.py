"""Blender: builds the EP01 scene (bright 100 m track + cartoon Rupert) around one replay.

Claude runs this; Tom just opens the .blend it saves.
  blender -b --python blender/build_scene.py -- <replay .npz> <attempt> <out .blend>

Rupert's look (decisions.md): no face, tall and thin, orange beanie, orange shoes.
Made "a touch cartoony": bigger head, chunkier mitten hands and shoes. Limb LENGTHS and
joint positions are exactly the physics body's (blender/rupert_shapes.json), so his feet
really meet the ground. Only thicknesses and the head are exaggerated, for looks.

Coordinates: metres, Z up, track runs along +X from the start line at X = 0,
Rupert's lane is centred on Y = 0. The side camera sits on the -Y side.
"""
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from import_replay import import_replay  # noqa: E402

# ---- look --------------------------------------------------------------------------
HEAD_SCALE = 1.45        # cartoon head
HAND_SCALE = 1.35        # mitten hands
SHOE_SCALE = 1.4         # chunky shoes
LIMB_SCALE = 1.1         # slightly chunkier limbs
TORSO_RADIUS = {"torso": 0.075, "lower_waist": 0.1, "butt": 0.1}
SKIP_SHAPES = {"upper_waist"}   # replaced by one smooth chest piece
CHEST = dict(radius=0.115, top=0.0, bottom=-0.19)   # vertical capsule in the torso body

COL = {
    "body": (0.9, 0.88, 0.84),
    "orange": (1.0, 0.33, 0.02),
    "track": (0.02, 0.28, 0.85),     # blue track: makes the orange shoes pop
    "line": (1.0, 1.0, 1.0),
    "grass": (0.12, 0.65, 0.08),
    "finish": (1.0, 0.85, 0.0),
    "post": (0.95, 0.95, 0.95),
    "stand": (0.38, 0.22, 0.8)   # purple, so white Rupert stands out,
}
BOARD_COLOURS = [(1.0, 0.1, 0.45), (1.0, 0.8, 0.0), (0.0, 0.8, 0.95), (0.55, 0.15, 0.95), (0.4, 0.9, 0.1)]

LANE_W = 1.22
N_LANES = 8
RUPERT_LANE = 3                      # lanes 1 (nearest camera) .. 8
TRACK_X = (-30.0, 230.0)
FINISH_X = 100.0


def lane_edge(i):
    """Y of the line on the near side of lane i (lane RUPERT_LANE is centred on 0)."""
    return (i - RUPERT_LANE - 0.5) * LANE_W


# ---- helpers -----------------------------------------------------------------------
def material(name, rgb, rough=0.5, emit=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def mesh_object(name, bm, mat, coll, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    return obj


def capsule_bm(radius, half_length, axis="z"):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=15, radius=radius)
    for v in bm.verts:
        v.co.z += half_length if v.co.z > 0 else -half_length
    if axis == "x":
        bmesh.ops.rotate(bm, verts=bm.verts, matrix=_rot_y90())
    return bm


def _rot_y90():
    from mathutils import Matrix
    return Matrix.Rotation(math.radians(90), 3, "Y")


def sphere_bm(radius, top_half_only=False):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=16, radius=radius)
    if top_half_only:
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-4], context="VERTS")
    return bm


def box(name, size, loc, mat, coll):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    obj = mesh_object(name, bm, mat, coll, smooth=False)
    obj.location = loc
    return obj


def text(name, body, loc, rot, size, mat, coll, extrude=0.01):
    c = bpy.data.curves.new(name, "FONT")
    c.body, c.size, c.extrude = body, size, extrude
    c.align_x, c.align_y = "CENTER", "CENTER"
    obj = bpy.data.objects.new(name, c)
    obj.data.materials.append(mat)
    obj.location, obj.rotation_euler = loc, [math.radians(a) for a in rot]
    coll.objects.link(obj)
    return obj


def attach(obj, parent, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    obj.location, obj.rotation_quaternion = pos, quat


# ---- Rupert ------------------------------------------------------------------------
def build_rupert(empties, coll):
    body_m = material("rupert_body", COL["body"], rough=0.45)
    orange = material("rupert_orange", COL["orange"], rough=0.6)
    shapes = json.loads((HERE / "rupert_shapes.json").read_text())
    top, bot = CHEST["top"], CHEST["bottom"]
    chest = mesh_object("rupert_chest", capsule_bm(CHEST["radius"], (top - bot) / 2), body_m, coll)
    attach(chest, empties["torso"], (0, 0, (top + bot) / 2))
    for s in shapes:
        if s["name"] in SKIP_SHAPES:
            continue
        parent = empties[s["body"]]
        r, h = s["radius"], s["half_length"]
        pos = list(s["pos"])
        mat = body_m
        if s["name"] == "head":
            r *= HEAD_SCALE
            pos[2] += 0.6 * (r - s["radius"])          # grow upwards, not into the chest
            head = mesh_object("rupert_head", sphere_bm(r), body_m, coll)
            attach(head, parent, pos)
            # beanie: dome + turned-up cuff + bobble
            dome = mesh_object("rupert_beanie", sphere_bm(r * 1.06, top_half_only=True), orange, coll)
            attach(dome, parent, (pos[0], pos[1], pos[2] + 0.12 * r))
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=r * 1.1, radius2=r * 1.1, depth=r * 0.35)
            cuff = mesh_object("rupert_beanie_cuff", bm, orange, coll)
            attach(cuff, parent, (pos[0], pos[1], pos[2] + 0.2 * r))
            bob = mesh_object("rupert_bobble", sphere_bm(r * 0.32), orange, coll)
            attach(bob, parent, (pos[0], pos[1], pos[2] + 1.22 * r))
            continue
        if s["type"] == "sphere":                       # hands
            r *= HAND_SCALE
            bm = sphere_bm(r)
        else:
            if s["name"] in TORSO_RADIUS:
                r = TORSO_RADIUS[s["name"]]
            elif s["body"].endswith("foot"):
                pos[2] += r * (SHOE_SCALE - 1)           # sit the bigger shoe on the floor
                r *= SHOE_SCALE
                mat = orange
            else:
                r *= LIMB_SCALE
            bm = capsule_bm(r, h)
        obj = mesh_object(f"rupert_{s['name']}", bm, mat, coll)
        attach(obj, parent, pos, s["quat"])
    for e in empties.values():
        e.hide_render = True


# ---- stadium -----------------------------------------------------------------------
def build_track(coll):
    track_m = material("track", COL["track"], rough=0.8)
    line_m = material("line", COL["line"], rough=0.5, emit=0.3)
    grass_m = material("grass", COL["grass"], rough=0.9)
    length = TRACK_X[1] - TRACK_X[0]
    mid_x = sum(TRACK_X) / 2
    y0, y1 = lane_edge(1), lane_edge(N_LANES + 1)

    box("grass", (length + 400, 400, 0.02), (mid_x, 0, -0.012), grass_m, coll)
    box("track", (length, y1 - y0 + 1.0, 0.01), (mid_x, (y0 + y1) / 2, -0.004), track_m, coll)
    for i in range(1, N_LANES + 2):
        box(f"lane_line_{i}", (length, 0.05, 0.004), (mid_x, lane_edge(i), 0.002), line_m, coll)
    box("start_line", (0.08, y1 - y0, 0.004), (0, (y0 + y1) / 2, 0.003), line_m, coll)
    # finish: chequered strip + yellow gate with banner
    for k in range(int((y1 - y0) / 0.25)):
        if k % 2 == 0:
            box(f"finish_check_{k}", (0.25, 0.25, 0.004), (FINISH_X, y0 + 0.125 + k * 0.25, 0.003), line_m, coll)
    # bright boards along the far side (also show speed in tracking shots)
    # every 10 m a white board shows the distance; FINISH board at 100 m
    num_m = material("sign_text", (0.05, 0.1, 0.4), rough=0.5)
    fin_m = material("finish_board", COL["finish"], rough=0.4, emit=0.2)
    for k, x in enumerate(range(int(TRACK_X[0]), int(TRACK_X[1]), 5)):
        if x % 10 == 0 and 0 < x <= 200:
            label, mat = ("FINISH", fin_m) if x == FINISH_X else (f"{x}m", material("sign", (1, 1, 1), rough=0.5))
            box(f"board_{x}m", (2.4, 0.15, 1.0), (x, y1 + 1.5, 0.5), mat, coll)
            text(f"board_{x}m_txt", label, (x, y1 + 1.41, 0.5), (90, 0, 0), 0.45 if x == FINISH_X else 0.55, num_m, coll)
            continue
        c = BOARD_COLOURS[k % len(BOARD_COLOURS)]
        box(f"board_{k}", (2.5, 0.15, 1.0), (x, y1 + 1.5, 0.5),
            material(f"board_{k % len(BOARD_COLOURS)}", c, rough=0.4, emit=0.2), coll)
    # stepped stands behind
    stand_m = material("stand", COL["stand"], rough=0.7)
    for t in range(5):
        box(f"stand_{t}", (length, 2.0, 0.45 * (t + 1)), (mid_x, y1 + 4 + 2.0 * t, 0.225 * (t + 1)), stand_m, coll)


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
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.75, 0.92, 1.0, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.45, (0.08, 0.45, 1.0, 1)
    bg.inputs["Strength"].default_value = 0.8
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.6
    sun.angle = math.radians(3)
    obj = bpy.data.objects.new("sun", sun)
    obj.rotation_euler = (math.radians(40), 0, math.radians(25))   # from the camera side, high
    scene.collection.objects.link(obj)


def build_camera(empties):
    scene = bpy.context.scene
    cam = bpy.data.cameras.new("side_cam")
    cam.lens = 40
    obj = bpy.data.objects.new("side_cam", cam)
    obj.location = (0.0, -7.0, 1.15)
    obj.rotation_euler = (math.radians(86), 0, 0)
    c = obj.constraints.new("COPY_LOCATION")
    c.target = empties["torso"]
    c.use_y = c.use_z = False
    c.use_offset = True
    scene.collection.objects.link(obj)
    scene.camera = obj


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
    for attr, val in (("taa_render_samples", 32), ("use_shadows", True), ("use_raytracing", False)):
        try:
            setattr(scene.eevee, attr, val)
        except (AttributeError, TypeError):
            pass
    scene.view_settings.view_transform = "Standard"   # AgX washed the bright colours out
    scene.view_settings.look = "None"


def main():
    extra = sys.argv[sys.argv.index("--") + 1:]
    replay, attempt, out = extra[0], int(extra[1]), extra[2]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    empties = import_replay(replay, attempt)
    rupert = bpy.data.collections.new("rupert")
    stadium = bpy.data.collections.new("stadium")
    bpy.context.scene.collection.children.link(rupert)
    bpy.context.scene.collection.children.link(stadium)
    build_rupert(empties, rupert)
    build_track(stadium)
    build_world_and_light()
    build_camera(empties)
    render_settings()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out).resolve()))
    print(f"Saved scene {out}")


main()
