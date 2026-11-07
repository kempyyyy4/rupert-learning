"""Blender: builds the EP01 scene (stadium + Rupert in his costume) around one replay.

Claude runs this (usually via rupert/make_clip.py); Tom just opens the .blend it saves.
  blender -b --python blender/build_scene.py -- <replay .npz> --out <file.blend> [--attempt 0]
        [--costume ep01_70s] [--screen "TEXT ON BIG SCREEN"]

Rupert himself is built in rupert_model.py, his outfit for each video in costumes.py.

Coordinates: metres, Z up, track runs along +X from the start line at X = 0,
Rupert's lane is centred on Y = 0. Cameras sit on the -Y side.
"""
import argparse
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from bl_helpers import box, material, mesh_object, node_material, text  # noqa: E402
from import_replay import import_replay  # noqa: E402
from rupert_model import build_rupert  # noqa: E402
import attempts_board  # noqa: E402

random.seed(7)

# ---- look --------------------------------------------------------------------------
COL = {
    "track": (0.75, 0.13, 0.05),      # traditional red tartan, a bit brighter
    "track2": (0.62, 0.09, 0.03),
    "line": (1.0, 1.0, 1.0),
    "grass": (0.16, 0.55, 0.06),
    "grass2": (0.11, 0.45, 0.04),
    "stand": (0.55, 0.6, 0.7),
    "seat": (0.0, 0.35, 0.75),
    "screen": (0.02, 0.02, 0.03),
}
CROWD_COLOURS = [(0.9, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.9), (1, 0.8, 0.1), (0.1, 0.6, 0.2),
                 (0.95, 0.45, 0.1), (0.6, 0.2, 0.7), (0.15, 0.15, 0.18), (0.9, 0.5, 0.6)]
SKIN_TONES = [(0.96, 0.8, 0.69), (0.87, 0.65, 0.5), (0.63, 0.42, 0.29), (0.42, 0.27, 0.18)]

LANE_W = 1.22
N_LANES = 8
RUPERT_LANE = 3                      # lanes 1 (nearest camera) .. 8
TRACK_X = (-40.0, 240.0)
FINISH_X = 100.0

def lane_edge(i):
    """Y of the line on the near side of lane i (lane RUPERT_LANE is centred on 0)."""
    return (i - RUPERT_LANE - 0.5) * LANE_W




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

    by = y1 + 0.9                                       # signs stand on the grass verge
    # big raised distance signs every 10 m (readable in wide shots); yellow FINISH at 100 m
    white_m = material("sign_white", (1, 1, 1), rough=0.4, emit=0.15)
    fin_m = material("finish_board", (1.0, 0.8, 0.0), rough=0.4, emit=0.3)
    num_m = material("sign_text", (0.03, 0.05, 0.2), rough=0.5)
    post_m = material("sign_post", (0.85, 0.86, 0.88), rough=0.4)
    sy = by
    for d in range(10, 210, 10):
        fin = d == FINISH_X
        box(f"sign_{d}m", (3.2, 0.14, 1.6), (d, sy, 1.85), fin_m if fin else white_m, coll, bevel=0.04)
        for side in (-1, 1):
            box(f"sign_{d}m_post", (0.1, 0.1, 1.1), (d + side * 1.2, sy + 0.04, 0.55), post_m, coll)
        text(f"sign_{d}m_txt", "FINISH" if fin else f"{d}m", (d, sy - 0.08, 1.82), (90, 0, 0),
             0.85 if fin else 1.15, num_m, coll)

    # stands: stepped concrete with blue seats and a crowd
    stand_m = material("stand", COL["stand"], rough=0.8)
    seat_m = material("seat", COL["seat"], rough=0.6)
    tiers = []
    for t in range(9):                                  # front row where the ad boards used to be
        y, z = y1 + 2.4 + 1.6 * t, 0.9 + 0.55 * t
        box(f"stand_{t}", (length, 1.6, z), (mid_x, y + 0.8, z / 2), stand_m, coll)
        box(f"seats_{t}", (length, 0.5, 0.12), (mid_x, y + 1.1, z + 0.06), seat_m, coll)
        tiers.append((y + 0.75, z))
    box("stand_wall", (length, 0.4, 1.0), (mid_x, y1 + 2.0, 0.5), stand_m, coll)
    build_crowd(coll, TRACK_X[0], TRACK_X[1], tiers)
    back_y = y1 + 2.4 + 1.6 * 9 + 0.5
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
    bg.inputs["Strength"].default_value = 0.7
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 5.0
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
    # wide, fixed shot of the start area and the attempts board (for sped-up training montages)
    wide = bpy.data.objects.new("cam_wide_start", bpy.data.cameras.new("cam_wide_start"))
    wide.data.lens = 30
    wide.location = (0.5, -13.5, 4.4)
    look = bpy.data.objects.new("cam_wide_target", None)
    look.location = (2.5, 3.0, 1.4)
    scene.collection.objects.link(look)
    t = wide.constraints.new("TRACK_TO")
    t.target, t.track_axis, t.up_axis = look, "TRACK_NEGATIVE_Z", "UP_Y"
    scene.collection.objects.link(wide)
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
    ap = argparse.ArgumentParser(prog="build_scene.py")
    ap.add_argument("replay")
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--screen", default="TRAINING", help="text on the stadium's big screen")
    ap.add_argument("--costume", default="ep01_70s")
    ap.add_argument("--attempts", type=int, default=0, help="number on the stadium attempts board")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    replay, attempt, out, screen_text = args.replay, args.attempt, args.out, args.screen
    bpy.ops.wm.read_factory_settings(use_empty=True)
    empties = import_replay(replay, attempt)
    rupert = bpy.data.collections.new("rupert")
    stadium = bpy.data.collections.new("stadium")
    bpy.context.scene.collection.children.link(rupert)
    bpy.context.scene.collection.children.link(stadium)
    build_rupert(empties, rupert, args.costume)
    build_stadium(stadium, screen_text)
    y_far = lane_edge(N_LANES + 1)
    board = attempts_board.build_board(stadium, (-4.5, y_far + 1.0, 1.3), digit_h=0.8)
    attempts_board.set_number(board, 1, args.attempts)
    build_world_and_light()
    build_cameras(empties)
    render_settings()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out).resolve()))
    print(f"Saved scene {out}")


main()
