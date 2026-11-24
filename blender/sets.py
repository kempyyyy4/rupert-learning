"""Sets for the series-preview clips (rupert/previews.py has the poses and the numbers):
a bicycle, a candlelit table for two at night, and a trading desk with three screens.

Called from props.py:  "bike": {...}, "date": {...}, "trading": {...}, "night": true
"""
import math

import bmesh
import bpy
from mathutils import Vector

import bl_helpers as H


def _tube(name, a, b, r, mat, coll, parent=None):
    a, b = Vector(a), Vector(b)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=14, radius1=r, radius2=r, depth=(b - a).length)
    o = H.mesh_object(name, bm, mat, coll)
    o.location = (a + b) / 2
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (b - a).to_track_quat("Z", "Y")
    if parent:
        o.parent = parent
    return o


def _ring(name, radius, tube, mat, coll, segs=48):
    """A torus in the x-z plane (a wheel or tyre), centred on the origin."""
    bm = bmesh.new()
    rings = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        c = Vector((radius * math.sin(a), 0, radius * math.cos(a)))
        out = c.normalized()
        ring = []
        for j in range(10):
            b = 2 * math.pi * j / 10
            ring.append(bm.verts.new(c + tube * (math.cos(b) * out + math.sin(b) * Vector((0, 1, 0)))))
        rings.append(ring)
    for i in range(segs):
        r0, r1 = rings[i], rings[(i + 1) % segs]
        for j in range(10):
            bm.faces.new((r0[j], r0[(j + 1) % 10], r1[(j + 1) % 10], r1[j]))
    return H.mesh_object(name, bm, mat, coll)


def bike(spec, coll):
    """A bicycle (bike frame: x forward, z up, origin on the ground under the pedals), moved every
    frame by spec["keys"] = [[location, quaternion, wheel angle, crank angle], ...]."""
    paint = H.material("bike_paint", (0.85, 0.05, 0.05), rough=0.3)
    chrome = H.material("bike_chrome", (0.8, 0.8, 0.82), rough=0.15)
    chrome.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 1.0
    rubber = H.material("bike_tyre", (0.03, 0.03, 0.03), rough=0.8)
    black = H.material("bike_black", (0.05, 0.05, 0.05), rough=0.5)
    root = bpy.data.objects.new("bike_root", None)
    root.rotation_mode = "QUATERNION"
    coll.objects.link(root)
    R, crank, saddle, bars = spec["wheel_r"], Vector(spec["crank"]), Vector(spec["saddle"]), Vector(spec["bars"])
    rear, front = Vector((spec["wheelbase"][0], 0, R)), Vector((spec["wheelbase"][1], 0, R))
    seat_top = saddle + Vector((0.03, 0, -0.06))
    head_top = Vector((bars.x - 0.06, 0, bars.z - 0.1))
    head_bot = Vector((front.x - 0.12, 0, front.z + 0.22))
    for name, a, b in (("down", crank, head_bot), ("top", seat_top + Vector((0.02, 0, -0.06)), head_top),
                       ("seat", crank, seat_top), ("head", head_bot, head_top)):
        _tube(f"bike_{name}_tube", a, b, 0.02, paint, coll, root)
    for s in (-1, 1):                                   # stays and forks, both sides of the wheels
        off = Vector((0, s * 0.05, 0))
        _tube(f"bike_chain_stay_{s}", crank + off * 0.6, rear + off, 0.012, paint, coll, root)
        _tube(f"bike_seat_stay_{s}", seat_top + off * 0.3, rear + off, 0.011, paint, coll, root)
        _tube(f"bike_fork_{s}", head_bot + off * 0.6, front + off, 0.013, chrome, coll, root)
    _tube("bike_stem", head_top, bars, 0.015, chrome, coll, root)
    _tube("bike_bar", bars + Vector((0, -0.3, 0)), bars + Vector((0, 0.3, 0)), 0.013, chrome, coll, root)
    for s in (-1, 1):
        _tube(f"bike_grip_{s}", bars + Vector((0, s * 0.2, 0)), bars + Vector((0, s * 0.31, 0)), 0.02, black, coll, root)
    _tube("bike_seat_post", seat_top, saddle, 0.012, chrome, coll, root)
    sad = H.mesh_object("bike_saddle", H.rounded_box_bm((0.26, 0.15, 0.05), 0.02), black, coll)
    sad.location, sad.parent = saddle + Vector((0.02, 0, 0.02)), root
    wheels = []
    for name, c in (("rear", rear), ("front", front)):
        hub = bpy.data.objects.new(f"bike_{name}_hub", None)
        hub.location, hub.parent = c, root
        coll.objects.link(hub)
        _ring(f"bike_{name}_tyre", R - 0.02, 0.022, rubber, coll).parent = hub
        _ring(f"bike_{name}_rim", R - 0.05, 0.01, chrome, coll).parent = hub
        for k in range(8):
            a = math.pi * k / 8
            d = Vector((math.sin(a), 0, math.cos(a))) * (R - 0.05)
            _tube(f"bike_{name}_spoke_{k}", -d, d, 0.003, chrome, coll, hub)
        wheels.append(hub)
    cr = bpy.data.objects.new("bike_crank", None)
    cr.location, cr.parent = crank, root
    coll.objects.link(cr)
    L = spec["crank_l"]
    for s, ph in ((-1, 0.0), (1, math.pi)):
        tip = Vector((L * math.sin(ph), s * 0.1, L * math.cos(ph)))
        _tube(f"bike_crank_arm_{s}", Vector((0, s * 0.07, 0)), tip, 0.012, chrome, coll, cr)
        ped = H.mesh_object(f"bike_pedal_{s}", H.rounded_box_bm((0.1, 0.08, 0.02), 0.006), black, coll)
        ped.parent, ped.location = cr, tip + Vector((0, s * 0.03, 0))
    chainring = _ring("bike_chainring", 0.1, 0.008, chrome, coll)
    chainring.parent, chainring.location = cr, (0, -0.07, 0)
    for f, (loc, quat, wheel, crank_a) in enumerate(spec["keys"], start=1):
        root.location, root.rotation_quaternion = loc, quat
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_quaternion", frame=f)
        for w in wheels:
            w.rotation_euler = (0, wheel, 0)
            w.keyframe_insert("rotation_euler", frame=f)
        cr.rotation_euler = (0, crank_a, 0)
        cr.keyframe_insert("rotation_euler", frame=f)
        for p in [o for o in cr.children if o.name.startswith("bike_pedal")]:
            p.rotation_euler = (0, -crank_a, 0)                # pedals stay level
            p.keyframe_insert("rotation_euler", frame=f)
    return root


def night(spec=None):
    """Night in the stadium: dark sky, dim blue moonlight."""
    scene = bpy.context.scene
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = 0.04
    sun = bpy.data.objects.get("sun")
    if sun:
        sun.data.energy, sun.data.color = 0.25, (0.55, 0.65, 1.0)


def date(spec, coll):
    """A small round table for two with a white cloth, a candle, wine glasses, plates and a rose."""
    x, y = spec["table"]
    h = spec["table_h"]
    cloth = H.material("tablecloth", (0.95, 0.94, 0.9), rough=0.8, sheen=0.4)
    wood = H.material("chair_wood", (0.25, 0.12, 0.05), rough=0.5)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=40, radius1=0.5, radius2=0.45, depth=h)
    top = H.mesh_object("date_table", bm, cloth, coll)
    top.location = (x, y, h / 2)
    for s in (-1, 1):                                  # chairs either side
        cx = x + s * 0.8
        H.box(f"date_chair_seat_{s}", (0.42, 0.42, 0.05), (cx, y, 0.44), wood, coll, bevel=0.01)
        H.box(f"date_chair_back_{s}", (0.04, 0.42, 0.55), (cx + s * 0.2, y, 0.74), wood, coll, bevel=0.01)
        for dx in (-0.17, 0.17):
            for dy in (-0.17, 0.17):
                H.box(f"date_chair_leg_{s}", (0.04, 0.04, 0.44), (cx + dx, y + dy, 0.22), wood, coll)
    plate = H.material("plate", (0.98, 0.98, 0.98), rough=0.2)
    glass = H.material("wine_glass", (0.9, 0.95, 1.0), rough=0.05)
    glass.node_tree.nodes["Principled BSDF"].inputs["Transmission Weight"].default_value = 0.9
    wine = H.material("wine", (0.35, 0.0, 0.05), rough=0.1)
    for s in (-1, 1):
        px = x + s * 0.27
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.12, radius2=0.14, depth=0.015)
        p = H.mesh_object(f"date_plate_{s}", bm, plate, coll)
        p.location = (px, y, h + 0.01)
        gx, gy = px - s * 0.02, y - s * 0.2
        _tube(f"date_glass_stem_{s}", (gx, gy, h), (gx, gy, h + 0.11), 0.006, glass, coll)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=0.03, radius2=0.045, depth=0.09)
        cup = H.mesh_object(f"date_glass_{s}", bm, glass, coll)
        cup.location = (gx, gy, h + 0.155)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.028, radius2=0.036, depth=0.04)
        w = H.mesh_object(f"date_wine_{s}", bm, wine, coll)
        w.location = (gx, gy, h + 0.135)
    # candle with a flickering flame and its warm light
    wax = H.material("candle_wax", (0.95, 0.92, 0.8), rough=0.6)
    _tube("date_candle", (x, y + 0.05, h), (x, y + 0.05, h + 0.2), 0.022, wax, coll)
    flame_m = H.material("flame", (1.0, 0.6, 0.15), emit=25.0)
    flame = H.mesh_object("date_flame", H.sphere_bm(0.014, scale=(1, 1, 2.2)), flame_m, coll)
    flame.location = (x, y + 0.05, h + 0.235)
    light = bpy.data.lights.new("candle_light", "POINT")
    light.color, light.shadow_soft_size = (1.0, 0.62, 0.3), 0.05
    lo = bpy.data.objects.new("candle_light", light)
    lo.location = (x, y + 0.05, h + 0.3)
    coll.objects.link(lo)
    import random
    rnd = random.Random(4)
    for f in range(1, spec.get("frames", 200) + 1, 3):
        light.energy = rnd.uniform(28, 45)
        light.keyframe_insert("energy", frame=f)
        flame.scale = (1, 1, rnd.uniform(0.85, 1.15))
        flame.keyframe_insert("scale", frame=f)
    # a soft warm fill so his face reads (like the candle, from the front)
    fill = bpy.data.lights.new("date_fill", "AREA")
    fill.energy, fill.color, fill.size = 70.0, (1.0, 0.88, 0.75), 1.0
    fo = bpy.data.objects.new("date_fill", fill)
    fo.location = (x + 1.2, y - 0.6, h + 0.9)
    fo.rotation_euler = (0, math.radians(-60), math.radians(-25))
    coll.objects.link(fo)
    # a rose in a little vase
    vase = H.material("vase", (0.6, 0.8, 0.9), rough=0.1)
    _tube("date_vase", (x + 0.1, y + 0.3, h), (x + 0.1, y + 0.3, h + 0.12), 0.025, vase, coll)
    _tube("date_rose_stem", (x + 0.1, y + 0.3, h + 0.1), (x + 0.12, y + 0.3, h + 0.3), 0.004,
          H.material("stem", (0.05, 0.35, 0.08), rough=0.6), coll)
    rose = H.mesh_object("date_rose", H.sphere_bm(0.03), H.material("rose", (0.75, 0.0, 0.05), rough=0.5), coll)
    rose.location = (x + 0.12, y + 0.3, h + 0.32)
    # string lights hanging over the table
    bulb = H.material("bulb", (1.0, 0.8, 0.45), emit=12.0)
    for k in range(13):
        u = k / 12
        p = Vector((x - 1.6 + 3.2 * u, y + 1.0, 2.6 - 0.5 * math.sin(math.pi * u)))
        b = H.mesh_object(f"date_bulb_{k}", H.sphere_bm(0.035), bulb, coll)
        b.location = p


def screen(name, image_first, n_frames, loc, size, yaw_deg, coll, tilt_deg=8):
    """A monitor on a little stand, showing an image sequence."""
    w, h = size
    body = H.material("monitor_body", (0.04, 0.04, 0.05), rough=0.4)
    root = bpy.data.objects.new(f"{name}_root", None)
    root.location = loc
    root.rotation_euler = (math.radians(-tilt_deg), 0, math.radians(yaw_deg))
    coll.objects.link(root)
    case = H.box(f"{name}_case", (0.03, w + 0.03, h + 0.03), (0, 0, 0), body, coll, bevel=0.008)
    case.parent = root
    img = bpy.data.images.load(str(image_first))
    img.source = "SEQUENCE"
    mat = bpy.data.materials.new(f"{name}_screen")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.image_user.frame_duration, tex.image_user.frame_start = n_frames, 1
    tex.image_user.use_auto_refresh = True
    links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.6
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    face = bpy.context.active_object
    for c in face.users_collection:
        c.objects.unlink(face)
    coll.objects.link(face)
    face.name = f"{name}_face"
    face.data.materials.append(mat)
    face.parent = root
    face.location = (-0.017, 0, 0)
    face.rotation_euler = (math.radians(90), 0, math.radians(-90))     # faces -x (towards the trader)
    face.scale = (w, h, 1)
    H.box(f"{name}_neck", (0.04, 0.04, 0.22), (0.03, 0, -h / 2 - 0.1), body, coll).parent = root
    return root


def trading(spec, coll):
    """A long dark desk with three screens of green numbers going up, a keyboard and an office chair."""
    x, y = spec["desk"]
    top = H.material("trading_desk", (0.12, 0.12, 0.13), rough=0.35)
    H.box("trading_desk_top", (0.8, 1.9, 0.05), (x + 0.15, y, 0.74), top, coll, bevel=0.01)
    for dy in (-0.9, 0.9):
        H.box("trading_desk_leg", (0.7, 0.05, 0.72), (x + 0.15, y + dy, 0.36), top, coll)
    kb = H.material("keyboard", (0.15, 0.15, 0.16), rough=0.5)
    H.box("trading_keyboard", (0.17, 0.45, 0.025), (x - 0.12, y, 0.78), kb, coll, bevel=0.005)
    H.box("trading_mouse", (0.1, 0.06, 0.03), (x - 0.1, y - 0.36, 0.78), kb, coll, bevel=0.012)
    chair = H.material("office_chair", (0.05, 0.05, 0.06), rough=0.6)
    H.box("trading_chair_seat", (0.48, 0.48, 0.08), (x - 0.64, y, 0.44), chair, coll, bevel=0.03)
    H.box("trading_chair_back", (0.07, 0.46, 0.62), (x - 0.92, y, 0.82), chair, coll, bevel=0.03)
    _tube("trading_chair_post", (x - 0.64, y, 0.05), (x - 0.64, y, 0.42), 0.03, chair, coll)
    for k, (dy, yaw, kind) in enumerate(((-0.62, -28, 0), (0.0, 0, 1), (0.62, 28, 2))):
        screen(f"trading_screen_{k}", spec["images"][kind], spec["frames"],
               (x + 0.36, y + dy, 1.2), (0.62, 0.36), yaw, coll)
    mug = H.material("mug", (0.95, 0.95, 0.95), rough=0.3)
    _tube("trading_mug", (x + 0.0, y + 0.55, 0.765), (x + 0.0, y + 0.55, 0.87), 0.04, mug, coll)
