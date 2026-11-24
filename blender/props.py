"""Props and camera moves for the hand-animated shots (build_scene.py --props file.json):

    {"cameras": [{"name": "cam_opening", "fstop": 2.8,
                  "keys": [[frame, [x, y, z], [aim_x, aim_y, aim_z], lens_mm], ...]}],
     "whiteboard": {"image": "board.png", "centre": [x, y, z], "size": [w, h], "land": frame},
     "pointer": true,
     "camera": "cam_opening"}          # the scene's active camera
"""
import math

import bpy

import bl_helpers as H

STICK = 0.75        # same as rupert/poses.py


def keyframed_camera(spec):
    scene = bpy.context.scene
    name = spec["name"]
    cam = bpy.data.cameras.new(name)
    look = bpy.data.objects.new(f"{name}_target", None)
    scene.collection.objects.link(look)
    cam.dof.use_dof, cam.dof.focus_object = True, look
    cam.dof.aperture_fstop = spec.get("fstop", 4.0)
    obj = bpy.data.objects.new(name, cam)
    scene.collection.objects.link(obj)
    t = obj.constraints.new("TRACK_TO")
    t.target, t.track_axis, t.up_axis = look, "TRACK_NEGATIVE_Z", "UP_Y"
    for frame, loc, aim, lens in spec["keys"]:
        obj.location, look.location, cam.lens = loc, aim, lens
        obj.keyframe_insert("location", frame=frame)
        look.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("lens", frame=frame)
    return obj


def _parent(obj, root):
    obj.parent = root
    obj.matrix_parent_inverse = root.matrix_world.inverted()


def whiteboard(spec, coll):
    """A whiteboard that drops out of the sky on two cables, lands with a bounce at frame `land`,
    and faces -y (the camera side). The drawing is an image (rupert/board_art.py)."""
    (cx, cy, cz), (w, h) = spec["centre"], spec["size"]
    root = bpy.data.objects.new("whiteboard", None)
    root.location = (cx, cy, cz)
    coll.objects.link(root)
    bpy.context.view_layer.update()
    alu = H.material("board_frame", (0.72, 0.74, 0.77), rough=0.35)
    parts = [H.box("board_back", (w + 0.09, 0.035, h + 0.09), (cx, cy, cz), alu, coll, bevel=0.012),
             H.box("board_tray", (w * 0.6, 0.09, 0.03), (cx, cy - 0.06, cz - h / 2 - 0.03), alu, coll, bevel=0.006)]
    img = bpy.data.images.load(spec["image"])
    mat = bpy.data.materials.new("board_drawing")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    if spec.get("sequence"):                   # a moving drawing: f_0001.png, f_0002.png ... one per frame
        img.source = "SEQUENCE"
        tex.image_user.frame_duration = spec["sequence"]
        tex.image_user.frame_start = 1
        tex.image_user.frame_offset = 0
        tex.image_user.use_auto_refresh = True
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55          # a little sheen, but no sky glare
    links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.35  # keeps the white white and the ink crisp
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy - 0.019, cz), rotation=(math.radians(90), 0, 0))
    face = bpy.context.active_object
    face.name = "board_face"
    face.scale = (w, h, 1)
    face.data.materials.append(mat)
    for c in face.users_collection:
        c.objects.unlink(face)
    coll.objects.link(face)
    parts.append(face)
    cable = H.material("board_cable", (0.1, 0.1, 0.1), rough=0.5)
    for s in (-1, 1):
        x = cx + s * (w / 2 - 0.15)
        parts.append(H.tube_curve(f"board_cable_{s}", [(x, cy, cz + h / 2 + 0.04), (x, cy, cz + 30)], 0.006, cable, coll))
    for p in parts:
        _parent(p, root)
    land = spec["land"]
    fall = spec.get("fall_frames", 22)
    for frame, dz in ((1, 7.0), (land - fall, 7.0), (land, 0.0), (land + 5, -0.07), (land + 10, 0.025), (land + 15, 0.0)):
        root.location = (cx, cy, cz + dz)
        root.keyframe_insert("location", frame=frame)
    for fc in H.fcurves(root):                     # falls like a dropped weight, then springs
        for k in fc.keyframe_points:
            k.interpolation = "BEZIER"
        if len(fc.keyframe_points) > 2:
            fc.keyframe_points[1].interpolation = "QUAD"
            fc.keyframe_points[1].easing = "EASE_IN"
    return root


def pointer(empties, coll):
    """Teacher's pointer stick in his right hand, carrying on the line of his forearm."""
    import bmesh
    wood = H.material("pointer_wood", (0.45, 0.26, 0.1), rough=0.55)
    tip = H.material("pointer_tip", (0.05, 0.05, 0.05), rough=0.5)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.011, radius2=0.006, depth=STICK + 0.06)
    for v in bm.verts:                                    # along +y, from just behind the wrist to the tip
        v.co = (v.co.x, v.co.z + (STICK - 0.06) / 2, -v.co.y)
    stick = H.mesh_object("pointer_stick", bm, wood, coll)
    bm = H.sphere_bm(0.011)
    end = H.mesh_object("pointer_tip", bm, tip, coll)
    end.location = (0, STICK, 0)
    end.parent = stick
    c = stick.constraints.new("COPY_LOCATION")
    c.target = empties["right_hand"]
    d = stick.constraints.new("DAMPED_TRACK")
    d.target, d.track_axis = empties["right_lower_arm"], "TRACK_NEGATIVE_Y"   # -y at the elbow: +y along the forearm
    return stick


def _anchor(name, parent, pos, coll):
    e = bpy.data.objects.new(name, None)
    coll.objects.link(e)
    H.attach(e, parent, pos)
    return e


def whistle(spec, empties, coll):
    """A silver whistle hanging on a black cord round the neck. At frame `grab` the right hand
    takes it; it rides in the fist up to the mouth (frame `mouth`)."""
    import bmesh
    metal = H.material("whistle_metal", (0.82, 0.82, 0.85), rough=0.18)
    metal.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 1.0
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=0.013, radius2=0.013, depth=0.024)
    for v in bm.verts:                        # barrel across, sideways (along y)
        v.co = (v.co.x, v.co.z, v.co.y)
    w = H.mesh_object("whistle", bm, metal, coll)
    w.scale = (1.6, 1.6, 1.6)                 # cartoon-sized, so it reads on screen
    mp = H.mesh_object("whistle_mouthpiece", H.rounded_box_bm((0.012, 0.016, 0.034), 0.003), metal, coll)
    mp.parent = w                             # mouthpiece sticks up when it hangs (+z)
    mp.location = (0.0, 0.0, 0.024)
    torso = empties["torso"]
    chest = _anchor("whistle_on_chest", torso, spec.get("chest", (0.175, 0.0, -0.06)), coll)
    # a point in the fist: at the hand, pushed a little further along the forearm
    base = bpy.data.objects.new("whistle_hand_base", None)
    coll.objects.link(base)
    c = base.constraints.new("COPY_LOCATION"); c.target = empties["right_hand"]
    t = base.constraints.new("DAMPED_TRACK"); t.target, t.track_axis = empties["right_lower_arm"], "TRACK_NEGATIVE_Y"
    fist = bpy.data.objects.new("whistle_in_fist", None)
    coll.objects.link(fist)
    fist.parent = base
    fist.location = (0.0, spec.get("fist", 0.06), 0.0)
    a = w.constraints.new("COPY_LOCATION"); a.target = chest
    b = w.constraints.new("COPY_LOCATION"); b.target = fist
    r = w.constraints.new("COPY_ROTATION"); r.target, r.mix_mode = torso, "BEFORE"
    grab, mouth = spec["grab"], spec["mouth"]
    for f, v in ((1, 0.0), (grab, 0.0), (grab + 3, 1.0)):
        b.influence = v
        w.keyframe_insert(f'constraints["{b.name}"].influence', frame=f)
    # own rotation (in the chest's frame): hanging with the mouthpiece up, then pointing back into his mouth
    for f, rot in ((grab, (0, 0, 0)), (mouth, (0, math.radians(-90), 0))):
        w.rotation_euler = rot
        w.keyframe_insert("rotation_euler", frame=f)
    cord_m = H.material("whistle_cord", (0.02, 0.02, 0.02), rough=0.7)
    for s in (1, -1):
        root = _anchor(f"cord_root_{s}", torso, (0.05, s * 0.065, 0.16), coll)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=8, radius1=0.0025, radius2=0.0025, depth=1.0)
        for v in bm.verts:                    # along +y from the origin, 1 m long
            v.co = (v.co.x, v.co.z + 0.5, -v.co.y)
        cord = H.mesh_object(f"whistle_cord_{s}", bm, cord_m, coll)
        cc = cord.constraints.new("COPY_LOCATION"); cc.target = root
        st = cord.constraints.new("STRETCH_TO"); st.target, st.rest_length, st.volume = w, 1.0, "NO_VOLUME"
    return w


def race_clock(spec, coll):
    """Trackside race clock: amber LED 'TIME' board counting tenths of a second.
    spec: {"loc": [x, y, z bottom], "digit_h": 0.6, "frames": [[frame, tenths], ...]}"""
    import attempts_board
    board = attempts_board.build_board(coll, spec["loc"], digit_h=spec.get("digit_h", 0.6), name="race_clock",
                                       label="TIME", n_digits=3, point_after=2)
    board["min_digits"] = 2
    for frame, tenths in spec["frames"]:
        attempts_board.set_number(board, frame, tenths)
    attempts_board.snap_numbers(board)
    return board


def _poly_tube(name, points, radius, mat, coll):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution = radius, 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, co in zip(sp.points, points):
        p.co = (*co, 1.0)
    obj = bpy.data.objects.new(name, cu)
    cu.materials.append(mat)
    coll.objects.link(obj)
    return obj


def _key_visible(obj, frames_on, first, last):
    """Visible (in render) only on the frames in frames_on, between first and last."""
    for f in range(first - 1, last + 2):
        obj.hide_render = f not in frames_on
        obj.keyframe_insert("hide_render", frame=f)
    obj.hide_viewport = False


def _bone(name, a, b, radius, mat, coll):
    """A bone from empty a stretching to empty b (both ends follow the skeleton)."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=radius, radius2=radius, depth=1.0)
    for v in bm.verts:
        v.co = (v.co.x, v.co.z + 0.5, -v.co.y)
    o = H.mesh_object(name, bm, mat, coll)
    c = o.constraints.new("COPY_LOCATION"); c.target = a
    s = o.constraints.new("STRETCH_TO"); s.target, s.rest_length, s.volume = b, 1.0, "NO_VOLUME"
    return o


def xray_skeleton(empties, coll):
    """A cartoon skeleton for the X-ray flash frames: its own collection, hidden except in that pass."""
    import bmesh
    xr = bpy.data.collections.new("xray")
    bpy.context.scene.collection.children.link(xr)
    bone_m = H.material("xray_bone", (0.85, 0.95, 1.0), rough=0.5, emit=3.0)
    dark = H.material("xray_socket", (0.0, 0.0, 0.0), rough=0.9)
    E = empties
    for a, b, r in (("pelvis", "torso", 0.022), ("torso", "head", 0.02),
                    ("right_thigh", "right_shin", 0.03), ("left_thigh", "left_shin", 0.03),
                    ("right_shin", "right_foot", 0.025), ("left_shin", "left_foot", 0.025),
                    ("right_upper_arm", "right_lower_arm", 0.024), ("left_upper_arm", "left_lower_arm", 0.024),
                    ("right_lower_arm", "right_hand", 0.02), ("left_lower_arm", "left_hand", 0.02),
                    ("right_upper_arm", "torso", 0.018), ("left_upper_arm", "torso", 0.018),
                    ("right_thigh", "pelvis", 0.022), ("left_thigh", "pelvis", 0.022)):
        _bone(f"xray_{a}_{b}", E[a], E[b], r, bone_m, xr)
    for name in ("right_shin", "left_shin", "right_lower_arm", "left_lower_arm", "right_foot", "left_foot",
                 "right_hand", "left_hand", "right_upper_arm", "left_upper_arm", "right_thigh", "left_thigh"):
        j = H.mesh_object(f"xray_joint_{name}", H.sphere_bm(0.04 if "hand" in name or "foot" in name else 0.035),
                          bone_m, xr)
        H.attach(j, E[name])
    for k in range(5):                                       # ribs: rings round the chest
        rx, ry, z = 0.12 - 0.006 * k, 0.15 - 0.008 * k, 0.03 - 0.07 * k
        pts = [(rx * math.cos(a), ry * math.sin(a), z) for a in (2 * math.pi * i / 24 for i in range(25))]
        rib = _poly_tube(f"xray_rib_{k}", pts, 0.012, bone_m, xr)
        H.attach(rib, E["torso"])
    skull = H.mesh_object("xray_skull", H.sphere_bm(0.15, scale=(1.0, 0.95, 1.05)), bone_m, xr)
    H.attach(skull, E["head"], (0.0, 0.0, 0.08))
    for s in (1, -1):
        eye = H.mesh_object(f"xray_eye_{s}", H.sphere_bm(0.04), dark, xr)
        H.attach(eye, E["head"], (0.125, s * 0.055, 0.09))
    nose = H.mesh_object("xray_nose", H.sphere_bm(0.022), dark, xr)
    H.attach(nose, E["head"], (0.145, 0.0, 0.03))
    jaw = H.mesh_object("xray_jaw", H.rounded_box_bm((0.12, 0.17, 0.05), 0.02), bone_m, xr)
    H.attach(jaw, E["head"], (0.07, 0.0, -0.06))
    xr.hide_render = True
    return xr


def zap(spec, empties, coll):
    """ZAP: lightning bolts flickering down onto him, a flash, then hair standing on end, sooty
    skin and vest, and puffs of smoke. spec: {"start": frame, "end": frame}."""
    import random
    rnd = random.Random(5)
    f0, f1 = spec["start"], spec["end"]
    scene = bpy.context.scene
    scene.frame_set(f0)
    head = empties["head"].matrix_world.translation.copy()
    bolt_m = H.material("bolt", (0.75, 0.9, 1.0), emit=40.0)
    # bolts: two shapes each, swapped every couple of frames so they crackle
    for k in range(4):
        end = head + __import__("mathutils").Vector((rnd.uniform(-0.15, 0.15), rnd.uniform(-0.15, 0.15), 0.12))
        top = end + __import__("mathutils").Vector((rnd.uniform(-3, 3), rnd.uniform(-1, 3), 9.0))
        for v in range(2):
            pts = []
            for i in range(13):
                t = i / 12
                p = top.lerp(end, t)
                jit = 0.0 if i in (0, 12) else 0.35 * (1 - t * 0.6)
                pts.append((p.x + rnd.uniform(-jit, jit), p.y + rnd.uniform(-jit, jit), p.z))
            b = _poly_tube(f"bolt_{k}_{v}", pts, 0.025, bolt_m, coll)
            on = {f for f in range(f0, f1) if ((f - f0) // 2 + k + v) % 2 == 0 and rnd.random() < 0.85}
            _key_visible(b, on, f0, f1)
    flash = bpy.data.lights.new("zap_flash", "POINT")
    flash.color, flash.shadow_soft_size = (0.75, 0.88, 1.0), 0.3
    fl = bpy.data.objects.new("zap_flash", flash)
    fl.location = head + __import__("mathutils").Vector((0.8, -1.2, 0.6))
    coll.objects.link(fl)
    for f in range(f0 - 1, f1 + 2):
        flash.energy = 0.0 if f < f0 or f >= f1 else rnd.choice((800.0, 3000.0, 6000.0))
        flash.keyframe_insert("energy", frame=f)
    # after: hair on end (spikes replace his hair), soot, smoke
    hair = bpy.data.objects.get("rupert_hair")
    if hair:
        _key_visible(hair, set(range(1, f1)), 1, f1)
    import bmesh
    from mathutils import Vector
    bm = bmesh.new()
    for el in range(28, 90, 12):
        for az in range(0, 360, 360 // max(1, int(14 * math.cos(math.radians(el))) + 3)):
            if el < 50 and (az < 55 or az > 305):               # keep his face clear
                continue
            dvec = Vector((math.cos(math.radians(el)) * math.cos(math.radians(az)),
                           math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))))
            base = Vector((0, 0, 0.07)) + dvec * 0.14
            length = rnd.uniform(0.13, 0.22)
            before = set(bm.verts)
            bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.035, radius2=0.002, depth=length)
            new = [v for v in bm.verts if v not in before]
            rot = dvec.to_track_quat("Z", "Y")
            for v in new:
                v.co = base + rot @ (v.co + Vector((0, 0, length / 2)))
    frizz = H.mesh_object("zap_frizz", bm, H.material("frizz", (0.1, 0.06, 0.035), rough=0.9), coll)
    frizz.parent = empties["head"]
    for f, s in ((1, 0.0), (f1 - 1, 0.0), (f1 + 3, 1.25), (f1 + 7, 1.0)):
        frizz.scale = (s, s, s)
        frizz.keyframe_insert("scale", frame=f)
    for mname, sooty in (("rupert_skin", (0.32, 0.27, 0.25)), ("vest", (0.35, 0.35, 0.35))):
        m = bpy.data.materials.get(mname)
        if not m or not m.node_tree:
            continue
        inp = m.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
        if inp.is_linked:
            continue
        before = tuple(inp.default_value)
        for f, c in ((f1 - 1, before), (f1, (*sooty, 1.0))):
            inp.default_value = c
            inp.keyframe_insert("default_value", frame=f)
    smoke_m = H.material("smoke", (0.55, 0.55, 0.57), rough=1.0)
    for k in range(9):
        puff = H.mesh_object(f"smoke_{k}", H.sphere_bm(1.0), smoke_m, coll)
        start = f1 + k * 7
        scene.frame_set(start)
        p = empties["head"].matrix_world.translation + Vector((rnd.uniform(-0.1, 0.1), rnd.uniform(-0.1, 0.1), 0.25))
        for f, dz, s in ((1, 0, 0.0), (start, 0, 0.0), (start + 6, 0.15, 0.09), (start + 40, 1.1, 0.22), (start + 55, 1.4, 0.0)):
            puff.location = p + Vector((0.1 * dz, 0, dz))
            puff.scale = (s, s, s)
            puff.keyframe_insert("location", frame=f)
            puff.keyframe_insert("scale", frame=f)
    xray_skeleton(empties, coll)


def build(spec, empties, coll):
    for c in spec.get("cameras", []):
        keyframed_camera(c)
    if "whiteboard" in spec:
        whiteboard(spec["whiteboard"], coll)
    if spec.get("pointer"):
        pointer(empties, coll)
    if "whistle" in spec:
        whistle(spec["whistle"], empties, coll)
    if "clock" in spec:
        race_clock(spec["clock"], coll)
    if "zap" in spec:
        zap(spec["zap"], empties, coll)
    if any(k in spec for k in ("bike", "date", "trading", "night")):
        import sets
        if spec.get("night"):
            sets.night()
        for k in ("bike", "date", "trading"):
            if k in spec:
                getattr(sets, k)(spec[k], coll)
    if "camera" in spec:
        bpy.context.scene.camera = bpy.data.objects[spec["camera"]]
