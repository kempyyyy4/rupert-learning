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


def build(spec, empties, coll):
    for c in spec.get("cameras", []):
        keyframed_camera(c)
    if "whiteboard" in spec:
        whiteboard(spec["whiteboard"], coll)
    if spec.get("pointer"):
        pointer(empties, coll)
    if "camera" in spec:
        bpy.context.scene.camera = bpy.data.objects[spec["camera"]]
