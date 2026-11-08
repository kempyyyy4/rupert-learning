"""Small Blender building blocks shared by the scene, Rupert and his costumes."""
import math

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.geometry import intersect_point_line

METABALL_SCALE = 1 / 0.57   # Blender metaball: the surface sits at ~0.57 x the element radius


# ---- materials ---------------------------------------------------------------------
def fcurves(obj):
    """An object's animation curves (works before and after Blender 5's layered actions)."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    if hasattr(ad.action, "fcurves"):
        return ad.action.fcurves
    from bpy_extras.anim_utils import action_get_channelbag_for_slot
    bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return bag.fcurves if bag else []


def set_interpolation(obj, mode, frames=None):
    """Set how keys blend into the next one: 'CONSTANT' (snap), 'LINEAR' or 'BEZIER'.
    frames: only the keys on these frames (default: every key)."""
    for fc in fcurves(obj):
        for kp in fc.keyframe_points:
            if frames is None or int(round(kp.co.x)) in frames:
                kp.interpolation = mode


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


def node_material(name, build, rough=0.5, sheen=0.0):
    """Material whose node tree is built by build(nodes, links, bsdf)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = rough
    if sheen:
        bsdf.inputs["Sheen Weight"].default_value = sheen
    build(nt.nodes, nt.links, bsdf)
    return m


class Maths:
    """Tiny helper for chains of shader Math nodes: m.op('ADD', a, b) with sockets or numbers."""

    def __init__(self, nodes, links):
        self.nodes, self.links = nodes, links

    def op(self, operation, a, b=0.0, c=0.0):
        n = self.nodes.new("ShaderNodeMath")
        n.operation = operation
        for i, v in enumerate((a, b, c)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.links.new(v, n.inputs[i])
        return n.outputs[0]


def two_colour(nodes, links, fac, rgb_a, rgb_b, bsdf):
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*rgb_a, 1)
    mix.inputs["B"].default_value = (*rgb_b, 1)
    links.new(fac, mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])


# ---- meshes ------------------------------------------------------------------------
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


def sphere_bm(radius, top_half_only=False, scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=18, radius=radius)
    if top_half_only:
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-4], context="VERTS")
    bmesh.ops.scale(bm, vec=scale, verts=bm.verts)
    return bm


def tube_bm(radius, length, segments=40):
    """Open-ended tube along Z (add a Solidify modifier for thickness)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=segments, radius1=radius, radius2=radius, depth=length)
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


def text(name, body, loc, rot, size, mat, coll, extrude=0.005, font="arialbd.ttf"):
    c = bpy.data.curves.new(name, "FONT")
    c.body, c.size, c.extrude = body, size, extrude
    c.align_x, c.align_y = "CENTER", "CENTER"
    try:
        c.font = bpy.data.fonts.load(f"C:/Windows/Fonts/{font}", check_existing=True)
    except (RuntimeError, OSError):
        pass
    obj = bpy.data.objects.new(name, c)
    obj.data.materials.append(mat)
    obj.location, obj.rotation_euler = loc, [math.radians(a) for a in rot]
    coll.objects.link(obj)
    return obj


def tube_curve(name, points, radius, mat, coll):
    """A smooth round tube through 3D points (e.g. a smile)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution, cu.use_fill_caps = radius, 4, True
    sp = cu.splines.new("NURBS")
    sp.points.add(len(points) - 1)
    for p, co in zip(sp.points, points):
        p.co = (*co, 1.0)
    sp.order_u, sp.use_endpoint_u = 3, True
    obj = bpy.data.objects.new(name, cu)
    cu.materials.append(mat)
    coll.objects.link(obj)
    return obj


def attach(obj, parent, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    obj.parent = parent
    obj.rotation_mode = "QUATERNION"
    obj.location, obj.rotation_quaternion = pos, quat


# ---- metaball skins ----------------------------------------------------------------
def metaball_mesh(name, elements, coll, resolution=0.012):
    """elements: (kind 'ball'|'capsule', matrix, surface radius, half length along local Z).

    Returns a mesh object (the blobs melted into one smooth surface).
    """
    to_z = Quaternion((0, 1, 0), -math.pi / 2)     # metaball capsules run along X; ours along Z
    mb = bpy.data.metaballs.new(name)
    mb.resolution = mb.render_resolution = resolution
    mb.threshold = 0.6
    for kind, m, r, h in elements:
        e = mb.elements.new(type="CAPSULE" if kind == "capsule" else "BALL")
        e.co = m.to_translation()
        e.rotation = m.to_quaternion() @ to_z
        e.radius = r * METABALL_SCALE
        e.stiffness = 2.0
        if kind == "capsule":
            e.size_x = h
    obj = bpy.data.objects.new(name + "_mb", mb)
    coll.objects.link(obj)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    coll.objects.unlink(obj)
    bpy.data.objects.remove(obj)
    mesh_obj = bpy.data.objects.new(name, me)
    coll.objects.link(mesh_obj)
    for p in me.polygons:
        p.use_smooth = True
    return mesh_obj


def _dist(p, a, b):
    if (b - a).length < 1e-6:
        return (p - a).length
    _, t = intersect_point_line(p, a, b)
    t = min(max(t, 0.0), 1.0)
    return (p - (a + (b - a) * t)).length


def skin_to_skeleton(obj, bodies, segs, skeleton, blend=0.012):
    """Each surface point follows the body part it's closest to, blended at the joints."""
    vgs = {b: obj.vertex_groups.new(name=b) for b in bodies}
    for v in obj.data.vertices:
        d = {b: min(_dist(v.co, a, c) for a, c in segs[b]) for b in bodies if b in segs}
        dmin = min(d.values())
        w = {b: math.exp(-(x - dmin) / blend) for b, x in d.items()}
        total = sum(w.values())
        for b, x in w.items():
            if x / total > 0.02:
                vgs[b].add([v.index], x / total, "REPLACE")
    mod = obj.modifiers.new("skeleton", "ARMATURE")
    mod.object = skeleton


def store_rest_height(obj, z0, name="rest_z"):
    """Remember each point's standing-pose height (for stripes that move with the cloth)."""
    attr = obj.data.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", [v.co.z - z0 for v in obj.data.vertices])


def rest_matrix(rest, body):
    r = rest[body]
    return Matrix.Translation(r["pos"]) @ Quaternion(r["quat"]).to_matrix().to_4x4()


def shape_matrix(s):
    return Matrix.Translation(s["pos"]) @ Quaternion(s["quat"]).to_matrix().to_4x4()


def look_along(direction, axis=(1, 0, 0)):
    """Quaternion turning the given local axis to point along direction."""
    return Vector(axis).rotation_difference(Vector(direction).normalized())
