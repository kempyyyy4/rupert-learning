"""Rupert himself: off-white skin, the face from the channel's profile pictures
(E:/AI Channel/Channel Pictures/pfp_rupert_*.png, watermark_rupert_head.png), caramel hair,
and an ink outline like the drawings. Clothes come from costumes.py.

His skin is ONE smooth surface built around the physics body's limbs
(blender/rupert_shapes.json) and bent at the joints by a skeleton that copies the replay,
so limb lengths and joints are exactly the body that learned. Only thickness and the
head are exaggerated, for looks.
"""
import json
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import bl_helpers as H

HERE = Path(__file__).parent

SKIN = (0.78, 0.63, 0.47)          # warm off-white / cream, so his white vest and socks stand out
HAIR = (0.33, 0.13, 0.05)
HAIR_DARK = (0.17, 0.06, 0.02)         # caramel brown (#B8785A in the drawings, darkened for the sun)
INK = (0.012, 0.01, 0.01)         # eyes, brows, mouth and outline
MOUTH_INSIDE = (0.05, 0.008, 0.01)
OUTLINE_M = 0.0045                # ink outline thickness (metres); 0 = off

HEAD_R = 0.14                     # physics head is 0.098; cartoon head
HEAD_LIFT = 0.045                 # head centre sits this far above the physics head centre (cartoon neck)
# Head shape (character sheet): round dome, wide flat cheekbones, straight jawlines down to a narrow chin
HEAD_SHAPE = dict(chin_length=1.1, chin_width=0.34, cheekbone=0.2, face_flat=0.9)
SKIN_RADIUS = {
    "thigh": 0.072, "shin": 0.052, "upper_arm": 0.045, "lower_arm": 0.037, "hand": 0.05,
    "lower_waist": 0.09, "butt": 0.095, "torso": 0.07,
}
CHEST = dict(offset=-0.09, radius=0.115, half=0.1)   # one smooth chest instead of stacked bars
SKIN_GROUPS = {
    "skin_core": ["torso", "head", "lower_waist", "pelvis"],
    "skin_leg_r": ["right_thigh", "right_shin"],
    "skin_leg_l": ["left_thigh", "left_shin"],
    "skin_arm_r": ["right_upper_arm", "right_lower_arm", "right_hand"],
    "skin_arm_l": ["left_upper_arm", "left_lower_arm", "left_hand"],
}


def direction(elev_deg, azim_deg):
    """Unit vector in the head's frame (x = forward/face, y = his left, z = up)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


class Body:
    """Everything a costume needs: rest pose, shapes, bone lines, skeleton, empties."""

    def __init__(self, empties, coll):
        data = json.loads((HERE / "rupert_shapes.json").read_text())
        self.shapes, self.rest = data["shapes"], data["rest"]
        self.empties, self.coll = empties, coll
        self.pieces = self._skin_pieces()
        self.segs = self._bone_lines()
        self.skeleton = self._build_skeleton()
        self.head_bvh = None
        self.hair_bvh = None

    # rest-pose helpers
    def m(self, body):
        return H.rest_matrix(self.rest, body)

    def shape(self, name):
        return next(s for s in self.shapes if s["name"] == name)

    def shape_m(self, name):
        s = self.shape(name)
        return self.m(s["body"]) @ H.shape_matrix(s)

    def chest_m(self):
        return self.m("torso") @ Matrix.Translation((0, 0, CHEST["offset"]))

    def neck_m(self):
        return self.m("torso") @ Matrix.Translation((0, 0, self._head_z() * 0.62))

    def _head_z(self):
        return (self.m("torso").inverted() @ Vector(self.rest["head"]["pos"])).z

    def head_surface(self, d, out=0.0):
        """Point on the skin of the head along direction d (head frame), pushed out by `out`."""
        centre = Vector((0, 0, HEAD_LIFT))
        d = d.normalized()
        hit, _, _, _ = self.head_bvh.ray_cast(centre, d)
        return (hit if hit else centre + d * HEAD_R) + d * out

    def outer_surface(self, d, out=0.0):
        """Outermost point (hair or skin) of the head along direction d (head frame)."""
        d = d.normalized()
        skin = self.head_surface(d)
        centre = Vector((0, 0, HEAD_LIFT))
        if self.hair_bvh:
            hit, _, _, _ = self.hair_bvh.ray_cast(centre + d * 0.6, -d)
            if hit and (hit - centre).length > (skin - centre).length:
                skin = hit
        return skin + d * out

    def limb_point(self, body, child, frac):
        """Point frac of the way from a body's joint to its child's joint (in the body's frame)."""
        return (self.m(body).inverted() @ Vector(self.rest[child]["pos"])) * frac

    def skinned(self, name, elements, bodies, mat, blend=0.012):
        obj = H.metaball_mesh(name, elements, self.coll)
        obj.data.materials.append(mat)
        H.skin_to_skeleton(obj, bodies, self.segs, self.skeleton, blend)
        return obj

    # internals
    def _skin_pieces(self):
        """(group, body, kind, rest matrix, surface radius, half length) for every skin blob."""
        pieces = []
        for s in self.shapes:
            body, name = s["body"], s["name"]
            if body.endswith("foot") or name == "upper_waist":
                continue
            group = next(g for g, bodies in SKIN_GROUPS.items() if body in bodies)
            if name == "head":
                continue                    # the head is sculpted separately (build_head)
            key = next((k for k in SKIN_RADIUS if name.endswith(k)), None)
            pieces.append((group, body, "ball" if s["type"] == "sphere" else "capsule",
                           self.m(body) @ H.shape_matrix(s), SKIN_RADIUS.get(key, s["radius"]), s["half_length"]))
        pieces.append(("skin_core", "torso", "capsule", self.chest_m(), CHEST["radius"], CHEST["half"]))
        pieces.append(("skin_core", "torso", "capsule", self.neck_m(), 0.045, self._head_z() * 0.4))
        return pieces

    def _bone_lines(self):
        segs = {}
        for _, body, kind, m, r, h in self.pieces:
            segs.setdefault(body, []).append((m @ Vector((0, 0, -h)), m @ Vector((0, 0, h))))
        for side in ("right", "left"):
            p = Vector(self.rest[f"{side}_foot"]["pos"])
            segs[f"{side}_foot"] = [(p, p)]
        return segs

    def _build_skeleton(self):
        """One bone per body part, each glued to that part's replay empty."""
        data = bpy.data.armatures.new("rupert_skeleton")
        arm = bpy.data.objects.new("rupert_skeleton", data)
        self.coll.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        for body, r in self.rest.items():
            if body in self.empties:
                b = data.edit_bones.new(body)
                b.head = Vector(r["pos"])
                b.tail = b.head + Vector((0, 0, 0.08))
        bpy.ops.object.mode_set(mode="OBJECT")
        for body in self.rest:
            if body in self.empties:
                c = arm.pose.bones[body].constraints.new("CHILD_OF")
                c.target = self.empties[body]
                c.inverse_matrix = self.m(body).inverted()
        arm.hide_render = True
        return arm


# ---- skin ---------------------------------------------------------------------------
def build_skin(body):
    skin = H.material("rupert_skin", SKIN, rough=0.5, sheen=0.15)
    for group, bodies in SKIN_GROUPS.items():
        els = [(k, m, r, h) for g, _, k, m, r, h in body.pieces if g == group]
        body.skinned(f"rupert_{group}", els, bodies, skin)


def _head_point(d):
    """Sculpt: where the head surface is along unit direction d (head frame, centre at origin)."""
    hs = HEAD_SHAPE
    p = d * HEAD_R
    x, y, z = p.x, p.y, p.z
    if z > 0:
        z *= 0.95                                         # slightly flatter dome
    else:
        s = min(1.0, -z / HEAD_R)                         # 0 at the equator .. 1 at the chin
        z *= hs["chin_length"]
        y *= 1 - (1 - hs["chin_width"]) * s ** 1.15       # straight jawlines to a narrow chin
        x *= (1 - 0.22 * s) if x > 0 else (1 - 0.6 * s)   # jaw tucked in at the back
    # cheekbones: widest point a little below the eyes, on the front half of the face
    bump = math.exp(-((z / HEAD_R + 0.12) / 0.13) ** 2) * min(1.0, max(0.0, x / HEAD_R + 0.35))
    y *= 1 + hs["cheekbone"] * bump
    if x > 0:
        x *= hs["face_flat"]                              # flatter face
    return Vector((x, y, z))


def build_head(body):
    """Rupert's head: a sculpted shape (not a blob) so the cheekbones and jaw read clearly."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=72, v_segments=48, radius=1.0)
    for v in bm.verts:
        v.co = _head_point(v.co.normalized()) + Vector((0, 0, HEAD_LIFT))
    head = H.mesh_object("rupert_head", bm, bpy.data.materials["rupert_skin"], body.coll)
    head.parent = body.empties["head"]
    bm = bmesh.new()
    bm.from_mesh(head.data)                 # head-frame surface, for placing face, hair, bands
    body.head_bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return head


# ---- face (decals stuck onto the head surface) -------------------------------------
def face_decal(body, name, poly_uv, elev, azim, mat, out=0.0015, rings=4):
    """A flat shape (u = sideways, v = up, metres) wrapped onto the head around (elev, azim)."""
    def on_head(u, v):
        return body.head_surface(direction(elev + math.degrees(v / HEAD_R), azim + math.degrees(u / HEAD_R)), out)

    cu = sum(u for u, _ in poly_uv) / len(poly_uv)
    cv = sum(v for _, v in poly_uv) / len(poly_uv)
    bm = bmesh.new()
    centre = bm.verts.new(on_head(cu, cv))
    rings_v = [[bm.verts.new(on_head(cu + f * (u - cu), cv + f * (v - cv))) for u, v in poly_uv]
               for f in (k / rings for k in range(1, rings + 1))]
    n = len(poly_uv)
    for i in range(n):
        bm.faces.new((centre, rings_v[0][i], rings_v[0][(i + 1) % n]))
    for a, b in zip(rings_v, rings_v[1:]):
        for i in range(n):
            bm.faces.new((a[i], b[i], b[(i + 1) % n], a[(i + 1) % n]))
    obj = H.mesh_object(f"rupert_face_{name}", bm, mat, body.coll)
    obj.parent = body.empties["head"]
    return obj


def ellipse(rx, ry, n=24):
    return [(rx * math.cos(2 * math.pi * i / n), ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def build_face(body, expression="happy"):
    """Wide-set oval eyes, little brows and an open 'D' smile, like the profile pictures."""
    ink = H.material("rupert_ink", INK, rough=0.35)
    for side in (1, -1):
        face_decal(body, f"eye_{side}", ellipse(0.0135, 0.0195), 0, 27 * side, ink)
        brow = []
        for i in range(9):
            t = -1 + i / 4
            brow.append((0.017 * t, 0.0035 * (1 - t * t) + 0.0034 * (1 - 0.4 * t * t)))
        for i in range(8, -1, -1):
            t = -1 + i / 4
            brow.append((0.017 * t, 0.0035 * (1 - t * t) - 0.0034 * (1 - 0.4 * t * t)))
        face_decal(body, f"brow_{side}", brow, 12, 27 * side, ink)
    if expression == "happy":
        w, depth = 0.036, 0.03
        top = [(w * t, 0.004 * t * t) for t in (-1 + i / 6 for i in range(13))]
        bottom = [(w * t, -depth * math.sqrt(max(0.0, 1 - t * t))) for t in (1 - i / 8 for i in range(1, 16))]
        face_decal(body, "mouth", top + bottom, -24, 0, H.material("rupert_mouth", MOUTH_INSIDE, rough=0.4))
    else:   # small wavy 'curious' mouth
        pts = [(0.022 * t, 0.004 * math.sin(3 * t)) for t in (-1 + i / 8 for i in range(17))]
        line = pts + [(u, v - 0.004) for u, v in reversed(pts)]
        face_decal(body, "mouth", line, -24, 0, ink)


# ---- hair ---------------------------------------------------------------------------
def _hair_streaks(nodes, links, bsdf):
    """Light and dark streaks running front-to-back, so the swept-back flow reads."""
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (18.0, 18.0, 45.0)     # fine streaks, mostly vertical-ish
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 3.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.35, (*HAIR_DARK, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.65, (*HAIR, 1)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _lock(body, v0, v1, els, lift=0.02, wave=0.007, phase=0.0, flick=0.016, r0=0.027, r1=0.012, out0=0.005, n=10):
    """One wavy lock of hair: from the root (v0) over the head to the tip (v1), lifted at the
    root, waving in and out along its length, tip flicking outward."""
    pts = []
    for i in range(n + 1):
        t = i / n
        d = v0.slerp(v1, t)
        out = (out0 + lift * math.sin(math.pi * min(1.0, t * 1.25))
               + wave * math.sin(2 * math.pi * 1.3 * t + phase)
               + flick * max(0.0, (t - 0.7) / 0.3) ** 2)
        pts.append(body.head_surface(d, max(out, 0.0)))
    for i, (p, q) in enumerate(zip(pts, pts[1:])):
        r = r0 + (r1 - r0) * ((i + 0.5) / n) ** 1.3
        m = Matrix.Translation((p + q) / 2) @ H.look_along(q - p, axis=(0, 0, 1)).to_matrix().to_4x4()
        els.append(("capsule", m, r, (q - p).length / 2))


def build_hair(body):
    """Medium-length, wavy 'flow' (Tom's reference): lifted at the front and swept back and
    out to the sides, covering the tops of the ears, down to the collar with flicked-out ends."""
    els = []
    # base layer: covers the scalp so no skin shows between locks (stops at the hairline in front)
    for az in range(0, 360, 24):
        a = (az + 180) % 360 - 180
        tip_el = 34 if abs(a) <= 50 else (-8 if abs(a) < 120 else -38)
        _lock(body, direction(80, az), direction(tip_el, az), els, lift=0.0, wave=0.0, flick=0.0,
              r0=0.032, r1=0.026, out0=0.0, n=7)
    for side in (1, -1):
        # front: lifted off the forehead, sweeping back and out to the temples
        for k in range(6):
            _lock(body, direction(40 - 1.5 * k, side * (3 + 7 * k)), direction(6 - 4 * k, side * (80 + 11 * k)), els,
                  lift=0.034 - 0.003 * k, wave=0.006, phase=0.7 * k, flick=0.018)
        # top and sides: from the crown, flowing back and down over the ears
        for k in range(6):
            _lock(body, direction(52 + 7 * k, side * 10), direction(-6 - 8 * k, side * (100 + 13 * k)), els,
                  lift=0.024, wave=0.008, phase=1.1 * k + side, flick=0.02, r0=0.029)
    # back: from the crown down to the collar, ends flicking out
    for k in range(-3, 4):
        _lock(body, direction(74, 180 + 16 * k), direction(-50, 180 + 22 * k), els,
              lift=0.016, wave=0.009, phase=0.9 * k, flick=0.024, r0=0.029, r1=0.013)
    hair = H.metaball_mesh("rupert_hair", els, body.coll, resolution=0.0065)
    hair.data.materials.append(H.node_material("rupert_hair", _hair_streaks, rough=0.6, sheen=0.3))
    sm = hair.modifiers.new("smooth", "SMOOTH")        # soften, but keep the wavy locks
    sm.factor, sm.iterations = 0.8, 4
    hair.parent = body.empties["head"]
    bm = bmesh.new()
    bm.from_mesh(hair.data)                    # head-frame surface, for hats and headbands
    body.hair_bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return hair


# ---- ink outline (like the drawings) -----------------------------------------------
def add_outline(obj, thickness=OUTLINE_M):
    """Classic 'inverted hull' outline: a slightly bigger, inside-out black copy."""
    ink = bpy.data.materials.get("rupert_outline")
    if ink is None:
        ink = H.material("rupert_outline", INK, rough=1.0)
        ink.use_backface_culling = True
    obj.data.materials.append(ink)
    sol = obj.modifiers.new("outline", "SOLIDIFY")
    sol.thickness, sol.offset = thickness, 1.0
    sol.use_flip_normals, sol.use_rim = True, False
    sol.material_offset = len(obj.data.materials) - 1


def build_rupert(empties, coll, costume):
    import costumes
    body = Body(empties, coll)
    build_skin(body)
    build_head(body)
    outfit = costumes.COSTUMES[costume]
    build_face(body, outfit.get("expression", "happy"))
    build_hair(body)
    outfit["build"](body)
    if OUTLINE_M > 0:
        for obj in list(coll.objects):
            if obj.type == "MESH" and not obj.name.startswith("rupert_face"):
                add_outline(obj)
    for e in empties.values():
        e.hide_render = True
        e.hide_viewport = True
    return body
