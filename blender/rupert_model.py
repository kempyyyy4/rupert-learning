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

HEAD_R = 0.165                    # physics head is 0.098; big cartoon head
HEAD_LIFT = 0.07                 # head centre sits this far above the physics head centre (cartoon neck)
# Head shape (Tom's drawings): round dome, flat face, strong square jaw, flat chin
HEAD_SHAPE = dict(jaw_taper=0.22, jaw_length=1.08, chin_flat=0.86, cheekbone=0.07, face_flat=0.9)
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


def _smooth_min(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k / 4


def _head_point(d):
    """Sculpt (Tom's drawings): big round-topped head, flat face, strong SQUARE jaw with defined
    corners at the back, flat chin. Returns the surface point along unit direction d."""
    hs = HEAD_SHAPE
    p = d * HEAD_R
    x, y, z = p.x, p.y, p.z
    if z > 0:
        z *= 0.93                                          # slightly flatter dome
    else:
        s = min(1.0, -z / HEAD_R)                          # 0 at the eyes' level .. 1 at the bottom
        y *= 1 - hs["jaw_taper"] * s ** 3                  # stays wide right down to the jaw corners
        if x < 0:
            x *= 1 - 0.55 * s ** 1.5                       # back of the jaw tucks in -> jaw corner
        else:
            x *= 1 - 0.1 * s
        z = -_smooth_min(-z * hs["jaw_length"], hs["chin_flat"] * HEAD_R, 0.04)   # flat chin / jawline
    bump = math.exp(-((z / HEAD_R + 0.15) / 0.14) ** 2) * min(1.0, max(0.0, x / HEAD_R + 0.3))
    y *= 1 + hs["cheekbone"] * bump                        # cheekbones
    if x > 0:
        x *= hs["face_flat"]
    return Vector((x, y, z))


def build_head(body):
    """Rupert's head: a sculpted shape (not a blob) so the jaw and cheekbones read clearly."""
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


def ellipse(rx, ry, n=24, cx=0.0, cy=0.0):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def bar(u0, v0, u1, v1, thick, n=8):
    """A thin stroke from (u0, v0) to (u1, v1) with rounded-ish ends."""
    du, dv = u1 - u0, v1 - v0
    length = math.hypot(du, dv)
    nu, nv = -dv / length * thick / 2, du / length * thick / 2
    top = [(u0 + du * i / n + nu * math.sin(math.pi * i / n) ** 0.3, v0 + dv * i / n + nv * math.sin(math.pi * i / n) ** 0.3) for i in range(n + 1)]
    bot = [(u0 + du * i / n - nu * math.sin(math.pi * i / n) ** 0.3, v0 + dv * i / n - nv * math.sin(math.pi * i / n) ** 0.3) for i in range(n, -1, -1)]
    return top + bot[1:-1]


def build_face(body, expression="determined"):
    """Tom's drawings: narrow almond eyes with a pupil, straight stern brows angled down to the
    nose, and a straight, slightly smirking mouth. ('happy' = the profile picture's D smile.)"""
    ink = H.material("rupert_ink", INK, rough=0.35)
    white = H.material("rupert_eye_white", (0.92, 0.92, 0.9), rough=0.3)
    ex, ey = 25, 2                                   # eye position (degrees across, up)
    for side in (1, -1):
        if expression == "happy":
            face_decal(body, f"eye_{side}", ellipse(0.0135, 0.0195), 0, 27 * side, ink)
            continue
        face_decal(body, f"eye_rim_{side}", ellipse(0.0215, 0.0105), ey, ex * side, ink, out=0.0012)
        face_decal(body, f"eye_white_{side}", ellipse(0.018, 0.0072), ey, ex * side, white, out=0.0018)
        face_decal(body, f"eye_pupil_{side}", ellipse(0.0062, 0.0062, cx=-0.004 * side), ey, ex * side, ink, out=0.0024)
        # brow: inner end (towards the nose) lower = determined look
        face_decal(body, f"brow_{side}", bar(-0.02, -0.004 * side, 0.02, 0.004 * side, 0.0065), ey + 12, ex * side, ink)
    if expression == "happy":
        w, depth = 0.036, 0.03
        top = [(w * t, 0.004 * t * t) for t in (-1 + i / 6 for i in range(13))]
        bottom = [(w * t, -depth * math.sqrt(max(0.0, 1 - t * t))) for t in (1 - i / 8 for i in range(1, 16))]
        face_decal(body, "mouth", top + bottom, -24, 0, H.material("rupert_mouth", MOUTH_INSIDE, rough=0.4))
    else:
        face_decal(body, "mouth", bar(-0.032, -0.002, 0.03, 0.004, 0.0048), -27, 2, ink)


# ---- hair ---------------------------------------------------------------------------
def _hair_streaks(nodes, links, bsdf):
    """Light and dark streaks, so the individual locks read."""
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (10.0, 30.0, 30.0)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 2.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.3, (*HAIR_DARK, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.7, (*HAIR, 1)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])


def _hairline(az):
    """Lowest elevation (deg) the hair cap reaches at this azimuth: high on the forehead,
    down past the ears at the sides, to the nape at the back."""
    a = abs((az + 180) % 360 - 180)
    if a <= 40:
        return 36.0
    if a <= 100:
        return 36.0 - (a - 40) / 60 * 44
    if a <= 150:
        return -8.0 - (a - 100) / 50 * 34
    return -42.0


def _cap(body, bm, out=0.022, rows=16):
    """Smooth hair cap over the scalp, under the locks (so no skin shows between them).
    Its lower edge follows the hairline exactly, so the edge is clean."""
    azs = list(range(0, 360, 5))
    grid = []
    for a in azs:
        h = _hairline(a)
        grid.append([bm.verts.new(body.head_surface(direction(h + (89.5 - h) * (i / rows) ** 0.8, a), out))
                     for i in range(rows + 1)])
    centre = Vector((0, 0, HEAD_LIFT))
    for j in range(len(azs)):
        c0, c1 = grid[j], grid[(j + 1) % len(azs)]
        for i in range(rows):
            f = bm.faces.new((c0[i], c1[i], c1[i + 1], c0[i + 1]))
            f.normal_update()
            if f.normal.dot(f.calc_center_median() - centre) < 0:
                f.normal_flip()


def _ribbon(body, bm, v0, v1, width, lift=0.025, flick=0.02, out0=0.026, n=14, curl=0.0):
    """One chunky lock: a tapering ribbon lying on the head from root (v0) to a pointed tip (v1),
    lifted near the root for volume, tip flicking outward."""
    centre = Vector((0, 0, HEAD_LIFT))
    pts = []
    for i in range(n + 1):
        t = i / n
        d = v0.slerp(v1, t)
        out = out0 + lift * math.sin(math.pi * min(1.0, t * 1.3)) + flick * max(0.0, (t - 0.65) / 0.35) ** 2
        pts.append(body.head_surface(d, out))
    rows = []
    for i, p in enumerate(pts):
        t = i / n
        tangent = (pts[min(i + 1, n)] - pts[max(i - 1, 0)]).normalized()
        normal = (p - centre).normalized()
        side = tangent.cross(normal).normalized()
        w = width * (1 - t ** 1.5) * (0.75 + 0.25 * math.sin(math.pi * min(1.0, t * 2)))
        bow = normal * (w * 0.18)                  # slightly rounded cross-section
        rows.append((bm.verts.new(p + side * w / 2), bm.verts.new(p + bow + curl * side * w),
                     bm.verts.new(p - side * w / 2)))
    for a, b in zip(rows, rows[1:]):
        for k in range(2):
            f = bm.faces.new((a[k], a[k + 1], b[k + 1], b[k]))
            f.normal_update()
            if f.normal.dot(f.calc_center_median() - centre) < 0:
                f.normal_flip()


def build_hair(body):
    """Chunky swept-back hair (Tom's reference + drawings): locks rise off the forehead and sweep
    back over the top, sides swept back past the ears, longer at the back with pointed tips
    flicking out at the nape, a couple of locks framing the face."""
    bm = bmesh.new()
    _cap(body, bm)
    for az in range(-48, 49, 12):                                       # top: swept straight back
        _ribbon(body, bm, direction(37, az), direction(-36 - 8 * abs(az) / 48, 180 - 1.3 * az),
                width=0.07, lift=0.032, flick=0.02)
    for az in (-30, -10, 10, 30):                                        # second layer on top
        _ribbon(body, bm, direction(55, az), direction(-12, 180 - 1.7 * az), width=0.065,
                lift=0.024, flick=0.03, out0=0.036)
    for side in (1, -1):
        for k in range(3):                                               # sides, back past the ears
            _ribbon(body, bm, direction(26 - 9 * k, side * (50 + 9 * k)), direction(-42 - 5 * k, side * (148 + 8 * k)),
                    width=0.06, lift=0.012, flick=0.022)
        _ribbon(body, bm, direction(40, side * 40), direction(-36, side * 90), width=0.05,   # framing the face
                lift=0.01, flick=0.018, out0=0.024)
    for k in range(-2, 3):                                               # back: long, tips flick out
        _ribbon(body, bm, direction(30, 180 + 22 * k), direction(-58, 180 + 26 * k), width=0.07,
                lift=0.01, flick=0.024, out0=0.028)
    hair = H.mesh_object("rupert_hair", bm, H.node_material("rupert_hair", _hair_streaks, rough=0.55, sheen=0.3), body.coll)
    sol = hair.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = 0.01, -1.0
    sub = hair.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    hair.parent = body.empties["head"]
    bm = bmesh.new()
    bm.from_mesh(hair.data)
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
    build_face(body, outfit.get("expression", "determined"))
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
