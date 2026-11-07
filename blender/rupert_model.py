"""Rupert himself: white skin, face and caramel mop hair (from the channel's character sheet,
E:/AI Channel/shorts3d/characters/rupert_3d_sheet_v1.png). Clothes come from costumes.py.

His skin is ONE smooth surface built around the physics body's limbs
(blender/rupert_shapes.json) and bent at the joints by a skeleton that copies the replay,
so limb lengths and joints are exactly the body that learned. Only thickness and the
head size are exaggerated, for looks.
"""
import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

import bl_helpers as H

HERE = Path(__file__).parent

SKIN = (0.92, 0.89, 0.84)          # white, like the channel drawings
HAIR = (0.33, 0.13, 0.05)         # caramel brown (#B8785A on the character sheet, darkened for the bright sun)
FEATURES = (0.015, 0.015, 0.018)  # eyes and smile: near-black

HEAD_R = 0.14                     # physics head is 0.098; cartoon head
HEAD_LIFT = 0.02                  # head sphere sits this far above the physics head centre
# skin thickness (surface radius, m) per physics shape
SKIN_RADIUS = {
    "thigh": 0.072, "shin": 0.052, "upper_arm": 0.045, "lower_arm": 0.037, "hand": 0.05,
    "lower_waist": 0.09, "butt": 0.095, "torso": 0.07,
}
CHEST = dict(offset=-0.09, radius=0.115, half=0.1)   # one smooth chest instead of stacked bars
# each group melts into one smooth piece; groups meet at hip and shoulder
SKIN_GROUPS = {
    "skin_core": ["torso", "head", "lower_waist", "pelvis"],
    "skin_leg_r": ["right_thigh", "right_shin"],
    "skin_leg_l": ["left_thigh", "left_shin"],
    "skin_arm_r": ["right_upper_arm", "right_lower_arm", "right_hand"],
    "skin_arm_l": ["left_upper_arm", "left_lower_arm", "left_hand"],
}


class Body:
    """Everything a costume needs: rest pose, shapes, bone lines, skeleton, empties."""

    def __init__(self, empties, coll):
        data = json.loads((HERE / "rupert_shapes.json").read_text())
        self.shapes, self.rest = data["shapes"], data["rest"]
        self.empties, self.coll = empties, coll
        self.pieces = self._skin_pieces()
        self.segs = self._bone_lines()
        self.skeleton = self._build_skeleton()

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

    def head_point(self, elev_deg, azim_deg, out=0.0):
        """A point on the head surface in the head's own frame (x = forward, z = up)."""
        e, a = math.radians(elev_deg), math.radians(azim_deg)
        r = HEAD_R + out
        return Vector((r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), HEAD_LIFT + r * math.sin(e)))

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
                pieces.append((group, body, "ball", self.m(body) @ Matrix.Translation((0, 0, HEAD_LIFT)), HEAD_R, 0))
                continue
            key = next((k for k in SKIN_RADIUS if name.endswith(k)), None)
            pieces.append((group, body, "ball" if s["type"] == "sphere" else "capsule",
                           self.m(body) @ H.shape_matrix(s), SKIN_RADIUS.get(key, s["radius"]), s["half_length"]))
        pieces.append(("skin_core", "torso", "capsule", self.chest_m(), CHEST["radius"], CHEST["half"]))
        head_z = (self.m("torso").inverted() @ Vector(self.rest["head"]["pos"])).z
        pieces.append(("skin_core", "torso", "capsule", self.m("torso") @ Matrix.Translation((0, 0, head_z * 0.55)),
                       0.045, head_z * 0.3))   # neck
        return pieces

    def _bone_lines(self):
        segs = {}
        for _, body, kind, m, r, h in self.pieces:
            segs.setdefault(body, []).append((m @ Vector((0, 0, -h)), m @ Vector((0, 0, h))))
        # the ankle joint, so the shin skin knows where the foot starts
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


def build_skin(body):
    skin = H.material("rupert_skin", SKIN, rough=0.45, sheen=0.2)
    for group, bodies in SKIN_GROUPS.items():
        els = [(k, m, r, h) for g, _, k, m, r, h in body.pieces if g == group]
        body.skinned(f"rupert_{group}", els, bodies, skin)


def build_face(body):
    """Little dot eyes and a small smile, like the channel drawings."""
    head = body.empties["head"]
    ink = H.material("rupert_features", FEATURES, rough=0.25)
    for side in (1, -1):
        p = body.head_point(6, 24 * side, out=-0.004)
        eye = H.mesh_object(f"rupert_eye_{'l' if side > 0 else 'r'}", H.sphere_bm(1.0, scale=(0.006, 0.0125, 0.019)), ink, body.coll)
        H.attach(eye, head, p, H.look_along(p - Vector((0, 0, HEAD_LIFT))))
    pts = []
    for i in range(9):
        a = -16 + 4 * i
        e = -22 + 6 * (a / 16) ** 2
        pts.append(body.head_point(e, a, out=0.001))
    smile = H.tube_curve("rupert_smile", pts, 0.0035, ink, body.coll)
    smile.parent = head


def build_hair(body, fringe_down_to=26):
    """Shoulder-length caramel mop: lumpy blobs over the top, sides and back, face left open."""
    rng = random.Random(3)
    els = []
    for elev in range(-42, 91, 11):
        n = max(1, int(22 * math.cos(math.radians(elev))))
        for k in range(n):
            azim = 360 * k / n + rng.uniform(-6, 6)
            a = (azim + 180) % 360 - 180
            front = abs(a) < 62
            if front and elev < fringe_down_to:
                continue                                  # leave the face open
            if abs(a) < 95 and elev < -10:
                continue                                  # no hair under the cheeks
            p = body.head_point(elev, azim, out=-0.006)
            r = rng.uniform(0.03, 0.042) * (1.15 if elev < 0 else 1.0)
            els.append(("ball", Matrix.Translation(p), r, 0))
    hair = H.metaball_mesh("rupert_hair", els, body.coll, resolution=0.008)
    hair.data.materials.append(H.material("rupert_hair", HAIR, rough=0.8))
    lumps = bpy.data.textures.new("hair_lumps", "CLOUDS")
    lumps.noise_scale = 0.035
    d = hair.modifiers.new("lumps", "DISPLACE")
    d.texture, d.strength, d.mid_level, d.texture_coords = lumps, 0.018, 0.5, "LOCAL"
    hair.parent = body.empties["head"]
    return hair


def build_rupert(empties, coll, costume):
    import costumes
    body = Body(empties, coll)
    build_skin(body)
    build_face(body)
    outfit = costumes.COSTUMES[costume]
    build_hair(body, **outfit.get("hair", {}))
    outfit["build"](body)
    for e in empties.values():
        e.hide_render = True
        e.hide_viewport = True
    return body
