"""Other characters, built on the same body as Rupert (same skeleton, same cartoon head and face
style) but with their own skin, hair, face details and clothes.

  judge   the style judge (EP01): grey horseshoe hair, glasses, moustache, stern face, navy blazer,
          white shirt, red tie. Sits at a judges' desk and can hold up a score card.
  runner  the real runner the judge studied (EP01): darker skin, short black hair, blue kit.
"""
import math

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import bl_helpers as H
import costumes
import rupert_model as R

NAVY, SHIRT, TIE = (0.02, 0.035, 0.11), (0.92, 0.92, 0.9), (0.6, 0.03, 0.03)
GREY_HAIR = (0.42, 0.42, 0.43)
BLUE, WHITE = (0.03, 0.18, 0.75), (0.95, 0.95, 0.95)

LOOKS = {
    "judge": dict(skin=(0.74, 0.55, 0.43), hair=GREY_HAIR, expression="determined"),
    "runner": dict(skin=(0.36, 0.22, 0.14), hair=(0.02, 0.015, 0.01), expression="determined"),
    # the world-record sprinter in the race: yellow vest, green shorts, no name or logo anywhere
    "sprinter": dict(skin=(0.24, 0.14, 0.085), hair=(0.015, 0.012, 0.01), expression="determined"),
}
YELLOW, GREEN, GOLD = (1.0, 0.78, 0.0), (0.0, 0.42, 0.16), (0.85, 0.62, 0.12)


# ---- body, head and hair with this character's own materials ------------------------
def _skin_and_head(body, name, rgb, covered=()):
    """covered: body parts fully hidden by clothes (no skin built there, so none pokes through)."""
    skin = H.material(f"{name}_skin", rgb, rough=0.5, sheen=0.15)
    for group, bodies in R.SKIN_GROUPS.items():
        is_neck = lambda b, r: b == "torso" and abs(r - 0.045) < 1e-6     # the neck piece (see rupert_model)
        els = [(k, m, r, h) for g, b, k, m, r, h in body.pieces if g == group and (b not in covered or is_neck(b, r))]
        if not els:
            continue
        body.skinned(f"{name}_{group}", els, bodies, skin)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=72, v_segments=48, radius=1.0)
    for v in bm.verts:
        v.co = R._head_point(v.co.normalized()) + Vector((0, 0, R.HEAD_LIFT))
    head = H.mesh_object(f"{name}_head", bm, skin, body.coll)
    head.parent = body.empties["head"]
    bm = bmesh.new()
    bm.from_mesh(head.data)
    body.head_bvh = BVHTree.FromBMesh(bm)
    bm.free()


def _hair_patch(body, name, rgb, azs, low, high, out=0.012, rows=10, thick=0.014, wrap=False):
    """A smooth layer of hair on the scalp between elevations low(az) and high(az)."""
    bm = bmesh.new()
    grid = [[bm.verts.new(body.head_surface(R.direction(low(a) + (high(a) - low(a)) * i / rows, a), out))
             for i in range(rows + 1)] for a in azs]
    centre = Vector((0, 0, R.HEAD_LIFT))
    for j in range(len(azs) if wrap else len(azs) - 1):
        c0, c1 = grid[j], grid[(j + 1) % len(azs)]
        for i in range(rows):
            f = bm.faces.new((c0[i], c1[i], c1[i + 1], c0[i + 1]))
            f.normal_update()
            if f.normal.dot(f.calc_center_median() - centre) < 0:
                f.normal_flip()
    obj = H.mesh_object(f"{name}_hair", bm, H.material(f"{name}_hair", rgb, rough=0.8, sheen=0.3), body.coll)
    sol = obj.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = thick, -1.0
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    obj.parent = body.empties["head"]
    return obj


# ---- judge details --------------------------------------------------------------------
def _tufts(body, rgb):
    """Wispy grey tufts round the sides and back that stick out a bit (professor hair)."""
    import random
    rnd = random.Random(3)
    bm = bmesh.new()
    for side in (1, -1):
        for k in range(9):
            az = side * (62 + k * 13 + rnd.uniform(-3, 3))
            el = rnd.uniform(10, 26)
            v0 = R.direction(el, az)
            v1 = R.direction(el - rnd.uniform(18, 30), az + side * rnd.uniform(8, 18))
            R._ribbon(body, bm, v0, v1, width=rnd.uniform(0.035, 0.05), lift=0.01,
                      flick=rnd.uniform(0.03, 0.05), out0=0.008, n=8)
    obj = H.mesh_object("judge_tufts", bm, bpy.data.materials["judge_hair"], body.coll)
    sol = obj.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = 0.008, -1.0
    sub = obj.modifiers.new("smooth", "SUBSURF")
    sub.levels = sub.render_levels = 1
    obj.parent = body.empties["head"]



def _glasses(body):
    frame = H.material("judge_glasses", (0.02, 0.02, 0.02), rough=0.4)
    w, h, t = 0.031, 0.022, 0.006
    for side in (1, -1):
        for name, (u0, v0, u1, v1) in {"top": (-w, h, w, h), "bottom": (-w, -h, w, -h),
                                       "in": (-w, -h, -w, h), "out": (w, -h, w, h)}.items():
            R.face_decal(body, f"glasses_{name}_{side}", R.bar(u0, v0, u1, v1, t), 2, 25 * side, frame, out=0.007)
    R.face_decal(body, "glasses_bridge", R.bar(-0.012, 0.004, 0.012, 0.004, t), 2, 0, frame, out=0.009)


def _moustache(body, rgb):
    # bushy walrus moustache: wide on top, drooping at the ends
    pts = [(0.06 * math.cos(math.pi * i / 20), 0.004 + 0.016 * math.sin(math.pi * i / 20)) for i in range(21)]
    pts += [(-0.06 + 0.12 * i / 12, -0.012 - 0.008 * math.cos(math.pi * (i / 12 - 0.5)) ** 2) for i in range(1, 12)]
    R.face_decal(body, "moustache", pts, -16, 0, H.material("judge_moustache", (0.3, 0.3, 0.31), rough=0.9), out=0.006)


def _blazer(body):
    arms = ["right_upper_arm", "right_lower_arm", "left_upper_arm", "left_lower_arm"]
    lw = body.shape("lower_waist")
    els = [("capsule", body.chest_m(), 0.14, 0.11),
           ("capsule", body.shape_m("torso"), 0.09, body.shape("torso")["half_length"]),   # across the shoulders
           ("capsule", body.shape_m("lower_waist"), 0.112, lw["half_length"])]
    for s in ("right", "left"):
        els.append(("ball", H.rest_matrix(body.rest, f"{s}_upper_arm"), 0.062, 0.0))
    els += [(k, m, r + 0.015, h) for _, b, k, m, r, h in body.pieces if b in arms]
    obj = H.metaball_mesh("judge_blazer", els, body.coll)
    obj.data.materials.append(H.material("judge_blazer", NAVY, rough=0.7, sheen=0.2))
    H.skin_to_skeleton(obj, ["torso", "lower_waist", *arms], body.segs, body.skeleton)
    # trousers (same navy suit)
    legs = ["pelvis", "right_thigh", "right_shin", "left_thigh", "left_shin"]
    els = [(k, m, r + 0.012, h) for _, b, k, m, r, h in body.pieces if b in legs]
    trousers = H.metaball_mesh("judge_trousers", els, body.coll)
    trousers.data.materials.append(bpy.data.materials["judge_blazer"])
    H.skin_to_skeleton(trousers, legs, body.segs, body.skeleton)
    # shirt V and tie, laid onto the front of the jacket at the collar (torso frame: x forward,
    # y = his left, z up). Each point is pushed onto the jacket surface so they hug the chest.
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bm_j = bmesh.new()
    bm_j.from_mesh(obj.data)
    bvh = BVHTree.FromBMesh(bm_j)
    torso = body.empties["torso"]
    M = body.m("torso")
    Mi, back = M.inverted(), M.to_3x3() @ Vector((-1, 0, 0))
    for name, pts, rgb, out in (("shirt", [(0.066, 0.16), (-0.066, 0.16), (0.0, -0.08)], SHIRT, 0.004),
                                ("tie", [(0.017, 0.155), (-0.017, 0.155), (-0.012, 0.125), (-0.022, -0.045),
                                         (0.0, -0.068), (0.022, -0.045), (0.012, 0.125)], TIE, 0.007)):
        bm = bmesh.new()
        vs = [bm.verts.new((0.4, y, z)) for y, z in pts]
        bm.faces.new(vs)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=5, use_grid_fill=True)
        for v in bm.verts:
            hit, normal, _, _ = bvh.ray_cast(M @ v.co, back)
            v.co = Mi @ (hit + normal * out) if hit else Vector((0.14 + out, v.co.y, v.co.z))
        o = H.mesh_object(f"judge_{name}", bm, H.material(f"judge_{name}", rgb, rough=0.6), body.coll, smooth=True)
        o.parent = torso
    bm_j.free()


def build_desk(at, coll, label="JUDGE", yaw_deg=0.0):
    """Judges' desk in front of a seated character whose pelvis is at `at` (world x, y), facing
    +x turned by yaw_deg (90 = facing +y)."""
    before = set(coll.objects)
    _desk_parts(at, coll, label)
    if yaw_deg:
        root = bpy.data.objects.new("desk_root", None)
        root.location = (at[0], at[1], 0.0)
        coll.objects.link(root)
        bpy.context.view_layer.update()
        for o in set(coll.objects) - before - {root}:
            if o.parent is None:
                o.parent = root
                o.matrix_parent_inverse = root.matrix_world.inverted()
        root.rotation_euler = (0, 0, math.radians(yaw_deg))


def _desk_parts(at, coll, label):
    x, y = at
    wood = H.material("desk_wood", (0.32, 0.17, 0.08), rough=0.45)
    panel = H.material("desk_panel", (0.9, 0.9, 0.88), rough=0.5)
    gold = H.material("desk_gold", (0.85, 0.6, 0.12), rough=0.3)
    H.box("desk_top", (0.75, 1.5, 0.05), (x + 0.6, y, 0.705), wood, coll, bevel=0.01)
    H.box("desk_front", (0.04, 1.5, 0.68), (x + 0.96, y, 0.34), panel, coll, bevel=0.005)
    for s in (1, -1):
        H.box(f"desk_side_{s}", (0.7, 0.04, 0.68), (x + 0.6, y + s * 0.73, 0.34), panel, coll)
    H.text("desk_label", label, (x + 0.985, y, 0.42), (90, 0, 90), 0.2, gold, coll, extrude=0.004)
    H.box("judge_chair_seat", (0.45, 0.45, 0.05), (x - 0.02, y, 0.44), wood, coll, bevel=0.01)
    H.box("judge_chair_back", (0.05, 0.45, 0.6), (x - 0.26, y, 0.75), wood, coll, bevel=0.01)
    H.box("judge_chair_post", (0.06, 0.06, 0.42), (x - 0.02, y, 0.21), wood, coll)
    # a stack of spare score cards on the desk
    H.box("card_stack", (0.22, 0.3, 0.03), (x + 0.75, y + 0.45, 0.745), H.material("card_white", (0.95, 0.95, 0.93), rough=0.6), coll)


def build_score_card(empties, coll, score, show_from, face_yaw_deg=-25.0, until=None, back=None, flip=None,
                     name="score_card"):
    """A white score card with a big number, held in the right hand from frame `show_from`
    (until frame `until`). back: text on the other side; flip: frame he turns it round (0.4 s)."""
    card = H.box(name, (0.012, 0.42, 0.32), (0, 0, 0), H.material("card_white", (0.95, 0.95, 0.93), rough=0.6), coll, bevel=0.004)
    ink = H.material("card_ink", (0.01, 0.01, 0.01), rough=0.8)
    size = 0.32 if len(score) <= 2 else 0.32 * 2 / len(score)
    num = H.text(f"{name}_number", score, (0.008, 0, 0), (90, 0, 90), size, ink, coll, extrude=0.002)
    num.parent = card
    if back:
        red = H.material("card_red", (0.8, 0.02, 0.02), rough=0.6)
        size = min(0.25, 0.62 / max(1, len(back)))
        b = H.text(f"{name}_back", back, (-0.008, 0, 0), (90, 0, -90), size, red, coll, extrude=0.002)
        b.parent = card
    c = card.constraints.new("COPY_LOCATION")
    c.target, c.use_offset = empties["right_hand"], True
    card.location = (0.0, -0.05, 0.2)          # just above the fist
    card.rotation_euler = (0, 0, math.radians(face_yaw_deg))
    if flip:
        for f, yaw in ((flip, face_yaw_deg), (flip + 20, face_yaw_deg + 180)):
            card.rotation_euler = (0, 0, math.radians(yaw))
            card.keyframe_insert("rotation_euler", frame=f)
    keys = [(1, 1.0)] if show_from <= 1 else [(1, 0.0), (show_from, 0.0), (show_from + 8, 1.0)]
    if until:
        keys += [(until, 1.0), (until + 6, 0.0)]
    for f, s in keys:
        card.scale = (s, s, s)
        card.keyframe_insert("scale", frame=f)
    return card


def build_character(empties, coll, look):
    style = LOOKS[look]
    body = R.Body(empties, coll)
    covered = ("right_upper_arm", "right_lower_arm", "left_upper_arm", "left_lower_arm",
               "torso", "lower_waist", "pelvis", "right_thigh", "right_shin", "left_thigh", "left_shin") if look == "judge" else ()
    _skin_and_head(body, look, style["skin"], covered)
    R.build_face(body, style["expression"])
    if look == "judge":
        # horseshoe: grey at the sides and back, bald on top
        _hair_patch(body, look, style["hair"], range(55, 306, 5), lambda a: min(R._hairline(a), 6.0),
                    lambda a: 28.0, out=0.006, thick=0.01)
        _tufts(body, style["hair"])
        _glasses(body)
        _moustache(body, style["hair"])
        _blazer(body)
    elif look == "runner":
        _hair_patch(body, look, style["hair"], range(0, 360, 5), lambda a: R._hairline(a), lambda a: 89.5,
                    out=0.006, thick=0.008, wrap=True)
        costumes.trimmed_vest(body, BLUE, WHITE, name="runner_vest")
        costumes.loose_shorts(body, BLUE, WHITE, name="runner_shorts")
        costumes.running_shoes(body, WHITE, BLUE, BLUE, name="runner_shoe")
    elif look == "sprinter":
        _hair_patch(body, look, style["hair"], range(0, 360, 5), lambda a: R._hairline(a) + 4, lambda a: 89.5,
                    out=0.004, thick=0.005, wrap=True)          # cropped close
        costumes.trimmed_vest(body, YELLOW, GREEN, name="sprinter_vest")
        costumes.loose_shorts(body, GREEN, YELLOW, name="sprinter_shorts")
        costumes.running_shoes(body, GOLD, WHITE, WHITE, name="sprinter_shoe")
    for e in empties.values():
        e.hide_render = True
        e.hide_viewport = True
    return body
