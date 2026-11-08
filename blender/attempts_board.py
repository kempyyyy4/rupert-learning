"""The physical LED attempts board standing in the stadium (for the wide, sped-up training shots).

Amber seven-segment digits and a dot-matrix 'ATTEMPTS' label, same design as the corner
overlay (rupert/led.py). Each digit segment is its own object whose glow is switched by
its object colour, so the number can change frame by frame:

    board = build_board(coll, location)
    set_number(board, frame=1, number=45723)
    set_number(board, frame=50, number=91220)      # ...as often as you like
"""
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

import bl_helpers as H

sys.path.insert(0, str(Path(__file__).parent.parent))
from rupert import led  # noqa: E402

AMBER = (1.0, 0.42, 0.02)
AMBER_OFF = (0.03, 0.017, 0.005)
N_DIGITS = 8


def _led_material():
    m = bpy.data.materials.get("led_segment")
    if m:
        return m

    def build(nodes, links, bsdf):
        info = nodes.new("ShaderNodeObjectInfo")
        sep = nodes.new("ShaderNodeSeparateColor")
        links.new(info.outputs["Color"], sep.inputs[0])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["A"].default_value = (*AMBER_OFF, 1)
        mix.inputs["B"].default_value = (*AMBER, 1)
        links.new(sep.outputs[0], mix.inputs["Factor"])
        links.new(mix.outputs["Result"], bsdf.inputs["Emission Color"])
        strength = H.Maths(nodes, links).op("MULTIPLY_ADD", sep.outputs[0], 7.0, 0.3)
        links.new(strength, bsdf.inputs["Emission Strength"])
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)

    return H.node_material("led_segment", build, rough=0.4)


def _flat_poly(name, pts, mat, coll, to3d):
    bm = bmesh.new()
    verts = [bm.verts.new(to3d(x, y)) for x, y in pts]
    f = bm.faces.new(verts)
    f.normal_update()
    if f.normal.y > 0:                 # face the camera side (-y)
        f.normal_flip()
    return H.mesh_object(name, bm, mat, coll, smooth=False)


def build_board(coll, loc, digit_h=0.42, name="attempts_board"):
    """Free-standing LED board facing the -y side (the cameras). loc = bottom-centre of the panel."""
    dw, dh, t, sp = digit_h * 0.55, digit_h, digit_h * 0.125, digit_h * 0.17
    pitch = digit_h * 0.065
    pad = digit_h * 0.3
    inner_w = N_DIGITS * dw + (N_DIGITS - 1) * sp
    w = inner_w + 2 * pad
    h = pad + 7 * pitch + digit_h * 0.25 + dh + pad
    x0, z0, y0 = loc[0] - w / 2, loc[2] + h, loc[1]
    face_y = y0 - 0.061

    H.box(f"{name}_panel", (w, 0.12, h), (loc[0], y0, loc[2] + h / 2), H.material("board_black", (0.012, 0.012, 0.014), rough=0.5), coll, bevel=0.02)
    H.box(f"{name}_frame", (w + 0.1, 0.1, h + 0.1), (loc[0], y0 + 0.03, loc[2] + h / 2), H.material("board_frame", (0.25, 0.26, 0.28), rough=0.4), coll, bevel=0.03)
    post_m = H.material("board_post", (0.35, 0.36, 0.38), rough=0.4)
    for side in (-1, 1):
        H.box(f"{name}_post", (0.12, 0.12, loc[2] + 0.05), (loc[0] + side * w * 0.35, y0 + 0.05, (loc[2] + 0.05) / 2), post_m, coll)

    def to3d(px, py):                  # board drawing coords (y down) -> world
        return Vector((x0 + px, face_y, z0 - py))

    mat = _led_material()
    board = {"digits": []}
    # label: lit dots and faint unlit grid
    label = "ATTEMPTS"
    lx = (w - led.text_width(label, pitch)) / 2
    ly = pad
    lit = set(led.dot_positions(label, lx, ly, pitch))
    r = pitch * 0.36
    for group, on in (("lit", True), ("off", False)):
        bm = bmesh.new()
        for i in range(len(label)):
            for row in range(7):
                for col in range(5):
                    c = (lx + (i * 6 + col) * pitch, ly + row * pitch)
                    if (c in lit) != on:
                        continue
                    bmesh.ops.create_circle(bm, cap_ends=True, segments=10, radius=r,
                                            matrix=_disc_matrix(to3d(*c)))
        obj = H.mesh_object(f"{name}_label_{group}", bm, mat, coll, smooth=False)
        obj.color = (1, 1, 1, 1) if on else (0, 0, 0, 1)
    # digits
    dy = ly + 7 * pitch + digit_h * 0.25
    for i in range(N_DIGITS):
        dx = pad + i * (dw + sp)
        segs = {}
        for seg, poly in led.segment_polygons(dx, dy, dw, dh, t).items():
            segs[seg] = _flat_poly(f"{name}_d{i}_{seg}", poly, mat, coll, to3d)
        board["digits"].append(segs)
    return board


def _disc_matrix(centre):
    from mathutils import Matrix
    import math
    return Matrix.Translation(centre) @ Matrix.Rotation(math.radians(90), 4, "X")


def set_number(board, frame, number):
    """Show `number` from `frame` onwards (holds until the next call)."""
    prefs = bpy.context.preferences.edit
    old = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = "CONSTANT"
    for ch, segs in zip(led.digits_for(number, N_DIGITS), board["digits"]):
        for seg, obj in segs.items():
            obj.color = (1, 1, 1, 1) if seg in led.SEGMENTS[ch] else (0, 0, 0, 1)
            obj.keyframe_insert("color", frame=frame)
    prefs.keyframe_new_interpolation_type = old


def snap_numbers(board):
    """Make every number change on the board instant (call once, after all set_number calls)."""
    for segs in board["digits"]:
        for obj in segs.values():
            H.set_interpolation(obj, "CONSTANT")
