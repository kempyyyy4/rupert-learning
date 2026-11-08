"""Blender: import one Rupert replay as animated empties (one per body part).

How to use (Claude will usually run this for you):
  blender --python blender/import_replay.py -- <path to step_XXXX.npz> [attempt number]
or open Blender > Scripting tab > open this file > edit REPLAY / ATTEMPT below > Run.

Each body part (torso, head, thighs, shins, feet, arms, hands...) becomes an Empty
named "rupert_<part>" with a keyframe for position and rotation on every frame.
Rupert's 3D model is then parented to these empties. Units are metres, Z is up
(same as MuJoCo), the track runs along +X starting at X = 0.
"""
import sys

import bpy
import numpy as np

REPLAY = None   # set a path here if running from the Scripting tab
ATTEMPT = 0
SKIP_BODIES = {"world"}


def load_args():
    replay, attempt = REPLAY, ATTEMPT
    if "--" in sys.argv:
        extra = sys.argv[sys.argv.index("--") + 1:]
        if extra:
            replay = extra[0]
        if len(extra) > 1:
            attempt = int(extra[1])
    if not replay:
        raise SystemExit("No replay file given.")
    return replay, attempt


def _keys(obj, path, vals, interp):
    """Key every frame of one property at once (much faster than one keyframe_insert per frame)."""
    from bl_helpers import fcurves
    obj.keyframe_insert(path, frame=1)
    n = len(vals)
    frames = np.arange(1, n + 1, dtype=np.float64)
    for fc in fcurves(obj):
        if fc.data_path != path:
            continue
        kp = fc.keyframe_points
        kp.add(n - 1)
        co = np.empty(2 * n)
        co[0::2], co[1::2] = frames, vals[:, fc.array_index]
        kp.foreach_set("co", co)
        kp.foreach_set("interpolation", interp)
        fc.update()


def import_replay(path, attempt=0, prefix="rupert"):
    """Creates the animated empties. Returns {body name: empty}.

    Sequences and practice swarms (rupert/make_sequence.py) contain hard cuts or restarts:
    'cuts' (frame numbers where a new piece starts, same for everyone) and/or 'resets'
    (frames x attempts: True on the last frame before that attempt restarts). Those use
    straight-line keys, and the key before a cut holds still, so nothing smears across it."""
    data = np.load(path)
    xpos = data["xpos"][:, attempt]      # (frames, bodies, 3)
    xquat = data["xquat"][:, attempt]    # (frames, bodies, 4)  w, x, y, z
    alive = data["alive"][:, attempt]
    names = [str(n) for n in data["body_names"]]
    fps = int(round(float(data["fps"])))

    # stop the clip one second after he falls (the rest is him lying still)
    n_frames = len(alive)
    if not alive.all():
        n_frames = min(n_frames, int(np.argmin(alive)) + fps)

    hold = None
    if "cuts" in data or "resets" in data:
        hold = np.zeros(n_frames, bool)
        if "cuts" in data:
            hold[[int(c) - 1 for c in data["cuts"] if 0 < c <= n_frames]] = True
        if "resets" in data:
            hold |= data["resets"][:n_frames, attempt]
        interp = np.where(hold, 0, 1).astype(np.int32)      # 0 = CONSTANT, 1 = LINEAR
    else:
        interp = np.full(n_frames, 2, np.int32)              # 2 = BEZIER (smooth), as before

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, max(scene.frame_end if attempt else 0, n_frames)

    empties = {}
    coll = bpy.data.collections.new(f"{prefix}_replay_attempt{attempt}")
    scene.collection.children.link(coll)
    for b, name in enumerate(names):
        if name in SKIP_BODIES:
            continue
        obj = bpy.data.objects.new(f"{prefix}_{name}", None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 0.1
        obj.rotation_mode = "QUATERNION"
        coll.objects.link(obj)
        empties[name] = obj
        _keys(obj, "location", xpos[:n_frames, b].astype(np.float64), interp)
        _keys(obj, "rotation_quaternion", xquat[:n_frames, b].astype(np.float64), interp)
    print(f"Imported {path}, attempt {attempt}: {n_frames} frames at {fps} fps")
    return empties


if __name__ == "__main__":
    import_replay(*load_args())
