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


def main():
    path, attempt = load_args()
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

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, n_frames

    coll = bpy.data.collections.new(f"rupert_replay_attempt{attempt}")
    scene.collection.children.link(coll)
    for b, name in enumerate(names):
        if name in SKIP_BODIES:
            continue
        obj = bpy.data.objects.new(f"rupert_{name}", None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 0.1
        obj.rotation_mode = "QUATERNION"
        coll.objects.link(obj)
        for f in range(n_frames):
            obj.location = [float(v) for v in xpos[f, b]]
            obj.rotation_quaternion = [float(v) for v in xquat[f, b]]
            obj.keyframe_insert("location", frame=f + 1)
            obj.keyframe_insert("rotation_quaternion", frame=f + 1)
    print(f"Imported {path}, attempt {attempt}: {n_frames} frames at {fps} fps")


main()
