"""Renders a finished-look clip of one test attempt: Blender scene + costume + attempt counter.

  python -m rupert.make_clip --run outputs/rupert_20261106_2220 --step 345047040 --start 10 --seconds 5

Steps: (1) test-run that checkpoint if it hasn't been (evaluate.py), (2) build the Blender
scene around the replay (cached in <run>/blender/), (3) render the frames with Blender,
(4) lay the attempt counter over them and save an mp4 (plus the counter PNG for Premiere).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

from rupert.attempts import attempts_by_step
from rupert.hud import make_hud

ROOT = Path(__file__).parent.parent
BLENDER = os.environ.get("RUPERT_BLENDER", r"E:\Blender\blender.exe")
FPS = 50


def run(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(f"Failed: {cmd[0]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True, help="checkpoint step number")
    ap.add_argument("--replay", type=int, default=0, help="which of the test attempts to film")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the attempt")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--camera", default="cam_three_quarter")
    ap.add_argument("--costume", default="ep01_70s")
    ap.add_argument("--scale", type=int, default=100, help="resolution percent of 1920x1080")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the Blender scene even if cached")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run)
    stem = f"step_{a.step:013d}"
    npz = run_dir / "eval" / f"{stem}.npz"
    if not npz.exists():
        run([sys.executable, "-m", "rupert.evaluate", "--run", run_dir, "--checkpoint",
             run_dir / "checkpoints" / f"{stem}.pkl", "--attempts", 8])

    attempt = attempts_by_step(run_dir, a.step)
    blend = run_dir / "blender" / f"{stem}_{a.costume}_r{a.replay}.blend"
    scripts_time = max(p.stat().st_mtime for p in (ROOT / "blender").glob("*.py"))
    if a.rebuild or not blend.exists() or blend.stat().st_mtime < scripts_time:
        run([BLENDER, "-b", "--python", ROOT / "blender" / "build_scene.py", "--", npz,
             "--attempt", a.replay, "--out", blend, "--costume", a.costume, "--screen", f"ATTEMPT {attempt:,}"])

    first = 1 + int(round(a.start * FPS))
    last = first + int(round(a.seconds * FPS)) - 1
    out = Path(a.out) if a.out else ROOT / "renders" / f"{run_dir.name}_{stem}_{a.camera}_{a.start:g}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1920 * a.scale // 100, 1080 * a.scale // 100
    with tempfile.TemporaryDirectory() as tmp:
        settings = (f"import bpy; s=bpy.context.scene; s.render.resolution_percentage={a.scale}; "
                    f"s.camera=s.objects['{a.camera}']")
        run([BLENDER, "-b", blend, "--python-expr", settings, "-o", Path(tmp) / "f_####",
             "-s", first, "-e", last, "-a"])
        hud = make_hud(out.with_suffix(".hud.png"), w, h, attempt)
        run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", FPS,
             "-start_number", first, "-i", Path(tmp) / "f_%04d.png", "-i", hud,
             "-filter_complex", "overlay", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out])
    print(f"Saved {out}  (attempt counter: about #{attempt:,}; overlay also saved as {hud.name})")


if __name__ == "__main__":
    main()
