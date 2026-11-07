"""On-screen overlay (the 'HUD'): attempt counter in the top-left corner.

Makes a transparent PNG the size of the video. make_clip.py lays it over the render;
Tom can also drop the PNG into Premiere on its own track.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path("C:/Windows/Fonts")
ORANGE = (255, 120, 20, 255)
PANEL = (16, 20, 36, 205)


def _font(size, bold=True):
    for name in ("bahnschrift.ttf", "ariblk.ttf", "arialbd.ttf"):
        try:
            f = ImageFont.truetype(str(FONTS / name), size)
            if name == "bahnschrift.ttf":
                f.set_variation_by_name("Bold" if bold else "SemiBold")
            return f
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def make_hud(path, width, height, attempt, label="ATTEMPT"):
    s = height / 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    small, big = _font(int(30 * s)), _font(int(72 * s))
    number = f"#{attempt:,}"
    nw = d.textlength(number, font=big)
    lw = d.textlength(label, font=small)
    x0, y0 = int(48 * s), int(44 * s)
    w = int(max(nw, lw) + 64 * s)
    h = int(150 * s)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=int(18 * s), fill=PANEL)
    d.rounded_rectangle((x0, y0, x0 + int(10 * s), y0 + h), radius=int(5 * s), fill=ORANGE)
    d.text((x0 + int(34 * s), y0 + int(18 * s)), label, font=small, fill=ORANGE)
    d.text((x0 + int(32 * s), y0 + int(52 * s)), number, font=big, fill=(255, 255, 255, 255))
    img.save(path)
    return path
