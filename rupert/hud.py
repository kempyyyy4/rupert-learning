"""On-screen overlay (the 'HUD'): an LED attempts board in the top-left corner, styled like
a trackside timing board (amber seven-segment digits, dot-matrix label, unlit segments
faintly visible).

Makes a transparent PNG the size of the video. make_clip.py lays it over the render;
Tom can also drop the PNG into Premiere on its own track.
"""
from PIL import Image, ImageDraw, ImageFilter

from rupert import led

AMBER = (255, 172, 20)
AMBER_OFF = (48, 34, 14)
PANEL = (12, 12, 14, 235)
BEZEL = (70, 72, 78, 255)
N_DIGITS = 8


def make_hud(path, width, height, attempt, label="ATTEMPTS"):
    s = height / 1080
    dw, dh, t, sp = 44 * s, 80 * s, 10 * s, 14 * s           # digit width/height/thickness/spacing
    pitch = 5.2 * s                                            # dot-matrix dot spacing
    pad = 26 * s
    inner_w = N_DIGITS * dw + (N_DIGITS - 1) * sp + 8 * s
    x0, y0 = 48 * s, 44 * s
    w, h = inner_w + 2 * pad, pad + 7 * pitch + 20 * s + dh + pad

    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle((x0 - 6 * s, y0 - 6 * s, x0 + w + 6 * s, y0 + h + 6 * s), radius=10 * s, fill=BEZEL)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=6 * s, fill=PANEL)

    lit = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lit)
    # label in dot matrix, centred; unlit dots of the full 5x7 grid faintly shown
    lx = x0 + (w - led.text_width(label, pitch)) / 2
    ly = y0 + pad
    r = pitch * 0.36
    for i in range(len(label)):
        for row in range(7):
            for col in range(5):
                cx, cy = lx + (i * 6 + col) * pitch, ly + row * pitch
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=AMBER_OFF)
    for cx, cy in led.dot_positions(label, lx, ly, pitch):
        ld.ellipse((cx - r, cy - r, cx + r, cy + r), fill=AMBER)
    # digits
    dy = ly + 7 * pitch + 20 * s
    for i, ch in enumerate(led.digits_for(attempt, N_DIGITS)):
        dx = x0 + pad + 4 * s + i * (dw + sp)
        for seg, poly in led.segment_polygons(dx, dy, dw, dh, t).items():
            d.polygon(poly, fill=AMBER_OFF)
            if seg in led.SEGMENTS[ch]:
                ld.polygon(poly, fill=AMBER)
    glow = lit.filter(ImageFilter.GaussianBlur(6 * s))
    out = Image.alpha_composite(Image.alpha_composite(base, glow), lit)
    out.save(path)
    return path


if __name__ == "__main__":
    make_hud("hud_test.png", 1920, 1080, 2392852)
