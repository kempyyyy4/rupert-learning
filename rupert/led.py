"""Shapes for a stadium-style LED board: seven-segment digits and a 5x7 dot-matrix font.

Pure Python (no libraries) so both the 2D overlay (hud.py, Pillow) and the 3D board in
Blender (blender/build_scene.py) draw exactly the same digits.
Units are whatever you pass in; y grows DOWNWARD (image style).
"""
import math

SEGMENTS = {          # which segments light for each digit (a=top, then clockwise, g=middle)
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
    "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg", " ": "",
}

GLYPHS = {            # 5x7 dot matrix, rows top to bottom
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    " ": ["00000"] * 7,
}


def segment_polygons(x, y, w, h, t, slant_deg=6.0, gap=None):
    """Hexagonal bar for each of the 7 segments of a digit whose box starts at (x, y)."""
    gap = t * 0.18 if gap is None else gap
    ht = t / 2
    mid = y + h / 2

    def horiz(cy):
        x0, x1 = x + ht + gap, x + w - ht - gap
        return [(x0, cy), (x0 + ht, cy - ht), (x1 - ht, cy - ht), (x1, cy), (x1 - ht, cy + ht), (x0 + ht, cy + ht)]

    def vert(cx, y0, y1):
        y0, y1 = y0 + gap, y1 - gap
        return [(cx, y0), (cx + ht, y0 + ht), (cx + ht, y1 - ht), (cx, y1), (cx - ht, y1 - ht), (cx - ht, y0 + ht)]

    polys = {
        "a": horiz(y + ht), "g": horiz(mid), "d": horiz(y + h - ht),
        "f": vert(x + ht, y + ht, mid), "b": vert(x + w - ht, y + ht, mid),
        "e": vert(x + ht, mid, y + h - ht), "c": vert(x + w - ht, mid, y + h - ht),
    }
    k = math.tan(math.radians(slant_deg))
    bottom = y + h
    return {s: [(px + (bottom - py) * k, py) for px, py in pts] for s, pts in polys.items()}


def dot_positions(textline, x, y, pitch):
    """Centres of the lit dots for a line of dot-matrix text (letters 6 dots apart)."""
    dots = []
    for i, ch in enumerate(textline.upper()):
        rows = GLYPHS.get(ch, GLYPHS[" "])
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    dots.append((x + (i * 6 + c) * pitch, y + r * pitch))
    return dots


def text_width(textline, pitch):
    return (len(textline) * 6 - 1) * pitch


def digits_for(number, n_digits):
    s = str(int(number))[-n_digits:]
    return s.rjust(n_digits)
