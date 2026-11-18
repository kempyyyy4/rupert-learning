"""Teacher Rupert's whiteboard drawings, in our own hand-drawn marker style (not screenshots).

Each board is a function that draws onto a 2200 x 1400 image (1 px = 1 mm of a 2.2 x 1.4 m board)
and returns named anchor points (fractions across and down) that Rupert's pointer stick taps.

    python -m rupert.board_art points out.png
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2200, 1400
FONT = r"C:\Windows\Fonts\Inkfree.ttf"
INK = {"black": (15, 17, 22), "blue": (15, 50, 170), "red": (190, 15, 20), "green": (10, 120, 40),
       "orange": (235, 95, 0)}


class Marker:
    """Wobbly, round-ended marker strokes."""

    def __init__(self, img, seed=0):
        self.d = ImageDraw.Draw(img)
        self.rng = np.random.default_rng(seed)

    def line(self, pts, colour="black", width=18, wobble=2.5):
        pts = np.asarray(pts, float)
        dense = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 6))
            dense += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
        dense.append(pts[-1])
        dense = np.array(dense)
        noise = np.cumsum(self.rng.normal(0, wobble * 0.35, dense.shape), axis=0)
        noise -= np.linspace(0, 1, len(dense))[:, None] * noise[-1]          # ends stay put
        dense += np.clip(noise, -wobble * 2, wobble * 2)
        c = INK[colour]
        for p, q in zip(dense[:-1], dense[1:]):
            self.d.line([tuple(p), tuple(q)], fill=c, width=width)
        r = width / 2
        for p in dense:
            self.d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=c)

    def circle(self, cx, cy, r, colour="black", width=12):
        t = np.linspace(0, 2 * np.pi * 1.04, 60)
        self.line(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)], colour, width, wobble=1.5)

    def text(self, xy, s, size=120, colour="black", anchor="la"):
        self.d.text(xy, s, font=ImageFont.truetype(FONT, size), fill=INK[colour], anchor=anchor,
                    stroke_width=max(2, size // 22), stroke_fill=INK[colour])          # fat marker

    def arrow(self, a, b, colour="black", width=16, head=60):
        a, b = np.asarray(a, float), np.asarray(b, float)
        self.line([a, b], colour, width)
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        self.line([b - head * u + head * 0.6 * n, b, b - head * u - head * 0.6 * n], colour, width)

    def underline(self, x0, x1, y, colour="black"):
        self.line([(x0, y), ((x0 + x1) / 2, y + 8), (x1, y - 4)], colour, 10)

    def runner(self, x, y, s=1.0, colour="black"):
        """Little stick Rupert mid-stride (x, y = feet), with his orange headband."""
        L = lambda pts: [(x + s * px, y - s * py) for px, py in pts]
        self.circle(x + s * 8, y - s * 205, s * 32, colour, 16)
        self.line(L([(-30, 214), (46, 214)]), "orange", 18)                     # headband
        self.line(L([(5, 170), (-10, 90)]), colour, 17)                          # body
        self.line(L([(-10, 90), (35, 45), (25, 0)]), colour, 17)                 # front leg
        self.line(L([(-10, 90), (-45, 50), (-75, 30)]), colour, 17)              # back leg
        self.line(L([(0, 150), (40, 120), (60, 145)]), colour, 17)               # arms
        self.line(L([(0, 150), (-35, 125), (-50, 100)]), colour, 17)

    def faller(self, x, y, s=1.0, colour="black"):
        """Little stick Rupert flat on his face (x, y = feet end, lying to the right)."""
        L = lambda pts: [(x + s * px, y - s * py) for px, py in pts]
        self.circle(x + s * 230, y - s * 32, s * 32, colour, 16)
        self.line(L([(208, 72), (240, 75)]), "orange", 18)
        self.line(L([(80, 18), (195, 28)]), colour, 17)                         # body
        self.line(L([(80, 18), (20, 30), (-10, 10)]), colour, 17)
        self.line(L([(80, 18), (30, 2), (0, 0)]), colour, 17)
        self.line(L([(170, 28), (200, 75), (180, 110)]), colour, 17)             # an arm flung up
        self.line(L([(150, 24), (190, 5)]), colour, 17)
        for k in range(3):                                                       # dizzy stars
            self.text((x + s * (210 + 55 * k), y - s * (140 + 25 * (k % 2))), "*", 70, "orange", "mm")


def points(img):
    """'How Rupert scores': moving forward = + points; falling over = attempt over."""
    m = Marker(img, 1)
    m.text((110, 70), "How Rupert scores", 130, "blue")
    m.underline(110, 1150, 230, "blue")
    m.runner(330, 620, 1.25)
    m.arrow((520, 520), (900, 520), "green", 18, 70)
    m.text((950, 520), "+ POINTS", 150, "green", "lm")
    m.faller(250, 1120, 1.25)
    m.line([(600, 860), (720, 1000)], "red", 18)
    m.line([(720, 860), (600, 1000)], "red", 18)
    m.text((800, 940), "ATTEMPT OVER", 135, "red", "lm")
    m.text((1430, 1260), "goal: MOST points", 85, "black", "lm")
    return {"points": (1500 / W, 520 / H), "attempt_over": (1500 / W, 940 / H)}


BOARDS = {"points": points}


def draw(name, path):
    img = Image.new("RGB", (W, H), (250, 250, 247))
    anchors = BOARDS[name](img)
    img.save(path)
    return anchors


if __name__ == "__main__":
    print(draw(sys.argv[1], sys.argv[2]))
