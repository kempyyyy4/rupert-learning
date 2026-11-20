"""Teacher Rupert's whiteboard drawings, in our own hand-drawn marker style (not screenshots).

Each board draws onto a 2200 x 1400 image (1 px = 1 mm of a 2.2 x 1.4 m board) and names the spots
(fractions across and down) that Rupert's pointer stick taps or traces. A board can have moving parts
(the boat going round, the graph drawing itself): those are drawn as one picture per video frame.

Rupert stands just off the right-hand edge, so everything his stick touches is in the right half.

    python -m rupert.board_art points out.png
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2200, 1400
FONT = r"C:\Windows\Fonts\Inkfree.ttf"
PAPER = (250, 250, 247)
INK = {"black": (15, 17, 22), "blue": (15, 50, 170), "red": (190, 15, 20), "green": (10, 120, 40),
       "orange": (235, 95, 0), "grey": (95, 100, 112), "water": (110, 160, 215)}
ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(None)
def font(size):
    return ImageFont.truetype(FONT, size)


class Marker:
    """Wobbly, round-ended marker strokes."""

    def __init__(self, img, seed=0):
        self.d = ImageDraw.Draw(img)
        self.rng = np.random.default_rng(seed)

    def wobbly(self, pts, wobble=2.5):
        """The hand-drawn version of a polyline: densely sampled, with a gentle drift."""
        pts = np.asarray(pts, float)
        dense = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 6))
            dense += [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
        dense.append(pts[-1])
        dense = np.array(dense)
        noise = np.cumsum(self.rng.normal(0, wobble * 0.35, dense.shape), axis=0)
        noise -= np.linspace(0, 1, len(dense))[:, None] * noise[-1]          # ends stay put
        return dense + np.clip(noise, -wobble * 2, wobble * 2)

    def stroke(self, dense, colour="black", width=18):
        c = INK[colour]
        for p, q in zip(dense[:-1], dense[1:]):
            self.d.line([tuple(p), tuple(q)], fill=c, width=width)
        r = width / 2
        for p in dense:
            self.d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=c)

    def line(self, pts, colour="black", width=18, wobble=2.5):
        self.stroke(self.wobbly(pts, wobble), colour, width)

    def circle(self, cx, cy, r, colour="black", width=12, fill=None):
        if fill:
            self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        t = np.linspace(0, 2 * np.pi * 1.04, max(24, int(r / 4)))
        self.line(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)], colour, width, wobble=1.5)

    def text(self, xy, s, size=120, colour="black", anchor="la"):
        self.d.text(xy, s, font=font(size), fill=INK[colour], anchor=anchor,
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


def frac(x, y):
    return (x / W, y / H)


# ---- Lesson 1: how he scores ------------------------------------------------------------------
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
    return {"points": frac(1500, 520), "attempt_over": frac(1500, 940)}


# ---- Lesson 2: his brain (VO: "about 135,000 numbers, all random at the start") ---------------
# Real count for EXP-010's brain: 120 senses in, layers of 256, 256 and 128, 42 out
# (each of his 21 muscles gets a setting and a wobble) = 135,082 numbers.
NET_X = [820, 1080, 1340, 1600, 1860]
NET_N = [6, 7, 7, 6, 5]
NET_Y = 640


def _nodes(k):
    n = NET_N[k]
    return [(NET_X[k], NET_Y + (i - (n - 1) / 2) * 105) for i in range(n)]


def brain(img):
    m = Marker(img, 2)
    m.text((110, 70), "Rupert's brain", 130, "blue")
    m.underline(110, 960, 230, "blue")
    for k in range(len(NET_X) - 1):                       # every line is one of the numbers
        for a in _nodes(k):
            for b in _nodes(k + 1):
                m.line([a, b], "grey", 5, wobble=1.0)
    for k in range(len(NET_X)):
        for (x, y) in _nodes(k):
            m.circle(x, y, 30, "black", 10, fill=PAPER)
    for (x, y), s in zip([(950, 395), (1215, 905), (1470, 360), (1735, 880), (1210, 470)],
                         ["0.37", "-1.2", "0.08", "-0.61", "2.4"]):
        m.text((x, y), s, 52, "orange", "mm")
    m.text((110, 560), "what he", 80, "black", "lm")
    m.text((110, 650), "feels", 80, "black", "lm")
    m.arrow((390, 650), (740, 650), "black", 14, 50)
    m.text((1860, 300), "muscles", 80, "black", "mm")
    m.arrow((1860, 345), (1860, 395), "black", 12, 35)
    m.text((1350, 1110), "135,000 numbers", 150, "red", "mm")
    m.text((1350, 1265), "all random at the start!", 80, "black", "mm")
    zig = [(1080, 560), (1340, 700), (1600, 580), (1860, 640)]
    return {"network": [frac(x, y) for x, y in zig], "numbers": frac(1680, 1110)}


# ---- Lesson 3: his real score graph (EXP-010), drawing itself on ------------------------------
G0, GX, GY0, GY1, GMAX = (520, 1180), 1560, 1180, 400, 1700     # origin, 1 billion steps wide, score scale


def _graph_data():
    p = json.loads((ROOT / "outputs" / "rupert_20261112_1312" / "progress.json").read_text())
    steps = np.array([r["step"] for r in p], float)
    score = np.array([r["score"] for r in p], float)
    return steps, score


def _gxy(step, score):
    return (G0[0] + step / 1e9 * GX, GY0 - max(score, -20) / GMAX * (GY0 - GY1))


_GRAPH = {}


def _typical(score, k=4):
    """His typical score: the middle value of the 9 check-ins around each one (ignores one-off bad days)."""
    return np.array([np.median(score[max(0, i - k): i + k + 1]) for i in range(len(score))])


def graph_static(img):
    """Thin light line: every check-in's real score. Bold line: his typical score at the time."""
    m = Marker(img, 3)
    m.text((110, 70), "Rupert's score", 130, "blue")
    m.underline(110, 950, 230, "blue")
    m.arrow((G0[0], G0[1] + 20), (G0[0], 300), "black", 14, 45)                # points axis
    m.arrow((G0[0] - 20, G0[1]), (G0[0] + GX + 90, G0[1]), "black", 14, 45)     # practice axis
    m.text((G0[0] - 40, 330), "points", 75, "black", "rm")
    m.text((G0[0] + GX / 2, 1335), "practice (moments)", 65, "black", "mm")
    for step, label in ((0, "0"), (5e8, "500 million"), (1e9, "1 billion")):
        x = _gxy(step, 0)[0]
        m.line([(x, G0[1] - 15), (x, G0[1] + 20)], "black", 10)
        m.text((x, 1245), label, 55, "black", "mm")
    steps, score = _graph_data()
    trend = _typical(score)
    _GRAPH["steps"] = steps
    _GRAPH["real"] = Marker(img, 4).wobbly([_gxy(s, v) for s, v in zip(steps, score)], wobble=1.0)
    _GRAPH["trend"] = Marker(img, 8).wobbly([_gxy(s, v) for s, v in zip(steps, trend)], wobble=2.0)
    at = lambda st: _gxy(st, float(np.interp(st, steps, trend)))
    first = at(79626240)
    dips = [_gxy(st, float(score[steps == st][0])) for st in (630128640.0, 708771840.0)]
    best = int(np.argmax(score))
    top = _gxy(steps[best], score[best])
    _GRAPH["notes"] = [
        (79626240, "first finish!", "green", (680, 330), [((720, 380), (first[0] + 8, first[1] - 30))]),
        (630128640, "bad days!", "red", (1150, 1010),
         [((1420, 1045), (dips[0][0], dips[0][1] - 30)), ((1470, 1010), (dips[1][0] - 5, dips[1][1] - 30))]),
        (float(steps[best]), "best score!", "green", (1855, 300), [((1855, 345), (top[0], top[1] - 25))])]
    return {"bad_days": frac(dips[0][0], dips[0][1] - 25), "best": frac(top[0], top[1] - 15)}


def graph_dynamic(img, t, t0=1.3, t1=4.3):
    """The lines draw themselves left to right between t0 and t1; each note pops up as they reach it."""
    if "real" not in _GRAPH:
        graph_static(Image.new("RGB", (W, H)))
    s = np.clip((t - t0) / (t1 - t0), 0, 1)
    x_reached = G0[0] + s * (GX * _GRAPH["steps"][-1] / 1e9)
    m = Marker(img, 5)
    for key, colour, width in (("real", "water", 7), ("trend", "blue", 17)):
        dense = _GRAPH[key]
        k = int(np.searchsorted(np.maximum.accumulate(dense[:, 0]), x_reached))
        if k >= 2:
            m.stroke(dense[:k], colour, width)
    for step, label, colour, xy, arrows in _GRAPH["notes"]:
        if x_reached >= _gxy(step, 0)[0]:
            m.text(xy, label, 70, colour, "lm" if xy[0] < 1700 else "mm")
            for a, b in arrows:
                m.arrow(a, b, colour, 10, 30)


# ---- Lesson 4: the boat that went round in circles (CoastRunners, drawn as our own cartoon) ---
LAGOON, RING, LAP_S = (1690, 790), 175, 1.5
LOOP_T = (2.0, 5.0)             # Rupert's stick chases the boat round twice, just behind it
TARGET_ANGLES = (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)


def _rounded_rect(x0, y0, x1, y1, r, n=10):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -np.pi / 2), (x1 - r, y1 - r, 0),
                       (x0 + r, y1 - r, np.pi / 2), (x0 + r, y0 + r, np.pi)):
        pts += [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a0 + np.pi / 2, n)]
    return pts + [pts[0]]


def _ring(a):
    return (LAGOON[0] + RING * np.cos(a), LAGOON[1] - RING * np.sin(a))   # anticlockwise on screen


def boat_static(img):
    m = Marker(img, 6)
    m.text((110, 70), "Points for hitting targets", 115, "blue")
    m.underline(110, 1420, 215, "blue")
    m.line(_rounded_rect(110, 330, 1260, 1300, 200), "black", 14)          # the race course (a river)
    m.line(_rounded_rect(290, 510, 1080, 1120, 60), "black", 14)
    m.circle(*LAGOON, 300, "black", 14)                                    # the side lagoon
    ImageDraw.Draw(img).rectangle([1245, 735, 1410, 845], fill=PAPER)        # open the channel
    m.line([(1255, 735), (1398, 735)], "black", 14)
    m.line([(1255, 845), (1398, 845)], "black", 14)
    for x, y in ((420, 420), (900, 1210), (190, 800), (1170, 620), (1560, 1010), (1820, 600)):
        m.text((x, y), "~", 70, "water", "mm")
    for i in range(6):                                                     # chequered finish line
        for j in range(2):
            if (i + j) % 2 == 0:
                x, y = 560 + j * 45, 335 + i * 29
                ImageDraw.Draw(img).rectangle([x, y, x + 45, y + 29], fill=INK["black"])
    m.line([(560, 330), (560, 520)], "black", 8)
    m.line([(650, 330), (650, 520)], "black", 8)
    m.text((605, 280), "FINISH", 60, "black", "mm")
    m.arrow((880, 420), (720, 420), "grey", 10, 30)                       # the way the race goes
    m.text((910, 420), "the race", 55, "grey", "lm")
    m.text((1690, 1190), "finished the race?", 75, "black", "mm")
    m.text((1690, 1310), "NEVER", 110, "red", "mm")
    for a in TARGET_ANGLES:
        x, y = _ring(a)
        m.circle(x, y, 34, "green", 10, fill=PAPER)
        m.circle(x, y, 10, "green", 10)
    a0 = _boat_angle(LOOP_T[0]) - 0.55
    laps = (LOOP_T[1] - LOOP_T[0]) / LAP_S
    chase = [frac(*_ring(a)) for a in np.linspace(a0, a0 + 2 * np.pi * laps, int(32 * laps) + 1)]
    return {"loop": chase, "never": frac(1880, 1310)}


def _boat_angle(t):
    return 2 * np.pi * t / LAP_S


def boat_dynamic(img, t):
    m = Marker(img, 7)
    a = _boat_angle(t)
    hits = int(np.floor((a - TARGET_ANGLES[0]) / (2 * np.pi / 3))) + 1
    m.text((1660, 330), "SCORE", 70, "black", "rm")
    m.text((1690, 330), f"{1000 + 100 * max(hits, 0):,}", 110, "orange", "lm")
    for k, ta in enumerate(TARGET_ANGLES):                                   # "+100" pops as each is hit
        since = ((a - ta) % (2 * np.pi)) / (2 * np.pi) * LAP_S
        if since < 0.45 and t > 0.3:
            x, y = _ring(ta)
            m.text((x, y - 70 - since * 120), "+100", 60, "green", "mm")
    x, y = _ring(a)
    dx, dy = -np.sin(a), -np.cos(a)                                          # heading (anticlockwise)
    rot = lambda px, py: (x + px * dx - py * dy, y + px * dy + py * dx)
    hull = [rot(*p) for p in ((60, 0), (22, 27), (-50, 24), (-58, 0), (-50, -24), (22, -27))]
    ImageDraw.Draw(img).polygon(hull, fill=INK["orange"])
    m.line(hull + [hull[0]], "black", 8, wobble=0.8)
    for s in (-1, 1):                                                        # wake
        m.line([rot(-70, s * 18), rot(-130, s * 34)], "water", 8, wobble=0.5)


# ---- the boards -------------------------------------------------------------------------------
# taps: (spot, start_s, end_s) in shot seconds; the board lands at 0.9 s.
BOARDS = {
    "points": dict(static=points, seconds=7.0, screen="LESSON 1",
                   taps=[("points", 2.0, 3.4), ("attempt_over", 3.9, 5.4)]),
    "brain": dict(static=brain, seconds=7.5, screen="LESSON 2",
                  taps=[("network", 2.0, 4.0), ("numbers", 4.6, 6.2)]),
    "graph": dict(static=graph_static, dynamic=graph_dynamic, seconds=9.0, screen="LESSON 3",
                  taps=[("bad_days", 4.9, 6.2), ("best", 6.7, 8.0)]),
    "boat": dict(static=boat_static, dynamic=boat_dynamic, seconds=8.5, screen="LESSON 4",
                 taps=[("loop", *LOOP_T), ("never", 5.7, 7.2)]),
}


def draw(name, path):
    """A still board as one picture. Returns the spots."""
    img = Image.new("RGB", (W, H), PAPER)
    spots = BOARDS[name]["static"](img)
    img.save(path)
    return spots


def draw_frames(name, folder, n, fps, scale=0.6):
    """A board with moving parts: one picture per video frame (f_0001.png ...), drawn smaller
    to keep Blender's memory down (still sharper than the board is on screen). Returns the spots."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGB", (W, H), PAPER)
    spots = BOARDS[name]["static"](base)
    size = (int(W * scale), int(H * scale))
    for f in range(1, n + 1):
        img = base.copy()
        BOARDS[name]["dynamic"](img, (f - 1) / fps)
        img.resize(size, Image.LANCZOS).save(folder / f"f_{f:04d}.png", compress_level=1)
    return spots


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    if BOARDS[name].get("dynamic") and len(sys.argv) > 3:       # preview one moment: ... boat out.png 3.2
        img = Image.new("RGB", (W, H), PAPER)
        print(BOARDS[name]["static"](img))
        BOARDS[name]["dynamic"](img, float(sys.argv[3]))
        img.save(out)
    else:
        print(draw(name, out))
