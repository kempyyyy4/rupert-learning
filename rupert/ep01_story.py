"""EP01's story, chapter by chapter: which training run (and which part of it) each
'timelapse -> breakthrough -> cheat -> fix' chapter is cut from, and the on-screen attempt
counter, which is a RUNNING TOTAL over all of Rupert's training, in story order.

Story order differs a little from the order the runs happened: cheat #1 (standing still) first
happened in the stock setup (EXP-002, no replays saved), so its footage comes from the start of
EXP-006, the other run where he did it. Every attempt is counted once.
"""
from pathlib import Path

from rupert.attempts import attempts_by_step

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

# (name, run folder, first step, step of the cheat clip, last step this chapter accounts for)
CHAPTERS = [
    ("cheat1_stands_still", "rupert_20261107_1159", 0, 23592960, 23592960),
    ("cheat2_leaning_back", "colab_run3", 0, 117964800, None),
    ("cheat3_pogo", "rupert_20261106_2121", 0, 101253120, None),
    ("cheat4_straight_legs", "rupert_20261106_2220", 0, 345047040, None),
    ("cheat5_knees", "rupert_20261107_1159", 23592960, 728432640, None),
    ("cheat6_trex", "rupert_20261107_1323", 0, 875888640, None),
    ("cheat7_limp", "rupert_20261107_2130", 0, 757923840, None),
    ("cheat8_gallop", "rupert_20261108_0008", 0, 698941440, None),
]


def _last_step(run):
    import json
    return max(r["step"] for r in json.loads((OUT / run / "progress.json").read_text()))


def chapter(name):
    i = next(k for k, c in enumerate(CHAPTERS) if c[0] == name)
    return i, CHAPTERS[i]


def offset(i):
    """Attempts made in all chapters before chapter i (each chapter counts its run up to its end)."""
    total = 0
    for name, run, s0, _, s_end in CHAPTERS[:i]:
        end = s_end if s_end is not None else _last_step(run)
        total += attempts_by_step(OUT / run, end) - attempts_by_step(OUT / run, s0)
    return total


def counter(name, step):
    """The on-screen total at this training step of this chapter's run."""
    i, (_, run, s0, _, _) = chapter(name)
    return offset(i) + attempts_by_step(OUT / run, step) - attempts_by_step(OUT / run, s0) + 1


def eval_steps(name, upto_cheat=True):
    """Check-ins of this chapter's run that have saved replays, from its first step to the cheat."""
    _, (_, run, s0, cheat, _) = chapter(name)
    steps = sorted(int(p.stem.split("_")[1]) for p in (OUT / run / "eval").glob("step_*.npz"))
    return [s for s in steps if s0 <= s <= (cheat if upto_cheat else 10 ** 12)]


if __name__ == "__main__":
    for name, run, s0, cheat, _ in CHAPTERS:
        print(f"{name:22s} {run:22s} counter {counter(name, s0):>11,} -> {counter(name, cheat):>11,}"
              f"  replays: {len(eval_steps(name))}")
    print(f"Grand total after all eight: {offset(len(CHAPTERS)):,}")
