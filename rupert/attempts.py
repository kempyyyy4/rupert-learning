"""How many attempts Rupert had made by a given training step (for the on-screen counter).

An attempt = one try from the start line until he falls or the 20 s runs out.
Training doesn't count these directly, so we estimate: between two check-ins,
attempts = physics steps taken / average attempt length at that check-in.
The average attempt length comes from progress.json (time upright in the test attempts).
It's an estimate (test attempts use his best guess, practice attempts are a bit wobblier,
so the true number is somewhat HIGHER). Say "over" or "about" on screen, never exact.

  python -m rupert.attempts --run outputs/rupert_20261106_2220 --step 345047040
"""
import argparse
import json
from pathlib import Path

CONTROL_DT = 0.02       # seconds per control step (env ctrl_dt)
MAX_ATTEMPT_S = 20.0    # episode length


def attempts_by_step(run_dir, step):
    rows = json.loads((Path(run_dir) / "progress.json").read_text())
    rows = sorted(rows, key=lambda r: r["step"])
    total = 0.0
    for a, b in zip(rows, rows[1:]):
        if a["step"] >= step:
            break
        steps = min(step, b["step"]) - a["step"]
        seconds = min(max(a.get("time_upright_s", MAX_ATTEMPT_S), CONTROL_DT), MAX_ATTEMPT_S)
        total += steps / (seconds / CONTROL_DT)
    return int(total) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args()
    print(f"About {attempts_by_step(a.run, a.step):,} attempts by step {a.step:,}")


if __name__ == "__main__":
    main()
