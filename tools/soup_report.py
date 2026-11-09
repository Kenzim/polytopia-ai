#!/usr/bin/env python3
"""W-D-L of one agent by map preset and map size, from arena or head-to-head JSONL files.

    /kvm/polytopia/.venv/bin/python ai/tools/soup_report.py --agent avg_h50 FILE.jsonl [FILE2.jsonl ...]

The agent is the seat whose name contains --agent. A draw is a capped game (150-turn cap, no winner). Rows are
presets, then sizes, then the total; each row gives W-D-L, the score (draw = half) with a Wilson 95% interval, and,
for draws, how many of the capped games the agent was ahead on score at the cap.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import PRESET_NAMES, SIZE_NAMES, wilson  # noqa: E402


def rows(files: list[str], agent: str):
    for f in files:
        for line in open(f):
            g = json.loads(line)
            seats = [i for i, a in enumerate(g["agents"]) if agent in a]
            if len(seats) != 1:
                continue
            s = seats[0]
            res = "d" if g["winner"] is None else ("w" if g["winner"] == s else "l")
            ahead = res == "d" and g["scores"][s] > g["scores"][1 - s]
            yield g["setup"]["preset"], g["setup"]["size"], res, ahead


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    tab = defaultdict(lambda: {"w": 0, "d": 0, "l": 0, "ahead": 0})
    for preset, size, res, ahead in rows(a.files, a.agent):
        for key in (("preset", PRESET_NAMES.get(preset, preset)), ("size", SIZE_NAMES.get(size, size)), ("all", "all")):
            tab[key][res] += 1
            tab[key]["ahead"] += ahead
    print(f"{'cell':<20} {'W-D-L':>12} {'score':>6}  95% CI       drawn-ahead")
    for kind in ("preset", "size", "all"):
        for key in sorted(k for k in tab if k[0] == kind):
            t = tab[key]
            n = t["w"] + t["d"] + t["l"]
            sc = t["w"] + 0.5 * t["d"]
            lo, hi = wilson(sc, n)
            print(f"{key[1]:<20} {t['w']:>3}-{t['d']:>3}-{t['l']:<4} {100 * sc / n:5.1f}%  {100 * lo:4.1f} to {100 * hi:4.1f}"
                  f"  {t['ahead']}/{t['d']}")


if __name__ == "__main__":
    main()
