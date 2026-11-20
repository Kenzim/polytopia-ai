"""Write the gate results of analysis value checkpoints as the Markdown table of docs/analysis.md and the short JSON of
results/calibration.

usage: python -m polyai.gate_report --gate NAME=DIR [...] --md FILE --json FILE

Each DIR holds a `gate.json` written by `polyai.train_value --report` (variants `uncalibrated`, `calibrated` and
`given`). The table lists, per game kind and stage, the given variant's verdict of every checkpoint, then for each
checkpoint the Brier score, the base rate's Brier score and the expected calibration error of the calibrated variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mark(s: dict) -> str:
    v = s["verdict"]
    return "pass" if v == "pass" else "-" if v in ("too few games", "no number given") else v.replace("fail: ", "fail ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True, help="NAME=DIR with gate.json")
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    gates = {}
    for spec in a.gate:
        name, d = spec.split("=", 1)
        gates[name] = json.loads((Path(d) / "gate.json").read_text())
    names = list(gates)
    kinds = sorted({k for g in gates.values() for k in g["calibrated"]})
    lines = []
    short = {}
    for kind in kinds:
        lines += [f"**{kind}**: " + ", ".join(f"{n} {gates[n]['given'][kind]['status']}" for n in names if kind in gates[n]["given"]), ""]
        head = "| turns | games | base rate | " + " | ".join(f"{n} raw Brier / ECE | {n} Brier / ECE | {n}" for n in names) + " | base Brier |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        first = gates[names[-1]]["uncalibrated"][kind]
        for i, s0 in enumerate(first["stages"]):
            row = [s0["stage"], str(s0.get("games", 0)), f"{s0['base_rate']:.3f}" if "base_rate" in s0 else ""]
            for n in names:
                raw = gates[n]["uncalibrated"].get(kind, {"stages": [{}] * len(first["stages"])})["stages"][i]
                cal = gates[n]["calibrated"].get(kind, {"stages": [{}] * len(first["stages"])})["stages"][i]
                giv = gates[n]["given"].get(kind, {"stages": [{"verdict": "too few games"}] * len(first["stages"])})["stages"][i]
                row.append(f"{raw['brier']:.4f} / {raw['ece']:.3f}" if "brier" in raw else "")
                row.append(f"{cal['brier']:.4f} / {cal['ece']:.3f}" if "brier" in cal else "")
                row.append(mark(giv) if giv.get("given") or giv["verdict"] != "too few games" else "-")
            row.append(f"{s0['base_brier']:.4f}" if "base_brier" in s0 else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for n in names:
            g = gates[n]["given"].get(kind)
            if g is None:
                continue
            short.setdefault(n, {})[kind] = {
                "status": g["status"],
                "stages": {s["stage"]: {"verdict": s["verdict"], "games": s.get("games"), "rows": s.get("rows"),
                                        **{k: round(s[k], 4) for k in ("brier", "base_brier", "ece", "worst_gap", "base_rate")
                                           if k in s}}
                           for s in g["stages"]}}
    Path(a.md).write_text("\n".join(lines) + "\n")
    Path(a.json).write_text(json.dumps(short, indent=1) + "\n")
    print(f"wrote {a.md} and {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
