#!/usr/bin/env python3
"""Automatic training loop: self-play round, gating match against the champion, promote on a clear win, repeat.

    autoloop.py run      [--dir DIR] [--dry-run]   # the loop (systemd: polyai-autoloop.service)
    autoloop.py status   [--dir DIR]               # print status.txt and the last rounds

Each round trains a candidate from the current champion with a new seed (the recipe below), then plays
  * the gate: 400 games head to head against the champion on a fresh block of evaluation seeds per round,
  * a guard: 200 games against retail Crazy on fixed seeds, compared with the champion on the same seeds.
The candidate is promoted when its gate score is at least GATE_SCORE (0.55 over 400 games: z = 2.0 against 0.5,
one-sided p about 0.023) and its Crazy score is no more than CRAZY_DROP below the champion's.

Every phase is idempotent (rl.py --resume, the arena tools skip finished games), and the decision for a round is
written before the champion changes, so a crash or reboot at any point resumes where it stopped. The loop never
touches polyai.serve: promotions are logged and reported for a person to act on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
PY = "/kvm/polytopia/.venv/bin/python"
DATA = Path("/kvm/polytopia-ai-data")
NOTIFY_TOKEN = Path("/root/.config/polytopia/autoloop_notify_token")
NOTIFY_URL = "https://forge.stackken.com/notify"


@dataclass
class Config:
    runs_dir: str = str(DATA / "runs")
    champion: str = str(DATA / "runs/rl3_b6c96/ckpt_u00480.pt")
    champion_id: str = "rl3_b6c96@480"
    ref: str = str(DATA / "runs/bc3_b6c96_v11/ckpt_final.pt")
    prefix: str = "auto_r"
    # Training recipe: the might1 config plus the advantage filter (rec_advf, the best recipe A/B). Engine features
    # and the belief auxiliary are off: no features or belief model has beaten rl3 u480 head to head.
    updates: int = 300
    games: int = 384
    rollout_steps: int = 32768
    train_threads: int = 12
    torch_threads: int = 4
    need_gb: float = 1.5
    features: bool = False
    belief_weight: float = 0.0
    extra_train_args: list[str] = field(default_factory=list)
    # Gate and guard.
    gate_seeds: int = 200  # each seed is played twice, seats swapped
    gate_seed_base: int = 2_000_000  # round k uses seeds gate_seed_base + 1000 k onwards: a fresh block every round
    gate_score: float = 0.55
    crazy_seeds: int = 100
    crazy_seed_start: int = 1_000_000
    crazy_drop: float = 0.08
    eval_threads: int = 8
    keep_failed_runs: bool = False  # prune a candidate that was not promoted down to ckpt_final.pt


DRY = dict(prefix="dry_r", updates=2, games=16, rollout_steps=512, need_gb=1.0, gate_seeds=4, crazy_seeds=4)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score_file(path: Path) -> dict:
    """W-L-D of the agent in seat `swap` (the candidate in a gate file, the model in an arena file)."""
    w = l = d = 0
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["winner"] is None:
                d += 1
            elif r["winner"] == r["swap"]:
                w += 1
            else:
                l += 1
    n = w + l + d
    s = (w + 0.5 * d) / n if n else 0.0
    lo, hi = wilson(s, n)
    return {"games": n, "wins": w, "losses": l, "draws": d, "score": s, "ci95": [lo, hi]}


def fmt(r: dict) -> str:
    return (f"{r['wins']}-{r['losses']}-{r['draws']} of {r['games']}, {100 * r['score']:.1f}% "
            f"(95% CI {100 * r['ci95'][0]:.1f} to {100 * r['ci95'][1]:.1f})")


class Loop:
    def __init__(self, root: Path, dry: bool):
        self.root, self.dry = root, dry
        root.mkdir(parents=True, exist_ok=True)
        (root / "rounds").mkdir(exist_ok=True)
        (root / "champions").mkdir(exist_ok=True)
        cfg_path = root / "config.json"
        if cfg_path.exists():
            self.cfg = Config(**json.loads(cfg_path.read_text()))
        else:
            self.cfg = Config(**(DRY if dry else {}))
            if dry:
                self.cfg.runs_dir = str(root / "runs")
            write_json(cfg_path, asdict(self.cfg))
        Path(self.cfg.runs_dir).mkdir(parents=True, exist_ok=True)
        st = root / "state.json"
        if not st.exists():
            write_json(st, {"round": 1, "champion": self.cfg.champion, "champion_id": self.cfg.champion_id,
                            "champion_since": now()})
        self.phase = "starting"

    # -- state ------------------------------------------------------------------------------------------------
    @property
    def state(self) -> dict:
        return json.loads((self.root / "state.json").read_text())

    def log(self, msg: str) -> None:
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(self.root / "loop.log", "a") as f:
            f.write(line + "\n")

    def status(self, extra: str = "") -> None:
        st = self.state
        lines = [f"updated:   {now()}", f"round:     {st['round']}  phase: {self.phase}",
                 f"champion:  {st['champion_id']}  ({st['champion']})  since {st['champion_since']}"]
        hist = self.root / "history.jsonl"
        if hist.exists():
            rows = [json.loads(x) for x in hist.read_text().splitlines() if x.strip()]
            if rows:
                last = rows[-1]
                lines.append(f"last gate: round {last['round']} {last['candidate_id']}: gate {fmt(last['gate'])}; "
                             f"crazy {100 * last['crazy']['score']:.1f}% vs champion "
                             f"{100 * last['champion_crazy']['score']:.1f}%; "
                             f"{'PROMOTED' if last['promoted'] else 'not promoted'}")
                lines.append(f"rounds:    {len(rows)} decided, {sum(r['promoted'] for r in rows)} promoted")
        if extra:
            lines.append(extra)
        tmp = self.root / "status.txt.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, self.root / "status.txt")

    def notify(self, text: str) -> None:
        if self.dry or not NOTIFY_TOKEN.exists():
            self.log(f"notify (not sent): {text.splitlines()[0]}")
            return
        token = NOTIFY_TOKEN.read_text().strip()
        req = urllib.request.Request(NOTIFY_URL, data=text.encode(), method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"})
        try:
            urllib.request.urlopen(req, timeout=15).read()
        except Exception as e:  # never let a failed notification stop the loop
            self.log(f"notify failed: {type(e).__name__}")

    # -- subprocesses -----------------------------------------------------------------------------------------
    def run(self, phase: str, args: list[str], log: Path) -> None:
        """Run a phase until it exits 0; on failure back off (10 min, doubling to 1 h) and retry."""
        self.phase = phase
        self.status()
        delay, fails = 600, 0
        env = dict(os.environ, TMPDIR="/kvm/tmp/tmp", PYTHONPATH=str(AI / "python"),
                   OMP_NUM_THREADS=str(self.cfg.torch_threads), MKL_NUM_THREADS=str(self.cfg.torch_threads))
        while True:
            self.log(f"{phase}: start")
            with open(log, "a") as f:
                f.write(f"\n=== {now()} {' '.join(args)}\n")
                f.flush()
                rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, cwd=AI / "python", env=env)
            if rc == 0:
                self.log(f"{phase}: done")
                return
            fails += 1
            tail = "\n".join(log.read_text().splitlines()[-15:])
            self.log(f"{phase}: exit {rc} (failure {fails}), retrying in {delay // 60} min")
            self.status(f"last error: {phase} exit {rc}, retry {fails} at +{delay // 60} min")
            if fails == 3:
                self.notify(f"autoloop: {phase} failed 3 times (exit {rc}); still retrying.\n\n{tail}")
            if self.dry and fails >= 2:
                raise SystemExit(f"dry run: {phase} failed")
            time.sleep(delay)
            delay = min(delay * 2, 3600)

    # -- phases -----------------------------------------------------------------------------------------------
    def train(self, k: int, champ: str) -> Path:
        c = self.cfg
        name = f"{c.prefix}{k:04d}"
        run = Path(c.runs_dir) / name
        final = run / "ckpt_final.pt"
        if final.exists():
            return final
        args = [PY, "-m", "polyai.rl", "--name", name, "--init", champ, "--ref", c.ref, "--runs-dir", c.runs_dir,
                "--updates", str(c.updates), "--games", str(c.games), "--rollout-steps", str(c.rollout_steps),
                "--n-players", "2:55,3:25,4:20", "--turn-limit", "150", "--mode", "might",
                "--pool-weights",
                "retail:easy=0.05,retail:normal=0.15,retail:hard=0.15,retail:crazy=0.15,bc=0.15,self=0.35",
                "--lr", "3e-05", "--ent-coef", "0.0005", "--kl-coef", "0.02", "--kl-coef-end", "0.02",
                "--critic-warmup", "2", "--snapshot-every", "10", "--max-snapshots", "6", "--ckpt-every", "20",
                "--adv-filter", "0.5", "--seed", str(k), "--threads", str(c.train_threads),
                "--torch-threads", str(c.torch_threads), "--need-gb", str(c.need_gb), "--device", "cuda"]
        if c.features:
            args.append("--features")
        if c.belief_weight > 0:
            args += ["--belief-weight", str(c.belief_weight), "--belief-warmup", "5"]
        args += c.extra_train_args
        if (run / "ckpt_last.pt").exists():
            args.append("--resume")
        self.run(f"train {name}", args, self.root / "rounds" / f"r{k:04d}" / "train.log")
        if not final.exists():
            raise RuntimeError(f"{final} missing after training")
        return final

    def gate(self, k: int, cand: str, champ: str) -> dict:
        c, rd = self.cfg, self.root / "rounds" / f"r{k:04d}"
        out = rd / "gate.jsonl"
        if score_file(out)["games"] < 2 * c.gate_seeds:
            self.run("gate", [PY, "-m", "polyai.h2h", "--a", cand, "--b", champ, "--seeds", str(c.gate_seeds),
                              "--seed-start", str(c.gate_seed_base + 1000 * k), "--mode", "might",
                              "--threads", str(c.eval_threads), "--device", "cpu", "--out", str(out)],
                     rd / "gate.log")
        return score_file(out)

    def crazy(self, path: str, out: Path, log: Path, phase: str) -> dict:
        c = self.cfg
        if score_file(out)["games"] < 2 * c.crazy_seeds:
            self.run(phase, [PY, "-m", "polyai.arena", "--model", path, "--opponent", "retail:crazy",
                             "--seeds", str(c.crazy_seeds), "--seed-start", str(c.crazy_seed_start),
                             "--mode", "might", "--threads", str(c.eval_threads), "--device", "cpu",
                             "--out", str(out)], log)
        return score_file(out)

    def champion_crazy(self, st: dict) -> dict:
        safe = st["champion_id"].replace("/", "_").replace("@", "_u")
        out = self.root / "champions" / f"{safe}_crazy.jsonl"
        return self.crazy(st["champion"], out, self.root / "champions" / f"{safe}_crazy.log",
                          f"champion crazy {st['champion_id']}")

    def prune(self, run: Path) -> None:
        for p in run.glob("ckpt_*.pt"):
            if p.name != "ckpt_final.pt":
                p.unlink()
        shutil.rmtree(run / "pool", ignore_errors=True)

    # -- the loop ---------------------------------------------------------------------------------------------
    def one_round(self) -> dict:
        st, c = self.state, self.cfg
        k = st["round"]
        rd = self.root / "rounds" / f"r{k:04d}"
        rd.mkdir(exist_ok=True)
        info = rd / "round.json"
        if not info.exists():
            write_json(info, {"round": k, "champion": st["champion"], "champion_id": st["champion_id"],
                              "started": now(), "config": asdict(c)})
        meta = json.loads(info.read_text())
        champ = meta["champion"]
        result_path = rd / "result.json"
        if not result_path.exists():
            self.log(f"round {k}: champion {meta['champion_id']}")
            champ_crazy = self.champion_crazy(st)
            cand = self.train(k, champ)
            cand_id = f"{c.prefix}{k:04d}@{c.updates}"
            g = self.gate(k, str(cand), champ)
            cz = self.crazy(str(cand), rd / "crazy.jsonl", rd / "crazy.log", "crazy guard")
            promoted = g["score"] >= c.gate_score and cz["score"] >= champ_crazy["score"] - c.crazy_drop
            result = {"round": k, "decided": now(), "champion_id": meta["champion_id"], "champion": champ,
                      "candidate_id": cand_id, "candidate": str(cand), "gate": g, "crazy": cz,
                      "champion_crazy": champ_crazy, "gate_score": c.gate_score, "crazy_drop": c.crazy_drop,
                      "promoted": promoted}
            write_json(result_path, result)
            with open(self.root / "history.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
            self.log(f"round {k}: gate {fmt(g)}; crazy {fmt(cz)}; "
                     f"{'PROMOTED' if promoted else 'not promoted'}")
        result = json.loads(result_path.read_text())
        new = dict(st, round=k + 1)
        if result["promoted"] and st["champion"] != result["candidate"]:
            new.update(champion=result["candidate"], champion_id=result["candidate_id"], champion_since=now())
            with open(self.root / "promotions.jsonl", "a") as f:
                f.write(json.dumps({"round": k, "at": now(), "from": result["champion_id"],
                                    "to": result["candidate_id"], "path": result["candidate"],
                                    "gate": result["gate"], "crazy": result["crazy"]}) + "\n")
            write_json(self.root / "state.json", new)
            self.notify(f"autoloop PROMOTION, round {k}: {result['candidate_id']} replaces {result['champion_id']}.\n"
                        f"gate {fmt(result['gate'])}\ncrazy {fmt(result['crazy'])} (champion "
                        f"{100 * result['champion_crazy']['score']:.1f}%)\npath {result['candidate']}\n"
                        f"polyai.serve is unchanged.")
        else:
            write_json(self.root / "state.json", new)
            if not result["promoted"] and not c.keep_failed_runs:
                self.prune(Path(result["candidate"]).parent)
        self.status()
        return result

    def forever(self, rounds: int | None) -> None:
        done = 0
        while rounds is None or done < rounds:
            self.one_round()
            done += 1
        self.phase = "stopped (round limit)"
        self.status()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--dir", default=str(DATA / "autoloop"))
    ap.add_argument("--dry-run", action="store_true", help="tiny settings; use with a scratch --dir")
    ap.add_argument("--rounds", type=int, default=None, help="stop after this many rounds (default: forever)")
    a = ap.parse_args(argv)
    root = Path(a.dir)
    if a.cmd == "status":
        print((root / "status.txt").read_text() if (root / "status.txt").exists() else "no status yet")
        hist = root / "history.jsonl"
        if hist.exists():
            for line in hist.read_text().splitlines()[-5:]:
                r = json.loads(line)
                print(f"  round {r['round']}: {r['candidate_id']} gate {fmt(r['gate'])}, crazy "
                      f"{100 * r['crazy']['score']:.1f}% -> {'promoted' if r['promoted'] else 'kept champion'}")
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Loop(root, a.dry_run).forever(a.rounds)


if __name__ == "__main__":
    main()
