"""The `evaluate` request of the policy service (docs/interfaces.md section 1.2.3): win chances per player and the
model's preferred moves along a web engine line.

A line is a polytopia-position file and the steps played from it, each by its command text. The line is replayed
once; at every requested point the position is kept (a copy), and every asked-for seat's own fogged view is encoded
there (`Game.encode_for`), so a seat's number never depends on what it has not seen. The value network and the
policy are then run in batches over all points at once.

A win chance is a number only where the analysis value's calibration layer has a temperature for the game's kind and
stage (`polyai.analysis`); elsewhere it is null with the reason. It is exact where the game has ended (1 for the
winners, 0 for the others) and 0 for a player already eliminated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .analysis import Calibration, ValueNet, calibrated_win, kind_of, load_value, stage_of
from .policy import collate, observe_games

NATURE = 255
MAX_AT = 5000
MAX_MOVES = 50
MAX_STEPS = 200_000
VALUE_BATCH = 256
POLICY_BATCH = 64
UNSUPPORTED = "unsupported position: "


class EvaluateError(Exception):
    """A request `evaluate` refuses; the message goes to the client."""


@dataclass
class _ValueJob:
    """One view to score: where the answer goes (`slot`), the encoded view, and its temperature."""
    tiles: np.ndarray
    globals_: np.ndarray
    kinds: np.ndarray
    mover: bool
    ab: tuple[float, float]
    out: dict
    key: str


@dataclass
class _Point:
    at: int
    game: object
    turn: int
    mover: int
    over: bool
    win: dict = field(default_factory=dict)
    why: dict = field(default_factory=dict)
    moves: list = field(default_factory=list)


class Evaluator:
    """The policy's moves and the analysis value's win chances. `value_path` is an analysis value checkpoint
    (`polyai.train_value`); without one every win chance is null."""

    def __init__(self, policy, value_path: str | Path | None = None, device: torch.device | str | None = None):
        import polyai_engine as pe

        self.pe = pe
        self.policy = policy
        self.device = torch.device(device) if device is not None else getattr(policy, "device", torch.device("cpu"))
        self.value: ValueNet | None = None
        self.cal = Calibration()
        self.value_id = None
        if value_path is not None:
            self.value, self.cal, meta = load_value(value_path, self.device)
            self.value_id = meta.get("name") or Path(value_path).stem

    # -- request ----------------------------------------------------------------------------------------------------

    def evaluate(self, req: dict) -> dict:
        t0 = time.perf_counter()
        position, steps = _line(req.get("line"))
        at = req.get("at")
        if at is None:
            at = [len(steps)]
        if not isinstance(at, list) or not at or len(at) > MAX_AT \
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i <= len(steps) for i in at):
            raise EvaluateError(f"'at' must be null or a list of 1 to {MAX_AT} step counts from 0 to {len(steps)}")
        seats = req.get("seats")
        if seats is not None and (not isinstance(seats, list) or any(isinstance(s, bool) or not isinstance(s, int) for s in seats)):
            raise EvaluateError("'seats' must be null or a list of engine player ids")
        k = req.get("moves", 0)
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= MAX_MOVES:
            raise EvaluateError(f"'moves' must be an integer from 0 to {MAX_MOVES}")
        after, played = req.get("after", False), req.get("played", False)
        if not isinstance(after, bool) or not isinstance(played, bool):
            raise EvaluateError("'after' and 'played' must be true or false")

        unsupported = None
        try:
            game = self.pe.Game.from_position(position)
        except ValueError as e:
            if not str(e).startswith(UNSUPPORTED):
                raise EvaluateError(str(e)) from None
            unsupported = str(e)[len(UNSUPPORTED):]
            game = self.pe.Game.from_position(position, strict=False)

        points = self._replay(game, steps, sorted(set(at)))
        info = self._facts(game)
        jobs: list[_ValueJob] = []
        policy_points: list[_Point] = []
        for p in points.values():
            self._seat_values(p, seats, info, unsupported, jobs)
            if k > 0 and not p.over and p.mover != NATURE and unsupported is None:
                policy_points.append(p)
        self._policy_moves(policy_points, steps, k, played)
        if after:
            for p in policy_points:
                for m in p.moves:
                    self._after_value(p, m, info, jobs)
        self._run_values(jobs)
        evals = []
        for i in at:
            p = points[i]
            e = {"at": i, "turn": p.turn, "mover": p.mover, "win": p.win, "why": p.why}
            if k > 0:
                e["moves"] = p.moves
            evals.append(e)
        return {"policy_id": self.policy.policy_id, "value_id": self.value_id, "engine": self.pe.ENGINE_COMMIT,
                "think_ms": (time.perf_counter() - t0) * 1000.0, "evals": evals}

    # -- replay -----------------------------------------------------------------------------------------------------

    def _replay(self, game, steps: list[str], ats: list[int]) -> dict[int, _Point]:
        points, want, last = {}, set(ats), ats[-1]
        for i in range(last + 1):
            if i in want:
                g = game.clone()
                points[i] = _Point(i, g, int(g.turn), int(g.mover_id), bool(g.is_over()))
            if i < last:
                try:
                    game.play_text(steps[i])
                except ValueError as e:
                    raise EvaluateError(f"step {i}: {e}") from None
        return points

    @staticmethod
    def _facts(game) -> dict:
        s = game.settings()
        return {"mode": int(s["rules_mode"]), "n_players": len(game.players())}

    # -- win chances ------------------------------------------------------------------------------------------------

    def _seat_values(self, p: _Point, seats, info: dict, unsupported: str | None, jobs: list) -> None:
        players = {pid: alive for pid, alive, _ in p.game.players()}
        winners = set(p.game.winner_ids()) if p.over else set()
        wanted = seats if seats is not None else [pid for pid, alive in players.items() if alive]
        for pid in wanted:
            key = str(pid)
            if pid not in players:
                p.win[key] = None
                p.why[key] = f"{pid} is not a player of this game"
            elif p.over:
                p.win[key] = 1.0 if pid in winners else 0.0
            elif not players[pid]:
                p.win[key] = 0.0
            elif unsupported is not None:
                p.win[key] = None
                p.why[key] = unsupported
            else:
                self._queue(p.game, pid, info, p.win, p.why, key, jobs)

    def _queue(self, g, pid: int, info: dict, win: dict, why: dict, key: str, jobs: list) -> None:
        """Queue the view of `pid` in `g` for the value net, or set the null and its reason."""
        if self.value is None:
            win[key] = None
            why[key] = "the service has no calibrated value net loaded"
            return
        kinds = np.asarray(g.slot_kinds(pid))
        kind = kind_of(info["mode"], info["n_players"], kinds)
        stage = int(stage_of(int(g.turn)))
        mover = int(g.mover_id) == pid
        ab = self.cal.get(kind, stage, mover)
        if ab is None:
            win[key] = None
            why[key] = self.cal.reason(kind, stage)
            return
        tiles, globals_ = g.encode_for(pid)
        win[key] = None
        jobs.append(_ValueJob(tiles, globals_, kinds, mover, ab, win, key))

    def _after_value(self, p: _Point, m: dict, info: dict, jobs: list) -> None:
        c = p.game.clone()
        m["win"] = None
        try:
            c.play_text(m["text"])
        except ValueError as e:
            m["why"] = str(e)
            return
        players = {pid: alive for pid, alive, _ in c.players()}
        if c.is_over():
            m["win"] = 1.0 if p.mover in set(c.winner_ids()) else 0.0
        elif not players.get(p.mover, False):
            m["win"] = 0.0
        else:
            why: dict = {}
            self._queue(c, p.mover, info, m, why, "win", jobs)
            if "win" in why:
                m["why"] = why["win"]

    @torch.inference_mode()
    def _run_values(self, jobs: list[_ValueJob]) -> None:
        if not jobs:
            return
        by_shape: dict[tuple, list[_ValueJob]] = {}
        for j in jobs:
            by_shape.setdefault(j.tiles.shape, []).append(j)
        for group in by_shape.values():
            for s in range(0, len(group), VALUE_BATCH):
                chunk = group[s:s + VALUE_BATCH]
                put = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).to(self.device)  # noqa: E731
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.value(put([j.tiles for j in chunk]), put([j.globals_ for j in chunk]),
                                        put([j.kinds for j in chunk]), put([np.int8(j.mover) for j in chunk]))
                ab = torch.tensor([j.ab for j in chunk], dtype=torch.float32, device=self.device)
                p = calibrated_win(logits.float(), ab[:, 0], ab[:, 1]).cpu().numpy()
                for j, x in zip(chunk, p):
                    j.out[j.key] = round(float(x), 4)

    # -- moves ------------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def _policy_moves(self, points: list[_Point], steps: list[str], k: int, played: bool) -> None:
        model = self.policy.model
        feats_on = bool(model.cfg.features)
        for s in range(0, len(points), POLICY_BATCH):
            chunk = points[s:s + POLICY_BATCH]
            obs = observe_games([p.game for p in chunk], features=feats_on)
            sel = np.arange(len(chunk))
            tiles, globals_, actions, mask = collate(obs, sel, self.device)
            feats = {}
            if feats_on:
                from .policy import collate_features
                feats = collate_features(obs, sel, self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = model(tiles, globals_, actions, mask, None, **feats)
            probs = F.softmax(out.policy_logits.float().masked_fill(~mask, float("-inf")), dim=1).cpu().numpy()
            for row, p in enumerate(chunk):
                texts = p.game.legal_texts()
                pr = probs[row, :len(texts)]
                order = np.argsort(-pr, kind="stable")[:k]
                nxt = steps[p.at] if p.at < len(steps) else None
                p.moves = [{"text": texts[i], "p": round(float(pr[i]), 4), "win": None, "played": texts[i] == nxt}
                           for i in order]
                if played and nxt is not None and not any(m["played"] for m in p.moves):
                    j = next((i for i, t in enumerate(texts) if t == nxt), None)
                    p.moves.append({"text": nxt, "p": None if j is None else round(float(pr[j]), 4), "win": None,
                                    "played": True, "outside_top": True})


def _line(line) -> tuple[str, list[str]]:
    if not isinstance(line, dict):
        raise EvaluateError("'line' must be an object with 'position' and 'steps'")
    position = line.get("position")
    if isinstance(position, dict):
        position = json.dumps(position)
    elif not isinstance(position, str):
        raise EvaluateError("'line.position' must be a polytopia-position object")
    steps = line.get("steps", [])
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise EvaluateError(f"'line.steps' must be a list of at most {MAX_STEPS} steps")
    texts = []
    for i, st in enumerate(steps):
        if not isinstance(st, dict) or not isinstance(st.get("text"), str):
            raise EvaluateError(f"step {i}: each step must be an object with a 'text' string")
        texts.append(st["text"])
    return position, texts
