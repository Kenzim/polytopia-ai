"""The `evaluate` request of the policy service (docs/interfaces.md section 1.3.1): win chances per player and the
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
from .serve import RequestError, parse_line

NATURE = 255
MAX_AT = 5000
MAX_MOVES = 50
MAX_STEPS = 200_000
VALUE_BATCH = 256
POLICY_BATCH = 64
MAX_PLAYOUTS = 64
MAX_PLAYOUT_STEPS = 2000
UNSUPPORTED = "unsupported position: "


class EvaluateError(RequestError):
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
    broken: str | None = None
    refused: int = -1
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
        position, steps = parse_line(req.get("line"))
        if len(steps) > MAX_STEPS:
            raise EvaluateError(f"'line.steps' holds {len(steps)} steps; at most {MAX_STEPS}")
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
        turns = _turns(req.get("line"), len(steps))
        try:
            game = self.pe.Game.from_position(position)
        except ValueError as e:
            if not str(e).startswith(UNSUPPORTED):
                raise EvaluateError(str(e)) from None
            unsupported = str(e)[len(UNSUPPORTED):]
            game = self.pe.Game.from_position(position, strict=False)

        points = self._replay(game, steps, sorted(set(at)), turns, strict=unsupported is None)
        info = self._facts(game)
        jobs: list[_ValueJob] = []
        policy_points: list[_Point] = []
        for p in points.values():
            if p.broken is not None:
                for pid in seats if seats is not None else [pid for pid, _, _ in game.players()]:
                    p.win[str(pid)] = None
                    p.why[str(pid)] = p.broken
                continue
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
        out = {"policy_id": self.policy.policy_id, "value_id": self.value_id, "engine": self.pe.ENGINE_COMMIT,
               "think_ms": (time.perf_counter() - t0) * 1000.0, "evals": evals}
        refused = sorted({p.refused for p in points.values() if p.broken is not None})
        if refused:
            out["refused"] = refused
        return out

    # -- replay -----------------------------------------------------------------------------------------------------

    def _replay(self, game, steps: list[str], ats: list[int], turns: dict[int, str], strict: bool = True) -> dict[int, _Point]:
        """The position at every step count of `ats`. Where `turns` holds a stored position for a step count, the
        replay continues from it. Without `turns` a refused step is an error; with them, the points from a refused
        step up to the next stored position are marked broken."""
        points, want, last = {}, set(ats), ats[-1]
        broken: tuple[int, str] | None = None
        for i in range(last + 1):
            if i in turns and (i > 0 or broken is not None):
                try:
                    game = self.pe.Game.from_position(turns[i], strict=strict)
                except ValueError as e:
                    raise EvaluateError(f"line.turns at {i}: {e}") from None
                broken = None
            if i in want:
                g = game.clone()
                p = _Point(i, g, int(g.turn), int(g.mover_id), bool(g.is_over()))
                if broken is not None:
                    p.refused, p.broken = broken
                points[i] = p
            if i < last and broken is None:
                try:
                    game.play_text(steps[i])
                except ValueError as e:
                    if not turns:
                        raise EvaluateError(f"step {i}: {e}") from None
                    broken = (i, f"step {i} cannot be replayed on engine {self.pe.ENGINE_COMMIT} ({e}); no number until "
                                 "the next stored turn position")
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
    def _probs(self, games: list) -> list[np.ndarray]:
        """The policy's probability over each game's legal list (in `legal_texts` order), batched."""
        model = self.policy.model
        feats_on = bool(model.cfg.features)
        out_probs: list[np.ndarray] = []
        for s in range(0, len(games), POLICY_BATCH):
            chunk = games[s:s + POLICY_BATCH]
            obs = observe_games(chunk, features=feats_on)
            sel = np.arange(len(chunk))
            tiles, globals_, actions, mask = collate(obs, sel, self.device)
            feats = {}
            if feats_on:
                from .policy import collate_features
                feats = collate_features(obs, sel, self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = model(tiles, globals_, actions, mask, None, **feats)
            probs = F.softmax(out.policy_logits.float().masked_fill(~mask, float("-inf")), dim=1).cpu().numpy()
            lens = np.diff(np.asarray(obs["act_off"]))
            out_probs.extend(probs[row, :lens[row]] for row in range(len(chunk)))
        return out_probs

    def _policy_moves(self, points: list[_Point], steps: list[str], k: int, played: bool) -> None:
        probs = self._probs([p.game for p in points])
        for p, pr in zip(points, probs):
            texts = p.game.legal_texts()
            order = np.argsort(-pr, kind="stable")[:k]
            nxt = steps[p.at] if p.at < len(steps) else None
            p.moves = [{"text": texts[i], "p": round(float(pr[i]), 4), "win": None, "played": texts[i] == nxt}
                       for i in order]
            if played and nxt is not None and not any(m["played"] for m in p.moves):
                j = next((i for i, t in enumerate(texts) if t == nxt), None)
                p.moves.append({"text": nxt, "p": None if j is None else round(float(pr[j]), 4), "win": None,
                                "played": True, "outside_top": True})

    # -- search -----------------------------------------------------------------------------------------------------

    def search(self, req: dict) -> dict:
        """The deeper check (docs/analysis.md section 5): for the mover at one point, its candidate moves (the
        policy's top `k` and the move played) each played out `playouts` times to the start of the mover's next turn
        (the mover and people by the sampled policy, bots by the retail AI), the leaves scored by the calibrated
        value from the mover's own view. Play-out j of every candidate uses the same random draws."""
        t0 = time.perf_counter()
        position, steps = parse_line(req.get("line"))
        at = req.get("at", len(steps))
        if isinstance(at, bool) or not isinstance(at, int) or not 0 <= at <= len(steps):
            raise EvaluateError(f"'at' must be one step count from 0 to {len(steps)}")
        k, n = req.get("k", 6), req.get("playouts", 16)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise EvaluateError("'k' must be an integer from 1 to 20")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_PLAYOUTS:
            raise EvaluateError(f"'playouts' must be an integer from 1 to {MAX_PLAYOUTS}")
        temperature = req.get("temperature", 1.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.05 <= temperature <= 5:
            raise EvaluateError("'temperature' must be a number from 0.05 to 5")
        seed = req.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluateError("'seed' must be an integer")
        extra = req.get("candidates", [])
        if not isinstance(extra, list) or not all(isinstance(t, str) for t in extra) or len(extra) > 20:
            raise EvaluateError("'candidates' must be a list of at most 20 command texts")
        try:
            game = self.pe.Game.from_position(position)
        except ValueError as e:
            raise EvaluateError(str(e)) from None
        root = self._replay(game, steps, [at], _turns(req.get("line"), len(steps)))[at]
        base = {"at": at, "turn": root.turn, "mover": root.mover}
        if root.broken is not None:
            raise EvaluateError(root.broken)
        if root.over or root.mover == NATURE:
            raise EvaluateError("the game is over or Nature is to move at that point; there is nothing to search")
        info = self._facts(root.game)
        why: dict = {}
        now: dict = {}
        jobs: list[_ValueJob] = []
        self._queue(root.game, root.mover, info, now, why, "win", jobs)
        if "win" in why:
            return base | {"win": None, "why": why["win"], "candidates": [], **self._ids(t0)}

        texts = root.game.legal_texts()
        pr = self._probs([root.game])[0]
        nxt = steps[at] if at < len(steps) else None
        order = [int(i) for i in np.argsort(-pr, kind="stable")[:k]]
        cands = [texts[i] for i in order]
        for t in ([nxt] if nxt is not None else []) + extra:
            if t not in cands:
                cands.append(t)
        p_of = {t: float(pr[i]) for i, t in enumerate(texts)}
        people = {pid for pid, _, human in root.game.players() if human}
        policy_seats = people | {root.mover}

        runs = []  # (candidate index, play-out index, game, rng)
        bad: dict[int, str] = {}
        for c, text in enumerate(cands):
            for j in range(n):
                g = root.game.clone()
                try:
                    g.play_text(text)
                except ValueError as e:
                    bad[c] = str(e)
                    break
                runs.append([c, j, g, np.random.default_rng([seed, j])])
        self._play_out(runs, root.mover, root.turn, policy_seats, float(temperature))

        leaf: dict[tuple[int, int], dict] = {}
        for c, j, g, _ in runs:
            out: dict = {}
            alive = {pid: a for pid, a, _ in g.players()}
            if g.is_over():
                out["win"] = 1.0 if root.mover in set(g.winner_ids()) else 0.0
            elif not alive.get(root.mover, False):
                out["win"] = 0.0
            else:
                lw: dict = {}
                self._queue(g, root.mover, info, out, lw, "win", jobs)
                if "win" in lw:
                    out["why"] = lw["win"]
            leaf[(c, j)] = out
        self._run_values(jobs)

        played_c = cands.index(nxt) if nxt in cands else None
        ref_c = played_c if played_c is not None else 0
        results = []
        for c, text in enumerate(cands):
            r = {"text": text, "p": round(p_of[text], 4) if text in p_of else None, "played": text == nxt}
            if c in bad:
                results.append(r | {"win": None, "why": f"cannot play it: {bad[c]}"})
                continue
            w = np.array([leaf[(c, j)].get("win") for j in range(n)], dtype=object)
            ref = np.array([leaf[(ref_c, j)].get("win") for j in range(n)], dtype=object)
            ok = np.array([x is not None and y is not None for x, y in zip(w, ref)])
            if not ok.any():
                reason = next((leaf[(c, j)]["why"] for j in range(n) if "why" in leaf[(c, j)]), None)
                results.append(r | {"win": None, "why": reason or "no leaf of this move has a calibrated number"})
                continue
            wv, rv = w[ok].astype(float), ref[ok].astype(float)
            d = wv - rv
            m = len(wv)
            results.append(r | {
                "win": round(float(wv.mean()), 4),
                "se": round(float(wv.std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0, 4),
                "diff": round(float(d.mean()), 4),
                "diff_se": round(float(d.std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0, 4),
                "better": round(float((d > 1e-9).mean()), 3),
                "worse": round(float((d < -1e-9).mean()), 3),
                "n": m,
            })
        return base | {"win": now["win"], "compared_to": cands[ref_c], "candidates": results, **self._ids(t0)}

    def _ids(self, t0: float) -> dict:
        return {"policy_id": self.policy.policy_id, "value_id": self.value_id, "engine": self.pe.ENGINE_COMMIT,
                "think_ms": (time.perf_counter() - t0) * 1000.0}

    def _play_out(self, runs: list, mover: int, turn: int, policy_seats: set, temperature: float) -> None:
        """Step every run until the mover's next turn starts, the game ends or the mover is out; batched policy calls
        for the seats in `policy_seats`, the retail AI (and Nature) for the rest."""
        active = list(runs)
        for _ in range(MAX_PLAYOUT_STEPS):
            still, ask = [], []
            for r in active:
                g = r[2]
                if g.is_over() or (int(g.turn) > turn and int(g.mover_id) == mover):
                    continue
                if not any(pid == mover and a for pid, a, _ in g.players()):
                    continue
                still.append(r)
                if int(g.mover_id) in policy_seats:
                    ask.append(r)
                else:
                    g.play_ai_move()
            if not still:
                return
            if ask:
                for r, pr in zip(ask, self._probs([r[2] for r in ask])):
                    q = np.power(np.maximum(pr, 0), 1.0 / temperature)
                    q = q / q.sum()
                    r[2].step(int(r[3].choice(len(q), p=q)))
            active = still

def _turns(line: dict, n_steps: int) -> dict[int, str]:
    """`line.turns` (optional): `[{"at": step count, "position": <polytopia-position object>}]`, positions the game
    was recorded in (the web keeps one at the start of every turn), as {step count: position JSON text}."""
    turns = line.get("turns") if isinstance(line, dict) else None
    if turns is None:
        return {}
    if not isinstance(turns, list) or len(turns) > MAX_STEPS:
        raise EvaluateError("'line.turns' must be a list of {\"at\": step count, \"position\": object}")
    out = {}
    for j, t in enumerate(turns):
        if not isinstance(t, dict) or isinstance(t.get("at"), bool) or not isinstance(t.get("at"), int) \
                or not 0 <= t["at"] <= n_steps or not isinstance(t.get("position"), dict):
            raise EvaluateError(f"'line.turns' entry {j} must be {{\"at\": a step count from 0 to {n_steps}, "
                                "\"position\": a polytopia-position object}")
        out[t["at"]] = json.dumps(t["position"], separators=(",", ":"))
    return out
