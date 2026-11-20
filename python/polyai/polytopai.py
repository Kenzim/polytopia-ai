"""Games against the PolytopAI website (polytopai.com), played by one of our networks (docs/polytopai.md).

The site plays its bot in the browser without an account: a page asks `/api/play/new` for a game, sends the human's
moves to `/api/play/move` and has the bot answer through `/api/play/bot`. Every answer carries the whole state and the
mover's legal moves. This module is that page for a network of ours:

* `SiteClient` makes the calls (one play id, one connection).
* `SiteGame` is one game. For every decision it turns the site's state into an engine position
  (`polytopai_state.position`), has the engine encode it and list the legal actions, and keeps the actions the site
  offers too (`polytopai_state.match_actions`). The chosen action goes back as the site's own move. The engine never
  steps the game, so a rule the two disagree on costs at most a missing option, never a wrong board.
* `SiteArm` plays several games at once on worker threads, with one batched forward pass for the decisions that are
  waiting. It records the decisions like a rollout does (`rl.SizeStore`, GAE per game), so `rl.train` can take them
  in next to its own games (`--polytopai-games`), and it counts results for an evaluation (`main`).

We are always the first player: the site gives the human that seat.

    python -m polyai.polytopai --ckpt models/auto_r0077 --games 100 --parallel 4 --out results.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import secrets
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import polytopai_state as T
from . import schema as S
from .serve import Batcher

DEFAULT_SITE = "https://polytopai.com"
USER_AGENT = "polyai-bridge"
DEFAULT_SIZES = (11, 14, 16)
DEFAULT_MAPS = (1, 2, 3, 4, 6)  # drylands, lakes, continents, archipelago, pangea (5 is water world)
STANDARD = 3  # the site's full-strength setting; 1 is "easier"
TURN_CAP = 150  # a game that reaches it is left as a draw, like a capped training game
BUSY_STATUS = (429, 502, 503, 504)


class SiteError(Exception):
    """The site could not be reached, refused a call, or answered with something the bridge cannot use."""


class _Stopped(Exception):
    """The arm was closed while a worker was waiting."""


# ---------------------------------------------------------------------------------------------------------------
# The calls
# ---------------------------------------------------------------------------------------------------------------


class SiteClient:
    """The site's play API for one play id (the site keeps one game per id). Not for use from several threads."""

    def __init__(self, base: str = DEFAULT_SITE, play_id: str | None = None, timeout: float = 120.0, retries: int = 4):
        u = urlsplit(base)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError(f"not a site address: {base!r}")
        self._cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self._host, self._prefix = u.netloc, u.path.rstrip("/") + "/api/play/"
        self.play_id = play_id or secrets.token_hex(12)
        self.timeout, self.retries = timeout, retries
        self._conn: http.client.HTTPConnection | None = None
        self.calls = 0
        self.seconds: Counter = Counter()  # by call name

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def call(self, name: str, body: dict | None = None, *, repeatable: bool = False) -> dict:
        """One call; a `body` makes it a POST. A busy answer (429, 502 to 504) is retried after a pause. A broken
        connection is retried only for a `repeatable` call: a move may have arrived, and sending it twice would play
        it twice."""
        headers = {"x-play-id": self.play_id, "user-agent": USER_AGENT, "accept-encoding": "gzip"}
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["content-type"] = "application/json"
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                if self._conn is None:
                    self._conn = self._cls(self._host, timeout=self.timeout)
                self._conn.request("GET" if body is None else "POST", self._prefix + name, body=payload, headers=headers)
                r = self._conn.getresponse()
                raw = r.read()
            except (http.client.HTTPException, OSError) as e:
                self.close()
                if not (repeatable or body is None) or attempt == self.retries:
                    raise SiteError(f"{name}: {type(e).__name__}: {e}") from e
                time.sleep(min(2.0**attempt, 30.0))
                continue
            if r.status in BUSY_STATUS and attempt < self.retries:
                wait = r.getheader("retry-after")
                time.sleep(min(float(wait) if wait and wait.isdigit() else 2.0 ** (attempt + 1), 60.0))
                continue
            if r.status != 200:
                raise SiteError(f"{name}: HTTP {r.status}: {raw[:200]!r}")
            try:
                data = gzip.decompress(raw) if r.getheader("content-encoding") == "gzip" else raw
                out = json.loads(data)
            except (OSError, ValueError) as e:
                raise SiteError(f"{name}: unreadable answer: {e}") from e
            self.calls += 1
            self.seconds[name] += time.perf_counter() - t0
            return out
        raise SiteError(f"{name}: the site stayed busy")


# ---------------------------------------------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------------------------------------------


class Decision:
    """A position of ours awaiting a choice: a one-position observation whose legal list holds only what the site
    offers as well, and for every row the site's move (`True` for a stay, which needs none)."""

    def __init__(self, obs: dict, moves: list):
        self.obs, self.moves = obs, moves


class SiteGame:
    """One game on the site. `start()`, then `decision()` and `play()` in turn until `over`."""

    def __init__(self, client, pe, setup: dict, *, features: bool = False, truth: bool = False,
                 stats: Counter | None = None, trace: list | None = None):
        self.client, self.pe, self.setup = client, pe, setup
        self.features, self.truth = features, truth
        self.stats = stats if stats is not None else Counter()
        self.trace = trace
        self.view: dict = {}
        self.me = 1
        self.stayed: set[int] = set()  # tiles of our units told to stay this turn
        self._turn_key: tuple | None = None
        self.decisions = 0

    # --- what the site says ---------------------------------------------------------------------------------

    @property
    def over(self) -> bool:
        return bool(self.view["play"]["over"])

    @property
    def turn(self) -> int:
        return int(self.view["state"]["settings"]["turn"])

    @property
    def mover(self) -> int:
        return int(self.view["state"]["settings"]["currentPlayerTurnId"])

    @property
    def seed(self) -> int:
        return int(self.view["state"]["settings"].get("seed", 0))

    @property
    def won(self) -> bool:
        return self.view["play"].get("winner") == self.me

    def _take(self, view: dict, what: str, sent: dict | None = None) -> None:
        if not isinstance(view, dict) or view.get("error") or "state" not in view or "play" not in view:
            raise SiteError(f"{what}: {view.get('error') if isinstance(view, dict) else view!r}")
        self.view = view
        if self.trace is not None:
            self.trace.append({"call": what, "sent": sent, "view": view})
        key = (self.turn, self.mover)
        if key != self._turn_key:
            self._turn_key = key
            self.stayed.clear()

    def _bot(self) -> None:
        """Have the site's bot play until it is our move again, or the game is over."""
        for _ in range(400):
            if self.over or self.mover == self.me:
                return
            self._take(self.client.call("bot", {"turn": True}), "bot")
        raise SiteError("the bot did not finish its turn")

    def start(self) -> None:
        s = self.setup
        body = {"mode": "play", "tribes": list(s["tribes"]), "map_type": s["map"], "size": s["size"], "skill": None,
                "steered": False, "level": s["level"], "levels": None, "win": None}
        self._take(self.client.call("new", body, repeatable=True), "new", body)
        self.me = int(self.view["play"]["human"])
        self._bot()

    def _send(self, move: dict) -> None:
        self._take(self.client.call("move", move), "move", move)
        self._bot()

    # --- our decisions --------------------------------------------------------------------------------------

    def decision(self) -> Decision | None:
        """The position to decide, or `None` when the two legal lists had nothing in common and the bridge moved by
        itself (it ends the turn)."""
        legal = self.view.get("legalMoves") or []
        if not legal:
            raise SiteError("it is our move and the site lists none")
        doc, notes = T.position(self.view["state"], legal, frozenset(self.stayed))
        for n in notes:
            self.stats["untranslated: " + n] += 1
        try:
            g = self.pe.Game.from_position(json.dumps(doc))
            acts = np.asarray(g.legal_actions())
        except ValueError as e:
            raise SiteError(f"the engine did not take the position: {e}") from e
        if g.mover_id != self.me:
            raise SiteError(f"the engine has player {g.mover_id} to move, the site player {self.me}")
        matched = T.match_actions(acts, legal)
        keep = [i for i, m in enumerate(matched) if m is not None]
        self._count(acts, legal, matched)
        if not keep:
            self.stats["bridge moves"] += 1
            self._send(next((m for m in legal if m.get("moveType") == T.M_END), legal[0]))
            return None
        tiles, glob = g.encode()
        obs = {
            "tiles": np.asarray(tiles)[None], "globals": np.asarray(glob)[None],
            "act_off": np.array([0, len(keep)], np.int32), "actions": acts[keep],
            "turn": np.array([self.turn], np.int16),
        }
        if self.features:
            ft, fg, fa = g.features()
            obs.update(f_tiles=np.asarray(ft)[None], f_globals=np.asarray(fg)[None], f_actions=np.asarray(fa)[keep])
        if self.truth:
            tr, mask, tg = g.truth()
            obs.update(truth=np.asarray(tr)[None], truth_mask=np.asarray(mask)[None], truth_g=np.asarray(tg)[None])
        return Decision(obs, [matched[i] for i in keep])

    def _count(self, acts: np.ndarray, legal: list[dict], matched: list) -> None:
        st = self.stats
        st["positions"] += 1
        used = {id(m) for m in matched if isinstance(m, dict)}
        ours = [a for a, m in zip(acts.tolist(), matched) if m is None]
        # (the site still lists the moves of a unit we told to stay; that is no disagreement)
        theirs = [m for m in legal if id(m) not in used and m.get("src") not in self.stayed]
        st["positions with the same legal list"] += not ours and not theirs
        st["engine actions"] += len(acts)
        st["site moves"] += len(legal)
        for a in ours:
            st["engine only: " + S.ACTION_KINDS[a[0] - 1]] += 1
        for m in theirs:
            st[f"site only: move type {m.get('moveType')}" + (f" ability {m.get('type')}" if m.get("moveType") == T.M_ABILITY else "")] += 1

    def play(self, d: Decision, index: int) -> None:
        self.decisions += 1
        move = d.moves[index]
        if move is True:
            self.stayed.add(int(d.obs["actions"][index][2]))
        else:
            self._send(move)


# ---------------------------------------------------------------------------------------------------------------
# Several games at once
# ---------------------------------------------------------------------------------------------------------------


class SiteArm:
    """`n_games` games against the site in flight, all played by `policy`.

    The decisions are made only between `resume()` and `pause()`, so the policy's network may be trained in between;
    the calls to the site carry on. With `record`, every decision is stored like a rollout's and `take()` hands over
    what has gathered: one `SizeStore.finalize()` dict per board size (the `extra` of `rl.ppo_update`) and the
    finished games. A game still running at `take()` is cut there like a rollout's.
    """

    def __init__(self, pe, policy, n_games: int, *, level: int = STANDARD, sizes=DEFAULT_SIZES, maps=DEFAULT_MAPS,
                 temperature: float | None = 1.0, gamma: float = 1.0, lam: float = 0.95, credit_per_turn: bool = False,
                 features: bool = False, truth: bool = False, record: bool = True, seed: int = 0,
                 max_games: int | None = None, turn_cap: int = TURN_CAP, base: str = DEFAULT_SITE,
                 trace_dir: str | Path | None = None, client_factory=None, pause_wait: float = 3.0):
        if not hasattr(pe.Game, "from_position"):
            raise SystemExit("this engine module has no Game.from_position; rebuild the bindings (docs/polytopai.md)")
        self.pe, self.policy, self.n_games = pe, policy, n_games
        self.level, self.sizes, self.maps = level, tuple(sizes), tuple(maps)
        self.tribes = tuple(int(t) for t in pe.TRIBES)
        self.temperature, self.gamma, self.lam, self.credit_per_turn = temperature, gamma, lam, credit_per_turn
        self.features, self.truth, self.record = features, truth, record
        self.max_games, self.turn_cap, self.pause_wait = max_games, turn_cap, pause_wait
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.entry = f"site:polytopai{level}"
        self._client = client_factory or (lambda k: SiteClient(base))
        self._rng = np.random.default_rng(seed)  # the setups, drawn under the lock
        self._lock = threading.Lock()  # stores, trajectories, results, counters
        self._cv = threading.Condition()  # the gate of the decisions
        self._open = False
        self._closed = False
        self._deciding = 0
        self._talking = 0  # workers inside a call to the site
        self._stores: dict[tuple, object] = {}
        self._traj: dict[int, tuple] = {}
        self._n_traj = 0
        self._results: list[dict] = []
        self.stats: Counter = Counter()
        self.seconds: Counter = Counter()
        self.started = self.finished = self.aborted = self.decisions = 0
        self.errors: list[str] = []
        self.batcher = Batcher(policy, window_ms=2.0, max_jobs=max(1, n_games))
        self.batcher.start()
        self._threads = [threading.Thread(target=self._work, args=(k,), name=f"polytopai-{k}", daemon=True)
                         for k in range(n_games)]
        for t in self._threads:
            t.start()

    # --- the gate -------------------------------------------------------------------------------------------

    def resume(self) -> None:
        with self._cv:
            self._open = True
            self._cv.notify_all()

    def pause(self) -> None:
        """Stop the decisions. Waits for those in flight, and up to `pause_wait` seconds for the calls to the site in
        flight to come back, so that a game ending there is still counted with its last moves."""
        deadline = time.perf_counter() + self.pause_wait
        with self._cv:
            self._open = False
            while self._deciding:
                self._cv.wait()
            while self._talking and (left := deadline - time.perf_counter()) > 0:
                self._cv.wait(left)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._open = False
            self._cv.notify_all()
        self.batcher.stop()

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @contextmanager
    def _decide(self):
        with self._cv:
            while not self._open and not self._closed:
                self._cv.wait()
            if self._closed:
                raise _Stopped
            self._deciding += 1
        try:
            yield
        finally:
            with self._cv:
                self._deciding -= 1
                self._cv.notify_all()

    @contextmanager
    def _talk(self):
        with self._cv:
            if self._closed:
                raise _Stopped
            self._talking += 1
        try:
            yield
        finally:
            with self._cv:
                self._talking -= 1
                self._cv.notify_all()

    # --- the workers ----------------------------------------------------------------------------------------

    def _next_setup(self) -> dict | None:
        with self._lock:
            if self.max_games is not None and self.started >= self.max_games:
                return None
            self.started += 1
            rng = self._rng
            return {"tribes": [int(rng.choice(self.tribes)), int(rng.choice(self.tribes))], "map": int(rng.choice(self.maps)),
                    "size": int(rng.choice(self.sizes)), "level": self.level}

    def _work(self, k: int) -> None:
        client = self._client(k)
        failures = 0
        try:
            while not self._closed:
                setup = self._next_setup()
                if setup is None:
                    return
                trace = [] if self.trace_dir is not None else None
                game = SiteGame(client, self.pe, setup, features=self.features, truth=self.truth, trace=trace)
                t0 = time.perf_counter()
                try:
                    self._play(k, game)
                    failures = 0
                except _Stopped:
                    raise
                except Exception as e:  # noqa: BLE001 - whatever went wrong, this game is dropped and the worker goes on
                    failures += 1
                    with self._lock:
                        self.aborted += 1
                        self.started -= 1  # the game does not count towards max_games
                        self._traj.pop(k, None)
                        self.stats.update(game.stats)
                        if len(self.errors) < 50:
                            self.errors.append(f"{setup}: {type(e).__name__}: {e}")
                    with self._cv:  # a pause between attempts that grows while the site keeps failing
                        self._cv.wait_for(lambda: self._closed, timeout=min(5.0 * 2 ** min(failures, 6), 300.0))
                finally:
                    with self._lock:
                        self.seconds.update(client.seconds)
                        self.seconds["games"] += time.perf_counter() - t0
                    client.seconds.clear()
                    if trace:
                        self._write_trace(k, game, trace)
        except _Stopped:
            pass
        finally:
            client.close()

    def _play(self, k: int, game: SiteGame) -> None:
        with self._talk():
            game.start()
        while not game.over and game.turn < self.turn_cap:
            with self._talk():
                d = game.decision()
            if d is None:
                continue
            with self._decide():
                index, logp, value, _ = self.batcher.submit(d.obs, self.temperature, 300.0).result
                self._store(k, d.obs, index, logp, value)
            with self._talk():
                game.play(d, index)
                if game.over or game.turn >= self.turn_cap:
                    self._finish(k, game)
                    return
        with self._talk():  # over before our first decision
            self._finish(k, game)

    def _store(self, k: int, obs: dict, index: int, logp: float, value: float) -> None:
        with self._lock:
            self.decisions += 1
            if not self.record:
                return
            from .rl import SizeStore

            hw = tuple(obs["tiles"].shape[1:3])
            store = self._stores.setdefault(hw, SizeStore())
            ids = store.add(obs, np.array([index]), np.array([logp], np.float32), np.array([value], np.float32))
            self._traj.setdefault(k, (hw, []))[1].append(int(ids[0]))

    def _flush(self, k: int, outcome: float, terminal: bool) -> None:
        """GAE over the stored decisions of worker `k`'s game, as `rl.Collector._flush`. Needs the lock."""
        from .rl import compute_gae

        tr = self._traj.pop(k, None)
        if tr is None:
            return
        hw, ids = tr
        ids = np.asarray(ids)
        store = self._stores[hw]
        store.set_traj(ids, self._n_traj, outcome, terminal)
        self._n_traj += 1
        turns = store.turn[ids] if self.credit_per_turn else None
        adv, ret = compute_gae(store.value[ids], outcome, terminal, self.gamma, self.lam, turns)
        store.mark(ids[: len(adv)], adv, ret)

    def _finish(self, k: int, game: SiteGame) -> None:
        capped = not game.over
        outcome = 0.5 if capped else (1.0 if game.won else 0.0)
        s = game.setup
        rec = {"seed": game.seed, "n_players": 2, "learner_seat": 0, "opponent": self.entry,
               "opponent_policy_id": self.entry, "outcome": outcome, "capped": capped, "turns": game.turn, "weight": 1.0,
               "decisions": game.decisions, "tribes": s["tribes"], "map": s["map"], "size": s["size"]}
        curve = game.view["play"].get("curve")
        if curve:  # the site's own estimate that its side wins, turn by turn
            rec["site_curve"] = [round(float(c[-1]), 3) for c in curve]
        with self._lock:
            self._flush(k, outcome, terminal=True)
            self._results.append(rec)
            self.finished += 1
            self.stats.update(game.stats)

    def _write_trace(self, k: int, game: SiteGame, trace: list) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.trace_dir / f"w{k:02d}_{game.seed}.json.gz", "wt") as f:
            json.dump(trace, f)

    # --- for the trainer ------------------------------------------------------------------------------------

    def take(self) -> tuple[list[dict], list[dict]]:
        """What gathered since the last call: the stored decisions per board size and the finished games. Call it
        paused."""
        with self._lock:
            for k in list(self._traj):
                self._flush(k, 0.0, terminal=False)
            data = [st.finalize() for st in self._stores.values() if st.n]
            results, self._stores, self._results = self._results, {}, []
        return data, results

    def results(self) -> list[dict]:
        with self._lock:
            return list(self._results)


# ---------------------------------------------------------------------------------------------------------------
# Evaluation from the command line
# ---------------------------------------------------------------------------------------------------------------


def summary(results: list[dict], arm: SiteArm, seconds: float) -> dict:
    n = len(results)
    wins = sum(r["outcome"] == 1.0 for r in results)
    draws = sum(r["outcome"] == 0.5 for r in results)
    score = (wins + 0.5 * draws) / n if n else 0.0
    by_size = {}
    for size in sorted({r["size"] for r in results}):
        rs = [r["outcome"] for r in results if r["size"] == size]
        by_size[str(size)] = {"games": len(rs), "score": round(float(np.mean(rs)), 4)}
    sec = arm.seconds
    return {
        "opponent": arm.entry, "games": n, "wins": wins, "draws": draws, "losses": n - wins - draws,
        "score": round(score, 4), "score_se": round(float(np.sqrt(score * (1 - score) / n)), 4) if n else None,
        "mean_turns": round(float(np.mean([r["turns"] for r in results])), 1) if n else None,
        "by_size": by_size, "aborted": arm.aborted, "decisions": arm.decisions,
        "seconds": round(seconds, 1), "games_per_hour": round(3600 * n / max(seconds, 1e-9), 1),
        "seconds_per_game": round(sec["games"] / max(n + arm.aborted, 1), 1),
        "site_seconds_per_game": {k: round(v / max(n + arm.aborted, 1), 2) for k, v in sorted(sec.items()) if k != "games"},
        "legal_lists": dict(sorted(arm.stats.items())), "errors": arm.errors[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="run directory or checkpoint of the network that plays")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=4, help="games in flight; every one costs the site a search per turn")
    ap.add_argument("--level", type=int, default=STANDARD, help="the site's strength: 1 easier, 3 standard")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--maps", default=",".join(map(str, DEFAULT_MAPS)), help="1 drylands, 2 lakes, 3 continents, 4 archipelago, 5 water world, 6 pangea")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 plays the top move, a positive value samples")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turn-cap", type=int, default=TURN_CAP)
    ap.add_argument("--site", default=DEFAULT_SITE)
    ap.add_argument("--out", default="", help="write one line per game here, and the summary beside it")
    ap.add_argument("--trace", default="", help="keep every call of every game in this folder (large)")
    a = ap.parse_args(argv)

    import polyai_engine as pe
    import torch

    from .policy import ModelPolicy

    torch.set_num_threads(max(1, a.torch_threads))
    policy = ModelPolicy(a.ckpt, a.device, seed=a.seed)
    arm = SiteArm(pe, policy, a.parallel, level=a.level, sizes=[int(x) for x in a.sizes.split(",")],
                  maps=[int(x) for x in a.maps.split(",")], temperature=a.temperature if a.temperature > 0 else None,
                  features=policy.features, record=False, seed=a.seed, max_games=a.games, turn_cap=a.turn_cap,
                  base=a.site, trace_dir=a.trace or None)
    out = open(a.out, "a", buffering=1) if a.out else None
    t0 = time.perf_counter()
    written = 0

    def report() -> None:
        nonlocal written
        for r in arm.results()[written:]:
            if out is not None:
                out.write(json.dumps({"policy": policy.policy_id, **r}) + "\n")
            print(f"game {written + 1}/{a.games}: {'won' if r['outcome'] == 1.0 else 'lost' if r['outcome'] == 0.0 else 'capped'} "
                  f"at turn {r['turns']}, {r['size']}x{r['size']} map {r['map']}, tribes {r['tribes']}", flush=True)
            written += 1

    arm.resume()
    try:
        while arm.alive:
            time.sleep(2.0)
            report()
            if arm.aborted > max(20, 2 * a.games):
                print("too many games failed; stopping", flush=True)
                break
    finally:
        arm.close()
    report()
    s = {"policy": policy.policy_id, **summary(arm.results(), arm, time.perf_counter() - t0)}
    if out is not None:
        out.close()
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
