"""The policy service: one `Policy` on a Unix socket, for the headless client (docs/interfaces.md section 1.2).

    python -m polyai.serve --model /kvm/polytopia-ai-data/runs/NAME [--device cpu|cuda] [--socket PATH]
    python -m polyai.serve --check FILE

`--model` is a checkpoint file or a run directory (then `ckpt_final.pt`). The socket defaults to
`/kvm/tmp/polyai/policy.sock` (mode 0600). The service reads checkpoints with torch's restricted unpickler
(`weights_only=True`). `--check FILE` only loads FILE that way and prints one JSON line (`ok`, `policy_id`, `step`,
`encoding_version`, `features`, `params`, or `ok: false` and `error`); its exit status is 0 when the file loads.

Protocol (version 1): each message is a 4-byte little-endian length and that many bytes of UTF-8 JSON; byte fields are
base64. One request, one response, on a connection that may stay open. A `choose` request carries the server's
`initial` and `current` state blobs and the engine player id `seat` of our account; the service rebuilds the game with
`polyai_engine.Game.from_server` (which checks that the engine agrees with the server's state), encodes it, asks the
policy, and answers with the chosen command as `RawCmd::to_bytes` in `raw`. Requests that arrive together are batched
into one forward pass (per map size and temperature). Anything the service cannot do is answered with `ok: false` and
an `error`; a bad request never takes the service down.

A position can also be given as a web engine line instead of the blobs: `line` = `{"position": <polytopia-position
object>, "steps": [{"text", "ai"}]}` with `seat` the engine player id to move. The game is rebuilt with
`polyai_engine.Game.from_line`, which plays each step by its command text (`format!("{cmd:?}")`, the text the web's
engine service records). A `choose` answer also carries the chosen command's `text`. `turn` (line or blobs, `seat`,
`temperature`, `max_commands`) plays the model for `seat` on its own copy of the game until it ends its turn, the game
is over or another player is to move, and answers with `cmds`: `[{"text", "p", "value"}]` in order. Either way the
network sees only the mover's encoding, which keeps to what that player has explored (docs/encoding.md section 2).

`PolicyClient` is the small blocking client the tests use.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import math
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .policy import ModelPolicy, Policy, observe_games

PROTOCOL = 1
DEFAULT_SOCKET = "/kvm/tmp/polyai/policy.sock"
MAX_MESSAGE = 64 * 1024 * 1024  # bytes of JSON; a game's two blobs are a few hundred kilobytes
MAX_PER_TURN = 400  # the retail AI ends its turn after this many commands (polyai_core::game::MAX_PER_TURN)
K_END_TURN = 26  # encoding.md section 4
DEFAULT_TIMEOUT = 10.0

log = logging.getLogger("polyai.serve")


class RequestError(Exception):
    """A request the service refuses; the message goes to the client."""


def _fatal(e: BaseException) -> bool:
    """KeyboardInterrupt and SystemExit end the service; everything else (a Rust panic included, which pyo3 raises as a
    BaseException) is an error of one request."""
    return isinstance(e, (KeyboardInterrupt, SystemExit))


# ---------------------------------------------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------------------------------------------


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """`n` bytes, or None when the peer closed before the first byte; ConnectionError when it closed half way."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            if not buf:
                return None
            raise ConnectionError("connection closed in the middle of a message")
        buf += chunk
    return bytes(buf)


def send_message(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack("<I", len(data)) + data)


def _decode_b64(req: dict, key: str) -> bytes:
    v = req.get(key)
    if not isinstance(v, str):
        raise RequestError(f"'{key}' must be a base64 string")
    try:
        out = base64.b64decode(v, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"'{key}' is not valid base64") from None
    if not out:
        raise RequestError(f"'{key}' is empty")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Micro-batching
# ---------------------------------------------------------------------------------------------------------------


class _Job:
    def __init__(self, obs: dict, temperature: float | None):
        self.obs, self.temperature = obs, temperature
        self.shape = tuple(obs["tiles"].shape[1:3])
        self.done = threading.Event()
        self.result = None  # (index, logp, value, top)
        self.error: str | None = None
        self.infer_ms = 0.0


class Batcher(threading.Thread):
    """Runs the policy on a worker thread. The first job opens a window of `window_ms`; jobs that arrive within it (up
    to `max_jobs`) share one forward pass per (map size, temperature)."""

    def __init__(self, policy: Policy, window_ms: float = 3.0, max_jobs: int = 32):
        super().__init__(name="polyai-batcher", daemon=True)
        self.policy, self.window, self.max_jobs = policy, window_ms / 1000.0, max_jobs
        self.jobs: queue.Queue[_Job | None] = queue.Queue()
        self.batches = 0
        self.largest_batch = 0

    def submit(self, obs: dict, temperature: float | None, timeout: float) -> _Job:
        job = _Job(obs, temperature)
        self.jobs.put(job)
        if not job.done.wait(timeout):
            raise RequestError("the model did not answer in time")
        if job.error is not None:
            raise RequestError(job.error)
        return job

    def stop(self) -> None:
        self.jobs.put(None)

    def run(self) -> None:
        while True:
            first = self.jobs.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.window
            stop = False
            while len(batch) < self.max_jobs:
                left = deadline - time.perf_counter()
                try:
                    j = self.jobs.get(timeout=max(left, 0.0)) if left > 0 else self.jobs.get_nowait()
                except queue.Empty:
                    break
                if j is None:
                    stop = True
                    break
                batch.append(j)
            self._run_batch(batch)
            if stop:
                return

    def _run_batch(self, batch: list[_Job]) -> None:
        self.batches += 1
        self.largest_batch = max(self.largest_batch, len(batch))
        groups: dict[tuple, list[_Job]] = {}
        for j in batch:
            groups.setdefault((j.shape, j.temperature), []).append(j)
        for (_, temperature), jobs in groups.items():
            t0 = time.perf_counter()
            try:
                obs = _stack([j.obs for j in jobs])
                c = self.policy.choose(obs, temperature=temperature)
                ms = (time.perf_counter() - t0) * 1000.0
                for i, j in enumerate(jobs):
                    j.result = (int(c.index[i]), float(c.logp[i]), float(c.value[i]), c.top[i])
                    j.infer_ms = ms
            except BaseException as e:  # noqa: BLE001  (one bad group must not stop the worker)
                if _fatal(e):
                    raise
                log.exception("the policy failed on a batch of %d", len(jobs))
                for j in jobs:
                    j.error = f"the model failed: {type(e).__name__}: {e}"
            for j in jobs:
                j.done.set()


def _stack(obs_list: list[dict]) -> dict:
    """Single-position observation groups of one map size -> one group."""
    if len(obs_list) == 1:
        return obs_list[0]
    lens = np.concatenate([np.diff(o["act_off"]) for o in obs_list])
    out = {
        "tiles": np.concatenate([o["tiles"] for o in obs_list]),
        "globals": np.concatenate([o["globals"] for o in obs_list]),
        "act_off": np.concatenate([[0], np.cumsum(lens)]).astype(np.int32),
        "actions": np.concatenate([o["actions"] for o in obs_list]),
    }
    for k in ("f_tiles", "f_globals", "f_actions"):  # the engine features, when the observations carry them
        if k in obs_list[0]:
            out[k] = np.concatenate([o[k] for o in obs_list])
    return out


# ---------------------------------------------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------------------------------------------


class PolicyServer:
    """`policy` behind a Unix socket. `start()` serves on a background thread, `serve_forever()` on this one."""

    def __init__(self, policy: Policy, socket_path: str | Path = DEFAULT_SOCKET, *, window_ms: float = 3.0,
                 max_jobs: int = 32, rebuild_workers: int = 4, request_timeout: float = 30.0, read_timeout: float = 30.0):
        import polyai_engine as pe

        self.pe = pe
        self.policy = policy
        self.path = Path(socket_path)
        self.request_timeout, self.read_timeout = request_timeout, read_timeout
        self.batcher = Batcher(policy, window_ms, max_jobs)
        self.rebuild_slots = threading.BoundedSemaphore(max(1, rebuild_workers))
        self.sock: socket.socket | None = None
        self.closing = threading.Event()
        self.requests = 0
        self.thread: threading.Thread | None = None

    # -- life cycle -----------------------------------------------------------------------------------------------

    def bind(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            probe = socket.socket(socket.AF_UNIX)
            try:
                probe.settimeout(1.0)
                probe.connect(str(self.path))
            except OSError:
                self.path.unlink()  # nobody listens: a stale socket file
            else:
                raise RuntimeError(f"another service is already listening on {self.path}")
            finally:
                probe.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            s.bind(str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        s.listen(64)
        s.settimeout(0.5)
        self.sock = s

    def serve_forever(self) -> None:
        if self.sock is None:
            self.bind()
        if not self.batcher.is_alive():
            self.batcher.start()
        log.info("serving %s on %s", self.policy.policy_id, self.path)
        while not self.closing.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._connection, args=(conn,), daemon=True).start()

    def start(self) -> PolicyServer:
        self.bind()
        self.batcher.start()
        self.thread = threading.Thread(target=self.serve_forever, name="polyai-serve", daemon=True)
        self.thread.start()
        return self

    def close(self) -> None:
        self.closing.set()
        self.batcher.stop()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(2.0)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> PolicyServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- one connection -------------------------------------------------------------------------------------------

    def _connection(self, conn: socket.socket) -> None:
        try:
            while not self.closing.is_set():
                conn.settimeout(None)
                try:
                    head = _recv_exact(conn, 4)
                except OSError:
                    return
                if head is None:
                    return
                n = struct.unpack("<I", head)[0]
                if n > MAX_MESSAGE:
                    send_message(conn, self._fail(f"the message is {n} bytes; the limit is {MAX_MESSAGE}"))
                    return  # the stream cannot be re-synchronised
                conn.settimeout(self.read_timeout)
                body = _recv_exact(conn, n) if n else b""
                send_message(conn, self.handle_bytes(body or b""))
        except (OSError, ConnectionError):
            pass
        except BaseException as e:  # noqa: BLE001
            if _fatal(e):
                raise
            log.exception("connection failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # -- one request ----------------------------------------------------------------------------------------------

    def _fail(self, error: str) -> dict:
        return {"v": PROTOCOL, "ok": False, "error": error}

    def handle_bytes(self, body: bytes) -> dict:
        """The response to one message body. Never raises."""
        t0 = time.perf_counter()
        self.requests += 1
        try:
            try:
                req = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise RequestError("the request is not valid UTF-8 JSON") from None
            if not isinstance(req, dict):
                raise RequestError("the request must be a JSON object")
            if req.get("v") != PROTOCOL or isinstance(req.get("v"), bool):
                raise RequestError(f"unsupported protocol version {req.get('v')!r}; this service speaks {PROTOCOL}")
            op = req.get("op")
            if op == "info":
                return self._info()
            if op == "choose":
                return self._choose(req, t0)
            if op == "turn":
                return self._turn(req, t0)
            raise RequestError(f"unknown op {op!r}; use 'choose', 'turn' or 'info'")
        except RequestError as e:
            return self._fail(str(e))
        except BaseException as e:  # noqa: BLE001  (a Rust panic arrives here too)
            if _fatal(e):
                raise
            log.exception("request failed")
            return self._fail(f"internal error: {type(e).__name__}: {e}")

    def _info(self) -> dict:
        return {"v": PROTOCOL, "ok": True, "policy_id": self.policy.policy_id, "encoding": self.pe.ENCODING_VERSION,
                "engine": self.pe.ENGINE_COMMIT, "device": str(getattr(self.policy, "device", "?")),
                "requests": self.requests}

    @staticmethod
    def _seat(req: dict) -> int:
        seat = req.get("seat")
        if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat <= 255:
            raise RequestError("'seat' must be an engine player id, an integer from 0 to 255")
        return seat

    @staticmethod
    def _temperature(req: dict) -> float | None:
        temperature = req.get("temperature")
        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                    or not math.isfinite(temperature) or temperature <= 0:
                raise RequestError("'temperature' must be null (greedy) or a positive number")
            temperature = float(temperature)
        return temperature

    def _game(self, req: dict, seat: int):
        """(game, steps already played this turn) from the request's `line`, else its server blobs. The game is the
        whole position; `seat` must be the player to move."""
        line = req.get("line")
        if line is None:
            initial, current = _decode_b64(req, "initial"), _decode_b64(req, "current")
            with self.rebuild_slots:
                try:
                    return self.pe.Game.from_server(initial, current, seat), None
                except ValueError as e:
                    raise RequestError(str(e)) from None
        if not isinstance(line, dict) or not isinstance(line.get("position"), dict):
            raise RequestError("'line' must be an object with a 'position' object and a 'steps' list")
        steps = line.get("steps", [])
        if not isinstance(steps, list) or not all(isinstance(s, dict) and isinstance(s.get("text"), str) for s in steps):
            raise RequestError("'line.steps' must be a list of objects with a 'text' string")
        texts = [s["text"] for s in steps]
        with self.rebuild_slots:
            try:
                game = self.pe.Game.from_line(json.dumps(line["position"], separators=(",", ":")), texts)
            except ValueError as e:
                raise RequestError(str(e)) from None
        if game.is_over():
            raise RequestError("the game is over")
        if game.mover_id != seat:
            raise RequestError(f"it is player {game.mover_id}'s turn, not seat {seat}'s")
        return game, len(texts)

    def _choose(self, req: dict, t0: float) -> dict:
        from .model import eval_pawns

        game_id = req.get("game_id", "?")
        if not isinstance(game_id, str):
            raise RequestError("'game_id' must be a string")
        seat = self._seat(req)
        per_turn = req.get("per_turn")
        if per_turn is not None and (isinstance(per_turn, bool) or not isinstance(per_turn, int) or per_turn < 0):
            raise RequestError("'per_turn' must be a non-negative integer")
        temperature = self._temperature(req)

        t1 = time.perf_counter()
        game, line_steps = self._game(req, seat)
        if per_turn is None:
            per_turn = line_steps or 0
        obs = observe_games([game], features=bool(getattr(self.policy, "features", False)))
        n_legal = len(obs["actions"])
        t2 = time.perf_counter()
        job = self.batcher.submit(obs, temperature, self.request_timeout)
        index, logp, value, top = job.result

        forced = None
        if per_turn + 1 > MAX_PER_TURN:
            ends = np.nonzero(obs["actions"][:, 0] == K_END_TURN)[0]
            if len(ends):
                if int(ends[-1]) != index:
                    forced = f"per_turn {per_turn}: the turn is ended after {MAX_PER_TURN} commands"
                    index = int(ends[-1])
                    logp = 0.0  # a forced move is certain
        desc = game.describe(index)
        top_out = [{"index": i, "desc": game.describe(i), "p": p} for i, p in top]
        try:
            raw = base64.b64encode(game.raw_command(index)).decode()
        except ValueError as e:
            if line_steps is None:  # a server game needs the retail bytes; a web line plays the text
                raise RequestError(f"the chosen move has no retail encoding: {e}") from None
            raw = None
        resp = {
            "v": PROTOCOL, "ok": True, "raw": raw, "desc": desc, "text": desc, "index": index,
            "n_legal": n_legal, "logp": logp, "value": value, "eval": float(eval_pawns(value)), "top": top_out,
            "policy_id": self.policy.policy_id, "encoding": self.pe.ENCODING_VERSION, "engine": self.pe.ENGINE_COMMIT,
            "rebuild_ms": (t2 - t1) * 1000.0, "infer_ms": job.infer_ms,
        }
        if forced:
            resp["forced"] = forced
        resp["think_ms"] = (time.perf_counter() - t0) * 1000.0
        log.info("%s seat %d turn %d: %s (%d legal, p=%.2f, value %.2f) %.0f ms", game_id, seat, game.turn, desc,
                 n_legal, math.exp(logp), value, resp["think_ms"])
        return resp

    def _turn(self, req: dict, t0: float) -> dict:
        """The model's commands for `seat` until it ends its turn, the game is over or another player is to move."""
        game_id = req.get("game_id", "?")
        if not isinstance(game_id, str):
            raise RequestError("'game_id' must be a string")
        seat = self._seat(req)
        temperature = self._temperature(req)
        max_commands = req.get("max_commands", MAX_PER_TURN)
        if isinstance(max_commands, bool) or not isinstance(max_commands, int) or not 1 <= max_commands <= MAX_PER_TURN:
            raise RequestError(f"'max_commands' must be an integer from 1 to {MAX_PER_TURN}")
        deadline = t0 + self.request_timeout
        game, line_steps = self._game(req, seat)
        per_turn = line_steps or 0
        features = bool(getattr(self.policy, "features", False))
        cmds: list[dict] = []
        infer_ms = 0.0
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                raise RequestError(f"the turn took longer than {self.request_timeout:g} s")
            obs = observe_games([game], features=features)
            job = self.batcher.submit(obs, temperature, left)
            index, logp, value, _ = job.result
            infer_ms += job.infer_ms
            if per_turn + 1 >= MAX_PER_TURN or len(cmds) + 1 >= max_commands:
                ends = np.nonzero(obs["actions"][:, 0] == K_END_TURN)[0]
                if len(ends) and int(ends[-1]) != index:
                    index, logp = int(ends[-1]), 0.0
            text = game.describe(index)
            cmds.append({"text": text, "p": math.exp(logp), "value": value})
            game.step(index)
            per_turn += 1
            if text == "EndTurn" or game.is_over() or game.mover_id != seat or len(cmds) >= max_commands:
                break
        think_ms = (time.perf_counter() - t0) * 1000.0
        log.info("%s seat %d turn %d: %d commands, %.0f ms (%.0f ms in the model)", game_id, seat, game.turn,
                 len(cmds), think_ms, infer_ms)
        return {"v": PROTOCOL, "ok": True, "cmds": cmds, "policy_id": self.policy.policy_id,
                "encoding": self.pe.ENCODING_VERSION, "engine": self.pe.ENGINE_COMMIT, "think_ms": think_ms,
                "infer_ms": infer_ms}


# ---------------------------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------------------------


class PolicyClient:
    """A blocking client with one connection kept open (and re-opened after a failure)."""

    def __init__(self, path: str | Path = DEFAULT_SOCKET, timeout: float = DEFAULT_TIMEOUT):
        self.path, self.timeout = str(path), timeout
        self.sock: socket.socket | None = None

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def request_raw(self, payload: bytes) -> dict:
        """Send one message body and read the response (for tests that send malformed JSON)."""
        if self.sock is None:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect(self.path)
            self.sock = s
        try:
            self.sock.sendall(struct.pack("<I", len(payload)) + payload)
            head = _recv_exact(self.sock, 4)
            if head is None:
                raise ConnectionError("the service closed the connection")
            data = _recv_exact(self.sock, struct.unpack("<I", head)[0])
            return json.loads(data.decode("utf-8"))
        except (OSError, ConnectionError):
            self.close()
            raise

    def request(self, obj: dict) -> dict:
        return self.request_raw(json.dumps(obj).encode())

    def info(self) -> dict:
        return self.request({"v": PROTOCOL, "op": "info"})

    def choose(self, game_id: str, seat: int, initial: bytes, current: bytes, per_turn: int = 0,
               temperature: float | None = None) -> dict:
        return self.request({
            "v": PROTOCOL, "op": "choose", "game_id": game_id, "seat": seat,
            "initial": base64.b64encode(initial).decode(), "current": base64.b64encode(current).decode(),
            "per_turn": per_turn, "temperature": temperature,
        })

    def choose_line(self, game_id: str, seat: int, line: dict, temperature: float | None = None) -> dict:
        return self.request({"v": PROTOCOL, "op": "choose", "game_id": game_id, "seat": seat, "line": line,
                             "temperature": temperature})

    def turn(self, game_id: str, seat: int, line: dict, temperature: float | None = None,
             max_commands: int = MAX_PER_TURN) -> dict:
        return self.request({"v": PROTOCOL, "op": "turn", "game_id": game_id, "seat": seat, "line": line,
                             "temperature": temperature, "max_commands": max_commands})


def check(path: str | Path) -> dict:
    """Load a checkpoint the way the service does (restricted unpickler, CPU) and describe it, or say why it fails."""
    try:
        policy = ModelPolicy(path, "cpu", weights_only=True)
    except (Exception, SystemExit) as e:  # noqa: BLE001  (load_checkpoint reports old files with SystemExit)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "policy_id": policy.policy_id, "step": policy.step, "encoding_version": policy.encoding_version,
            "features": policy.features, "params": sum(p.numel() for p in policy.model.parameters())}


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--model", help="checkpoint file or run directory (ckpt_final.pt)")
    what.add_argument("--check", metavar="FILE", help="only load FILE as the service would and print what it is")
    ap.add_argument("--name", help="name of the model in policy_id (default: the run directory's name)")
    ap.add_argument("--request-seconds", type=float, default=30.0, help="time limit of one request (a whole turn)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    ap.add_argument("--socket", default=DEFAULT_SOCKET, help=f"Unix socket path (default {DEFAULT_SOCKET})")
    ap.add_argument("--threads", type=int, default=4, help="torch threads (at most 8)")
    ap.add_argument("--gpu-gb", type=float, default=1.0, help="cap on this process's GPU memory when --device cuda")
    ap.add_argument("--batch-ms", type=float, default=3.0, help="how long the first request waits for others to batch with")
    ap.add_argument("--max-batch", type=int, default=32, help="most requests in one batch")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sampling generator")
    ap.add_argument("--tta", action="store_true", help="average the policy over the board symmetries (8x the inference)")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request")
    a = ap.parse_args(argv)

    if a.check:
        torch.set_num_threads(1)
        out = check(a.check)
        print(json.dumps(out), flush=True)
        return 0 if out["ok"] else 1
    logging.basicConfig(stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO if a.verbose else logging.WARNING)
    torch.set_num_threads(max(1, min(a.threads, 8)))
    device = torch.device(a.device)
    if device.type == "cuda":
        from .train_bc import check_gpu_memory

        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        check_gpu_memory(a.gpu_gb, device)
    policy = ModelPolicy(a.model, device, seed=a.seed, tta=a.tta, weights_only=True, name=a.name)
    server = PolicyServer(policy, a.socket, window_ms=a.batch_ms, max_jobs=a.max_batch,
                          request_timeout=a.request_seconds)
    server.bind()
    signal.signal(signal.SIGTERM, lambda *_: server.closing.set())
    print(f"{policy.policy_id} ({policy.path}) on {a.device}, listening on {server.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
