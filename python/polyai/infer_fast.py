"""Faster rollout inference: the network on a few fixed padded shapes, optionally `torch.compile`d and/or captured in CUDA
graphs (docs/rl.md "Fast inference"). Opt-in (`--infer-graphs`, `--infer-compile`); with both off nothing here runs.

Why: late in a run most inference calls are small (a few positions per network and map size) and the cost is the ~200
kernel launches of the forward pass, not the arithmetic. A CUDA graph replays all of them with one launch, and
`torch.compile` fuses the elementwise kernels between the convolutions. Both need fixed shapes, so a call is padded to a
bucket: rows to the next power of two and the flat list of legal actions of all rows to the next power of two, while the
board keeps its own size (the collector already pads mixed boards to a few sizes). Padded rows and actions are never
read back, and every row is independent in eval mode (BatchNorm on running statistics), so the real rows get the same
logits up to rounding. The forward is `PolyNet.forward_flat`, which takes the legal actions as a flat list instead of
finding them with `nonzero` (a data-dependent shape cannot be captured); the legal list is not padded per row, so the
policy head costs what it does in the eager path.

Memory: all graphs of a device share one memory pool (they replay one at a time on one stream) and one set of input
buffers per shape, so the cost is the largest graph's activations, not the sum. Shapes above `max_actions` and
anything that fails to capture fall back to the eager path, counted in `stats`.
"""

from __future__ import annotations

import torch

from . import schema as S
from .model import MASK_FILL

MIN_ROWS = 4
MIN_ACTIONS = 128
MAX_ACTIONS = 1 << 19  # flat legal actions of one padded call; larger ones run eagerly

_POOLS: dict[torch.device, tuple] = {}
_BUFFERS: dict[tuple, "_Buffers"] = {}


def _pow2(n: int, floor: int) -> int:
    return max(floor, 1 << (max(n, 1) - 1).bit_length())


def bucket_rows(n: int) -> int:
    """The padded batch size for `n` rows: the next power of two, at least `MIN_ROWS`."""
    return _pow2(n, MIN_ROWS)


def bucket_actions(n: int) -> int:
    """The padded length of the flat legal-action list for `n` actions: the next power of two, at least `MIN_ACTIONS`."""
    return _pow2(n, MIN_ACTIONS)


def _graph_pool(device: torch.device):
    """The memory pool all graphs of a device share. A trivial graph is kept alive with it: the allocator forgets a pool
    whose last graph is gone (a dropped snapshot's graphs), and capturing into the forgotten handle then fails."""
    if device not in _POOLS:
        pool = torch.cuda.graph_pool_handle()
        keep = torch.cuda.CUDAGraph()
        with torch.cuda.graph(keep, pool=pool, capture_error_mode="thread_local"):
            x = torch.zeros(1, device=device) + 1
        _POOLS[device] = (pool, keep, x)
    return _POOLS[device][0]


class _Buffers:
    """Static input tensors of one padded shape (the graphs read them in place). Padding entries keep whatever an earlier
    call left there (always valid indices and finite values); nothing reads their outputs."""

    def __init__(self, key: tuple, tiles, globals_, feats: dict[str, torch.Tensor], device: torch.device):
        bq, tq, h, w = key[:4]
        self.tiles = torch.zeros((bq, h, w, tiles.shape[3]), dtype=tiles.dtype, device=device)
        self.globals = torch.zeros((bq, globals_.shape[1]), dtype=globals_.dtype, device=device)
        self.bmask = torch.ones((bq, h, w), dtype=torch.bool, device=device)
        self.act_b = torch.zeros(tq, dtype=torch.long, device=device)
        self.actions = torch.zeros((tq, 4), dtype=torch.int16, device=device)
        self.feats = {k: torch.zeros((bq if k != "f_actions" else tq, *v.shape[(2 if k == "f_actions" else 1):]),
                                     dtype=v.dtype, device=device) for k, v in feats.items()}

    def fill(self, tiles, globals_, bmask, bi, actions, feats) -> None:
        b, t = tiles.shape[0], bi.numel()
        self.tiles[:b].copy_(tiles)
        self.globals[:b].copy_(globals_)
        if bmask is None:
            self.bmask.fill_(True)
        else:
            self.bmask[:b].copy_(bmask)
        self.act_b[:t].copy_(bi)
        self.actions[:t].copy_(actions)
        for k, v in feats.items():
            self.feats[k][: b if k != "f_actions" else t].copy_(v)

    def args(self) -> tuple[tuple, dict]:
        return (self.tiles, self.globals, self.bmask, self.act_b, self.actions), self.feats


class FastForward:
    """`PolyNet.forward_flat` of one network on padded buckets, compiled and/or graph-captured per bucket.

    `__call__` takes what `ModelPolicy.choose` has collated for a chunk and returns (policy_logits [B, A],
    winner_logits [B, 9]) as `PolyNet.forward` would, or None when the chunk is not handled (the caller then runs the
    eager path). Outputs of a graph replay live in the shared pool and are copied out before the next call."""

    def __init__(self, model, device: torch.device, graphs: bool = False, compile: bool = False,
                 max_actions: int = MAX_ACTIONS):
        self.model = model
        self.device = torch.device(device)
        self.graphs = graphs and self.device.type == "cuda"
        self.compile = compile
        self.max_actions = max_actions
        self.fn = model.forward_flat
        if compile:
            import torch._dynamo.config as dyn

            dyn.cache_size_limit = max(dyn.cache_size_limit, 512)  # one entry per padded shape
            dyn.accumulated_cache_size_limit = max(dyn.accumulated_cache_size_limit, 4096)
            self.fn = torch.compile(model.forward_flat, dynamic=False)
        self.entries: dict[tuple, tuple | None] = {}
        self.last_error = ""
        self.stats = {"graph_replays": 0, "graph_captures": 0, "padded_calls": 0, "eager_calls": 0, "capture_failures": 0}

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def __call__(self, tiles, globals_, actions, mask, bmask, feats):
        b, h, w = tiles.shape[:3]
        bi, ai = mask.nonzero(as_tuple=True)
        t = bi.numel()
        if t > self.max_actions:
            self.stats["eager_calls"] += 1
            return None
        bq, tq = bucket_rows(b), bucket_actions(t)
        key = (bq, tq, h, w, tiles.dtype, tuple(sorted(feats)), self.device)
        bufs = _BUFFERS.get(key)
        if bufs is None:
            bufs = _BUFFERS[key] = _Buffers(key, tiles, globals_, feats, self.device)
        flat_feats = {k: (v[bi, ai] if k == "f_actions" else v) for k, v in feats.items()}
        bufs.fill(tiles, globals_, bmask, bi, actions[bi, ai], flat_feats)
        pos, kw = bufs.args()
        out = None
        if self.graphs:
            entry = self.entries.get(key, False)
            if entry is False:
                entry = self.entries[key] = self._capture(pos, kw)
            if entry is not None:
                entry[0].replay()
                self.stats["graph_replays"] += 1
                out = entry[1]
        if out is None:
            with self._autocast():
                out = self.fn(*pos, **kw)
            self.stats["padded_calls"] += 1
        flat, winner = out
        logits = torch.full(mask.shape, MASK_FILL, dtype=torch.float32, device=mask.device)
        return logits.index_put((bi, ai), flat[:t]), winner[:b].clone()

    def _capture(self, pos, kw):
        try:
            pool = _graph_pool(self.device)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side), self._autocast():
                for _ in range(2):  # warm-up: kernels, cuDNN algorithms and, with compile, the compilation
                    self.fn(*pos, **kw)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"), self._autocast():
                out = self.fn(*pos, **kw)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.stats["capture_failures"] += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        self.stats["graph_captures"] += 1
        return graph, out
