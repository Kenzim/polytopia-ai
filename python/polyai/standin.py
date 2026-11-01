"""A deterministic stand-in for a network policy: the same position always gets the same move, whatever it is batched with.

Used to test that the two rollout collectors (python and rust, docs/collector.md) record the same rollout, and to time
a collector without a network. `choose` has the signature of `ModelPolicy.choose` and returns `Choices`.
"""

from __future__ import annotations

import numpy as np

from .policy import Choices


class StandInPolicy:
    """Picks index `h % n` of a position's `n` legal actions, `h` a hash of the position (seed, turn, seat, legal list).
    `logp` is the uniform log-probability, `value` another hash in (0, 1). `salt` makes different instances differ."""

    def __init__(self, policy_id: str = "standin", salt: int = 0):
        self.policy_id = policy_id
        self.salt = salt
        self.calls = 0
        self.rows = 0

    def choose(self, obs: dict, *, temperature: float | None = None) -> Choices:
        act_off = np.asarray(obs["act_off"], dtype=np.int64)
        n = np.diff(act_off)
        acts = np.asarray(obs["actions"]).astype(np.int64)
        per_action = (acts * np.array([3, 5, 7, 11], dtype=np.int64)).sum(1) if len(acts) else np.zeros(0, np.int64)
        sums = np.add.reduceat(per_action, act_off[:-1]) if len(acts) else np.zeros(len(n), np.int64)
        h = (np.asarray(obs["seed"], dtype=np.int64) * 1000003 + np.asarray(obs["turn"], dtype=np.int64) * 10007
             + np.asarray(obs["seat"], dtype=np.int64) * 131 + sums * 31 + n * 7 + self.salt * 977) % (1 << 31)
        h = (h * 2654435761) % (1 << 31)
        self.calls += 1
        self.rows += len(n)
        return Choices(index=(h % n).astype(np.int32), logp=(-np.log(n)).astype(np.float32),
                       value=(((h >> 7) % 1000 + 0.5) / 1000).astype(np.float32), top=[[] for _ in n])
