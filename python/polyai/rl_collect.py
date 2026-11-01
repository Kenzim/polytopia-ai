"""The rollout collector with its data plane in Rust (`--collector rust`, docs/collector.md).

`RustCollector` is a `rl.Collector` whose arms, scheduling, inference batches, learner rows and GAE live in the engine
module's `RolloutCore` (crates/polyai-py/src/collector.rs). Python keeps every random draw (which games to activate,
the roles of fresh games), the pool statistics and the networks: per inference round it makes one `next_round` call,
one `choose` per batch and one `finish_round` call. The recorded rollout has the content of the Python collector's.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .rl import Arm, Collector, Pool, RLConfig
from .rl_pool import NetPolicy


class RustArm(Arm):
    """An `Arm` whose game bookkeeping lives in the core: `active` and `n_active` read from it."""

    core = None
    idx = -1

    @property
    def active(self) -> np.ndarray:
        return self.core.active_mask(self.idx)

    @active.setter
    def active(self, _value) -> None:  # `Arm.__init__` assigns it; the core owns it
        pass

    @property
    def n_active(self) -> int:
        return self.core.n_active(self.idx)

    @n_active.setter
    def n_active(self, _value) -> None:
        pass


class RustStore:
    """The rows of one board size in the layout of `SizeStore.finalize()`, taken from the core."""

    def __init__(self, data: dict):
        self._data = data
        self.n = len(data["chosen"])

    @property
    def value(self) -> np.ndarray:
        return self._data["value"]

    def finalize(self) -> dict:
        return dict(self._data)


class RustCollector(Collector):
    def __init__(self, cfg: RLConfig, pe, learner: NetPolicy, pool: Pool, rng: np.random.Generator,
                 executor: ThreadPoolExecutor, games_log=None):
        super().__init__(cfg, pe, learner, pool, rng, executor, games_log)
        self.core = pe.RolloutCore(executor._max_workers, cfg.max_lead, cfg.min_infer, cfg.gather_ms, cfg.pad_sizes,
                                   cfg.gamma, cfg.lam, cfg.drop_capped)
        self._arm_list: list[RustArm] = []
        self._names = {0: "learner"}  # policy index of the core -> pool entry

    # --- arms ---------------------------------------------------------------------------------------------

    def _make_arm(self, key: tuple) -> Arm:
        base = super()._make_arm(key)
        arm = RustArm(base.key, base.env, base.cap)
        arm.core = self.core
        arm.idx = self.core.add_arm(base.env, base.seat)
        self._arm_list.append(arm)
        return arm

    def _policy_index(self, entry: str) -> int:
        i = self.core.register(entry)
        self._names[i] = entry
        return i

    def _set_roles(self, arm: RustArm, slots: list[int], n_players: list[int]) -> None:
        """Draw the roles of fresh games (the random stream of `Collector._assign_roles`) and hand them to the core."""
        for slot, n in zip(slots, n_players):
            arm.learner[slot] = int(self.rng.integers(n))
            arm.opp[slot] = self.pool.sample_model_entry(self.rng)
        self.core.set_roles([arm.idx] * len(slots), slots, [int(arm.learner[s]) for s in slots],
                            [self._policy_index(str(arm.opp[s])) for s in slots])

    def set_quotas(self, quotas: dict[tuple, int]) -> None:
        for key, q in quotas.items():
            if q > 0 and key not in self.arms:
                self.arms[key] = self._make_arm(key)
            if key in self.arms:
                self._set_quota(self.arms[key], q)
        for key, arm in self.arms.items():
            if key not in quotas:
                self._set_quota(arm, 0)
            if arm.n_active < arm.quota:
                self._activate(arm)

    def _set_quota(self, arm: RustArm, quota: int) -> None:
        arm.quota = quota
        self.core.set_quota(arm.idx, quota)

    def _activate(self, arm: RustArm) -> None:
        need = arm.quota - arm.n_active
        groups = self.core.observe_slots(arm.idx)
        if need <= 0 or not groups:
            return
        active = arm.active
        idle_slots = np.concatenate([s[~active[s]] for s, _ in groups])
        chosen = list(self.rng.permutation(idle_slots)[:need])
        for slots, n_players in groups:
            idle = np.nonzero(~active[slots] & np.isin(slots, chosen))[0]
            if not len(idle):
                continue
            sub = slots[idle].tolist()
            if arm.kind == "model":
                self._set_roles(arm, sub, n_players[idle].tolist())
            self.core.activate(arm.idx, sub)

    # --- one round ----------------------------------------------------------------------------------------

    def _handle(self, events: list) -> None:
        """Finished games and the roles of fresh ones, in the order the Python collector meets them."""
        roles: list[tuple[int, int, int, int]] = []
        for e in events:
            arm = self._arm_list[e[1]]
            if e[0] == "fin":
                self._on_finished(arm, int(e[2]["game"]), e[2])
            else:
                slot = e[2]
                arm.learner[slot] = int(self.rng.integers(e[3]))
                arm.opp[slot] = self.pool.sample_model_entry(self.rng)
                roles.append((e[1], slot, int(arm.learner[slot]), self._policy_index(str(arm.opp[slot]))))
        if roles:
            self.core.set_roles(*map(list, zip(*roles)))

    def _flush(self, arm: Arm, slot: int, outcome: float, terminal: bool) -> None:
        pass  # GAE runs in the core when the game ends

    def tick(self) -> None:
        events, batches = self.core.next_round()
        self._handle(events)
        if not batches:
            return
        t0 = time.perf_counter()
        results = []
        for b in batches:
            if b["key"] == 0:
                ch = self.learner.choose(b["obs"], temperature=self.cfg.temperature)
            else:
                ch = self.pool.nets[self._names[b["key"]]].choose(b["obs"], temperature=self.cfg.opp_temperature)
            results.append((np.ascontiguousarray(ch.index, np.int32), np.ascontiguousarray(ch.logp, np.float32),
                            np.ascontiguousarray(ch.value, np.float32)))
        self.t_infer += time.perf_counter() - t0
        self.n_learner, self.n_opp = self.core.finish_round(results)

    def cut(self) -> None:
        self.core.cut()

    def in_use(self) -> set[str]:
        used = set()
        for arm in self._arm_list:
            if arm.kind == "model":
                used.update(str(e) for e in arm.opp[arm.active])
        return used

    def collect(self, steps: int) -> None:
        """Play until the learner has made `steps` decisions."""
        self.stores = {}
        self.results = []
        self.n_learner = self.n_opp = 0
        self.t_infer = self.t_step = 0.0
        self.core.begin_rollout()
        while self.n_learner < steps:
            self.tick()
        self._handle(self.core.drain())  # games finishing in the steps still in flight end in this rollout
        self.core.cut()
        self.t_step = self.core.counters()[2]
        self.stores = {hw: RustStore(d) for hw, d in self.core.take_stores()}
