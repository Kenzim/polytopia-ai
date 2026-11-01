"""Synthetic PPO inputs for the tests of `polyai.rl`: a rollout store filled with random positions (no engine needed)."""

import numpy as np
import torch

from polyai.policy import collate
from polyai.rl import SizeStore
from test_arena import fake_obs

CPU = torch.device("cpu")


def synthetic_stores(model, groups: int = 4, seed: int = 0) -> dict:
    """`groups` batches of 4 positions on 11x11 boards in one SizeStore. The behaviour log-probabilities come from
    `model` (ratio 1 before any step); advantages and returns are random, all positions are trainable."""
    rng = np.random.default_rng(seed)
    store = SizeStore()
    model.eval()
    for g in range(groups):
        obs = fake_obs(lens=(4, 6, 3, 5), size=11, seed=seed * 100 + g)
        tiles, glob, actions, mask = collate(obs, np.arange(4), CPU)
        with torch.no_grad():
            out = model(tiles, glob, actions, mask)
        lp = torch.log_softmax(out.policy_logits.float(), 1)
        chosen = np.array([0, 2, 1, 3])
        value = torch.softmax(out.winner_logits.float(), 1)[:, 0].numpy()
        ids = store.add(obs, chosen, lp.gather(1, torch.from_numpy(chosen)[:, None])[:, 0].numpy(), value)
        adv = rng.normal(size=4).astype(np.float32)
        ret = rng.uniform(size=4).astype(np.float32)
        store.mark(ids, adv, ret)
    return {(11, 11): store}
