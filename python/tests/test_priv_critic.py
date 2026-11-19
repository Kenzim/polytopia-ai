"""The privileged critic (docs/rl.md "Privileged critic"): a value network that also reads the truth under the fog
computes the advantages and is trained next to the policy; the policy itself never sees the truth."""

import importlib.util

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.model import critic_value, make_critic
from polyai.rl import trajectory_gae, train
from polyai.train_bc import load_checkpoint
from test_collector_rust import collector_classes, play
from test_rl import CPU, init_run, read_metrics, small_cfg, tiny_model  # noqa: F401

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")


def batch(n=3, h=11, w=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    tiles = torch.randint(0, 3, (n, h, w, S.F_TILE), generator=g)
    globals_ = torch.rand(n, S.G_DIM, generator=g)
    truth = torch.randint(1, 4, (n, h, w, S.TRUTH_K), generator=g)
    mask = (torch.rand(n, h, w, generator=g) < 0.5).float()
    truth_g = torch.rand(n, S.TRUTH_G, generator=g) * 50
    return tiles, globals_, truth, mask, truth_g


def test_a_new_critic_has_the_policy_value_and_reads_the_truth_only_on_hidden_tiles():
    model = tiny_model().eval()
    critic = make_critic(model).eval()
    assert critic.cfg.truth_input and not model.cfg.truth_input and "truth_input" not in model.cfg.to_dict()
    tiles, globals_, truth, mask, truth_g = batch()
    with torch.no_grad():
        n = tiles.shape[0]
        actions = torch.full((n, 1, 4), -1, dtype=torch.long)
        actions[..., 0] = 0
        pol = model(tiles, globals_, actions, torch.ones(n, 1, dtype=torch.bool)).winner_logits
        np.testing.assert_allclose(critic_value(critic, tiles, globals_, truth, mask, truth_g), pol, atol=1e-5)
        for p in list(critic.truth_embed.parameters()) + list(critic.truth_g_proj.parameters()):
            torch.nn.init.normal_(p, std=0.5)
        base = critic_value(critic, tiles, globals_, truth, mask, truth_g)
        # a different truth on hidden tiles changes the estimate; on visible tiles it is ignored
        other = torch.where(mask[..., None].bool(), truth % 3 + 1, truth)
        assert not torch.allclose(critic_value(critic, tiles, globals_, other, mask, truth_g), base)
        visible = torch.where(mask[..., None].bool(), truth, truth % 3 + 1)
        torch.testing.assert_close(critic_value(critic, tiles, globals_, visible, mask, truth_g), base)


@needs_engine
@pytest.mark.parametrize("per_turn", [False, True])
def test_trajectory_gae_over_the_stored_values_gives_back_the_collector_credit(per_turn):
    for cls in collector_classes():
        out, _ = play(cls, rollout_steps=900, games=6, turn_limit=8, credit_per_turn=per_turn, priv_critic=True)
        for d in out[0][0].values():
            assert {"truth", "truth_mask", "truth_g", "traj", "traj_out", "traj_term"} <= set(d)
            assert (d["traj"][d["train"]] >= 0).all()
            adv, ret = trajectory_gae(d, d["value"].astype(np.float64), small_cfg_like(per_turn))
            np.testing.assert_allclose(adv, d["adv"], atol=1e-5)
            np.testing.assert_allclose(ret, d["ret"], atol=1e-5)
            # other values give other credit on the trained rows
            adv2, _ = trajectory_gae(d, np.full(len(d["value"]), 0.5), small_cfg_like(per_turn))
            assert not np.allclose(adv2[d["train"]], d["adv"][d["train"]])


def small_cfg_like(per_turn):
    from polyai.rl import RLConfig

    return RLConfig(credit_per_turn=per_turn)


@needs_engine
@pytest.mark.parametrize("collector", ["python", "rust"])
def test_a_run_with_the_critic_trains_it_saves_it_and_resumes(tmp_path, init_run, collector):
    cfg = small_cfg(tmp_path, init_run, collector=collector, priv_critic=True, credit_per_turn=True, updates=2, lr=1e-3,
                    turn_limit=10)
    assert train(cfg)["updates"] == 2
    run = tmp_path / "t"
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    for r in recs:
        assert np.isfinite(r["critic_loss"]) and 0 <= r["critic_value_mean"] <= 1
    critic = load_checkpoint(run / "critic_final.pt")
    assert critic.cfg.truth_input
    assert any(p.abs().sum() > 0 for p in critic.truth_embed.parameters())  # the truth inputs were trained
    policy = load_checkpoint(run / "ckpt_final.pt")
    assert not policy.cfg.truth_input and policy.truth_embed is None
    assert train(small_cfg(tmp_path, init_run, collector=collector, priv_critic=True, credit_per_turn=True, updates=3,
                           resume=True, turn_limit=10))["updates"] == 3
    with pytest.raises(SystemExit):  # a run without the critic cannot be resumed with it
        plain = small_cfg(tmp_path, init_run, name="p", collector=collector, updates=1)
        train(plain)
        train(small_cfg(tmp_path, init_run, name="p", collector=collector, priv_critic=True, updates=2, resume=True))
