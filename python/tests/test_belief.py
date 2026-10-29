"""Auxiliary belief heads: truth keys in shards, the masked loss, training, and that checkpoints load everywhere."""

import json

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.belief import BeliefMeter, belief_loss, evaluate_belief, format_table
from polyai.data import BatchLoader, Shard, make_synthetic_shard, save_shard, to_torch
from polyai.model import ModelConfig, build_model
from polyai.policy import ModelPolicy
from polyai.train_bc import TrainConfig, compute_losses, load_checkpoint, save_checkpoint, train
from test_arena import fake_obs


def truth_shards(tmp_path, name, n_shards=1, n=64, size=11, seed=1):
    paths = []
    for k in range(n_shards):
        p = tmp_path / name / f"s{k}.npz"
        save_shard(p, make_synthetic_shard(n, size, seed * 100 + k, max_actions=20, with_truth=True))
        paths.append(p)
    return paths


def tiny_belief(hidden=8):
    return build_model(ModelConfig(blocks=3, channels=24, policy_hidden=32, value_hidden=32, name="tiny", belief_hidden=hidden))


# ---- data -----------------------------------------------------------------------------------------------------


def test_truth_keys_roundtrip_and_follow_the_symmetry(tmp_path):
    [path] = truth_shards(tmp_path, "t", n=40, size=11)
    shard = Shard.load(path)
    assert shard.has_truth
    plain = shard.batch(np.arange(40))
    assert plain["truth"].shape == (40, 11, 11, S.TRUTH_K) and plain["truth_mask"].shape == (40, 11, 11)
    # synthetic labels are climate % vocab on hidden tiles: that identity must survive every symmetry
    for seed in range(4):
        b = shard.batch(np.arange(40), np.random.default_rng(seed), augment=True)
        climate = b["tiles"][..., S.COLUMN["climate"]].astype(np.int64)
        hidden = b["tiles"][..., S.COLUMN["unexplored"]] == 1
        np.testing.assert_array_equal(b["truth_mask"].astype(bool), hidden)
        for k, v in enumerate(S.TRUTH_VOCABS):
            np.testing.assert_array_equal(b["truth"][..., k], (climate % v) * hidden)


def test_shards_without_truth_load_and_batch_as_before(shard16):
    assert not shard16.has_truth
    assert "truth" not in shard16.batch(np.arange(4))


def test_a_shard_with_only_one_truth_key_is_refused():
    a = make_synthetic_shard(8, 11, with_truth=True)
    del a["truth_mask"]
    with pytest.raises(ValueError, match="both truth and truth_mask"):
        Shard(a)


def test_loader_yields_truth_on_the_device(tmp_path):
    paths = truth_shards(tmp_path, "t", n=32)
    b = next(BatchLoader(paths, 16, prefetch=0).iterate(epochs=1))
    assert b["truth"].shape[0] == 16 and b["truth"].dtype == torch.int16 and b["truth_mask"].dtype == torch.uint8


# ---- loss and metrics -----------------------------------------------------------------------------------------


def perfect_logits(truth, wrong_on_visible=True):
    out = []
    for k, v in enumerate(S.TRUTH_VOCABS):
        lg = torch.nn.functional.one_hot(truth[..., k].long(), v).permute(0, 3, 1, 2).float() * 30.0
        out.append(lg)
    return tuple(out)


def test_loss_counts_only_hidden_tiles():
    truth = torch.randint(0, 2, (4, 5, 5, S.TRUTH_K)).short()
    mask = torch.zeros(4, 5, 5, dtype=torch.uint8)
    mask[:, :2] = 1
    logits = list(perfect_logits(truth))
    loss, per = belief_loss(tuple(logits), truth, mask)
    assert loss.item() < 1e-4 and set(per) == set(S.TRUTH_NAMES)
    # destroying the predictions on explored tiles changes nothing
    for lg in logits:
        lg[:, :, 2:] = torch.randn_like(lg[:, :, 2:]) * 10
    assert belief_loss(tuple(logits), truth, mask)[0].item() < 1e-4
    # destroying one hidden tile does
    logits[0][0, :, 0, 0] = torch.randn(S.TRUTH_VOCABS[0]) * 10
    assert belief_loss(tuple(logits), truth, mask)[0].item() > 1e-2
    # no hidden tile: no loss and no NaN
    zero = belief_loss(tuple(logits), truth, torch.zeros_like(mask))[0]
    assert zero.item() == 0.0


def test_uniform_prediction_costs_log_of_vocabulary():
    truth = torch.zeros(2, 4, 4, S.TRUTH_K, dtype=torch.int16)
    mask = torch.ones(2, 4, 4, dtype=torch.uint8)
    logits = tuple(torch.zeros(2, v, 4, 4) for v in S.TRUTH_VOCABS)
    _, per = belief_loss(logits, truth, mask)
    for k, v in enumerate(S.TRUTH_VOCABS):
        assert per[S.TRUTH_NAMES[k]].item() == pytest.approx(np.log(v), rel=1e-5)


def test_meter_scores_a_perfect_and_a_majority_predictor():
    g = torch.Generator().manual_seed(0)
    truth = torch.zeros(6, 8, 8, S.TRUTH_K, dtype=torch.int16)
    truth[..., 0] = torch.randint(1, 6, (6, 8, 8), generator=g)  # terrain: land and water
    truth[..., 4] = (torch.rand(6, 8, 8, generator=g) < 0.03).short() * 2
    truth[:, 0, :, 0] = 3
    mask = torch.ones(6, 8, 8, dtype=torch.uint8)
    perfect = BeliefMeter()
    perfect.add(perfect_logits(truth), truth, mask)
    r = perfect.result()
    for name, grp in r["groups"].items():
        assert grp["accuracy"] == pytest.approx(1.0), name
        assert grp["accuracy"] >= grp["baseline_accuracy"]
    cap = r["capital_location"]
    assert cap["top1"] == pytest.approx(1.0) and cap["top1_uniform"] < 0.3
    assert "capital location" in format_table(r)
    # a head that always says "most common value" matches the baseline exactly
    maj = BeliefMeter()
    logits = tuple(torch.zeros(6, v, 8, 8) for v in S.TRUTH_VOCABS)
    for k in range(S.TRUTH_K):
        maj_class = int(torch.bincount(truth[..., k].flatten().long(), minlength=S.TRUTH_VOCABS[k]).argmax())
        logits[k][:, maj_class] = 5.0
    maj.add(logits, truth, mask)
    for name, grp in maj.result()["groups"].items():
        assert grp["accuracy"] == pytest.approx(grp["baseline_accuracy"]), name


def test_capital_baseline_is_the_uniform_prior():
    from polyai.belief import _top_hit_baseline

    n, k = np.array([10.0, 4.0, 2.0]), np.array([1.0, 2.0, 1.0])
    np.testing.assert_allclose(_top_hit_baseline(n, k, 1), [0.1, 0.5, 0.5])
    np.testing.assert_allclose(_top_hit_baseline(n, k, 3), [0.3, 1.0, 1.0])


# ---- model ----------------------------------------------------------------------------------------------------


def test_a_model_without_heads_is_unchanged():
    m = build_model("tiny")
    assert m.belief_heads is None and "belief_hidden" not in m.cfg.to_dict()
    assert not any(k.startswith("belief") for k in m.state_dict())
    b = to_torch(Shard(make_synthetic_shard(4, 11, max_actions=10)).batch(np.arange(4)))
    out = m.eval()(b["tiles"], b["globals"], b["actions"], b["action_mask"], belief=True)
    assert len(out) == 2 and not hasattr(out, "belief")


def test_heads_add_outputs_and_leave_the_policy_alone():
    torch.manual_seed(0)
    plain = build_model("tiny").eval()
    for p in plain.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    withheads = tiny_belief().eval()
    withheads.load_state_dict(plain.state_dict(), strict=False)
    b = to_torch(Shard(make_synthetic_shard(4, 11, max_actions=10)).batch(np.arange(4)))
    args = (b["tiles"], b["globals"], b["actions"], b["action_mask"])
    a, c = plain(*args), withheads(*args)
    assert len(c) == 2  # heads are off unless asked for
    torch.testing.assert_close(a.policy_logits, c.policy_logits)
    out = withheads(*args, belief=True)
    assert [t.shape for t in out.belief] == [(4, v, 11, 11) for v in S.TRUTH_VOCABS]
    torch.testing.assert_close(a.policy_logits, out.policy_logits)


# ---- training and checkpoints ---------------------------------------------------------------------------------


def run_cfg(tmp_path, name, shards, val, **kw):
    base = dict(name=name, model="tiny", train=[str(p) for p in shards], val=[str(p) for p in val], batch_size=32,
                steps=40, log_every=10, eval_every=20, eval_batches=2, warmup_steps=2, lr=3e-3,
                runs_dir=str(tmp_path / "runs"), device="cpu", bf16=False, threads=2)
    base.update(kw)
    return TrainConfig(**base)


def test_belief_training_learns_and_the_checkpoint_loads_both_ways(tmp_path):
    shards = truth_shards(tmp_path, "train", n_shards=2, n=96, seed=1)
    val = truth_shards(tmp_path, "val", n=64, seed=2)
    train(run_cfg(tmp_path, "bel", shards, val, belief_weight=1.0, belief_hidden=16, steps=300, eval_every=100))
    run = tmp_path / "runs" / "bel"
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    trains = [r for r in recs if r["kind"] == "train"]
    assert trains[-1]["belief"] < 0.6 * trains[0]["belief"], (trains[0]["belief"], trains[-1]["belief"])
    val_rec = [r for r in recs if r["kind"] == "val"][-1]
    assert val_rec["belief"]["groups"]["terrain"]["accuracy"] > 0 and "val_belief" in val_rec

    cfg_json = json.loads((run / "config.json").read_text())
    assert cfg_json["model"]["belief_hidden"] == 16 and cfg_json["train"]["belief_weight"] == 1.0

    # play-time loaders get a plain network with the same policy ...
    full = load_checkpoint(run / "ckpt_final.pt", belief=True).eval()
    plain = load_checkpoint(run / "ckpt_final.pt").eval()
    assert full.belief_heads is not None and plain.belief_heads is None
    assert "belief_hidden" not in plain.cfg.to_dict() and plain.n_params() < full.n_params()
    b = to_torch(Shard.load(val[0]).batch(np.arange(8)))
    args = (b["tiles"], b["globals"], b["actions"], b["action_mask"])
    torch.testing.assert_close(full(*args).policy_logits, plain(*args).policy_logits)
    torch.testing.assert_close(full(*args).winner_logits, plain(*args).winner_logits)

    # ... in the policy, which only ever uses the policy and value
    obs = fake_obs(lens=(3, 5, 2))
    c = ModelPolicy(run / "ckpt_final.pt").choose(obs)
    assert len(c) == 3 and ((c.value > 0) & (c.value < 1)).all()

    # and the held-out belief report runs on the full model
    res = evaluate_belief(full, val, batch_size=32)
    assert res["groups"]["terrain"]["hidden_tiles"] > 0 and res["capital_location"] is not None


def test_weight_zero_trains_the_old_model_and_old_configs_load(tmp_path):
    shards = truth_shards(tmp_path, "train", n=64, seed=3)
    train(run_cfg(tmp_path, "zero", shards, [], steps=6, belief_weight=0.0))
    run = tmp_path / "runs" / "zero"
    cfg_json = json.loads((run / "config.json").read_text())
    assert "belief_hidden" not in cfg_json["model"]
    state = torch.load(run / "ckpt_final.pt", weights_only=False)
    assert not any(k.startswith("belief") for k in state["model"])
    assert load_checkpoint(run / "ckpt_final.pt", belief=True).belief_heads is None
    # a checkpoint written before the key existed has no belief_hidden at all
    assert "belief_hidden" not in state["model_config"]
    assert build_model(ModelConfig(**state["model_config"])).belief_heads is None


def test_belief_weight_needs_truth_shards(tmp_path):
    plain = tmp_path / "plain.npz"
    save_shard(plain, make_synthetic_shard(32, 11, max_actions=10))
    with pytest.raises(SystemExit, match="--truth"):
        train(run_cfg(tmp_path, "nt", [plain], [], belief_weight=0.5, steps=2))


def test_init_fine_tunes_from_a_checkpoint_with_or_without_heads(tmp_path):
    shards = truth_shards(tmp_path, "train", n=64, seed=4)
    torch.manual_seed(1)
    base = build_model("tiny")
    for p in base.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    ckpt = tmp_path / "base.pt"
    save_checkpoint(ckpt, base, 7)
    # steps=0: nothing is trained, so the saved weights are the loaded ones
    train(run_cfg(tmp_path, "a", shards, [], steps=0, init=str(ckpt)))
    train(run_cfg(tmp_path, "b", shards, [], steps=0, init=str(ckpt), belief_weight=0.5, belief_hidden=8))
    a = load_checkpoint(tmp_path / "runs" / "a" / "ckpt_final.pt")
    b = load_checkpoint(tmp_path / "runs" / "b" / "ckpt_final.pt")
    for (ka, va), (kb, vb) in zip(base.state_dict().items(), b.state_dict().items()):
        assert ka == kb
        torch.testing.assert_close(va, vb)
        torch.testing.assert_close(va, a.state_dict()[ka])
    # a checkpoint that has heads initialises another run with heads of the same width
    train(run_cfg(tmp_path, "c", shards, [], steps=0, init=str(tmp_path / "runs" / "b" / "ckpt_final.pt"),
                  belief_weight=0.5, belief_hidden=8))
    bf = load_checkpoint(tmp_path / "runs" / "b" / "ckpt_final.pt", belief=True)
    cf = load_checkpoint(tmp_path / "runs" / "c" / "ckpt_final.pt", belief=True)
    torch.testing.assert_close(bf.state_dict()["belief_heads.0.0.weight"], cf.state_dict()["belief_heads.0.0.weight"])
    # a different trunk is refused
    with pytest.raises(SystemExit, match="different model configuration"):
        train(run_cfg(tmp_path, "d", shards, [], steps=0, model="b6c96", init=str(ckpt)))


def test_compute_losses_without_belief_is_unchanged(tmp_path):
    shard = Shard.load(truth_shards(tmp_path, "t", n=16)[0])
    batch = to_torch(shard.batch(np.arange(16)))
    out = build_model("tiny")(batch["tiles"], batch["globals"], batch["actions"], batch["action_mask"])
    losses = compute_losses(out, batch, TrainConfig(belief_weight=1.0))  # no belief output: no belief term
    assert "belief" not in losses
