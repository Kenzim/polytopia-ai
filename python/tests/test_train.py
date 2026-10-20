import json
import math

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.data import BatchLoader, Shard, make_synthetic_shard, save_shard, to_torch, write_synthetic_shards
from polyai.model import MASK_FILL, ModelOutput, build_model
from polyai.train_bc import (
    Meter,
    TrainConfig,
    check_gpu_memory,
    compute_losses,
    evaluate,
    load_checkpoint,
    lr_at,
    make_optimizer,
    train,
)


def fake_output(batch, logits=None, winner=None) -> ModelOutput:
    b, a = batch["action_mask"].shape
    if logits is None:
        logits = torch.zeros(b, a)
    logits = logits.masked_fill(~batch["action_mask"], MASK_FILL)
    return ModelOutput(logits, winner if winner is not None else torch.zeros(b, S.N_SLOTS), torch.zeros(b),
                       torch.zeros(b))


@pytest.fixture
def batch():
    shard = Shard(make_synthetic_shard(32, 11, seed=2, max_actions=20))
    return to_torch(shard.batch(np.arange(32)))


def test_uniform_policy_costs_log_of_list_length(batch):
    cfg = TrainConfig()
    losses = compute_losses(fake_output(batch), batch, cfg)
    expected = torch.log(batch["action_mask"].sum(1).float()).mean()
    assert losses["policy_ce"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert losses["winner_ce"].item() == pytest.approx(math.log(S.N_SLOTS), rel=1e-5)


def test_policy_ce_vanishes_for_a_confident_correct_policy(batch):
    logits = torch.zeros_like(batch["action_mask"], dtype=torch.float32)
    logits.scatter_(1, batch["chosen"][:, None], 30.0)
    losses = compute_losses(fake_output(batch, logits), batch, TrainConfig())
    assert losses["policy_ce"].item() < 1e-4


def test_kl_is_zero_when_the_model_matches_the_candidate_distribution(batch):
    cfg = TrainConfig()
    b, a = batch["action_mask"].shape
    logits = torch.zeros(b, a)
    valid = batch["cand_idx"] >= 0
    assert valid.any()
    for i in range(b):
        for k in range(valid.shape[1]):
            if valid[i, k]:
                logits[i, batch["cand_idx"][i, k]] = batch["cand_score"][i, k] / cfg.kl_temperature
    good = compute_losses(fake_output(batch, logits), batch, cfg)["policy_kl"].item()
    flat = compute_losses(fake_output(batch), batch, cfg)["policy_kl"].item()
    assert good == pytest.approx(0.0, abs=1e-5)
    assert flat > 0.01


def test_kl_ignores_positions_without_candidates(batch):
    batch = dict(batch)
    batch["cand_idx"] = torch.full_like(batch["cand_idx"], -1)
    losses = compute_losses(fake_output(batch), batch, TrainConfig())
    assert losses["policy_kl"].item() == 0.0


def test_unknown_winner_is_ignored(batch):
    batch = dict(batch)
    batch["winner"] = torch.full_like(batch["winner"], -1)
    losses = compute_losses(fake_output(batch), batch, TrainConfig())
    assert losses["winner_ce"].item() == 0.0
    assert torch.isfinite(losses["loss"])


def test_margin_loss_uses_thousandths(batch):
    batch = dict(batch)
    batch["margin"] = torch.full_like(batch["margin"], 2000.0)
    losses = compute_losses(fake_output(batch), batch, TrainConfig())
    assert losses["margin_mse"].item() == pytest.approx(4.0)


def test_brier_by_turn_and_agreement(batch):
    batch = dict(batch)
    b = batch["chosen"].shape[0]
    batch["winner"] = torch.zeros(b, dtype=torch.long)  # I always win
    batch["turn"] = torch.where(torch.arange(b) < b // 2, 3, 25)
    winner = torch.full((b, S.N_SLOTS), -30.0)
    winner[:, 0] = 30.0
    winner[b // 2 :, 0] = -30.0
    winner[b // 2 :, 1] = 30.0  # wrong in the later half
    logits = torch.zeros_like(batch["action_mask"], dtype=torch.float32)
    logits.scatter_(1, batch["chosen"][:, None], 5.0)
    out = fake_output(batch, logits, winner)
    m = Meter()
    m.add(out, batch, compute_losses(out, batch, TrainConfig()))
    r = m.result()
    assert r["top1"] == 1.0 and r["top3"] == 1.0
    assert r["brier_by_turn"]["0-9"] == pytest.approx(0.0, abs=1e-6)
    assert r["brier_by_turn"]["20-29"] == pytest.approx(1.0, abs=1e-6)
    assert r["brier"] == pytest.approx(0.5, abs=1e-6)


def test_top3_is_at_least_top1(batch):
    out = fake_output(batch, torch.randn(*batch["action_mask"].shape))
    m = Meter()
    m.add(out, batch, compute_losses(out, batch, TrainConfig()))
    r = m.result()
    assert r["top1"] <= r["top3"] <= 1.0


def test_lr_schedule():
    cfg = TrainConfig(steps=1000, warmup_steps=100, lr=1e-3, min_lr_frac=0.02)
    assert lr_at(0, cfg) < lr_at(50, cfg) < lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(100, cfg) == pytest.approx(1e-3)
    assert lr_at(550, cfg) == pytest.approx(0.5 * 1e-3 * 1.02, rel=0.05)
    assert lr_at(1000, cfg) == pytest.approx(2e-5, rel=1e-3)
    assert lr_at(5000, cfg) == pytest.approx(2e-5, rel=1e-3)


def test_gpu_check_fails_clearly_when_memory_is_short():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    with pytest.raises(SystemExit, match="not enough free GPU memory"):
        check_gpu_memory(1000.0, torch.device("cuda"))


def test_cpu_device_skips_the_gpu_check():
    assert check_gpu_memory(1000.0, torch.device("cpu")) == 0


def test_overfits_a_tiny_synthetic_set(tmp_path):
    """64 positions, tiny net, no augmentation: the policy must memorise and the losses must fall a long way."""
    torch.manual_seed(0)
    path = tmp_path / "tiny.npz"
    save_shard(path, make_synthetic_shard(64, 11, seed=7, max_actions=20))
    cfg = TrainConfig(batch_size=64, lr=3e-3, steps=200, warmup_steps=10, augment=False, bf16=False,
                      device="cpu")
    model = build_model("tiny")
    opt = make_optimizer(model, cfg)
    loader = BatchLoader([path], 64, augment=False, shuffle=True, prefetch=0)
    first = None
    step = 0
    for batch in loader.iterate(epochs=None):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        out = model(batch["tiles"], batch["globals"], batch["actions"], batch["action_mask"])
        losses = compute_losses(out, batch, cfg)
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        first = first or {k: v.item() for k, v in losses.items()}
        step += 1
        if step >= cfg.steps:
            break
    last = {k: v.item() for k, v in losses.items()}
    assert last["loss"] < 0.5 * first["loss"], (first, last)
    assert last["policy_ce"] < 0.35 * first["policy_ce"], (first, last)
    assert last["winner_ce"] < 0.5 * first["winner_ce"], (first, last)

    model.eval()
    res = evaluate(model, BatchLoader([path], 64, augment=False, shuffle=False, prefetch=0), cfg,
                   torch.device("cpu"))
    assert res["top1"] > 0.8, res
    assert res["top3"] >= res["top1"]


def test_train_run_writes_config_metrics_and_checkpoints(tmp_path):
    shards = write_synthetic_shards(tmp_path / "train", 2, 64, size=11, seed=1, max_actions=20)
    val = write_synthetic_shards(tmp_path / "val", 1, 64, size=11, seed=2, max_actions=20)
    cfg = TrainConfig(name="smoke", model="tiny", train=[str(p) for p in shards], val=[str(p) for p in val],
                      batch_size=32, steps=20, log_every=5, eval_every=10, eval_batches=2, warmup_steps=2,
                      runs_dir=str(tmp_path / "runs"), device="cpu", bf16=False, threads=2)
    summary = train(cfg)
    run = tmp_path / "runs" / "smoke"
    assert (run / "config.json").exists() and (run / "ckpt_final.pt").exists() and (run / "ckpt_last.pt").exists()
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    kinds = [rec["kind"] for rec in lines]
    assert kinds.count("train") == 4 and kinds.count("val") == 2 and kinds[-1] == "summary"
    val_rec = next(rec for rec in lines if rec["kind"] == "val")
    assert {"top1", "top3", "brier", "brier_by_turn"} <= set(val_rec)
    assert summary["steps"] == 20 and summary["steps_per_sec"] > 0

    model = load_checkpoint(run / "ckpt_final.pt")
    ref = build_model("tiny")
    assert model.n_params() == ref.n_params()
    b = to_torch(Shard.load(val[0]).batch(np.arange(4)))
    model.eval()
    out = model(b["tiles"], b["globals"], b["actions"], b["action_mask"])
    assert torch.isfinite(out.policy_logits.masked_fill(~b["action_mask"], 0)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_training_stays_under_the_memory_budget(tmp_path):
    free, _ = torch.cuda.mem_get_info()
    if free < 2.5 * 1024**3:
        pytest.skip("not enough free GPU memory right now")
    shards = write_synthetic_shards(tmp_path / "train", 1, 256, size=16, seed=3, max_actions=60)
    cfg = TrainConfig(name="gpu", model="b6c96", train=[str(p) for p in shards], batch_size=128, steps=6,
                      log_every=3, runs_dir=str(tmp_path / "runs"), warmup_steps=1, threads=2)
    summary = train(cfg)
    assert 0 < summary["peak_mem_mb"] < 2.5 * 1024
