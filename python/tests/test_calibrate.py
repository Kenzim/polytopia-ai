"""`polyai.calibrate`: win estimates of a policy network or a privileged critic against the outcomes of finished games."""

import json

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.calibrate import collect, main, reliability, report, wilson, window_labels
from polyai.data import make_synthetic_shard
from polyai.model import make_critic
from polyai.train_bc import load_checkpoint, save_checkpoint
from test_rl import tiny_model


def test_reliability_buckets_scores_and_interval():
    p = np.array([0.05, 0.05, 0.85, 0.85, 0.85, 0.85, 1.0])
    y = np.array([0, 1, 1, 1, 1, 0, 1])
    r = reliability(p, y)
    assert r["n"] == 7 and [b["n"] for b in r["buckets"]] == [2, 0, 0, 0, 0, 0, 0, 0, 4, 1]
    low, high, top = r["buckets"][0], r["buckets"][8], r["buckets"][9]
    assert low["won"] == 0.5 and high["won"] == 0.75 and top["won"] == 1.0
    assert high["mean_p"] == pytest.approx(0.85) and high["lo"] < 0.75 < high["hi"]
    assert r["brier"] == pytest.approx(((p - y) ** 2).mean())
    assert r["ece"] == pytest.approx((2 * 0.45 + 4 * 0.10 + 0) / 7)
    assert r["accuracy"] == pytest.approx(5 / 7)
    assert wilson(0, 0) == (0.0, 1.0) and wilson(50, 100)[0] == pytest.approx(0.404, abs=1e-3)
    assert reliability(np.array([]), np.array([]))["n"] == 0


def test_windows_and_samples_are_one_side_of_one_game_per_window():
    assert window_labels([2, 5]) == ["<=2", "3-5", ">=6"]
    data = {
        "game": np.array([0, 0, 0, 0, 1, 1]), "won": np.array([1, 1, 0, 1, 0, 0]), "turn": np.array([0, 2, 2, 3, 6, 7]),
        "p": {"m": np.array([0.6, 0.8, 0.3, 0.9, 0.2, 0.4])},
    }
    rep = report(data, [2, 5])
    assert rep["games"] == 2 and rep["positions"] == 6
    first, mid, late = (rep["models"]["m"][k] for k in rep["windows"])
    assert first["n"] == 2 and mid["n"] == 1 and late["n"] == 1  # winner and loser of game 0; winner; loser of game 1
    assert first["brier"] == pytest.approx(((0.7 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
    assert late["brier"] == pytest.approx(0.3 ** 2)


def write_shard(path, n, seed, players, truth):
    a = make_synthetic_shard(n, size=11, seed=seed, with_truth=truth)
    a["game_key"] = np.array([f"human:g{seed}_{i % 4}" for i in range(n)])
    if truth:
        a["truth_g"] = np.random.default_rng(seed).random((n, S.TRUTH_G)).astype(np.float32)
    np.savez(path, **a)
    with open(path.parent / "meta.jsonl", "a") as f:
        for k in range(4):
            f.write(json.dumps({"id": f"g{seed}_{k}", "players": players[k]}) + "\n")
    return a


def test_models_are_run_on_the_decided_games_with_the_asked_number_of_players(tmp_path, capsys):
    a = write_shard(tmp_path / "s0.npz", 60, 1, [2, 2, 3, 2], truth=True)
    model = tiny_model().eval()
    critic = make_critic(model)
    for p in list(critic.truth_embed.parameters()) + [critic.truth_g_proj.weight]:
        torch.nn.init.normal_(p, std=0.5)
    save_checkpoint(tmp_path / "policy.pt", model, 1)
    save_checkpoint(tmp_path / "critic.pt", critic, 1)
    models = {"policy": load_checkpoint(tmp_path / "policy.pt").eval(), "critic": load_checkpoint(tmp_path / "critic.pt").eval()}
    assert models["critic"].cfg.truth_input and not models["policy"].cfg.truth_input
    data = collect([tmp_path / "s0.npz"], models, 2, torch.device("cpu"), batch=16, log=lambda _: None)
    keep = (a["winner"] >= 0) & (np.arange(60) % 4 != 2)
    assert len(data["game"]) == keep.sum() and len(np.unique(data["game"])) == 3
    np.testing.assert_array_equal(data["won"], (a["winner"][keep] == 0))
    np.testing.assert_array_equal(data["turn"], a["turn"][keep])
    rows = np.nonzero(keep)[0]
    with torch.no_grad():
        actions = torch.full((len(rows), 1, 4), -1, dtype=torch.long)
        actions[..., 0] = 0
        out = model(torch.from_numpy(a["tiles"][rows]), torch.from_numpy(a["globals"][rows]), actions,
                    torch.ones(len(rows), 1, dtype=torch.bool))
        want = torch.softmax(out.winner_logits, -1)[:, 0].numpy()
    np.testing.assert_allclose(data["p"]["policy"], want, atol=1e-5)
    assert not np.allclose(data["p"]["critic"], want, atol=1e-3)  # the critic reads the truth
    every = collect([tmp_path / "s0.npz"], {"policy": models["policy"]}, 0, torch.device("cpu"), log=lambda _: None)
    assert len(every["game"]) == (a["winner"] >= 0).sum()

    out = tmp_path / "rep.json"
    assert main(["--shards", str(tmp_path), "--model", f"policy={tmp_path / 'policy.pt'}", f"critic={tmp_path / 'critic.pt'}",
                 "--out", str(out), "--turns", "10,30"]) == 0
    rep = json.loads(out.read_text())
    assert rep["games"] == 3 and set(rep["models"]) == {"policy", "critic"} and rep["windows"] == ["<=10", "11-30", ">=31"]
    assert "policy, turns <=10" in capsys.readouterr().out


def test_a_critic_needs_shards_with_the_truth_and_a_player_filter_needs_the_meta_file(tmp_path):
    write_shard(tmp_path / "s0.npz", 20, 2, [2, 2, 2, 2], truth=False)
    critic = make_critic(tiny_model()).eval()
    with pytest.raises(SystemExit, match="truth"):
        collect([tmp_path / "s0.npz"], {"critic": critic}, 2, torch.device("cpu"), log=lambda _: None)
    (tmp_path / "meta.jsonl").unlink()
    with pytest.raises(SystemExit, match="meta.jsonl"):
        collect([tmp_path / "s0.npz"], {"policy": tiny_model().eval()}, 2, torch.device("cpu"), log=lambda _: None)
