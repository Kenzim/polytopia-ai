from pathlib import Path

import pytest
import torch

from polyai.arena import ModelAgent, Suite
from polyai.h2h import play_h2h
from polyai.model import build_model
from polyai.train_bc import save_checkpoint


def _ckpt(root: Path, name: str, seed: int) -> Path:
    run = root / name
    run.mkdir()
    torch.manual_seed(seed)
    model = build_model("tiny")
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    save_checkpoint(run / "ckpt_final.pt", model, seed)
    return run / "ckpt_final.pt"


def test_two_models_play_both_seatings_on_the_eval_suite(tmp_path):
    pytest.importorskip("polyai_engine")
    a = ModelAgent(_ckpt(tmp_path, "runa", 1), None, torch.device("cpu"))
    b = ModelAgent(_ckpt(tmp_path, "runb", 2), None, torch.device("cpu"))
    suite = Suite(seeds=3, sizes=[1], presets=[3], turn_limit=12, batch_games=2, threads=4)
    records = play_h2h(a, b, suite, set(), log=lambda m: None)
    assert len(records) == 6
    by = {(r["seed"], r["swap"]): r for r in records}
    for s in range(suite.seed_start, suite.seed_start + 3):
        assert by[(s, 0)]["agents"] == ["model:runa", "model:runb"]
        assert by[(s, 1)]["agents"] == ["model:runb", "model:runa"]
        assert by[(s, 0)]["tribes"] == by[(s, 1)]["tribes"] and by[(s, 0)]["setup"] == by[(s, 1)]["setup"]
    for r in records:
        assert r["seat_difficulties"] == [1, 1] and r["illegal"] == [0, 0] and r["turns"] <= 13
        assert r["winner"] in (None, 0, 1) and (r["winner"] is None or not r["capped"])
    done = {(r["seed"], r["swap"], tuple(r["agents"])) for r in records}
    assert play_h2h(a, b, suite, done, log=lambda m: None) == []
