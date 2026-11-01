"""`polyai.value_rank`: candidate generation, playouts and the analysis (needs the engine)."""

import math

import pytest
import torch

from polyai.model import build_model
from polyai.policy import ModelPolicy
from polyai.search import END_TURN, SearchConfig, SearchPolicy
from polyai.train_bc import save_checkpoint
from polyai import value_rank as vr

pe = pytest.importorskip("polyai_engine")


@pytest.fixture(scope="module")
def sp(tmp_path_factory):
    run = tmp_path_factory.mktemp("runs") / "tinyrun"
    run.mkdir()
    torch.manual_seed(0)
    model = build_model("tiny")
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    save_checkpoint(run / "ckpt_final.pt", model, 5)
    return SearchPolicy(ModelPolicy(run / "ckpt_final.pt", "cpu", seed=0), SearchConfig(k=4), seed=0)


def test_candidates_end_at_end_turn_and_keep_the_root(sp):
    roots = vr.make_roots(sp, [vr.DIAG_SEED_MIN + 1], 3, 4, [1], [3])
    assert roots and all(r.turn in vr.SAMPLE_TURNS for r in roots)
    r0 = roots[0]
    before = r0.game.legal_actions().tobytes()
    cands = vr.make_candidates(sp, roots[:1], 3, 2)
    assert r0.game.legal_actions().tobytes() == before  # candidates play on clones
    assert len(cands) == 5 and [c.kind for c in cands] == ["top"] * 3 + ["sample"] * 2
    for c in cands:
        assert c.value is not None and 0.0 <= c.value <= 1.0
        assert c.line and (c.terminal or c.end_index is None or c.line[-1][0] == END_TURN)
        assert c.logp <= c.first_logp + 1e-6 <= 1e-6
    tops = [c for c in cands if c.kind == "top"]
    assert tops[0].first_logp >= tops[1].first_logp >= tops[2].first_logp


def test_playouts_fill_outcomes(sp):
    rows = vr.run_seeds(sp, [vr.DIAG_SEED_MIN + 2], 3, 4, [1], [3], k=2, n_sample=1, playouts=2)
    assert rows
    for r in rows:
        assert len(r["outcomes"]) == 2 and all(o in (0.0, 0.5, 1.0) for o in r["outcomes"])
    a = vr.analyse(rows)
    assert a["candidates"] == len(rows)


def test_spearman_and_analysis_on_made_up_rows():
    assert vr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert vr.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert vr.spearman([1, 1, 1], [1, 2, 3]) is None
    rows = []
    for seed in range(5):
        for j, (v, r) in enumerate([(0.9, [1, 1, 1, 1]), (0.5, [1, 0, 1, 0]), (0.1, [0, 0, 0, 0])]):
            rows.append({"seed": seed, "turn": 4, "kind": "top", "j": j, "value": v, "first_logp": -j,
                         "logp": -j, "outcomes": r})
    a = vr.analyse(rows)
    assert a["rho_value"][0] == pytest.approx(1.0)
    assert a["pick"]["value"][0] == pytest.approx(1.0) and a["pick"]["greedy"][0] == pytest.approx(1.0)
    assert a["pick"]["mean"][0] == pytest.approx(0.5)
    assert not math.isnan(a["reliability_ceiling"])
