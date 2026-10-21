"""Tests for elo.py. Run: /kvm/polytopia/.venv/bin/python -m pytest ai/tools/test_elo.py"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import elo  # noqa: E402


def game(seed, a, b, winner, score_diff=0, capped=False):
    return {"seed": seed, "a": a, "b": b, "winner": winner, "score_diff": score_diff, "capped": capped, "wall_ms": 1000.0}


def test_wilson_known_values():
    lo, hi = elo.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = elo.wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.30
    assert elo.wilson(0, 0) == (0.0, 1.0)


def test_pair_stats_orients_by_name_and_counts_draws():
    games = [
        game(1, "b", "a", 0, score_diff=100),  # b (seat 0) beat a
        game(1, "a", "b", 0, score_diff=50),  # a beat b
        game(2, "a", "b", 1, score_diff=-200),  # b beat a
        game(2, "b", "a", None, capped=True),  # draw at the turn cap
    ]
    s = elo.pair_stats(games)[("a", "b")]
    assert (s["n"], s["wins_a"], s["wins_b"], s["draws"]) == (4, 1, 2, 1)
    assert abs(s["rate"] - 0.375) < 1e-9
    assert abs(s["score_diff"] - (-100 + 50 - 200 + 0) / 4) < 1e-9
    assert s["draw_rate"] == 0.25


def test_draws_count_half_in_the_fit_and_old_adjudicated_lines_are_draws(tmp_path):
    # 20 decisive games split evenly plus 20 draws: the two agents are equal.
    games = [game(s, "x", "y", s % 2) for s in range(20)] + [game(100 + s, "x", "y", None, capped=True) for s in range(20)]
    agents, e, _, _, _ = elo.elo_table(games, anchor="x", bootstrap=0)
    assert abs(e[agents.index("y")] - 1000) < 5
    lines = [
        {"seed": 1, "agents": ["x", "y"], "winner": 0, "scores": [10, 5], "adjudicated": True, "wall_ms": 1},  # old: score winner ignored
        {"seed": 2, "agents": ["x", "y"], "winner": 0, "scores": [10, 5], "capped": False, "wall_ms": 1},
        {"seed": 3, "agents": ["x", "y"], "winner": -1, "scores": [10, 5], "capped": True, "wall_ms": 1},
        {"seed": 4, "agents": ["x", "y"], "winner": 0, "scores": [10, 5], "capped": True, "wall_ms": 1},  # never a winner when capped
    ]
    path = tmp_path / "old.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    loaded = elo.load_games([str(path)])
    assert [g["winner"] for g in loaded] == [None, 0, None, None]
    assert [g["capped"] for g in loaded] == [True, False, True, True]


def test_bt_recovers_known_elo_gaps():
    # A beats B 75% of the time (+191 Elo), B beats C 75%.
    rng = np.random.default_rng(1)
    games = []
    for s in range(1500):
        for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
            p = {("A", "B"): 0.75, ("B", "C"): 0.75, ("A", "C"): 0.9}[(x, y)]
            games.append(game(s, x, y, 0 if rng.random() < p else 1))
    agents, e, lo, hi, anchor = elo.elo_table(games, anchor="B", bootstrap=40)
    d = dict(zip(agents, e))
    assert anchor == "B" and abs(d["B"] - 1000) < 1e-6
    expect = 400 * math.log10(3)
    assert abs(d["A"] - 1000 - expect) < 25 and abs(1000 - d["C"] - expect) < 25
    i = agents.index("A")
    assert lo[i] < d["A"] < hi[i]


def test_unbeaten_agent_gets_a_finite_rating_and_missing_anchor_falls_back():
    games = [game(s, "x", "y", 0) for s in range(20)]
    agents, e, lo, hi, anchor = elo.elo_table(games, anchor="retail:normal", bootstrap=10)
    assert anchor == "x" and np.all(np.isfinite(e)) and e[agents.index("y")] < 1000


def test_cli_reads_jsonl(tmp_path, capsys):
    path = tmp_path / "g.jsonl"
    with open(path, "w") as f:
        for s in range(10):
            for swap, agents in ((0, ["retail:easy", "retail:normal"]), (1, ["retail:normal", "retail:easy"])):
                f.write(json.dumps({"seed": 1000000 + s, "swap": swap, "agents": agents, "winner": 1 - swap, "scores": [1, 2], "capped": False, "wall_ms": 5}) + "\n")
    assert elo.main([str(path), "--bootstrap", "10"]) == 0
    out = capsys.readouterr().out
    assert "20 games, 10 seeds, 0.0% draws" in out and "retail:normal (anchor)" in out and "retail:easy vs retail:normal" in out
