import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import autoloop  # noqa: E402


def rec(swap, winner):
    return json.dumps({"swap": swap, "winner": winner})


def test_score_file_counts_the_seat_given_by_swap(tmp_path):
    p = tmp_path / "g.jsonl"
    # swap 0: candidate is seat 0; swap 1: candidate is seat 1
    p.write_text("\n".join([rec(0, 0), rec(1, 1), rec(0, 1), rec(1, None)]) + "\n")
    r = autoloop.score_file(p)
    assert (r["wins"], r["losses"], r["draws"], r["games"]) == (2, 1, 1, 4)
    assert r["score"] == 2.5 / 4


def test_missing_file_is_empty(tmp_path):
    assert autoloop.score_file(tmp_path / "none.jsonl")["games"] == 0


def test_gate_threshold_is_two_standard_errors_at_400_games():
    se = (0.25 / 400) ** 0.5
    assert abs((autoloop.Config().gate_score - 0.5) / se - 2.0) < 1e-9


def test_wilson_contains_the_estimate():
    lo, hi = autoloop.wilson(0.55, 400)
    assert lo < 0.55 < hi and 0.50 < lo
