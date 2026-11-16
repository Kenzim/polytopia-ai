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


def result(gate, crazy, champ_crazy=0.80, promoted=False):
    return {"candidate": "/r/cand.pt", "candidate_id": "cand@300", "promoted": promoted,
            "gate": {"score": gate}, "crazy": {"score": crazy}, "champion_crazy": {"score": champ_crazy}}


ST = {"champion": "/r/champ.pt", "champion_id": "champ@300"}


def test_line_is_off_by_default():
    assert autoloop.next_line(ST, result(0.45, 0.80), autoloop.Config()) == {}


def test_line_continues_from_a_candidate_that_fails_the_gate_but_holds_up():
    c = autoloop.Config(continue_line=True)
    assert autoloop.next_line(dict(ST, line_rounds=2), result(0.45, 0.78), c) == \
        {"line": "/r/cand.pt", "line_id": "cand@300", "line_rounds": 3}


def test_line_resets_to_the_champion_on_guard_failure_low_gate_or_too_many_rounds():
    c = autoloop.Config(continue_line=True)
    back = {"line": "/r/champ.pt", "line_id": "champ@300", "line_rounds": 0}
    assert autoloop.next_line(ST, result(0.45, 0.70), c) == back  # crazy 70% < 80% - 8
    assert autoloop.next_line(ST, result(0.25, 0.80), c) == back  # below line_floor
    assert autoloop.next_line(dict(ST, line_rounds=c.line_max_rounds - 1), result(0.45, 0.80), c) == back


def test_line_follows_a_promotion():
    c = autoloop.Config(continue_line=True)
    assert autoloop.next_line(dict(ST, line_rounds=3), result(0.60, 0.80, promoted=True), c) == \
        {"line": "/r/cand.pt", "line_id": "cand@300", "line_rounds": 0}


def test_default_pool_weights_are_half_retail():
    w = dict(kv.split("=") for kv in autoloop.Config().pool_weights.split(","))
    assert abs(sum(float(v) for k, v in w.items() if k.startswith("retail:")) - 0.5) < 1e-9


def test_wilson_contains_the_estimate():
    lo, hi = autoloop.wilson(0.55, 400)
    assert lo < 0.55 < hi and 0.50 < lo
