import json

import cells


def game(seed, swap, agents, winner, tribes=(4, 9), size=2, preset=2, capped=False):
    return {"seed": seed, "swap": swap, "agents": list(agents), "winner": winner, "tribes": list(tribes),
            "setup": {"size": size, "preset": preset}, "capped": capped, "turns": 30}


def write(path, games):
    path.write_text("".join(json.dumps(g) + "\n" for g in games))
    return str(path)


def test_outcome_from_either_seat_and_draws():
    g = game(1, 0, ["model:x", "retail:crazy"], 1)
    assert cells.outcome(g, "model:x")["score"] == 0.0
    assert cells.outcome(g, "retail:crazy")["score"] == 1.0
    g = game(1, 1, ["retail:crazy", "model:x"], 1, tribes=(9, 4))
    o = cells.outcome(g, "model:x")
    assert (o["score"], o["seat"], o["tribe"], o["opp_tribe"]) == (1.0, 1, 4, 9)
    assert cells.outcome(game(1, 0, ["model:x", "retail:crazy"], None, capped=True), "model:x")["score"] == 0.5
    assert cells.outcome(game(1, 0, ["a", "b"], 0), "model:x") is None


def test_wilson_and_sign_test():
    lo, hi = cells.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
    assert cells.sign_test(0, 0) == 1.0
    assert abs(cells.sign_test(5, 5) - 1.0) < 1e-12
    assert cells.sign_test(10, 0) < 0.01


def test_paired_matches_on_seed_and_swap(tmp_path, capsys):
    a = write(tmp_path / "a.jsonl", [game(s, w, ["model:a", "retail:crazy"] if w == 0 else ["retail:crazy", "model:a"],
                                          w) for s in range(10) for w in (0, 1)])
    b = write(tmp_path / "b.jsonl", [game(s, w, ["model:b", "retail:crazy"] if w == 0 else ["retail:crazy", "model:b"],
                                          1 - w) for s in range(10) for w in (0, 1)])
    cells.main(["--paired", a, b])
    out = capsys.readouterr().out
    assert "20 matched games, A 100.0%, B 0.0%, A better 20, B better 0" in out


def test_report_runs(tmp_path, capsys):
    f = write(tmp_path / "r.jsonl", [game(s, 0, ["model:a", "retail:hard"], s % 2) for s in range(20)])
    cells.main([f, "--min-games", "1"])
    out = capsys.readouterr().out
    assert "vs retail:hard: n=20 score 50.0%" in out and "Bardur v Luxidoor" in out
