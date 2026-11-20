"""The service on web engine lines (`line` input, `choose` with `text`, `turn`) and `--check`.

`web_line_tiny.json` is a line the web's engine service (`polyengine`) wrote: a tiny two-player Might game at
turn 3, part way through player 1's turn, with the texts of every legal command there (`texts`).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pe = pytest.importorskip("polyai_engine")

from polyai.policy import ModelPolicy  # noqa: E402
from polyai.serve import PolicyClient, PolicyServer, check  # noqa: E402

HERE = Path(__file__).parent
MODEL = HERE.parent.parent / "models" / "auto_r0077" / "ckpt_final.pt"
DATA = json.loads((HERE / "web_line_tiny.json").read_text())
pytestmark = [
    pytest.mark.skipif(not hasattr(pe.Game, "from_line"), reason="polyai_engine without Game.from_line"),
    pytest.mark.skipif(not MODEL.exists(), reason="needs models/auto_r0077"),
]


def from_line(line: dict):
    return pe.Game.from_line(json.dumps(line["position"]), [s["text"] for s in line["steps"]])


@pytest.fixture(scope="module")
def client():
    policy = ModelPolicy(MODEL, "cpu", weights_only=True, name="r77")
    with tempfile.TemporaryDirectory(dir="/kvm/tmp/tmp" if Path("/kvm/tmp/tmp").is_dir() else None) as d:
        with PolicyServer(policy, Path(d) / "p.sock", request_timeout=60.0) as server:
            with PolicyClient(server.path, timeout=60.0) as c:
                yield c


def test_the_line_rebuilds_with_the_web_engines_legal_commands():
    g = from_line(DATA["line"])
    assert g.mover_id == DATA["player"]
    assert set(g.legal_texts()) <= set(DATA["texts"])


def test_choose_on_a_line_answers_a_legal_text(client):
    r = client.choose_line("t", DATA["player"], DATA["line"])
    assert r["ok"], r
    assert r["text"] in DATA["texts"]
    assert r["policy_id"] == "r77@" + r["policy_id"].split("@")[1]


def test_a_turn_plays_legal_commands_to_the_end_of_the_turn(client):
    line = DATA["line"]
    r = client.turn("t", DATA["player"], line)
    assert r["ok"], r
    texts = [c["text"] for c in r["cmds"]]
    assert texts and texts[-1] == "EndTurn"
    assert not any(t.startswith("Stay {") for t in texts), "a Stay is the model's own; the web engine has none"
    assert all(0.0 <= c["p"] <= 1.0 and 0.0 <= c["value"] <= 1.0 for c in r["cmds"])
    g = from_line(line)
    for t in texts:  # each in turn is in the engine's own legal list, which the web plays texts against
        assert t in g.legal_texts() or t == "EndTurn", t
        g.play_text(t)
    assert g.mover_id != DATA["player"] or g.is_over()


def test_max_commands_ends_the_turn_early(client):
    r = client.turn("t", DATA["player"], DATA["line"], max_commands=1)
    assert r["ok"], r
    assert [c["text"] for c in r["cmds"]] == ["EndTurn"]


def test_bad_lines_are_refused_cleanly(client):
    wrong_seat = client.turn("t", 3 - DATA["player"], DATA["line"])
    assert not wrong_seat["ok"] and "turn" in wrong_seat["error"]
    bad = copy.deepcopy(DATA["line"])
    bad["steps"].append({"text": "Move { unit: 999, to: WorldCoordinates { x: 0, y: 0 } }", "ai": False})
    refused = client.turn("t", DATA["player"], bad)
    assert not refused["ok"] and "step" in refused["error"]
    assert not client.turn("t", DATA["player"], {"steps": []})["ok"]
    assert not client.turn("t", DATA["player"], DATA["line"], max_commands=0)["ok"]
    assert client.info()["ok"]  # the service is still up


class _Plant:
    """Unpickling this calls `Path.touch` on the marker: what a hostile checkpoint could do."""

    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (Path(self.marker),))


def test_check_accepts_our_model_and_refuses_code_in_a_pickle(tmp_path):
    ok = check(MODEL)
    assert ok["ok"] and ok["step"] > 0 and ok["params"] > 1_000_000
    marker = tmp_path / "ran"
    bad = tmp_path / "bad.pt"
    import torch

    torch.save({"model": {}, "model_config": {}, "step": 1, "payload": _Plant(str(marker))}, bad)
    assert not marker.exists()
    refused = check(bad)
    assert not refused["ok"] and not marker.exists()
    cli = subprocess.run([sys.executable, "-m", "polyai.serve", "--check", str(bad)], capture_output=True, text=True,
                         cwd=HERE.parent, timeout=120)
    assert cli.returncode == 1 and not json.loads(cli.stdout.strip().splitlines()[-1])["ok"]
    assert not marker.exists()
