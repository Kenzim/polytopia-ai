"""A shard written by the Rust `polyai-gen` carries the optional provenance keys with the generator values."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from polyai import schema as S
from polyai.data import Shard

BIN = Path(__file__).resolve().parents[2] / "target" / "release" / "polyai-gen"
pytestmark = pytest.mark.skipif(not BIN.exists(), reason="target/release/polyai-gen is not built")


def test_polyai_gen_writes_the_optional_keys(tmp_path):
    subprocess.run([str(BIN), "--out", str(tmp_path), "--games", "3", "--threads", "2", "--turn-limit", "12",
                    "--players", "2", "--seed-start", "40"], check=True, capture_output=True, timeout=300)
    paths = sorted(tmp_path.glob("shard-*.npz"))
    assert paths
    for p in paths:
        with np.load(p) as z:
            assert z["version"].tolist() == [S.ENCODING_VERSION]
            for k, spec in S.OPTIONAL_FIELDS.items():
                assert k in z.files, k
                if spec.dtype != "str":
                    assert z[k].dtype == np.dtype(spec.dtype), k
                else:
                    assert z[k].dtype.kind == "U", k
        s = Shard.load(p)
        assert (s.a["source"] == 0).all() and (s.a["ours"] == 0).all() and (s.a["policy"] == -1).all()
        assert np.isnan(s.a["behaviour_logp"]).all() and s.policy_ids == []
        assert list(s.game_key) == [f"gen:{x}" for x in s.a["seed"]]
        assert set(s.a["seed"]) <= {40, 41, 42}
