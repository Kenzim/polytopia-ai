"""Fixed extra opponents in the RL opponent pool (`--extra-opponents`)."""
import numpy as np
import pytest

from polyai.rl import Pool, RLConfig
from polyai.rl_pool import EntryInfo


def make_pool(weights: str) -> Pool:
    return Pool(RLConfig(pool_weights=weights), object(), EntryInfo("bc", "bc@0", "bc.pt"))


def test_extra_opponent_is_sampled_by_its_weight():
    pool = make_pool("retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0,bc=0,self=0,bpolait=1")
    pool.add_fixed("bpolait", object(), EntryInfo("bpolait", "bp@4000", "bp.pt"))
    assert "bpolait" in pool.model_entries()
    rng = np.random.default_rng(0)
    assert {pool.sample_model_entry(rng) for _ in range(20)} == {"bpolait"}


def test_extra_opponent_without_weight_is_not_played():
    pool = make_pool("bc=0.5,self=0.5")
    pool.add_fixed("bpolait", object(), EntryInfo("bpolait", "bp@4000", "bp.pt"))
    assert "bpolait" not in pool.model_entries()


@pytest.mark.parametrize("name", ["bc", "self", "self:u00010", "retail:crazy"])
def test_extra_opponent_name_clash(name):
    pool = make_pool("bc=1")
    with pytest.raises(ValueError):
        pool.add_fixed(name, object(), EntryInfo(name, "x@0", "x.pt"))
