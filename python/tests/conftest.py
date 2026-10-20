import numpy as np
import pytest

from polyai.data import Shard, make_synthetic_shard


@pytest.fixture(scope="session")
def shard16() -> Shard:
    return Shard(make_synthetic_shard(64, 16, seed=1))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
