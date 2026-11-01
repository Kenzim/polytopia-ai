"""Padded / compiled / CUDA-graph rollout inference (infer_fast.py): same choices as the eager path.

The CPU tests run everywhere (the padded dense forward needs no GPU); the graph test needs a CUDA device with free memory
and skips otherwise."""

import numpy as np
import pytest
import torch

from polyai import infer_fast, schema as S
from polyai.infer_fast import FastForward, bucket_actions, bucket_rows
from polyai.model import ModelConfig, add_features, build_model
from polyai.policy import collate, collate_features
from polyai.rl_pool import NetPolicy, merge_padded
from test_arena import fake_obs
from test_features import feature_batch, randomise_features, tiny

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def fresh_runners():
    infer_fast._RUNNERS.clear()
    infer_fast._BUFFERS.clear()
    yield


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    m = build_model("tiny")
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    return m.eval()


def pair(model, **kw):
    """An eager NetPolicy and a fast one around the same network, with the same sampling seed."""
    return (NetPolicy(model, "eager", CPU, seed=5), NetPolicy(model, "fast", CPU, seed=5, **kw))


def assert_same_choices(a, b, atol=1e-4):
    np.testing.assert_array_equal(a.index, b.index)
    np.testing.assert_allclose(a.logp, b.logp, atol=atol)
    np.testing.assert_allclose(a.value, b.value, atol=atol)


def test_buckets():
    assert [bucket_rows(n) for n in (1, 4, 5, 13, 100, 256, 257, 600)] == [4, 4, 8, 16, 128, 256, 512, 768]
    assert [bucket_actions(n) for n in (1, 128, 129, 1000, 5000)] == [128, 128, 192, 1024, 8192]


def test_forward_flat_equals_forward_on_the_listed_actions():
    model = tiny_model()
    obs = fake_obs(lens=(4, 9, 1, 6), size=11, seed=3)
    tiles, glob, actions, mask = collate(obs, np.arange(4), CPU)
    bi, ai = mask.nonzero(as_tuple=True)
    with torch.no_grad():
        ref = model(tiles, glob, actions, mask)
        flat, winner = model.forward_flat(tiles, glob, torch.ones(4, 11, 11, dtype=torch.bool), bi, actions[bi, ai])
    assert torch.allclose(ref.policy_logits[bi, ai], flat, atol=1e-5)
    assert torch.allclose(ref.winner_logits, winner, atol=1e-5)


def test_fast_policy_matches_eager_choices_sampled_and_greedy():
    model = tiny_model()
    eager, fast = pair(model, infer_graphs=True)  # no CUDA here: the padded flat path
    assert fast.fast is not None and not fast.fast.graphs
    obs = fake_obs(lens=(4, 9, 1, 6, 3), size=11, seed=1)
    assert_same_choices(eager.choose(obs, temperature=1.0), fast.choose(obs, temperature=1.0))
    g_eager, g_fast = eager.choose(obs), fast.choose(obs)
    assert_same_choices(g_eager, g_fast)
    assert g_eager.top == g_fast.top or all(
        [i for i, _ in x] == [i for i, _ in y] for x, y in zip(g_eager.top, g_fast.top))
    assert fast.fast.stats["padded_calls"] == 2 and fast.fast.stats["eager_calls"] == 0


def test_chunks_and_odd_row_counts_are_padded_to_a_bucket():
    model = tiny_model()
    eager, fast = pair(model, infer_graphs=True)
    eager.max_batch = fast.max_batch = 4
    obs = fake_obs(lens=tuple(range(1, 12)), size=11, seed=2)  # 11 rows: chunks of 4, 4, 3
    assert_same_choices(eager.choose(obs, temperature=0.8), fast.choose(obs, temperature=0.8))


def test_padded_boards_with_a_board_mask_match():
    model = tiny_model()
    eager, fast = pair(model, infer_graphs=True)
    obs = merge_padded([fake_obs(lens=(3, 5), size=11, seed=1), fake_obs(lens=(4, 2, 6), size=14, seed=2)])
    assert "board_mask" in obs
    assert_same_choices(eager.choose(obs, temperature=1.0), fast.choose(obs, temperature=1.0))


def test_engine_features_are_padded_too():
    model = randomise_features(add_features(tiny(False)))
    obs = feature_batch(b=3, size=11, lens=(4, 7, 2))
    eager, fast = pair(model, infer_graphs=True)
    assert_same_choices(eager.choose(obs, temperature=1.0), fast.choose(obs, temperature=1.0))


def test_oversized_calls_fall_back_to_eager():
    model = tiny_model()
    ff = FastForward(model, CPU, graphs=True, max_actions=20)
    obs = fake_obs(lens=(40, 3), size=11, seed=4)
    tiles, glob, actions, mask = collate(obs, np.arange(2), CPU)
    assert ff(tiles, glob, actions, mask, None, {}) is None and ff.stats["eager_calls"] == 1


def test_networks_of_one_architecture_share_a_runner_and_keep_their_own_weights():
    m1, m2 = tiny_model(1), tiny_model(2)
    e1, e2 = NetPolicy(m1, "e1", CPU, seed=5), NetPolicy(m2, "e2", CPU, seed=5)
    f1, f2 = (NetPolicy(m, "f", CPU, seed=5, infer_graphs=True) for m in (m1, m2))
    assert f1.fast.runner is f2.fast.runner and len(infer_fast._RUNNERS) == 1
    obs = fake_obs(lens=(4, 9, 1, 6), size=11, seed=1)
    for _ in range(2):  # alternating calls: each sees its own weights
        assert_same_choices(e1.choose(obs, temperature=1.0), f1.choose(obs, temperature=1.0))
        assert_same_choices(e2.choose(obs, temperature=1.0), f2.choose(obs, temperature=1.0))
    with torch.no_grad():  # in-place weight updates (the learner after a PPO step) are picked up
        for p in m1.parameters():
            p.mul_(1.1)
    assert_same_choices(e1.choose(obs, temperature=1.0), f1.choose(obs, temperature=1.0))


def test_off_by_default():
    assert NetPolicy(tiny_model(), "x", CPU).fast is None


def test_compiled_forward_matches_on_cpu():
    pytest.importorskip("torch._dynamo")
    model = tiny_model()
    eager, fast = pair(model, infer_compile=True)
    obs = fake_obs(lens=(4, 9, 1), size=11, seed=1)
    try:
        got = fast.choose(obs, temperature=1.0)
    except Exception as e:  # no working C++ toolchain / inductor in this environment
        pytest.skip(f"torch.compile is not usable here: {type(e).__name__}")
    assert_same_choices(eager.choose(obs, temperature=1.0), got, atol=1e-3)


def _free_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        return 0.0


needs_gpu = pytest.mark.skipif(_free_gib() < 1.5, reason="no CUDA device with 1.5 GiB free")


@needs_gpu
@pytest.mark.parametrize("compile_", [False, True])
def test_cuda_graphs_match_eager_within_bf16_and_follow_weight_updates(compile_):
    dev = torch.device("cuda")
    model = tiny_model().to(dev).to(memory_format=torch.channels_last)
    eager = NetPolicy(model, "eager", dev, seed=5)
    fast = NetPolicy(model, "fast", dev, seed=5, infer_graphs=True, infer_compile=compile_)
    obs = merge_padded([fake_obs(lens=(3, 5, 40, 7), size=11, seed=1), fake_obs(lens=(4, 2, 6), size=14, seed=2)])
    a, b = eager.choose(obs), fast.choose(obs)
    # bf16 autocast: the logits agree to about 1e-2; the greedy move may differ only when two moves are that close
    np.testing.assert_allclose(a.logp, b.logp, atol=0.1)
    np.testing.assert_allclose(a.value, b.value, atol=0.03)
    same = (a.index == b.index).mean()
    assert same >= 0.8, same
    b2 = fast.choose(obs)  # second call replays the captured graph
    np.testing.assert_allclose(b.logp, b2.logp, atol=1e-6)
    assert fast.fast.stats["graph_captures"] >= 1, fast.fast.last_error
    assert fast.fast.stats["graph_replays"] >= 2
    # the graph reads the live weights: a step on the network changes what the next replay returns
    with torch.no_grad():
        for p in model.policy_mlp.parameters():
            p.add_(0.2)
    c, d = eager.choose(obs), fast.choose(obs)
    np.testing.assert_allclose(c.logp, d.logp, atol=0.1)
    assert np.abs(d.logp - b2.logp).max() > 1e-3
