import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.data import Shard, make_synthetic_shard, to_torch
from polyai.model import CONFIGS, MASK_FILL, build_model, eval_pawns, win_probability


def batch_of(size: int, n: int = 4, seed: int = 0, max_actions: int = 30) -> dict:
    shard = Shard(make_synthetic_shard(n, size, seed, max_actions=max_actions))
    return to_torch(shard.batch(np.arange(n)))


def run(model, b, **kw):
    return model(b["tiles"], b["globals"], b["actions"], b["action_mask"], **kw)


@pytest.mark.parametrize("size", [11, 14, 16, 18, 20, 30])
def test_output_shapes_on_every_map_size(size):
    model = build_model("tiny").eval()
    b = batch_of(size, n=3)
    out = run(model, b)
    a = b["actions"].shape[1]
    assert out.policy_logits.shape == (3, a)
    assert out.winner_logits.shape == (3, S.N_SLOTS)
    assert len(out) == 2  # policy logits and the winner head, nothing else
    assert torch.isfinite(out.winner_logits).all()


def test_parameter_counts_are_in_the_designed_range():
    counts = {name: build_model(name).n_params() / 1e6 for name in ("b6c96", "b10c128", "b15c192")}
    assert 0.8 < counts["b6c96"] < 1.5, counts
    assert 2.5 < counts["b10c128"] < 4.0, counts
    assert 8.0 < counts["b15c192"] < 12.0, counts


def test_initial_policy_is_uniform_and_values_neutral():
    model = build_model("tiny").eval()
    b = batch_of(11)
    out = run(model, b)
    p = torch.softmax(out.policy_logits, -1)
    n = b["action_mask"].sum(1)
    for i in range(len(n)):
        assert torch.allclose(p[i, : n[i]], torch.full((int(n[i]),), 1.0 / float(n[i])), atol=1e-5)
    assert torch.allclose(win_probability(out), torch.full((4,), 1.0 / S.N_SLOTS), atol=1e-5)


def trained_ish_model():
    """Random non-zero heads, so that masking and invariance tests are not trivially satisfied."""
    torch.manual_seed(0)
    model = build_model("tiny")
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    # populate BatchNorm running statistics
    model.train()
    for s in range(3):
        run(model, batch_of(16, n=8, seed=s))
    return model.eval()


def test_padded_actions_are_masked_and_do_not_change_real_logits():
    model = trained_ish_model()
    b = batch_of(16, n=4)
    out = run(model, b)
    assert (out.policy_logits[~b["action_mask"]] == MASK_FILL).all()
    assert (out.policy_logits[b["action_mask"]] > MASK_FILL / 2).all()
    p = torch.softmax(out.policy_logits, -1)
    assert (p[~b["action_mask"]] == 0).all()
    np.testing.assert_allclose(p.sum(-1).detach().numpy(), 1.0, atol=1e-5)

    # extend the legal lists with 17 more padding rows: real logits must be identical
    pad = torch.zeros(4, 17, 4, dtype=b["actions"].dtype)
    pad[..., 2:] = -1
    actions = torch.cat([b["actions"], pad], 1)
    mask = torch.cat([b["action_mask"], torch.zeros(4, 17, dtype=torch.bool)], 1)
    out2 = model(b["tiles"], b["globals"], actions, mask)
    a = b["actions"].shape[1]
    torch.testing.assert_close(out2.policy_logits[:, :a], out.policy_logits)


def test_garbage_in_padded_action_rows_is_ignored():
    model = trained_ish_model()
    b = batch_of(16, n=4)
    out = run(model, b)
    actions = b["actions"].clone()
    actions[~b["action_mask"]] = torch.tensor([7, 3, 5, 9], dtype=actions.dtype)
    out2 = model(b["tiles"], b["globals"], actions, b["action_mask"])
    torch.testing.assert_close(out2.policy_logits, out.policy_logits)


def test_logits_depend_on_the_tiles_an_action_points_at():
    model = trained_ish_model()
    b = batch_of(16, n=2)
    actions = b["actions"].clone()
    actions[:, 0] = torch.tensor([2, 0, 17, 40], dtype=actions.dtype)
    base = model(b["tiles"], b["globals"], actions, b["action_mask"]).policy_logits[:, 0]
    swapped = actions.clone()
    swapped[:, 0, 2] = 200
    other = model(b["tiles"], b["globals"], swapped, b["action_mask"]).policy_logits[:, 0]
    assert not torch.allclose(base, other)


def test_board_mask_makes_padding_irrelevant():
    """An 11x11 position embedded in a 16x16 tensor with a board mask behaves like the 11x11 position alone."""
    model = trained_ish_model()
    small = batch_of(11, n=3, seed=9)
    ref = run(model, small)

    n, h = 3, 16
    tiles = torch.zeros(n, h, h, S.F_TILE, dtype=small["tiles"].dtype)
    tiles[:, :11, :11] = small["tiles"]
    tiles[:, 11:] = torch.randint(0, 3, tiles[:, 11:].shape, dtype=tiles.dtype)  # junk in the padding
    tiles[:, :, 11:] = torch.randint(0, 3, tiles[:, :, 11:].shape, dtype=tiles.dtype)
    board_mask = torch.zeros(n, h, h, dtype=torch.bool)
    board_mask[:, :11, :11] = True
    ys, xs = small["actions"][..., 2] // 11, small["actions"][..., 2] % 11
    yd, xd = small["actions"][..., 3] // 11, small["actions"][..., 3] % 11
    actions = small["actions"].clone()
    actions[..., 2] = torch.where(small["actions"][..., 2] >= 0, ys * h + xs, small["actions"][..., 2])
    actions[..., 3] = torch.where(small["actions"][..., 3] >= 0, yd * h + xd, small["actions"][..., 3])
    out = model(tiles, small["globals"], actions, small["action_mask"], board_mask=board_mask)
    torch.testing.assert_close(out.winner_logits, ref.winner_logits, atol=1e-4, rtol=1e-4)
    real = small["action_mask"]
    torch.testing.assert_close(out.policy_logits[real], ref.policy_logits[real], atol=1e-4, rtol=1e-4)


def test_gradients_flow_to_every_parameter():
    model = build_model("tiny")
    for p in model.parameters():  # heads start at zero, which would hide downstream gradients
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    b = batch_of(11, n=4)
    out = run(model, b)
    loss = out.policy_logits.masked_fill(~b["action_mask"], 0).sum() + out.winner_logits.sum()
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, dead


def test_all_configs_construct():
    for name in CONFIGS:
        assert build_model(name).cfg.name == name


def test_eval_pawns():
    assert float(eval_pawns(0.5)) == pytest.approx(0.0, abs=1e-6)
    assert float(eval_pawns(0.85)) == pytest.approx(4 * np.log10(0.85 / 0.15), abs=1e-5)
    assert 2.9 < float(eval_pawns(0.85)) < 3.1
    assert float(eval_pawns(1.0)) == pytest.approx(9.9)
    assert float(eval_pawns(0.0)) == pytest.approx(-9.9)
    assert float(eval_pawns(0.2)) == pytest.approx(-float(eval_pawns(0.8)), abs=1e-5)
    t = eval_pawns(torch.tensor([0.1, 0.5, 0.9]))
    assert t.shape == (3,) and t[0] < t[1] < t[2]


def test_large_global_values_are_compressed_and_the_rest_are_not():
    """Scores and stars reach the tens in the real data; asinh keeps them from swamping the other inputs."""
    model = build_model("tiny").eval()
    assert int(model.g_large.sum()) == len(S.G_LARGE) == 3 + S.N_OTHER_SLOTS
    assert model.g_large[S.G_SCORE] and model.g_large[S.G_SLOTS + S.SLOT_SCORE]
    assert not model.g_large[S.G_TECHS] and not model.g_large[S.G_TRIGGER]
    b = batch_of(11, n=2)
    b["globals"][:, S.G_SCORE] = 1e6
    assert torch.isfinite(run(model, b).winner_logits).all()
