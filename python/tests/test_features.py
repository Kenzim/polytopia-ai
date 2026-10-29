"""Engine features as optional model inputs (encoding.md section 12): shard keys, the zero-initialised inputs of the
model, plumbing through observations, self-play and the arena, and that a model that reads them can use them."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.arena import ModelAgent, Suite, play_pairing
from polyai.data import BatchLoader, Shard, make_synthetic_shard, save_shard, transform_tiles
from polyai.h2h import play_h2h
from polyai.model import ModelConfig, add_features, build_model, feature_inputs
from polyai.policy import ModelPolicy, collate, collate_features
from polyai.rl import Collector, Pool, RLConfig, ppo_update, train
from polyai.rl_pool import EntryInfo, NetPolicy, merge, merge_padded, plan_quotas, take
from polyai.train_bc import TrainConfig, load_checkpoint, load_init, make_optimizer, save_checkpoint
from polyai.train_bc import train as train_bc

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")
CPU = torch.device("cpu")
TINY = dict(blocks=3, channels=24, policy_hidden=32, value_hidden=32, name="tiny")


def tiny(features=False, seed=0):
    torch.manual_seed(seed)
    m = build_model(ModelConfig(**TINY, features=features))
    for n, p in m.named_parameters():
        if p.abs().sum() == 0 and not n.startswith("feat_"):
            torch.nn.init.normal_(p, std=0.1)  # non-zero heads, so that outputs differ from the neutral start
    return m.eval()


def randomise_features(model, std=0.3):
    for m in (model.feat_tile_conv, model.feat_global_proj, model.feat_action_proj):
        torch.nn.init.normal_(m.weight, std=std)
    return model


def feature_batch(b=3, size=11, lens=(4, 7, 2), seed=0):
    rng = np.random.default_rng(seed)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    obs = {
        "tiles": rng.integers(0, 3, (b, size, size, S.F_TILE)).astype(np.int16),
        "globals": rng.normal(size=(b, S.G_DIM)).astype(np.float32),
        "act_off": off,
        "actions": np.stack([rng.integers(1, 26, off[-1]), rng.integers(0, 5, off[-1]),
                             rng.integers(0, size * size, off[-1]), rng.integers(0, size * size, off[-1])], 1).astype(np.int16),
        "f_tiles": rng.integers(0, 5, (b, size, size, S.FEAT_T)).astype(np.int16),
        "f_globals": rng.normal(size=(b, S.FEAT_G)).astype(np.float32),
        "f_actions": rng.random((off[-1], S.FEAT_A)).astype(np.float32),
    }
    return obs


def run_model(model, obs, with_features=True, board_mask=None):
    sel = np.arange(len(obs["act_off"]) - 1)
    tiles, glob, actions, mask = collate(obs, sel, CPU)
    kw = feature_inputs(collate_features(obs, sel, CPU), model) if with_features else {}
    with torch.no_grad():
        return model(tiles, glob, actions, mask, board_mask, **kw)


# ---- schema -----------------------------------------------------------------------------------------------------


@needs_engine
def test_schema_matches_the_engine_module():
    import polyai_engine as pe

    assert (S.FEAT_T, S.FEAT_G, S.FEAT_A) == (pe.FEAT_T, pe.FEAT_G, pe.FEAT_A)
    assert S.FEAT_T_NAMES == list(pe.FEAT_T_NAMES) and S.FEAT_T_DIVISORS == list(pe.FEAT_T_DIVISORS)


# ---- the model --------------------------------------------------------------------------------------------------


def test_a_model_with_features_starts_exactly_equal_to_the_plain_one():
    plain = tiny(False)
    withf = add_features(plain)
    assert withf.cfg.features and not plain.cfg.features
    obs = feature_batch()
    a, b = run_model(plain, obs, with_features=False), run_model(withf, obs)
    assert torch.equal(a.policy_logits, b.policy_logits) and torch.equal(a.winner_logits, b.winner_logits)
    # also on a padded board with a mask
    bm = torch.zeros(3, 11, 11, dtype=torch.bool)
    bm[:, :8, :9] = True
    a, b = run_model(plain, obs, False, bm), run_model(withf, obs, True, bm)
    assert torch.equal(a.policy_logits, b.policy_logits) and torch.equal(a.winner_logits, b.winner_logits)


def test_trained_feature_inputs_change_the_outputs_and_get_gradients_from_the_start():
    obs = feature_batch()
    model = add_features(tiny(False)).train()
    sel = np.arange(3)
    tiles, glob, actions, mask = collate(obs, sel, CPU)
    out = model(tiles, glob, actions, mask, **feature_inputs(collate_features(obs, sel, CPU), model))
    (out.policy_logits.masked_fill(~mask, 0).pow(2).sum() + out.winner_logits.pow(2).sum()).backward()
    # the head weights are not zero, so every zero-initialised input receives a gradient on the first step
    for name in ("feat_tile_conv", "feat_global_proj", "feat_action_proj"):
        assert getattr(model, name).weight.grad.abs().sum() > 0, name
    base = run_model(tiny(False), obs, False)
    moved = run_model(randomise_features(add_features(tiny(False))), obs)
    assert not torch.allclose(base.policy_logits, moved.policy_logits)
    assert not torch.allclose(base.winner_logits, moved.winner_logits)


def test_features_do_nothing_for_a_model_without_them_and_are_required_by_one_with_them():
    obs = feature_batch()
    plain = tiny(False)
    sel = np.arange(3)
    assert feature_inputs(collate_features(obs, sel, CPU), plain) == {}
    without = {k: v for k, v in obs.items() if not k.startswith("f_")}
    assert collate_features(without, sel, CPU) == {}
    with pytest.raises(ValueError, match="engine features"):
        feature_inputs({}, add_features(plain))


def test_the_config_key_is_absent_unless_set_and_old_checkpoints_load_either_way(tmp_path):
    assert "features" not in build_model("tiny").cfg.to_dict() and "features" not in tiny(False).cfg.to_dict()
    assert tiny(True).cfg.to_dict()["features"] is True
    plain = tiny(False)
    save_checkpoint(tmp_path / "plain.pt", plain, 3)
    assert not load_checkpoint(tmp_path / "plain.pt").cfg.features
    withf = randomise_features(add_features(plain))
    save_checkpoint(tmp_path / "f.pt", withf, 4)
    back = load_checkpoint(tmp_path / "f.pt")
    assert back.cfg.features
    obs = feature_batch()
    assert torch.equal(run_model(withf, obs).policy_logits, run_model(back.eval(), obs).policy_logits)


def test_init_from_a_plain_checkpoint_gives_the_same_function(tmp_path):
    plain = tiny(False, seed=4)
    save_checkpoint(tmp_path / "plain.pt", plain, 3)
    withf = tiny(True, seed=9)
    load_init(withf, tmp_path / "plain.pt")
    obs = feature_batch()
    assert torch.equal(run_model(plain, obs, False).policy_logits, run_model(withf, obs).policy_logits)
    # the other way round would drop trained weights: refused
    save_checkpoint(tmp_path / "f.pt", withf, 3)
    with pytest.raises(SystemExit, match="do not fit"):
        load_init(tiny(False), tmp_path / "f.pt")


# ---- shards -----------------------------------------------------------------------------------------------------


def feature_shard(n=40, size=11, seed=1):
    return make_synthetic_shard(n, size, seed, max_actions=20, with_features=True)


def test_feature_keys_roundtrip_and_stay_aligned_under_the_symmetries(tmp_path):
    arrays = feature_shard()
    # tie one feature column to a tile field and one action column to the action's kind: both must survive any symmetry
    arrays["f_tiles"][..., 0] = arrays["tiles"][..., S.COLUMN["climate"]]
    arrays["f_actions"][:, 1] = arrays["actions"][:, 0]
    save_shard(tmp_path / "s.npz", arrays)
    shard = Shard.load(tmp_path / "s.npz")
    assert shard.has_features
    plain = shard.batch(np.arange(40))
    assert plain["f_tiles"].shape == (40, 11, 11, S.FEAT_T) and plain["f_globals"].shape == (40, S.FEAT_G)
    amax = plain["actions"].shape[1]
    assert plain["f_actions"].shape == (40, amax, S.FEAT_A)
    assert (plain["f_actions"][~plain["action_mask"]] == 0).all()
    for seed in range(4):
        b = shard.batch(np.arange(40), np.random.default_rng(seed), augment=True)
        np.testing.assert_array_equal(b["f_tiles"][..., 0], b["tiles"][..., S.COLUMN["climate"]])
        real = b["action_mask"]
        np.testing.assert_array_equal(b["f_actions"][..., 1][real], b["actions"][..., 0][real])
    # a symmetric position keeps its per-tile features where the tiles go
    np.testing.assert_array_equal(transform_tiles(plain["f_tiles"], 3)[..., 0], transform_tiles(plain["tiles"], 3)[..., S.COLUMN["climate"]])


def test_shards_without_features_are_unchanged(shard16):
    assert not shard16.has_features
    assert not any(k.startswith("f_") for k in shard16.batch(np.arange(4)))
    assert not any(k.startswith("f_") for k in make_synthetic_shard(8, 11))


def test_synthetic_shard_without_features_is_not_changed_by_the_option():
    a, b = make_synthetic_shard(16, 11, 5), make_synthetic_shard(16, 11, 5, with_features=True)
    assert np.array_equal(a["tiles"], b["tiles"]) and np.array_equal(a["globals"], b["globals"])


def test_a_partial_or_misshapen_feature_set_is_refused():
    a = feature_shard(8)
    del a["f_actions"]
    with pytest.raises(ValueError, match="all of"):
        Shard(a)
    a = feature_shard(8)
    a["f_globals"] = a["f_globals"][:, :5]
    with pytest.raises(ValueError, match="f_globals has shape"):
        Shard(a)


def test_loader_yields_features_on_the_device(tmp_path):
    save_shard(tmp_path / "s.npz", feature_shard(32))
    b = next(BatchLoader([tmp_path / "s.npz"], 16, prefetch=0).iterate(epochs=1))
    assert b["f_tiles"].dtype == torch.int16 and b["f_globals"].dtype == torch.float32
    assert b["f_actions"].dtype == torch.float32 and b["f_actions"].shape[:2] == b["actions"].shape[:2]


# ---- training on shards -----------------------------------------------------------------------------------------


def bc_cfg(tmp_path, name, shards, val, **kw):
    base = dict(name=name, model="tiny", train=[str(p) for p in shards], val=[str(p) for p in val], batch_size=32,
                steps=250, lr=3e-3, warmup_steps=10, eval_every=250, ckpt_every=1000, eval_batches=8, log_every=50,
                threads=2, runs_dir=str(tmp_path), device="cpu", bf16=False, augment=False)
    base.update(kw)
    return TrainConfig(**base)


def test_bc_with_features_learns_what_only_the_features_tell(tmp_path):
    # the synthetic label depends on the first action feature: a model that reads it should fit much better
    train_p = [tmp_path / "tr.npz"]
    val_p = [tmp_path / "va.npz"]
    save_shard(train_p[0], make_synthetic_shard(600, 11, 3, max_actions=20, with_features=True))
    save_shard(val_p[0], make_synthetic_shard(128, 11, 4, max_actions=20, with_features=True))
    plain = train_bc(bc_cfg(tmp_path, "plain", train_p, val_p))
    withf = train_bc(bc_cfg(tmp_path, "withf", train_p, val_p, features=True))
    ce_plain = plain["last"]["val"]["val_policy_ce"]
    ce_feat = withf["last"]["val"]["val_policy_ce"]
    assert ce_feat < ce_plain - 0.1, (ce_feat, ce_plain)
    cfg = json.loads((tmp_path / "withf" / "config.json").read_text())
    assert cfg["model"]["features"] is True and "features" not in json.loads((tmp_path / "plain" / "config.json").read_text())["model"]
    assert load_checkpoint(tmp_path / "withf" / "ckpt_final.pt").cfg.features


def test_bc_features_needs_shards_with_features(tmp_path):
    p = tmp_path / "s.npz"
    save_shard(p, make_synthetic_shard(32, 11, 1, max_actions=20))
    with pytest.raises(SystemExit, match="polyai-gen --features"):
        train_bc(bc_cfg(tmp_path, "x", [p], [], features=True, steps=1))


def test_bc_fine_tune_from_a_plain_checkpoint_starts_from_its_function(tmp_path):
    save_shard(tmp_path / "s.npz", feature_shard(64))
    plain = tiny(False, seed=2)
    save_checkpoint(tmp_path / "plain.pt", plain, 5)
    train_bc(bc_cfg(tmp_path, "ft", [tmp_path / "s.npz"], [], features=True, steps=0, init=str(tmp_path / "plain.pt")))
    ft = load_checkpoint(tmp_path / "ft" / "ckpt_final.pt")
    obs = feature_batch()
    assert torch.equal(run_model(plain, obs, False).policy_logits, run_model(ft.eval(), obs).policy_logits)


# ---- observations of the real engine ----------------------------------------------------------------------------


@needs_engine
def test_vecenv_observations_carry_aligned_features_and_survive_take_and_merge():
    import polyai_engine as pe

    env = pe.VecEnv(6, 1, "all", split="train", n_players=[2], sizes=[1, 2], threads=1, features=True)
    groups = env.observe()
    assert groups
    for g in groups:
        n = len(g["act_off"]) - 1
        assert g["f_tiles"].shape == (n, *g["tiles"].shape[1:3], S.FEAT_T) and g["f_globals"].shape == (n, S.FEAT_G)
        assert g["f_actions"].shape == (len(g["actions"]), S.FEAT_A)
    plain = pe.VecEnv(6, 1, "all", split="train", n_players=[2], sizes=[1, 2], threads=1).observe()
    assert all("f_tiles" not in g for g in plain)
    # the same games give the same observations with and without the option
    for g, p in zip(groups, plain):
        np.testing.assert_array_equal(g["tiles"], p["tiles"])
        np.testing.assert_array_equal(g["actions"], p["actions"])

    g = groups[0]
    n = len(g["act_off"]) - 1
    sel = np.arange(n)[::-1][: max(1, n // 2)]
    sub = take(g, sel)
    for i, j in enumerate(sel):
        a, b = g["act_off"][j], g["act_off"][j + 1]
        np.testing.assert_array_equal(sub["f_actions"][sub["act_off"][i]:sub["act_off"][i + 1]], g["f_actions"][a:b])
    np.testing.assert_array_equal(sub["f_tiles"], g["f_tiles"][sel])
    both = merge([g, g])
    assert both["f_actions"].shape[0] == 2 * g["f_actions"].shape[0] and both["f_tiles"].shape[0] == 2 * n
    # padded merge of two board sizes: features are padded like the tiles and the action rows stay aligned
    if len(groups) > 1:
        m = merge_padded(groups[:2])
        assert m["f_tiles"].shape[:3] == m["tiles"].shape[:3] and m["f_actions"].shape[0] == m["actions"].shape[0]
        assert not m["f_tiles"][~m["board_mask"]].any()


@needs_engine
def test_a_features_policy_needs_features_in_the_observation(tmp_path):
    import polyai_engine as pe

    (tmp_path / "f").mkdir()
    save_checkpoint(tmp_path / "f" / "ckpt_final.pt", randomise_features(add_features(tiny(False))), 2)
    pol = ModelPolicy(tmp_path / "f", CPU)
    assert pol.features
    with_f = pe.VecEnv(4, 1, "all", split="train", n_players=[2], sizes=[1], threads=1, features=True).observe()[0]
    without = pe.VecEnv(4, 1, "all", split="train", n_players=[2], sizes=[1], threads=1).observe()[0]
    ch = pol.choose(with_f)
    assert len(ch) == len(with_f["act_off"]) - 1 and np.isfinite(ch.logp).all()
    with pytest.raises(ValueError, match="features=True"):
        pol.choose(without)


# ---- self-play and the arena ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("featrl") / "plain"
    run.mkdir()
    save_checkpoint(run / "ckpt_final.pt", tiny(False, seed=6), 7)
    return run


def rl_cfg(tmp_path, run, **kw):
    base = dict(name="t", init=str(run), runs_dir=str(tmp_path), updates=2, games=8, rollout_steps=120, sizes="tiny",
                device="cpu", threads=3, torch_threads=1, minibatch=64, critic_warmup=1, snapshot_every=1, ckpt_every=1,
                max_snapshots=2, bf16=False, features=True)
    base.update(kw)
    return RLConfig(**base)


@needs_engine
def test_rollouts_with_features_store_them_and_the_behaviour_logp_matches(tmp_path, plain_run):
    import polyai_engine as pe

    cfg = rl_cfg(tmp_path, plain_run, games=10, rollout_steps=150, sizes="tiny,small")
    model = randomise_features(add_features(tiny(False, seed=6)))
    ref = tiny(False, seed=3)  # a plain reference and plain opponents play in the same observations
    pool = Pool(cfg, NetPolicy(ref, "ref@7", CPU, seed=1), EntryInfo("bc", "ref@7"))
    pool.add_snapshot("self:u00001", NetPolicy(randomise_features(add_features(tiny(False, seed=9))), "t@1", CPU, seed=2),
                      EntryInfo("self:u00001", "t@1"))
    ex = ThreadPoolExecutor(max_workers=2)
    col = Collector(cfg, pe, NetPolicy(model, "t@0", CPU), pool, np.random.default_rng(0), ex)
    try:
        col.set_quotas(plan_quotas(pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        ex.shutdown()
    assert col.n_learner >= cfg.rollout_steps and col.n_opp > 0
    for st in col.stores.values():
        d = st.finalize()
        assert d["f_tiles"].shape[0] == st.n and d["f_actions"].shape[0] == d["act_off"][-1]
        sel = d["train"][:32]
        tiles, glob, actions, mask = collate(d, sel, CPU)
        with torch.no_grad():
            out = model(tiles, glob, actions, mask, **collate_features(d, sel, CPU))
        lp = torch.log_softmax(out.policy_logits.float(), 1).gather(1, torch.from_numpy(d["chosen"][sel]).long()[:, None])
        np.testing.assert_allclose(lp[:, 0].numpy(), d["logp"][sel], atol=1e-4)
        # a PPO step moves the feature inputs (and the plain reference still works for the KL term)
        before = model.feat_global_proj.weight.detach().clone()  # f_actions can be all zero this early in a game
        cfg2 = rl_cfg(tmp_path, plain_run, lr=1e-2, critic_warmup=0)
        res = ppo_update(model, ref, make_optimizer(model, cfg2), col.stores, cfg2, 0, CPU, np.random.default_rng(0))
        assert res["minibatches"] > 0 and np.isfinite(res["loss"])
        assert not torch.equal(before, model.feat_global_proj.weight)
        break


@needs_engine
def test_a_run_with_features_starts_from_the_plain_checkpoint_and_writes_feature_checkpoints(tmp_path, plain_run):
    summary = train(rl_cfg(tmp_path, plain_run))
    run = tmp_path / "t"
    assert summary["updates"] == 2
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["model"]["features"] is True and cfg["train"]["features"] is True
    final = load_checkpoint(run / "ckpt_final.pt")
    assert final.cfg.features and load_checkpoint(run / "pool" / "snap_u00001.pt").cfg.features
    # resume keeps working with the feature model
    assert train(rl_cfg(tmp_path, plain_run, updates=3, resume=True))["updates"] == 3


@needs_engine
def test_arena_and_h2h_give_a_features_model_its_observations(tmp_path):
    f_dir, p_dir = tmp_path / "f", tmp_path / "p"
    f_dir.mkdir()
    p_dir.mkdir()
    save_checkpoint(f_dir / "ckpt_final.pt", randomise_features(add_features(tiny(False, seed=1))), 2)
    save_checkpoint(p_dir / "ckpt_final.pt", tiny(False, seed=2), 2)
    suite = Suite(seeds=3, sizes=[1], presets=[3], turn_limit=12, batch_games=2, threads=4)
    fa = ModelAgent(f_dir / "ckpt_final.pt", None, CPU)
    pa = ModelAgent(p_dir / "ckpt_final.pt", None, CPU)
    assert fa.features and not pa.features
    records = play_pairing(fa, "retail:easy", suite, log=lambda m: None)
    assert len(records) == 6 and all(r["illegal"] == [0, 0] for r in records)
    records = play_h2h(fa, pa, suite, set(), log=lambda m: None)
    assert len(records) == 6
    # a plain model in the same arena does not pay for features
    assert len(play_pairing(pa, "retail:easy", suite, log=lambda m: None)) == 6
