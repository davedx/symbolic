"""Training-stack tests: model invariants, schedule, stream determinism, resume
equivalence, budget accounting, controls, evaluation paths."""

import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from train.config import RunConfig
from train.data import TrainStream, answer_kind, train_queries
from train.model import SIZES, ModelConfig, Transformer, count_params, padding_mask
from train.trainer import Trainer, lr_at

ROOT = Path(__file__).resolve().parents[1]


def smoke_cfg(**over) -> RunConfig:
    cfg = RunConfig.from_yaml(ROOT / "configs/runs/smoke.yaml")
    cfg.world = str(ROOT / "configs/world/tiny.yaml")
    cfg.train.budget = 30_000
    cfg.train.n_evals = 2
    cfg.train.eval_max_per_split = 50
    cfg.train.final_eval_max_per_split = 50
    cfg.train.ckpt_every = 10_000
    for k, v in over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    return cfg


# ------------------------------------------------------------------------------ model
@pytest.mark.parametrize("size,target", [("S", 2e6), ("M", 8e6), ("L", 20e6)])
def test_param_counts_near_targets(size, target):
    m = Transformer(ModelConfig(vocab_size=1250, size=size))
    assert abs(count_params(m) - target) / target < 0.1


def test_named_sizes_in_spec_ranges():
    for name in ("S", "M", "L"):
        s = SIZES[name]
        assert 4 <= s["n_layers"] <= 6 and 4 <= s["n_heads"] <= 8


def test_forward_shapes_and_hidden():
    m = Transformer(ModelConfig(vocab_size=37, size="XS"))
    x = mx.array(np.random.default_rng(0).integers(0, 37, (3, 11)))
    logits, hidden = m(x, return_hidden=True)
    assert logits.shape == (3, 11, 37)
    assert len(hidden) == m.cfg.n_layers + 1 and hidden[-1].shape == (3, 11, m.cfg.d_model)


def test_causality():
    m = Transformer(ModelConfig(vocab_size=37, size="XS"))
    a = mx.array([[1, 2, 3, 4, 5, 6]])
    b = mx.array([[1, 2, 3, 9, 9, 9]])
    assert mx.allclose(m(a)[:, :3], m(b)[:, :3], atol=1e-5)


@pytest.mark.parametrize("device,atol", [("cpu", 1e-5), ("gpu", 1e-3)])
def test_left_padding_invariance(device, atol):
    # CPU checks the masking logic tightly. The GPU attention/matmul kernels differ
    # between the padded and unpadded shapes; on the M5 that gives fp32 noise up to
    # ~2e-4 (median 0). A masking bug would show differences around 1e-1.
    with mx.stream(getattr(mx, device)):
        mx.random.seed(0)
        m = Transformer(ModelConfig(vocab_size=37, size="XS"))
        x = mx.array([[5, 6, 7, 8]])
        xp = mx.array([[0, 0, 0, 5, 6, 7, 8]])
        pad = mx.array([[True, True, True, False, False, False, False]])
        assert mx.allclose(m(x), m(xp, mask=padding_mask(pad))[:, 3:], atol=atol)


def test_weight_tying():
    m = Transformer(ModelConfig(vocab_size=37, size="XS"))
    names = [k for k, _ in tree_flatten(m.parameters())]
    assert "embed.weight" in names
    assert not any("lm_head" in n or "unembed" in n for n in names)


def test_lr_schedule():
    tc = RunConfig().train
    assert lr_at(0.0, tc) == 0.0
    assert lr_at(tc.warmup_frac, tc) == pytest.approx(tc.lr)
    assert lr_at(1.0, tc) == pytest.approx(tc.lr * tc.min_lr_ratio)
    vals = [lr_at(p, tc) for p in np.linspace(tc.warmup_frac, 1, 50)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------------------- stream
def test_stream_deterministic_and_resumable(tiny):
    from encoders.dataset import make_encoder

    cfg = smoke_cfg()
    enc = make_encoder("prose", tiny.world)
    qs = train_queries(tiny, cfg.train)

    def take(stream, n):
        return [np.array(stream.next_batch().inputs).tobytes() for _ in range(n)]

    s1 = TrainStream(enc, qs, cfg.data, cfg.train)
    full = take(s1, 30)
    s2 = TrainStream(enc, qs, cfg.data, cfg.train)
    first = take(s2, 12)
    saved = s2.state
    s3 = TrainStream(enc, qs, cfg.data, cfg.train)
    s3.state = type(saved)(**vars(saved))
    assert first + take(s3, 18) == full


def test_stage2_mixes_kb_and_queries(tiny):
    from encoders.dataset import make_encoder

    cfg = smoke_cfg()
    s = TrainStream(make_encoder("symbolic", tiny.world), train_queries(tiny, cfg.train), cfg.data, cfg.train)
    assert s.state.stage == 1
    b = s.next_batch()
    assert b.n_kb_facts > 0 and b.n_queries == 0
    s.end_stage1()
    kb, q = 0, 0
    for _ in range(len(s.plan())):
        b = s.next_batch()
        kb += b.n_kb_facts
        q += b.n_queries
    assert q == len(train_queries(tiny, cfg.train))
    assert kb == len(tiny.world.facts)  # kb_per_query_epoch = 1.0


def test_context_regime_has_no_kb_docs(tiny):
    from encoders.dataset import make_encoder

    cfg = smoke_cfg(data__regime="context")
    s = TrainStream(make_encoder("triples", tiny.world), train_queries(tiny, cfg.train), cfg.data, cfg.train)
    assert s.state.stage == 2
    for _ in range(5):
        b = s.next_batch()
        assert b.n_kb_facts == 0 and b.n_ctx_facts > 0


def test_label_shuffle_preserves_marginals(tiny):
    cfg = smoke_cfg(train__label_shuffle=True)
    orig = train_queries(tiny, smoke_cfg().train)
    shuf = train_queries(tiny, cfg.train)
    assert [q.qid for q in orig] == [q.qid for q in shuf]
    for key in {(q.form, answer_kind(q)) for q in orig}:
        a = Counter(str(q.answer) for q in orig if (q.form, answer_kind(q)) == key)
        b = Counter(str(q.answer) for q in shuf if (q.form, answer_kind(q)) == key)
        assert a == b
    changed = sum(o.answer != s.answer for o, s in zip(orig, shuf))
    assert changed > 0.3 * len(orig)


def test_train_subset(tiny):
    qs = train_queries(tiny, smoke_cfg(train__train_subset=40).train)
    assert len(qs) == 40 and all(q.split == "train" for q in qs)


# ---------------------------------------------------------------------------- trainer
def _params(trainer):
    return {k: np.array(v) for k, v in tree_flatten(trainer.model.parameters())}


def test_resume_matches_uninterrupted(tmp_path):
    cfg = smoke_cfg(train__ckpt_every=7, train__n_evals=0)
    a = Trainer(cfg, runs_dir=tmp_path / "a")
    a.train(verbose=False)
    pa = _params(a)

    # interrupted run: stop after 7 steps (one checkpoint), then resume in a fresh Trainer
    b = Trainer(cfg, runs_dir=tmp_path / "b")
    orig_next = b.stream.next_batch
    calls = {"n": 0}

    def limited():
        if calls["n"] == 9:  # 2 steps past the checkpoint at step 7: those must be redone
            raise KeyboardInterrupt
        calls["n"] += 1
        return orig_next()

    b.stream.next_batch = limited
    with pytest.raises(KeyboardInterrupt):
        b.train(verbose=False)
    c = Trainer(cfg, runs_dir=tmp_path / "b")
    c.train(verbose=False)
    pc = _params(c)
    assert a.c.step == c.c.step and a.c.tokens == c.c.tokens
    for k in pa:
        np.testing.assert_allclose(pa[k], pc[k], atol=1e-5, err_msg=k)
    steps = [json.loads(l)["step"] for l in (tmp_path / "b" / c.run_dir.name / "log.jsonl").read_text().splitlines()]
    assert steps == sorted(set(steps))  # no duplicated log rows after resume


def test_budget_accounting_and_outputs(tmp_path):
    cfg = smoke_cfg()
    t = Trainer(cfg, runs_dir=tmp_path)
    res = t.train(verbose=False)
    assert t.c.tokens >= cfg.train.budget
    rd = t.run_dir
    for name in ("config.yaml", "meta.json", "log.jsonl", "eval.jsonl", "results.json", "predictions.jsonl"):
        assert (rd / name).exists(), name
    meta = json.loads((rd / "meta.json").read_text())
    assert meta["seed"] == cfg.train.seed and "git_hash" in meta
    rows = [json.loads(l) for l in (rd / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["tokens"] <= t.c.tokens
    assert all(k in rows[0] for k in ("loss", "tokens", "kb_facts_seen", "queries_seen", "wall_time"))
    assert "test_length" in res["accuracy"]
    assert 0.0 <= res["kb_recall"]["all"]["acc"] <= 1.0 and res["kb_recall"]["all"]["n"] > 0
    assert not any(k == "prop" for k in res["kb_recall"])  # boolean statements are excluded
    # rerunning a finished run is a no-op
    assert Trainer(cfg, runs_dir=tmp_path).train(verbose=False) == res


def test_items_budget(tmp_path):
    cfg = smoke_cfg(train__budget_kind="items", train__budget=600)
    t = Trainer(cfg, runs_dir=tmp_path)
    t.train(verbose=False)
    assert t.c.items >= 600
    assert t.c.items - 600 < 600  # stops within roughly one batch


def test_lr_takes_effect_in_compiled_step(tmp_path):
    cfg = smoke_cfg(train__lr=0.0, train__weight_decay=0.0, train__n_evals=0)
    t = Trainer(cfg, runs_dir=tmp_path)
    before = _params(t)
    t.train(verbose=False)
    after = _params(t)
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


def test_proof_before_generation_runs(tmp_path):
    cfg = smoke_cfg(data__proof="before", train__n_evals=0)
    res = Trainer(cfg, runs_dir=tmp_path).train(verbose=False)
    assert "train" in res["accuracy"]


def test_eval_single_step_equals_generation_for_answer_first(tiny):
    """For proof=none, the single-forward-pass prediction equals greedy decoding's first token."""
    from encoders.dataset import DataConfig, make_encoder
    from train.evaluate import Evaluator

    enc = make_encoder("symbolic", tiny.world)
    mx.random.seed(0)
    m = Transformer(ModelConfig(vocab_size=len(enc.vocab), size="XS"))
    ev = Evaluator(enc, DataConfig(encoding="symbolic"), batch_size=16)
    qs = tiny.queries[:40]
    single = ev.predict(m, qs)
    for r, q in zip(single, qs):
        ids, _ = ev._prompt(q)
        logits = m(mx.array([ids]))
        assert enc.vocab.decode([int(mx.argmax(logits[0, -1]))]) == r["pred"]
