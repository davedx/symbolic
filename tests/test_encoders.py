"""Encoder tests: round-trips (text <-> tokens <-> ids <-> fact/query), unambiguous parsing,
answer agreement across encodings, vocabulary coverage, example layout, contexts."""

import dataclasses

import numpy as np
import pytest

from encoders.base import ANS, BOS, EOS, PROOF, QUERY, SPECIALS
from encoders.dataset import (
    rename_example,
    ENCODERS,
    ContextSampler,
    DataConfig,
    extract_answer,
    kb_documents,
    make_encoder,
    query_example,
)
from world.closure import Closure

ENC_NAMES = list(ENCODERS)


def _sample(items, n, seed=0):
    if len(items) <= n:
        return list(items)
    idx = np.random.default_rng(seed).choice(len(items), size=n, replace=False)
    return [items[i] for i in sorted(idx)]


@pytest.fixture(scope="module")
def encs_tiny(tiny):
    return {n: make_encoder(n, tiny.world) for n in ENC_NAMES}


@pytest.fixture(scope="module")
def encs_default(default_ds):
    return {n: make_encoder(n, default_ds.world) for n in ENC_NAMES}


@pytest.fixture(params=["tiny", "default"])
def setup(request, tiny, default_ds, encs_tiny, encs_default):
    """(dataset, encoders, facts, queries); exhaustive on tiny, a sample on default."""
    if request.param == "tiny":
        return tiny, encs_tiny, tiny.world.facts, tiny.queries
    return default_ds, encs_default, _sample(default_ds.world.facts, 4000), _sample(default_ds.queries, 4000)


# ---------------------------------------------------------------------- round-trips
@pytest.mark.parametrize("name", ENC_NAMES)
def test_fact_roundtrip_all_variants(setup, name):
    _, encs, facts, _ = setup
    enc = encs[name]
    for f in facts:
        for v in range(enc.n_fact_variants(f)):
            toks = enc.fact(f, v)
            assert enc.tokenize(enc.detok(toks)) == toks
            assert enc.vocab.decode(enc.vocab.encode(toks)) == toks
            assert enc.parse_fact(toks) == f, (f, v, toks)


@pytest.mark.parametrize("name", ENC_NAMES)
def test_query_roundtrip_all_variants(setup, name):
    _, encs, _, queries = setup
    enc = encs[name]
    for q in queries:
        for v in range(enc.n_query_variants(q.form, q.args)):
            toks = enc.query(q.form, q.args, v)
            assert enc.tokenize(enc.detok(toks)) == toks
            assert enc.vocab.decode(enc.vocab.encode(toks)) == toks
            assert enc.parse_query(toks) == (q.form, q.args), (q.qid, v)


def test_all_encodings_agree_on_answers(setup):
    _, encs, _, queries = setup
    for q in queries:
        decoded = {n: e.parse_answer(e.answer(q.answer)) for n, e in encs.items()}
        assert all(a == q.answer for a in decoded.values()), (q.qid, decoded)
        assert all(len(e.answer(q.answer)) == 1 for e in encs.values())


def test_all_encodings_parse_to_same_query(setup):
    _, encs, _, queries = setup
    for q in queries[:500]:
        parsed = {n: e.parse_query(e.query(q.form, q.args)) for n, e in encs.items()}
        assert len(set(parsed.values())) == 1
        surfaces = {n: tuple(e.query(q.form, q.args)) for n, e in encs.items()}
        assert len(set(surfaces.values())) == len(encs)  # genuinely different renderings


def test_variant_counts(tiny, encs_tiny):
    w = tiny.world
    for f in w.facts[:200] + w.facts[-200:]:
        assert encs_tiny["prose"].n_fact_variants(f) == 3
        assert encs_tiny["symbolic"].n_fact_variants(f) == 1
        assert encs_tiny["triples"].n_fact_variants(f) == 1
    for q in tiny.queries[:300]:
        assert encs_tiny["prose"].n_query_variants(q.form, q.args) == 3


def test_prose_variants_distinct(tiny, encs_tiny):
    enc = encs_tiny["prose"]
    for f in tiny.world.facts[:300]:
        assert len({tuple(enc.fact(f, v)) for v in range(3)}) == 3


# ----------------------------------------------------------------------- vocabulary
def test_vocab_covers_corpus_and_has_no_unused_literals(tiny, encs_tiny):
    w = tiny.world
    for name, enc in encs_tiny.items():
        corpus = set()
        for f in w.facts:
            for v in range(enc.n_fact_variants(f)):
                corpus.update(enc.fact(f, v))
        for q in tiny.queries:
            for v in range(enc.n_query_variants(q.form, q.args)):
                corpus.update(enc.query(q.form, q.args, v))
            corpus.update(enc.answer(q.answer))
        vocab = set(enc.vocab.tokens)
        assert corpus <= vocab, corpus - vocab
        literals = {t for t, ty in zip(enc.vocab.tokens, enc.vocab.types) if ty in ("literal", "answer")}
        assert literals <= corpus, literals - corpus  # no dead template words


def test_names_never_collide_with_literals(encs_default, default_ds):
    names = set(default_ds.world.lexicon.values())
    for enc in encs_default.values():
        literals = {t for t, ty in zip(enc.vocab.tokens, enc.vocab.types) if ty in ("literal", "answer", "special")}
        assert not names & literals
        assert len(enc.vocab) == len(names) + len(literals)


def test_vocab_types(encs_default, default_ds):
    w = default_ds.world
    for enc in encs_default.values():
        types = dict(zip(enc.vocab.tokens, enc.vocab.types))
        assert all(types[w.lexicon[e]] == "entity" for e in w.entities)
        assert all(types[w.lexicon[c]] == "category" for c in w.categories)
        assert all(types[s] == "special" for s in SPECIALS)


def test_vocab_save_load(tmp_path, encs_tiny):
    from encoders.base import Vocab

    v = encs_tiny["prose"].vocab
    v.save(tmp_path / "v.json")
    v2 = Vocab.load(tmp_path / "v.json")
    assert v2.tokens == v.tokens and v2.types == v.types


# ------------------------------------------------------------------------- examples
@pytest.mark.parametrize("name", ENC_NAMES)
@pytest.mark.parametrize("regime", ["weights", "context"])
@pytest.mark.parametrize("proof", ["none", "after", "before"])
def test_example_layout(tiny, encs_tiny, name, regime, proof):
    enc = encs_tiny[name]
    cfg = DataConfig(encoding=name, regime=regime, proof=proof)
    sampler = ContextSampler(tiny.world)
    for q in tiny.queries[:200]:
        ex = query_example(enc, q, cfg, ctx_sampler=sampler)
        assert len(ex.tokens) == len(ex.loss_mask)
        assert ex.tokens[0] == BOS and ex.tokens[-1] == EOS
        assert ex.loss_mask == [0] * ex.prompt_len + [1] * (len(ex.tokens) - ex.prompt_len)
        assert ex.tokens[ex.prompt_len - 1] == (PROOF if proof == "before" else ANS)
        assert QUERY in ex.tokens[: ex.prompt_len]
        # the gold continuation yields the gold answer
        got = extract_answer(ex.tokens[ex.prompt_len :], proof)
        assert enc.parse_answer(got) == q.answer
        assert all(t in enc.vocab.index for t in ex.tokens)
        if proof != "none":
            assert PROOF in ex.tokens
        if regime == "weights":
            assert ex.n_facts == 0


def test_proof_target_parses_back(tiny, encs_tiny):
    """With proof as target, the proof tokens are the proof facts in order."""
    for name, enc in encs_tiny.items():
        cfg = DataConfig(encoding=name, proof="after", paraphrase="fixed")
        for q in tiny.queries[:200]:
            ex = query_example(enc, q, cfg)
            proof_toks = ex.tokens[ex.tokens.index(PROOF) + 1 : -1]
            rendered = [t for _, f in q.proof for t in enc.fact(f, 0)]
            assert proof_toks == rendered


def test_examples_deterministic(tiny, encs_tiny):
    enc = encs_tiny["prose"]
    cfg = DataConfig(encoding="prose", regime="context")
    s = ContextSampler(tiny.world)
    for q in tiny.queries[:100]:
        assert query_example(enc, q, cfg, 3, s).tokens == query_example(enc, q, cfg, 3, s).tokens


def test_paraphrase_resampled_across_epochs(tiny, encs_tiny):
    enc = encs_tiny["prose"]
    cfg = DataConfig(encoding="prose")
    variants = {tuple(query_example(enc, tiny.queries[0], cfg, epoch).tokens) for epoch in range(20)}
    assert len(variants) == 3
    fixed = DataConfig(encoding="prose", paraphrase="fixed")
    assert len({tuple(query_example(enc, tiny.queries[0], fixed, e).tokens) for e in range(5)}) == 1


# --------------------------------------------------------------------------- contexts
def test_context_contains_needed_facts_and_is_sufficient(ds):
    """Context = proof facts + overridden default + n distractors, all KB facts; an
    independent reasoner restricted to the context alone derives the same answer."""
    w = ds.world
    sampler = ContextSampler(w)
    cfg = DataConfig(regime="context")
    for q in _sample(ds.queries, 1500):
        ctx = sampler.sample(q, cfg)
        needed = sampler.needed(q)
        assert set(needed) <= set(ctx)
        assert len(ctx) == len(set(ctx)) == len(set(needed)) + cfg.n_distractors
        assert set(ctx) <= w.fact_set
        sub = dataclasses.replace(w, facts=ctx)
        assert Closure(sub).answer(q.form, q.args) == q.answer, q.qid


@pytest.mark.parametrize("rename", ["entities", "nodes"])
def test_renamed_context_is_sufficient_and_consistent(ds, rename):
    """Renaming is a consistent relabelling: the reasoner, given only the renamed context,
    derives the renamed answer; nodes really change; it is deterministic per (qid, epoch)."""
    w = ds.world
    sampler = ContextSampler(w)
    cfg = DataConfig(regime="context", rename=rename)
    changed = n_ent = 0
    for q in _sample(ds.queries, 1500):
        ctx = sampler.sample(q, cfg)
        q2, ctx2 = rename_example(w, q, ctx, cfg, 0)
        assert (q2, ctx2) == rename_example(w, q, ctx, cfg, 0)
        assert len(set(ctx2)) == len(ctx2)
        sub = dataclasses.replace(w, facts=ctx2)
        assert Closure(sub).answer(q2.form, q2.args) == q2.answer, q.qid
        changed += q2.args != q.args
        if q.form == "chain":
            n_ent += 1
            assert q2.answer in w.entities
        cats = lambda fs: sorted(x for f in fs for x in f if isinstance(x, str) and x in w.cat_level)
        if rename == "entities":
            assert cats(ctx2) == cats(ctx)
    assert changed > 0.95 * 1500 and n_ent > 0


def test_rename_off_by_default_and_ignored_in_weights(tiny, encs_tiny):
    enc = encs_tiny["symbolic"]
    q = tiny.queries[0]
    a = query_example(enc, q, DataConfig(regime="context"), 0).tokens
    b = query_example(enc, q, DataConfig(regime="context", rename="nodes"), 0).tokens
    assert a != b
    assert query_example(enc, q, DataConfig(rename="nodes"), 0).tokens == query_example(enc, q, DataConfig(), 0).tokens


def test_context_varies_with_epoch(tiny):
    sampler = ContextSampler(tiny.world)
    cfg = DataConfig(regime="context")
    q = tiny.queries[0]
    assert len({tuple(sampler.sample(q, cfg, e)) for e in range(10)}) > 1


# ------------------------------------------------------------------------- KB documents
@pytest.mark.parametrize("name", ENC_NAMES)
def test_kb_documents_cover_kb_once(tiny, encs_tiny, name):
    enc = encs_tiny[name]
    cfg = DataConfig(encoding=name)
    docs = kb_documents(enc, cfg, epoch=0)
    assert sum(d.n_facts for d in docs) == len(tiny.world.facts)
    # parse the documents back into facts by re-rendering per fact is not possible without
    # boundaries, so check token totals against per-fact renderings instead
    assert all(d.tokens[0] == BOS and d.tokens[-1] == EOS for d in docs)
    assert all(d.loss_mask[0] == 0 and all(d.loss_mask[1:]) for d in docs)
    docs1 = kb_documents(enc, cfg, epoch=1)
    assert [d.tokens for d in docs] != [d.tokens for d in docs1]
    assert [d.tokens for d in kb_documents(enc, cfg, epoch=0)] == [d.tokens for d in docs]


@pytest.mark.parametrize("name", ENC_NAMES)
def test_kb_documents_segment_into_kb_facts(tiny, encs_tiny, name):
    """Greedy segmentation of each document by the parser recovers exactly the KB."""
    enc = encs_tiny[name]
    lengths = sorted({len(t.tokens) for ts in enc.fact_templates.values() for t in ts})
    recovered = []
    for d in kb_documents(enc, DataConfig(encoding=name), epoch=2):
        toks = d.tokens[1:-1]
        i = 0
        while i < len(toks):
            for n in lengths:
                try:
                    recovered.append(enc.parse_fact(toks[i : i + n]))
                    i += n
                    break
                except Exception:
                    continue
            else:
                raise AssertionError(f"cannot segment at {toks[i:i + 8]}")
    assert sorted(map(repr, recovered)) == sorted(map(repr, tiny.world.facts))


# ------------------------------------------------------------- golden renderings (mini)
GOLDEN_FACTS = [
    (("isa", "E0", "C2"), "E0 is a C2.", "(isa E0 C2)", "<s> E0 <r> isa <o> C2"),
    (("sub", "C2", "C1"), "every C2 is a C1.", "(sub C2 C1)", "<s> C2 <r> sub <o> C1"),
    (("prop", "C1", "P0", True), "every C1 is P0.", "(has C1 P0)", "<s> C1 <r> P0 <o> true"),
    (("prop", "C2", "P0", False), "no C2 is P0.", "(not (has C2 P0))", "<s> C2 <r> P0 <o> false"),
    (("prop", "E3", "P0", True), "E3 is P0.", "(has E3 P0)", "<s> E3 <r> P0 <o> true"),
    (("prop", "E2", "P2", "V1"), "the P2 of E2 is V1.", "(val E2 P2 V1)", "<s> E2 <r> P2 <o> V1"),
    (("rel", "R0", "E0", "E1"), "the R0 of E0 is E1.", "(R0 E0 E1)", "<s> E0 <r> R0 <o> E1"),
    (("rel", "R4", "E0", "E1"), "E0 is R4 E1.", "(R4 E0 E1)", "<s> E0 <r> R4 <o> E1"),
]

GOLDEN_QUERIES = [
    ("prop", ("E0", "P0"), "is E0 P0?", "? (has E0 P0)", "<q> <s> E0 <r> P0 <o> ?"),
    ("prop", ("E0", "P2"), "what is the P2 of E0?", "? (val E0 P2 ?x)", "<q> <s> E0 <r> P2 <o> ?"),
    ("isa", ("E0", "C1"), "is E0 a C1?", "? (isa E0 C1)", "<q> <s> E0 <r> isa <o> C1"),
    (
        "chain",
        ("E0", ("R0", "R1")),
        "what is the R1 of the R0 of E0?",
        "? (R1 (R0 E0))",
        "<q> <s> E0 <r> R0 <o> ?x1 <s> ?x1 <r> R1 <o> ?",
    ),
    ("trans", ("E0", "R4", "E2"), "is E0 R4 E2?", "? (R4 E0 E2)", "<q> <s> E0 <r> R4 <o> E2"),
]


@pytest.mark.parametrize("fact,prose,symbolic,triples", GOLDEN_FACTS)
def test_golden_facts(mini, fact, prose, symbolic, triples):
    for name, expected in (("prose", prose), ("symbolic", symbolic), ("triples", triples)):
        enc = make_encoder(name, mini)
        assert enc.detok(enc.fact(fact, 0)) == expected


@pytest.mark.parametrize("form,args,prose,symbolic,triples", GOLDEN_QUERIES)
def test_golden_queries(mini, form, args, prose, symbolic, triples):
    for name, expected in (("prose", prose), ("symbolic", symbolic), ("triples", triples)):
        enc = make_encoder(name, mini)
        assert enc.detok(enc.query(form, args, 0)) == expected


def test_context_includes_overridden_default_for_exceptions(default_ds):
    sampler = ContextSampler(default_ds.world)
    cfg = DataConfig(regime="context")
    exc = [q for q in default_ds.queries if q.task_type == "exception"][:500]
    assert exc
    for q in exc:
        ctx = sampler.sample(q, cfg)
        assert q.overridden in ctx
        assert all(f in ctx for _, f in q.proof)


def test_ctx_loss_masks_context_tokens(tiny, encs_tiny):
    enc = encs_tiny["symbolic"]
    q = tiny.queries[0]
    a = query_example(enc, q, DataConfig(regime="context"), 0)
    b = query_example(enc, q, DataConfig(regime="context", ctx_loss=True), 0)
    assert a.tokens == b.tokens
    q_at = b.tokens.index(QUERY)
    assert b.loss_mask[:2] == [0, 0] and all(b.loss_mask[2:q_at])
    assert b.loss_mask[q_at:b.prompt_len] == [0] * (b.prompt_len - q_at)
    assert b.loss_mask[b.prompt_len:] == a.loss_mask[a.prompt_len:]


def test_distractor_ramp(ds):
    from encoders.dataset import n_distractors_at
    from train.evaluate import Evaluator
    from encoders.dataset import make_encoder

    cfg = DataConfig(regime="context", distractor_ramp_epochs=4)
    assert [n_distractors_at(cfg, e) for e in range(6)] == [0, 2, 3, 4, 6, 6]  # round(6 * e/4)
    from2 = DataConfig(regime="context", distractor_ramp_epochs=4, distractor_ramp_start=2)
    assert [n_distractors_at(from2, e) for e in range(6)] == [2, 3, 4, 5, 6, 6]
    sampler = ContextSampler(ds.world)
    q = ds.queries[0]
    assert len(sampler.sample(q, cfg, 0)) == len(sampler.needed(q))
    assert len(sampler.sample(q, cfg, 10)) == len(sampler.needed(q)) + 6
    ev = Evaluator(make_encoder("symbolic", ds.world), cfg)
    assert ev.dc.distractor_ramp_epochs == 0  # evaluation always sees the full context
