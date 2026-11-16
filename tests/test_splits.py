"""Split tests: no leakage, routing correctness, derivability and coverage of held-out queries."""

from collections import Counter, defaultdict

import pytest

from world.splits import SPLITS, TEST_CATEGORY, TEST_COMBO, TEST_IID, TEST_LENGTH, TRAIN, axes

TEST_SPLITS = [s for s in SPLITS if s != TRAIN]


def by_split(ds):
    out = defaultdict(list)
    for q in ds.queries:
        out[q.split].append(q)
    return out


def test_every_split_nonempty(ds):
    sizes = Counter(q.split for q in ds.queries)
    for s in SPLITS:
        assert sizes[s] > 0, s


def test_qids_unique_and_match_keys(ds):
    qids = [q.qid for q in ds.queries]
    assert len(qids) == len(set(qids))
    keys = [q.key for q in ds.queries]
    assert len(keys) == len(set(keys))


def test_no_query_in_two_splits(ds):
    seen = {}
    for q in ds.queries:
        assert q.key not in seen or seen[q.key] == q.split
        seen[q.key] = q.split


def test_hop_ranges(ds):
    sc = ds.world.config.splits
    for q in ds.queries:
        if q.split == TEST_LENGTH:
            assert sc.length_hops[0] <= q.hops <= sc.length_hops[1]
        else:
            assert q.hops <= sc.train_max_hops


def test_routing_matches_axes(ds):
    """Each query sits on exactly the axis its split names (train/iid: on none)."""
    for q in ds.queries:
        ax = axes(ds.world, ds.heldout, q)
        if q.split in (TRAIN, TEST_IID):
            assert ax == []
        else:
            assert ax == [q.split]


def test_train_never_touches_heldout_pairs(ds):
    w, h = ds.world, ds.heldout
    ce, cp = set(h.combo_entities), set(h.combo_predicates)
    cat_pairs = set(h.category_pairs)
    for q in ds.queries:
        if q.split != TRAIN:
            continue
        for ent, pred in q.touches:
            assert not (ent in ce and pred in cp), q.qid
            for c in w.entity_categories(ent):
                assert (c, pred) not in cat_pairs, q.qid


def test_touches_cover_every_relation_step(ds):
    """touches must list (entity, relation) for every hop of a chain, in order."""
    for q in ds.queries:
        if q.form == "chain":
            assert [p for _, p in q.touches] == list(q.args[1])
            assert q.touches[0][0] == q.args[0]


def test_heldout_queries_derivable_from_training_facts(ds):
    """The KB is training material in full; every fact a held-out proof uses must be in it,
    and the overridden default (for exceptions) as well."""
    kb = ds.world.fact_set
    for q in ds.queries:
        if q.split == TRAIN:
            continue
        for _, fact in q.proof:
            assert fact in kb, (q.qid, fact)
        if q.overridden is not None:
            assert q.overridden in kb


def _train_inventory(train):
    preds, rules, ents = set(), set(), set()
    pred_rules = set()
    for q in train:
        for e, p in q.touches:
            ents.add(e)
            preds.add(p)
        rules.update(q.rules)
        for p in {p for _, p in q.touches}:
            pred_rules.update((p, r) for r in q.rules)
    return preds, rules, ents, pred_rules


def test_heldout_coverage_in_train(ds):
    """Every predicate, entity, rule -- and every (predicate, rule) combination such as
    'symmetric relation used in reverse' -- seen in a test query was also seen in train."""
    splits = by_split(ds)
    preds, rules, ents, pred_rules = _train_inventory(splits[TRAIN])
    for s in TEST_SPLITS:
        for q in splits[s]:
            for e, p in q.touches:
                assert p in preds, (s, q.qid, p)
                assert e in ents, (s, q.qid, e)
            for r in q.rules:
                assert r in rules, (s, q.qid, r)
            if q.form == "chain":
                for p, (rule, _) in zip(q.args[1], q.proof):
                    assert (p, rule) in pred_rules, (s, q.qid, p, rule)


def test_heldout_combo_grid_is_compositional(ds):
    """Held-out entities and predicates each appear in train, just never together."""
    train = by_split(ds)[TRAIN]
    h = ds.heldout
    ents = {e for q in train for e, _ in q.touches}
    preds = {p for q in train for _, p in q.touches}
    assert set(h.combo_entities) <= ents
    combo_preds_used = {p for q in by_split(ds)[TEST_COMBO] for _, p in q.touches if p in h.combo_predicates}
    assert combo_preds_used and combo_preds_used <= preds


def test_heldout_category_pairs_are_compositional(ds):
    """For each held-out (category, predicate): the predicate is queried in train on entities
    outside the category, and entities inside the category are queried in train."""
    w = ds.world
    train = by_split(ds)[TRAIN]
    for c, pred in ds.heldout.category_pairs:
        inside = set(w.entities_under(c))
        assert any(p == pred and e not in inside for q in train for e, p in q.touches), (c, pred)
        assert any(e in inside for q in train for e, _ in q.touches), (c, pred)


def test_heldout_category_split_has_relations_and_properties(default_ds):
    test = by_split(default_ds)[TEST_CATEGORY]
    forms = Counter(q.form for q in test)
    assert forms["prop"] > 0 and (forms["chain"] + forms["trans"]) > 0


@pytest.mark.parametrize("split", TEST_SPLITS)
def test_default_split_sizes_reasonable(default_ds, split):
    assert len(by_split(default_ds)[split]) >= 300
