"""Generator tests: determinism, names, taxonomy, properties, relations."""

import json
from collections import Counter

import pytest

from world.build import build_dataset
from world.config import WorldConfig
from world.generator import generate_world
from world.kb import PROP_INDIVIDUAL, REL_FUNCTIONAL, REL_INVERSE, REL_SYMMETRIC, REL_TRANSITIVE
from world.names import NAME_RE, blocklist


def _serialize(ds):
    return json.dumps(
        {"world": ds.world.to_dict(), "queries": [q.to_dict() for q in ds.queries], "held": ds.heldout.to_dict()},
        sort_keys=True,
        default=list,
    )


# ------------------------------------------------------------------------ determinism
def test_same_seed_identical(tiny_cfg, tiny):
    assert _serialize(build_dataset(tiny_cfg)) == _serialize(tiny)


def test_different_seed_differs(tiny_cfg, tiny):
    cfg = WorldConfig.from_dict({**tiny_cfg.to_dict(), "seed": 1})
    other = build_dataset(cfg)
    assert other.world.facts != tiny.world.facts
    assert other.world.lexicon != tiny.world.lexicon


def test_save_load_roundtrip(tiny, tmp_path):
    from world.build import Dataset

    tiny.save(tmp_path)
    loaded = Dataset.load(tmp_path)
    assert _serialize(loaded) == _serialize(tiny)
    assert loaded.world.config.fingerprint() == tiny.world.config.fingerprint()


def test_component_streams_independent(tiny_cfg, tiny):
    """Changing relation settings must not perturb the taxonomy or entity placement."""
    d = tiny_cfg.to_dict()
    d["relations"]["n_functional"] += 1
    w = generate_world(WorldConfig.from_dict(d))
    assert w.cat_parent == tiny.world.cat_parent
    assert w.entity_leaf == tiny.world.entity_leaf


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError):
        WorldConfig.from_dict({"taxonomy": {"bogus": 1}})


# ------------------------------------------------------------------------------ names
def test_names_unique_pronounceable_not_words(ds):
    names = list(ds.world.lexicon.values())
    assert len(names) == len(set(names))
    block = blocklist()
    assert len(block) > 1000, "blocklist missing; run scripts/build_name_blocklist.py"
    for n in names:
        assert NAME_RE.match(n), n
        assert n not in block, n


def test_lexicon_covers_all_ids(ds):
    w = ds.world
    ids = set(w.entities) | set(w.categories) | set(w.properties) | set(w.relations)
    ids |= {v for p in w.properties.values() for v in p.get("values", [])}
    assert ids == set(w.lexicon)


# --------------------------------------------------------------------------- taxonomy
def test_taxonomy_is_tree_with_consistent_levels(ds):
    w = ds.world
    t = w.config.taxonomy
    assert abs(len(w.categories) - t.target_categories) <= t.tolerance
    for c in w.categories:
        p = w.cat_parent[c]
        if p is None:
            assert w.cat_level[c] == 1
        else:
            assert w.cat_level[c] == w.cat_level[p] + 1
        assert len(w.cat_ancestors(c)) == w.cat_level[c]  # no cycles
        assert w.cat_level[c] <= t.max_depth
    for leaf in w.leaves:
        assert w.cat_level[leaf] >= t.min_leaf_depth
    sub_facts = [f for f in w.facts if f[0] == "sub"]
    assert len(sub_facts) == len(w.categories) - t.n_top


def test_each_entity_in_exactly_one_leaf(ds):
    w = ds.world
    isa = Counter(f[1] for f in w.facts if f[0] == "isa")
    assert set(isa) == set(w.entities)
    assert set(isa.values()) == {1}
    leaves = set(w.leaves)
    for f in w.facts:
        if f[0] == "isa":
            assert f[2] in leaves
    counts = Counter(w.entity_leaf.values())
    assert min(counts[c] for c in leaves) >= w.config.taxonomy.min_entities_per_leaf


# ------------------------------------------------------------------------- properties
def test_property_counts(ds):
    w = ds.world
    pc = w.config.properties
    kinds = Counter(m["kind"] for m in w.properties.values())
    assert kinds == {"bool": pc.n_inherited_bool, "categorical": pc.n_categorical, "individual": pc.n_individual}


def test_individual_properties_only_on_entities(ds):
    w = ds.world
    for (node, p), v in w.assignments.items():
        if w.properties[p]["kind"] == PROP_INDIVIDUAL:
            assert node in w.entity_leaf
            assert isinstance(v, bool)


def test_prop_values_well_typed(ds):
    w = ds.world
    for (node, p), v in w.assignments.items():
        meta = w.properties[p]
        if meta["kind"] == "categorical":
            assert v in meta["values"]
        else:
            assert isinstance(v, bool)


def test_no_duplicate_prop_statements(ds):
    keys = [(f[1], f[2]) for f in ds.world.facts if f[0] == "prop"]
    assert len(keys) == len(set(keys))


def test_every_non_root_assignment_is_an_override(ds):
    """Statements below another statement of the same property always change its value
    (no redundant restatements), and top-most attachments of a property are disjoint."""
    w = ds.world
    assign = w.assignments
    for (node, p), v in assign.items():
        if w.properties[p]["kind"] == PROP_INDIVIDUAL:
            continue
        chain = w.entity_categories(node) if node in w.entity_leaf else w.cat_ancestors(node)[1:]
        above = [assign[(n, p)] for n in chain if (n, p) in assign]
        if above:
            assert above[0] != v, (node, p)


def test_every_inherited_property_has_short_attachment(ds):
    w = ds.world
    leaves = set(w.leaves)
    for p, meta in w.properties.items():
        if meta["kind"] == PROP_INDIVIDUAL:
            continue
        cats = [n for (n, q) in w.assignments if q == p and n in w.cat_level]
        assert any(c in leaves or set(w.cat_children[c]) & leaves for c in cats), p


# -------------------------------------------------------------------------- relations
def test_relation_counts(ds):
    w = ds.world
    rc = w.config.relations
    kinds = Counter(m["kind"] for m in w.relations.values())
    assert kinds[REL_FUNCTIONAL] == rc.n_functional
    assert kinds[REL_SYMMETRIC] == rc.n_symmetric
    assert kinds[REL_INVERSE] == 2 * rc.n_inverse_pairs
    assert kinds[REL_TRANSITIVE] == rc.n_transitive
    for r, m in w.relations.items():
        if m["kind"] == REL_INVERSE:
            assert w.relations[m["inverse"]]["inverse"] == r


def _edges(w, r):
    return [(f[2], f[3]) for f in w.facts if f[0] == "rel" and f[1] == r]


def test_functional_relations_total_and_irreflexive(ds):
    w = ds.world
    for r in w.relations_of_kind(REL_FUNCTIONAL):
        es = _edges(w, r)
        assert sorted(a for a, _ in es) == sorted(w.entities)
        assert all(a != b for a, b in es)


def test_symmetric_is_matching_stated_once(ds):
    w = ds.world
    for r in w.relations_of_kind(REL_SYMMETRIC):
        es = _edges(w, r)
        used = [x for e in es for x in e]
        assert len(used) == len(set(used))  # each entity in at most one pair, each pair stated once
        assert len(used) >= len(w.entities) - 1


def test_inverse_pair_is_derangement(ds):
    w = ds.world
    for r in w.relations_of_kind(REL_INVERSE):
        inv = w.relations[r]["inverse"]
        if r > inv:
            continue
        mapping = dict(_edges(w, r))
        for a, b in _edges(w, inv):
            assert b not in mapping  # each underlying edge stated exactly once
            mapping[b] = a
        assert sorted(mapping) == sorted(w.entities)
        assert sorted(mapping.values()) == sorted(w.entities)
        assert all(a != b for a, b in mapping.items())


def test_transitive_is_forest(ds):
    w = ds.world
    for r in w.relations_of_kind(REL_TRANSITIVE):
        parent = {}
        for a, b in _edges(w, r):
            assert a not in parent
            parent[a] = b
        for a in w.entities:
            seen, x = set(), a
            while x in parent:
                assert x not in seen
                seen.add(x)
                x = parent[x]
