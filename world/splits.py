"""Split assignment. Splits partition *queries*; the KB is shared training material.

Every query is routed by these rules (see README "Splits" for the rationale):

1. hops > max(length_hops)                         -> dropped
2. Each query is tested against three held-out axes:
     length    hops in length_hops
     category  touches a held-out (category, predicate) pair: some entity under the
               held-out category has the held-out predicate applied to it
     combo     touches a held-out (entity, predicate) pair from the grid E_h x P_h
   A query on more than one axis is dropped, so each test split isolates exactly
   one kind of generalization.
3. one axis                                       -> that test split
4. no axis, train_max_hops < hops                 -> dropped (gap between train and length hops)
5. no axis                                        -> test_iid with prob iid_fraction, else train

Consequences, all checked in tests/test_splits.py: query keys are disjoint across
splits; no train query touches a held-out pair; every test query has hops within
its split's range; every predicate, rule and entity appearing in a test query also
appears in some train query.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .generator import STREAM_SPLITS, rng_for
from .kb import PROP_INDIVIDUAL, World
from .queries import Query

TRAIN = "train"
TEST_IID = "test_iid"
TEST_COMBO = "test_heldout_combo"
TEST_LENGTH = "test_length"
TEST_CATEGORY = "test_heldout_category"
SPLITS = (TRAIN, TEST_IID, TEST_COMBO, TEST_LENGTH, TEST_CATEGORY)


@dataclass
class HeldOut:
    combo_entities: list[str]
    combo_predicates: list[str]
    category_pairs: list[tuple[str, str]]  # (category, predicate)

    def to_dict(self) -> dict:
        return {
            "combo_entities": self.combo_entities,
            "combo_predicates": self.combo_predicates,
            "category_pairs": [list(p) for p in self.category_pairs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeldOut":
        return cls(d["combo_entities"], d["combo_predicates"], [tuple(p) for p in d["category_pairs"]])


def choose_heldout(world: World) -> HeldOut:
    sc = world.config.splits
    rng = rng_for(world.config.seed, STREAM_SPLITS)

    n_e = round(sc.combo_entity_fraction * len(world.entities))
    combo_entities = sorted(rng.choice(world.entities, size=n_e, replace=False).tolist())
    preds = world.predicates
    n_p = round(sc.combo_predicate_fraction * len(preds))
    combo_predicates = sorted(rng.choice(preds, size=n_p, replace=False).tolist())

    level_cats = [c for c in world.categories if world.cat_level[c] == sc.heldout_category_level]
    pairs: list[tuple[str, str]] = []
    rels = list(world.relations)
    for j in rng.choice(len(rels), size=min(sc.n_heldout_category_relations, len(rels)), replace=False):
        c = level_cats[int(rng.integers(len(level_cats)))]
        pairs.append((c, rels[j]))

    # (category, property) pairs: the property must also be defined for entities outside the
    # category, otherwise it would never be queried in train at all
    inherited = [p for p, m in world.properties.items() if m["kind"] != PROP_INDIVIDUAL]
    domain = _property_domains(world, inherited)
    candidates = []
    for c in level_cats:
        inside = set(world.entities_under(c))
        for p in inherited:
            if domain[p] & inside and domain[p] - inside:
                candidates.append((c, p))
    used_props: set[str] = set()
    for j in rng.permutation(len(candidates)):
        if len(pairs) >= sc.n_heldout_category_relations + sc.n_heldout_category_properties:
            break
        c, p = candidates[j]
        if p in used_props:
            continue
        used_props.add(p)
        pairs.append((c, p))
    return HeldOut(combo_entities, combo_predicates, pairs)


def _property_domains(world: World, props: list[str]) -> dict[str, set[str]]:
    assign = world.assignments
    out = {p: set() for p in props}
    for e in world.entities:
        nodes = [e] + world.entity_categories(e)
        for p in props:
            if any((n, p) in assign for n in nodes):
                out[p].add(e)
    return out


def axes(world: World, held: HeldOut, q: Query) -> list[str]:
    sc = world.config.splits
    lo, hi = sc.length_hops
    out = []
    if lo <= q.hops <= hi:
        out.append(TEST_LENGTH)
    cat_pairs = set(held.category_pairs)
    if any(
        (c, pred) in cat_pairs for ent, pred in q.touches for c in world.entity_categories(ent)
    ):
        out.append(TEST_CATEGORY)
    ce, cp = set(held.combo_entities), set(held.combo_predicates)
    if any(ent in ce and pred in cp for ent, pred in q.touches):
        out.append(TEST_COMBO)
    return out


def assign_splits(world: World, queries: list[Query], held: HeldOut) -> dict:
    """Set ``q.split`` on every query (None = dropped). Returns drop statistics."""
    sc = world.config.splits
    rng = rng_for(world.config.seed, STREAM_SPLITS + 100)
    dropped = Counter()
    for q in queries:
        ax = axes(world, held, q)
        if q.hops > sc.length_hops[1]:
            q.split = None
            dropped["too_many_hops"] += 1
        elif len(ax) > 1:
            q.split = None
            dropped["multiple_axes:" + "+".join(ax)] += 1
        elif len(ax) == 1:
            q.split = ax[0]
        elif q.hops > sc.train_max_hops:
            q.split = None
            dropped["hop_gap"] += 1
        else:
            q.split = TEST_IID if rng.random() < sc.iid_fraction else TRAIN
    return dict(dropped)
