"""Deterministic, seeded generator of the synthetic world.

Every component draws from its own RNG stream, ``default_rng([seed, STREAM])``, so
changing e.g. relation settings does not perturb the taxonomy or the names.
"""

from __future__ import annotations

import numpy as np

from .config import WorldConfig
from .kb import (
    PROP_BOOL,
    PROP_CAT,
    PROP_INDIVIDUAL,
    REL_FUNCTIONAL,
    REL_INVERSE,
    REL_SYMMETRIC,
    REL_TRANSITIVE,
    World,
)
from .names import generate_names

STREAM_TAXONOMY = 1
STREAM_ENTITIES = 2
STREAM_PROPERTIES = 3
STREAM_EXCEPTIONS = 4
STREAM_RELATIONS = 5
STREAM_NAMES = 6
STREAM_QUERIES = 7
STREAM_SPLITS = 8


def rng_for(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def generate_world(cfg: WorldConfig) -> World:
    cat_parent, cat_level = _generate_taxonomy(cfg, rng_for(cfg.seed, STREAM_TAXONOMY))
    categories = list(cat_parent)
    entities = [f"E{i:04d}" for i in range(cfg.n_entities)]
    leaves = [c for c in categories if not any(cat_parent[d] == c for d in categories)]
    entity_leaf = _assign_entities(cfg, entities, leaves, rng_for(cfg.seed, STREAM_ENTITIES))

    world = World(
        config=cfg,
        lexicon={},
        entities=entities,
        categories=categories,
        cat_parent=cat_parent,
        cat_level=cat_level,
        entity_leaf=entity_leaf,
        properties={},
        relations={},
    )
    facts: list[tuple] = []
    facts += [("isa", e, entity_leaf[e]) for e in entities]
    facts += [("sub", c, cat_parent[c]) for c in categories if cat_parent[c] is not None]

    facts += _generate_properties(
        cfg, world, rng_for(cfg.seed, STREAM_PROPERTIES), rng_for(cfg.seed, STREAM_EXCEPTIONS)
    )
    facts += _generate_relations(cfg, world, rng_for(cfg.seed, STREAM_RELATIONS))
    world.facts = facts

    ids = entities + categories + list(world.properties)
    ids += [v for p in world.properties.values() for v in p.get("values", [])]
    ids += list(world.relations)
    names = generate_names(len(ids), rng_for(cfg.seed, STREAM_NAMES))
    world.lexicon = dict(zip(ids, names))
    return world


# ---------------------------------------------------------------------------- taxonomy
def _generate_taxonomy(cfg: WorldConfig, rng: np.random.Generator):
    t = cfg.taxonomy
    lo, hi = t.branching
    for _ in range(10_000):
        parents: list[int | None] = []
        levels: list[int] = []
        queue = []
        for _ in range(t.n_top):
            parents.append(None)
            levels.append(1)
            queue.append(len(levels) - 1)
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            lvl = levels[node]
            if lvl >= t.max_depth:
                continue
            if lvl >= t.min_leaf_depth and rng.random() < t.p_leaf:
                continue
            for _ in range(int(rng.integers(lo, hi + 1))):
                parents.append(node)
                levels.append(lvl + 1)
                queue.append(len(levels) - 1)
        if abs(len(levels) - t.target_categories) <= t.tolerance:
            width = len(str(len(levels) - 1))
            ids = [f"C{i:0{width}d}" for i in range(len(levels))]
            cat_parent = {ids[i]: (None if p is None else ids[p]) for i, p in enumerate(parents)}
            cat_level = {ids[i]: levels[i] for i in range(len(levels))}
            return cat_parent, cat_level
    raise RuntimeError("could not generate a taxonomy of the requested size; adjust taxonomy config")


def _assign_entities(cfg, entities, leaves, rng) -> dict[str, str]:
    m = cfg.taxonomy.min_entities_per_leaf
    if len(entities) < m * len(leaves):
        raise ValueError("too few entities for min_entities_per_leaf")
    order = [entities[i] for i in rng.permutation(len(entities))]
    out = {}
    for i, e in enumerate(order):
        if i < m * len(leaves):
            out[e] = leaves[i % len(leaves)]
        else:
            out[e] = leaves[int(rng.integers(len(leaves)))]
    return {e: out[e] for e in entities}


# -------------------------------------------------------------------------- properties
def _inherited_value(world: World, assign: dict, node_chain: list[str], prop: str):
    for n in node_chain:
        if (n, prop) in assign:
            return assign[(n, prop)]
    return None


def _other_value(prop_meta: dict, current, rng):
    if prop_meta["kind"] == PROP_BOOL:
        return not current
    choices = [v for v in prop_meta["values"] if v != current]
    return choices[int(rng.integers(len(choices)))]


def _generate_properties(cfg, world: World, rng, exc_rng) -> list[tuple]:
    pc = cfg.properties
    n_total = pc.n_inherited_bool + pc.n_categorical + pc.n_individual
    width = len(str(n_total - 1))
    props: dict[str, dict] = {}
    v_counter = 0
    for i in range(n_total):
        pid = f"P{i:0{width}d}"
        if i < pc.n_inherited_bool:
            props[pid] = {"kind": PROP_BOOL}
        elif i < pc.n_inherited_bool + pc.n_categorical:
            k = int(rng.integers(pc.categorical_values[0], pc.categorical_values[1] + 1))
            values = [f"V{v_counter + j:03d}" for j in range(k)]
            v_counter += k
            props[pid] = {"kind": PROP_CAT, "values": values}
        else:
            props[pid] = {"kind": PROP_INDIVIDUAL}
    world.properties = props

    cats = world.categories
    max_level = max(world.cat_level.values())
    by_level = {lvl: [c for c in cats if world.cat_level[c] == lvl] for lvl in range(1, max_level + 1)}
    leafset = set(world.leaves)
    # first attachment of every property goes on a leaf or a parent of a leaf, so that every
    # property has short (<= 3 hop) inheritance queries available for training
    near_leaf = [c for c in cats if c in leafset or any(ch in leafset for ch in world.cat_children[c])]

    assign: dict[tuple[str, str], object] = {}
    cat_facts: list[tuple] = []

    def related(a, b):
        return a in world.cat_ancestors(b) or b in world.cat_ancestors(a)

    attachments: list[tuple[str, str]] = []  # (category, prop) in creation order
    for pid, meta in props.items():
        if meta["kind"] == PROP_INDIVIDUAL:
            continue
        n_att = int(rng.integers(pc.attachments[0], pc.attachments[1] + 1))
        chosen: list[str] = [near_leaf[int(rng.integers(len(near_leaf)))]]
        for _ in range(n_att - 1):
            for _try in range(50):
                lvl = int(rng.integers(1, max_level + 1))
                c = by_level[lvl][int(rng.integers(len(by_level[lvl])))]
                if not any(related(c, x) for x in chosen):
                    chosen.append(c)
                    break
        for c in chosen:
            if meta["kind"] == PROP_BOOL:
                v = bool(rng.random() < 0.5)
            else:
                v = meta["values"][int(rng.integers(len(meta["values"])))]
            assign[(c, pid)] = v
            cat_facts.append(("prop", c, pid, v))
            attachments.append((c, pid))

    # category-level exceptions: walk each attachment's subtree top-down (pre-order)
    for c, pid in attachments:
        for d in world.descendants(c):
            if exc_rng.random() < pc.category_exception_rate:
                cur = _inherited_value(world, assign, world.cat_ancestors(d), pid)
                v = _other_value(props[pid], cur, exc_rng)
                assign[(d, pid)] = v
                cat_facts.append(("prop", d, pid, v))

    # entity-level exceptions
    ent_facts: list[tuple] = []
    inherited_ids = [p for p, m in props.items() if m["kind"] != PROP_INDIVIDUAL]
    for e in world.entities:
        chain = world.entity_categories(e)
        for pid in inherited_ids:
            cur = _inherited_value(world, assign, chain, pid)
            if cur is None:
                continue
            if exc_rng.random() < pc.entity_exception_rate:
                v = _other_value(props[pid], cur, exc_rng)
                assign[(e, pid)] = v
                ent_facts.append(("prop", e, pid, v))

    # individual properties (stated directly on entities, no category default)
    indiv_ids = [p for p, m in props.items() if m["kind"] == PROP_INDIVIDUAL]
    k = min(pc.individual_per_entity, len(indiv_ids))
    for e in world.entities:
        for j in sorted(rng.choice(len(indiv_ids), size=k, replace=False)):
            ent_facts.append(("prop", e, indiv_ids[j], bool(rng.random() < 0.5)))

    return cat_facts + ent_facts


# --------------------------------------------------------------------------- relations
def _generate_relations(cfg, world: World, rng) -> list[tuple]:
    rc = cfg.relations
    ents = world.entities
    n = len(ents)
    rels: dict[str, dict] = {}
    facts: list[tuple] = []
    counter = 0

    def new_rel(kind, inverse=None):
        nonlocal counter
        rid = f"R{counter}"
        counter += 1
        rels[rid] = {"kind": kind, "inverse": inverse}
        return rid

    for _ in range(rc.n_functional):
        r = new_rel(REL_FUNCTIONAL)
        for i, x in enumerate(ents):
            j = int(rng.integers(n - 1))
            j = j + 1 if j >= i else j  # any entity except x itself
            facts.append(("rel", r, x, ents[j]))

    for _ in range(rc.n_symmetric):
        r = new_rel(REL_SYMMETRIC)
        perm = rng.permutation(n)
        for k in range(0, n - 1, 2):
            a, b = ents[perm[k]], ents[perm[k + 1]]
            if rng.random() < 0.5:
                a, b = b, a
            facts.append(("rel", r, a, b))

    for _ in range(rc.n_inverse_pairs):
        r = new_rel(REL_INVERSE)
        r_inv = new_rel(REL_INVERSE, inverse=r)
        rels[r]["inverse"] = r_inv
        perm = list(rng.permutation(n))
        for i in range(n):  # remove fixed points (derangement)
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
        for i, x in enumerate(ents):
            y = ents[perm[i]]
            if rng.random() < 0.5:
                facts.append(("rel", r, x, y))
            else:
                facts.append(("rel", r_inv, y, x))

    for _ in range(rc.n_transitive):
        r = new_rel(REL_TRANSITIVE)
        order = [ents[i] for i in rng.permutation(n)]
        g = rc.transitive_group_size
        for start in range(0, n, g):
            group = order[start : start + g]
            for j in range(1, len(group)):
                back = 1 + int(rng.integers(min(rc.transitive_window, j)))
                facts.append(("rel", r, group[j], group[j - back]))  # child r parent

    world.relations = rels
    return facts
