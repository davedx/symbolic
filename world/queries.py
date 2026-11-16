"""Query objects and the deterministic query pool.

A query's semantic identity is ``(form, args)``; ``qid`` is a readable string form
of it and is shared by all encodings (it is the alignment key for CKA etc.).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .generator import STREAM_QUERIES, rng_for
from .kb import PROP_INDIVIDUAL, REL_INVERSE, REL_SYMMETRIC, REL_TRANSITIVE, World
from .oracle import Oracle, Undefined


@dataclass
class Query:
    qid: str
    form: str
    args: tuple
    answer: object
    task_type: str
    hops: int
    proof: list  # [(rule, fact), ...]
    rules: list[str]
    touches: list  # [(entity, predicate), ...]: every predicate applied to every entity
    overridden: tuple | None = None
    split: str | None = None

    @property
    def key(self) -> tuple:
        return (self.form, self.args)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Query":
        d = dict(d)
        d["args"] = _tuplify(d["args"])
        d["proof"] = [(r, tuple(f)) for r, f in d["proof"]]
        d["touches"] = [tuple(t) for t in d["touches"]]
        d["overridden"] = tuple(d["overridden"]) if d["overridden"] else None
        return cls(**d)


def _tuplify(x):
    return tuple(_tuplify(i) for i in x) if isinstance(x, list) else x


def make_qid(form: str, args: tuple) -> str:
    flat = []
    for a in args:
        flat.append(".".join(a) if isinstance(a, tuple) else a)
    return "|".join([form, *flat])


def touches_for(form: str, args: tuple, proof) -> list[tuple[str, str]]:
    if form == "prop":
        return [(args[0], args[1])]
    if form == "isa":
        return [(args[0], "isa")]
    if form == "chain":
        out = []
        x = args[0]
        for r, (_, fact) in zip(args[1], proof):
            out.append((x, r))
            # the entity reached after this step (subject or object depending on direction)
            x = fact[3] if fact[2] == x else fact[2]
        return out
    if form == "trans":
        return [(args[0], args[1])]
    raise ValueError(form)


def make_query(oracle: Oracle, form: str, args: tuple) -> Query:
    a = oracle.answer(form, args)
    return Query(
        qid=make_qid(form, args),
        form=form,
        args=args,
        answer=a.answer,
        task_type=a.task_type,
        hops=a.hops,
        proof=a.proof,
        rules=a.rules,
        touches=touches_for(form, args, a.proof),
        overridden=a.overridden,
    )


# ------------------------------------------------------------------------------ pool
def build_pool(world: World) -> list[Query]:
    """All prop and isa queries, plus sampled chain and trans queries; capped per stratum."""
    qc = world.config.queries
    rng = rng_for(world.config.seed, STREAM_QUERIES)
    oracle = Oracle(world)
    pool: dict[tuple, Query] = {}

    def add(form, args):
        key = (form, args)
        if key in pool:
            return False
        try:
            q = make_query(oracle, form, args)
        except Undefined:
            return False
        if q.hops > qc.max_hops:
            return False
        pool[key] = q
        return True

    # prop: every (entity, property) pair with a defined value
    for e in world.entities:
        for p, meta in world.properties.items():
            if meta["kind"] == PROP_INDIVIDUAL and (e, p) not in world.assignments:
                continue
            add("prop", (e, p))

    # isa: per entity and level, the true ancestor plus random same-level negatives
    levels = sorted(set(world.cat_level.values()))
    by_level = {lvl: [c for c in world.categories if world.cat_level[c] == lvl] for lvl in levels}
    for e in world.entities:
        for c in world.entity_categories(e):
            add("isa", (e, c))
            others = [d for d in by_level[world.cat_level[c]] if d != c]
            if not others:
                continue
            k = min(qc.isa_negatives_per_level, len(others))
            for j in sorted(rng.choice(len(others), size=k, replace=False)):
                add("isa", (e, others[j]))

    # chain: sampled compositions of functional relations, avoiding trivial back-and-forth
    chain_rels = world.chain_relations
    for k in range(1, qc.max_hops + 1):
        got, tries = 0, 0
        while got < qc.chain_per_hop and tries < 20 * qc.chain_per_hop:
            tries += 1
            e = world.entities[int(rng.integers(len(world.entities)))]
            rels: list[str] = []
            while len(rels) < k:
                r = chain_rels[int(rng.integers(len(chain_rels)))]
                if rels and _cancels(world, rels[-1], r):
                    continue
                rels.append(r)
            got += add("chain", (e, tuple(rels)))

    # trans: (a, b) with b an r-ancestor of a at distance k, plus reversed negatives
    for r in world.relations_of_kind(REL_TRANSITIVE):
        by_dist: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for a in world.entities:
            x, d = a, 0
            while True:
                ys = world.rel_by_subject.get((r, x))
                if not ys:
                    break
                x, d = ys[0], d + 1
                if d > qc.max_hops:
                    break
                by_dist[d].append((a, x))
        for d in sorted(by_dist):
            pairs = by_dist[d]
            take = min(qc.trans_per_hop, len(pairs))
            for j in sorted(rng.choice(len(pairs), size=take, replace=False)):
                a, b = pairs[j]
                add("trans", (a, r, b))
                add("trans", (b, r, a))

    queries = list(pool.values())
    if qc.max_per_stratum:
        queries = _cap_strata(queries, qc.max_per_stratum, rng)
    return queries


def _cancels(world: World, prev: str, r: str) -> bool:
    """r immediately undoes prev (would return to the previous entity)."""
    mp, mr = world.relations[prev], world.relations[r]
    if mp["kind"] == REL_SYMMETRIC and prev == r:
        return True
    if mp["kind"] == REL_INVERSE and mp["inverse"] == r:
        return True
    return False


def _cap_strata(queries: list[Query], cap: int, rng: np.random.Generator) -> list[Query]:
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, q in enumerate(queries):
        strata[(q.form, q.task_type, q.hops)].append(i)
    keep = set()
    for key in sorted(strata, key=str):
        idx = strata[key]
        if len(idx) > cap:
            idx = sorted(rng.choice(idx, size=cap, replace=False).tolist())
        keep.update(idx)
    return [q for i, q in enumerate(queries) if i in keep]
