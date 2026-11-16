"""Proof checker: verifies that a query's proof trace is a valid derivation of its answer
from KB facts (and that ``hops`` equals the number of facts used)."""

from __future__ import annotations

from .kb import REL_INVERSE, REL_SYMMETRIC, World
from .queries import Query


class ProofError(AssertionError):
    pass


def check_proof(world: World, q: Query) -> None:
    kb = world.fact_set
    if q.hops != len(q.proof):
        raise ProofError(f"{q.qid}: hops {q.hops} != proof length {len(q.proof)}")
    for _, fact in q.proof:
        if fact not in kb:
            raise ProofError(f"{q.qid}: proof fact not in KB: {fact}")
    {"prop": _prop, "isa": _isa, "chain": _chain, "trans": _trans}[q.form](world, q)


def _taxonomy_path(world: World, q: Query, steps, start: str) -> str:
    """Check isa/sub steps form a connected upward path from `start`; return the last node."""
    node = start
    for i, (rule, fact) in enumerate(steps):
        expected = "isa" if i == 0 else "sub"
        if rule != expected or fact[0] != expected or fact[1] != node:
            raise ProofError(f"{q.qid}: broken taxonomy path at step {i}: {fact}")
        node = fact[2]
    return node


def _prop(world: World, q: Query) -> None:
    e, p = q.args
    *path, (rule, fact) = q.proof
    node = _taxonomy_path(world, q, path, e)
    if fact[0] != "prop" or fact[1] != node or fact[2] != p or fact[3] != q.answer:
        raise ProofError(f"{q.qid}: final fact does not conclude the answer: {fact}")
    # specificity: nothing on the path below `node` may state a value for p
    below = [e] + [f[2] for _, f in path[:-1]] if path else []
    for n in below:
        if (n, p) in world.assignments:
            raise ProofError(f"{q.qid}: a more specific value exists at {n}")
    if rule == "override":
        if q.overridden is None or q.overridden[3] == q.answer:
            raise ProofError(f"{q.qid}: override without a differing overridden fact")
        if q.overridden not in world.fact_set:
            raise ProofError(f"{q.qid}: overridden fact not in KB")
        if q.overridden[1] not in world.entity_categories(e):
            raise ProofError(f"{q.qid}: overridden fact not on the entity's category path")
    elif rule not in ("inherit", "stated"):
        raise ProofError(f"{q.qid}: unexpected final rule {rule}")


def _isa(world: World, q: Query) -> None:
    e, c = q.args
    node = _taxonomy_path(world, q, q.proof, e)
    if world.cat_level[node] != world.cat_level[c]:
        raise ProofError(f"{q.qid}: path ends at level {world.cat_level[node]}, query level {world.cat_level[c]}")
    if (node == c) != q.answer:
        raise ProofError(f"{q.qid}: answer inconsistent with path")


def _chain(world: World, q: Query) -> None:
    x, rels = q.args
    if len(rels) != len(q.proof):
        raise ProofError(f"{q.qid}: one fact per relation step expected")
    for r, (rule, fact) in zip(rels, q.proof):
        meta = world.relations[r]
        if rule == "rel" and fact[1] == r and fact[2] == x:
            x = fact[3]
        elif rule == "symmetric" and meta["kind"] == REL_SYMMETRIC and fact[1] == r and fact[3] == x:
            x = fact[2]
        elif rule == "inverse" and meta["kind"] == REL_INVERSE and fact[1] == meta["inverse"] and fact[3] == x:
            x = fact[2]
        else:
            raise ProofError(f"{q.qid}: invalid chain step {rule} {fact} from {x}")
    if x != q.answer:
        raise ProofError(f"{q.qid}: chain ends at {x}, answer {q.answer}")


def _trans(world: World, q: Query) -> None:
    a, r, b = q.args
    start, end = (a, b) if q.answer else (b, a)
    x = start
    for rule, fact in q.proof:
        if rule != "trans" or fact[1] != r or fact[2] != x:
            raise ProofError(f"{q.qid}: broken transitive path at {fact}")
        x = fact[3]
    if x != end:
        raise ProofError(f"{q.qid}: path does not reach {end}")
    if not q.answer and "asymmetric" not in q.rules:
        raise ProofError(f"{q.qid}: negative transitive answer must cite asymmetry")
