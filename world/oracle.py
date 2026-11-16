"""Ground-truth simulator: answers any query with its proof trace.

Query forms (``args`` in parentheses):

    prop  (entity, property)            -> bool | value id
    isa   (entity, category)            -> bool
    chain (entity, (r1, ..., rk))       -> entity        r_k(...r_1(entity))
    trans (a, relation, b)              -> bool          does a r b hold (r transitive)?

A proof is a list of ``(rule, fact)`` steps; every fact is a KB fact. ``hops`` is
always ``len(proof)``: the number of KB facts the derivation uses.

Rules:
    stated     the fact is the answer (individual property)
    isa, sub   taxonomy edge traversed
    inherit    property value read from a category and inherited downward
    override   property value that overrides a different value stated higher up
    rel        relation edge used in its stated direction
    symmetric  edge used in reverse because the relation is symmetric
    inverse    edge stated with the inverse relation, used in reverse
    trans      direct link of a transitive relation

Negative answers that are not stated explicitly are derived from two structural
axioms (recorded in ``Answer.rules``):
    disjoint    each entity is in exactly one leaf; the taxonomy is a tree, so an
                entity is in exactly one category per level
    asymmetric  transitive relations are strict orders; if a r+ b then not b r+ a
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .kb import PROP_INDIVIDUAL, REL_INVERSE, REL_SYMMETRIC, REL_TRANSITIVE, World

TASK_DIRECT = "direct"
TASK_INHERITED = "inherited"
TASK_RELATION = "relation"
TASK_EXCEPTION = "exception"
TASK_TYPES = (TASK_DIRECT, TASK_INHERITED, TASK_RELATION, TASK_EXCEPTION)


class Undefined(Exception):
    """The query has no answer in this world (e.g. property not defined for the entity)."""


@dataclass
class Answer:
    answer: object
    task_type: str
    proof: list[tuple[str, tuple]]
    rules: list[str] = field(default_factory=list)  # sorted distinct rules + axioms used
    overridden: tuple | None = None  # for exceptions: the more general fact that was overridden

    @property
    def hops(self) -> int:
        return len(self.proof)


class Oracle:
    def __init__(self, world: World):
        self.w = world

    def answer(self, form: str, args: tuple) -> Answer:
        fn = {"prop": self.prop, "isa": self.isa, "chain": self.chain, "trans": self.trans}[form]
        return fn(*args)

    # ---------------------------------------------------------------- properties
    def prop(self, e: str, p: str) -> Answer:
        w = self.w
        assign = w.assignments
        nodes = [e] + w.entity_categories(e)
        proof: list[tuple[str, tuple]] = []
        found_at = None
        for i, node in enumerate(nodes):
            if (node, p) in assign:
                found_at = i
                break
            if i == 0:
                proof.append(("isa", ("isa", e, nodes[1])))
            elif i + 1 < len(nodes):
                proof.append(("sub", ("sub", node, nodes[i + 1])))
        if found_at is None:
            raise Undefined(f"{p} undefined for {e}")
        node = nodes[found_at]
        value = assign[(node, p)]
        fact = ("prop", node, p, value)

        overridden = None
        for higher in nodes[found_at + 1 :]:
            if (higher, p) in assign:
                if assign[(higher, p)] != value:
                    overridden = ("prop", higher, p, assign[(higher, p)])
                break

        if w.properties[p]["kind"] == PROP_INDIVIDUAL:
            rule, task = "stated", TASK_DIRECT
        elif overridden is not None:
            rule, task = "override", TASK_EXCEPTION
        else:
            rule, task = "inherit", TASK_INHERITED
        proof.append((rule, fact))
        return Answer(value, task, proof, _rules(proof), overridden)

    def isa(self, e: str, c: str) -> Answer:
        w = self.w
        target_level = w.cat_level[c]
        chain = w.entity_categories(e)
        if w.cat_level[chain[0]] < target_level:
            raise Undefined("category deeper than the entity's leaf")
        proof = [("isa", ("isa", e, chain[0]))]
        i = 0
        while w.cat_level[chain[i]] > target_level:
            proof.append(("sub", ("sub", chain[i], chain[i + 1])))
            i += 1
        ans = chain[i] == c
        rules = _rules(proof) + ([] if ans else ["disjoint"])
        task = TASK_DIRECT if len(proof) == 1 else TASK_INHERITED
        return Answer(ans, task, proof, sorted(rules))

    # ----------------------------------------------------------------- relations
    def step(self, r: str, x: str) -> tuple[str, str, tuple]:
        """Follow functional relation r from x: returns (y, rule, fact)."""
        w = self.w
        meta = w.relations[r]
        ys = w.rel_by_subject.get((r, x))
        if ys:
            return ys[0], "rel", ("rel", r, x, ys[0])
        if meta["kind"] == REL_SYMMETRIC:
            ys = w.rel_by_object.get((r, x))
            if ys:
                return ys[0], "symmetric", ("rel", r, ys[0], x)
        if meta["kind"] == REL_INVERSE:
            inv = meta["inverse"]
            ys = w.rel_by_object.get((inv, x))
            if ys:
                return ys[0], "inverse", ("rel", inv, ys[0], x)
        raise Undefined(f"{r}({x}) undefined")

    def chain(self, e: str, rels: tuple[str, ...]) -> Answer:
        x = e
        proof = []
        for r in rels:
            if self.w.relations[r]["kind"] == REL_TRANSITIVE:
                raise Undefined("transitive relations are not functional")
            x, rule, fact = self.step(r, x)
            proof.append((rule, fact))
        return Answer(x, TASK_RELATION, proof, _rules(proof))

    def _up_path(self, r: str, a: str, b: str):
        """Direct-link path a -> ... -> b following r upward, or None."""
        path = []
        x = a
        seen = {a}
        while True:
            ys = self.w.rel_by_subject.get((r, x))
            if not ys:
                return None
            y = ys[0]
            path.append(("trans", ("rel", r, x, y)))
            if y == b:
                return path
            if y in seen:  # defensive: forests have no cycles
                return None
            seen.add(y)
            x = y

    def trans(self, a: str, r: str, b: str) -> Answer:
        if self.w.relations[r]["kind"] != REL_TRANSITIVE:
            raise Undefined("not a transitive relation")
        path = self._up_path(r, a, b)
        if path is not None:
            return Answer(True, TASK_RELATION, path, _rules(path))
        path = self._up_path(r, b, a)
        if path is not None:
            return Answer(False, TASK_RELATION, path, sorted(_rules(path) + ["asymmetric"]))
        raise Undefined("pair not comparable under the order")


def _rules(proof) -> list[str]:
    return sorted({rule for rule, _ in proof})
