"""Independent reasoner used to cross-check the oracle.

The oracle walks *up* from the queried entity and stops at the first assignment.
This module instead forward-chains the whole KB to a fixpoint (top-down property
propagation, naive relation closure) and answers queries by lookup. The two
share no code beyond the ``World`` container, so agreement on every query is
meaningful evidence that both are right.
"""

from __future__ import annotations

from .kb import REL_INVERSE, REL_SYMMETRIC, REL_TRANSITIVE, World


class Closure:
    def __init__(self, world: World):
        self.w = world
        facts = world.facts

        # membership closure: (entity, category) pairs, iterated to fixpoint over "sub"
        sub = {(f[1], f[2]) for f in facts if f[0] == "sub"}
        member = {(f[1], f[2]) for f in facts if f[0] == "isa"}
        while True:
            new = {(e, p) for (e, c) in member for (c2, p) in sub if c == c2} - member
            if not new:
                break
            member |= new
        self.member = member

        # property values propagated top-down; more specific statements replace inherited ones
        stated: dict[tuple[str, str], object] = {}
        for f in facts:
            if f[0] == "prop":
                stated[(f[1], f[2])] = f[3]
        parent = {f[1]: f[2] for f in facts if f[0] == "sub"}
        depth = {}
        for c in world.categories:
            d, x = 0, c
            while x in parent:
                x, d = parent[x], d + 1
            depth[c] = d
        values: dict[str, dict[str, object]] = {}
        for c in sorted(world.categories, key=lambda c: depth[c]):
            inherited = dict(values[parent[c]]) if c in parent else {}
            inherited.update({p: v for (n, p), v in stated.items() if n == c})
            values[c] = inherited
        leaf = {f[1]: f[2] for f in facts if f[0] == "isa"}
        for e in world.entities:
            v = dict(values[leaf[e]]) if e in leaf else {}  # tolerate partial KBs (context subsets)
            v.update({p: val for (n, p), val in stated.items() if n == e})
            values[e] = v
        self.values = values

        # relation closure: stated edges + symmetric + inverse + transitive, to fixpoint
        edges: dict[str, set[tuple[str, str]]] = {r: set() for r in world.relations}
        for f in facts:
            if f[0] == "rel":
                edges[f[1]].add((f[2], f[3]))
        for r, meta in world.relations.items():
            if meta["kind"] == REL_SYMMETRIC:
                edges[r] |= {(b, a) for a, b in edges[r]}
        for r, meta in world.relations.items():
            if meta["kind"] == REL_INVERSE:
                inv = meta["inverse"]
                edges[r] = edges[r] | {(b, a) for a, b in edges[inv]}
        for r, meta in world.relations.items():
            if meta["kind"] == REL_TRANSITIVE:
                edges[r] = _transitive_closure(edges[r])
        self.edges = edges
        self.succ: dict[str, dict[str, list[str]]] = {}
        for r, es in edges.items():
            d: dict[str, list[str]] = {}
            for a, b in sorted(es):
                d.setdefault(a, []).append(b)
            self.succ[r] = d

    def answer(self, form: str, args: tuple):
        if form == "prop":
            e, p = args
            return self.values[e][p]
        if form == "isa":
            return args in self.member
        if form == "chain":
            x, rels = args
            for r in rels:
                ys = self.succ[r][x]
                if len(ys) != 1:
                    raise AssertionError(f"relation {r} not functional at {x}: {ys}")
                x = ys[0]
            return x
        if form == "trans":
            a, r, b = args
            return (a, b) in self.edges[r]
        raise ValueError(form)


def _transitive_closure(pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    closure = set(pairs)
    while True:
        succ: dict[str, set[str]] = {}
        for a, b in closure:
            succ.setdefault(a, set()).add(b)
        new = {(a, c) for a, b in closure for c in succ.get(b, ())} - closure
        if not new:
            return closure
        closure |= new
