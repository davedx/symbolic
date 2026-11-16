"""The world: entities, taxonomy, properties, relations and the knowledge base (KB).

Facts are plain tuples (hashable, JSON-friendly):

    ("isa",  entity, leaf_category)        entity belongs to a leaf category
    ("sub",  category, parent_category)    taxonomy edge
    ("prop", node, property, value)        node is an entity or category; value is a
                                           bool (boolean property) or a value id (categorical)
    ("rel",  relation, subject, object)    stated relation edge

The KB (``World.facts``) is the complete set of stated facts. It is shared training
material for every split; splits partition *queries* only (see ``world/splits.py``).

Internal ids are opaque (``E0001``, ``C07``, ``P012``, ``V003``, ``R2``); surface
names live in ``World.lexicon`` and are only used by the encoders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .config import WorldConfig

PROP_BOOL = "bool"  # inherited boolean property
PROP_CAT = "categorical"  # inherited categorical attribute
PROP_INDIVIDUAL = "individual"  # boolean property stated only on entities

REL_FUNCTIONAL = "functional"
REL_SYMMETRIC = "symmetric"
REL_INVERSE = "inverse"  # member of an inverse pair; `inverse` field names its partner
REL_TRANSITIVE = "transitive"


def fact_from_json(x) -> tuple:
    return tuple(x)


@dataclass
class World:
    config: WorldConfig
    lexicon: dict[str, str]
    entities: list[str]
    categories: list[str]
    cat_parent: dict[str, str | None]
    cat_level: dict[str, int]
    entity_leaf: dict[str, str]
    properties: dict[str, dict]  # id -> {"kind": ..., "values": [...]} (values only for categorical)
    relations: dict[str, dict]  # id -> {"kind": ..., "inverse": id | None}
    facts: list[tuple] = field(default_factory=list)

    # ------------------------------------------------------------------ indexes
    @cached_property
    def fact_set(self) -> frozenset[tuple]:
        return frozenset(self.facts)

    @cached_property
    def cat_children(self) -> dict[str, list[str]]:
        ch: dict[str, list[str]] = {c: [] for c in self.categories}
        for c in self.categories:
            p = self.cat_parent[c]
            if p is not None:
                ch[p].append(c)
        return ch

    @cached_property
    def leaves(self) -> list[str]:
        return [c for c in self.categories if not self.cat_children[c]]

    @cached_property
    def leaf_entities(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {c: [] for c in self.leaves}
        for e in self.entities:
            out[self.entity_leaf[e]].append(e)
        return out

    def cat_ancestors(self, c: str) -> list[str]:
        """[c, parent(c), ..., top-level category]."""
        out = []
        while c is not None:
            out.append(c)
            c = self.cat_parent[c]
        return out

    def entity_categories(self, e: str) -> list[str]:
        """Leaf category first, then its ancestors up to level 1."""
        return self.cat_ancestors(self.entity_leaf[e])

    def entity_ancestor_at(self, e: str, level: int) -> str | None:
        for c in self.entity_categories(e):
            if self.cat_level[c] == level:
                return c
        return None

    def descendants(self, c: str) -> list[str]:
        """Strict descendant categories of c (pre-order)."""
        out = []
        stack = list(reversed(self.cat_children[c]))
        while stack:
            d = stack.pop()
            out.append(d)
            stack.extend(reversed(self.cat_children[d]))
        return out

    def entities_under(self, c: str) -> list[str]:
        return [e for e in self.entities if c in self.entity_categories(e)]

    @cached_property
    def assignments(self) -> dict[tuple[str, str], object]:
        """(node, property) -> stated value."""
        out = {}
        for f in self.facts:
            if f[0] == "prop":
                out[(f[1], f[2])] = f[3]
        return out

    @cached_property
    def rel_by_subject(self) -> dict[tuple[str, str], list[str]]:
        """(relation, subject) -> objects, stated direction only."""
        out: dict[tuple[str, str], list[str]] = {}
        for f in self.facts:
            if f[0] == "rel":
                out.setdefault((f[1], f[2]), []).append(f[3])
        return out

    @cached_property
    def rel_by_object(self) -> dict[tuple[str, str], list[str]]:
        """(relation, object) -> subjects, stated direction only."""
        out: dict[tuple[str, str], list[str]] = {}
        for f in self.facts:
            if f[0] == "rel":
                out.setdefault((f[1], f[3]), []).append(f[2])
        return out

    def relations_of_kind(self, kind: str) -> list[str]:
        return [r for r, m in self.relations.items() if m["kind"] == kind]

    @cached_property
    def chain_relations(self) -> list[str]:
        """Relations usable in entity-valued composition chains (all are functional)."""
        return [r for r, m in self.relations.items() if m["kind"] != REL_TRANSITIVE]

    @property
    def predicates(self) -> list[str]:
        return list(self.properties) + list(self.relations)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "lexicon": self.lexicon,
            "entities": self.entities,
            "categories": self.categories,
            "cat_parent": self.cat_parent,
            "cat_level": self.cat_level,
            "entity_leaf": self.entity_leaf,
            "properties": self.properties,
            "relations": self.relations,
            "facts": [list(f) for f in self.facts],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        return cls(
            config=WorldConfig.from_dict(d["config"]),
            lexicon=d["lexicon"],
            entities=d["entities"],
            categories=d["categories"],
            cat_parent=d["cat_parent"],
            cat_level=d["cat_level"],
            entity_leaf=d["entity_leaf"],
            properties=d["properties"],
            relations=d["relations"],
            facts=[fact_from_json(f) for f in d["facts"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "World":
        return cls.from_dict(json.loads(Path(path).read_text()))
