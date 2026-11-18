"""Encoding C: graph triples with role markers.

Every fact is one edge ``<s> subject <r> label <o> object``; entities and categories
are untyped graph nodes, properties and relations are edge labels, boolean values are
the nodes ``true``/``false``. Queries are triple patterns: ``?`` marks the slot to
fill, ``?x1..`` are join variables for chains, and a fully bound pattern asks whether
the edge holds.

    <s> rex <r> isa <o> dog            <q> <s> rex <r> fur <o> ?        -> true
    <s> mammal <r> fur <o> true        <q> <s> rex <r> isa <o> mammal   -> true
    <q> <s> rex <r> r1 <o> ?x1 <s> ?x1 <r> r2 <o> ?                     -> entity
"""

from __future__ import annotations

from .base import Encoder


class TriplesEncoder(Encoder):
    name = "triples"
    TRUE = "true"
    FALSE = "false"

    FACT_TEMPLATES = {
        "isa": ["<s> {e:entity} <r> isa <o> {c:category}"],
        "sub": ["<s> {c:category} <r> sub <o> {p:category}"],
        "prop_true": ["<s> {n:node} <r> {P:bprop} <o> true"],
        "prop_false": ["<s> {n:node} <r> {P:bprop} <o> false"],
        "val": ["<s> {n:node} <r> {A:cprop} <o> {v:value}"],
        "rel": ["<s> {x:entity} <r> {r:frel} <o> {y:entity}", "<s> {x:entity} <r> {r:trel} <o> {y:entity}"],
    }

    QUERY_TEMPLATES = {
        "prop": ["<q> <s> {e:entity} <r> {P:bprop} <o> ?", "<q> <s> {e:entity} <r> {P:cprop} <o> ?"],
        "isa": ["<q> <s> {e:entity} <r> isa <o> {c:category}"],
        "trans": ["<q> <s> {a:entity} <r> {r:trel} <o> {b:entity}"],
    }

    def chain_templates(self, k: int) -> list[str]:
        parts = []
        subj = "{e:entity}"
        for i in range(1, k + 1):
            obj = "?" if i == k else f"?x{i}"
            parts.append(f"<s> {subj} <r> {{r{i}:frel}} <o> {obj}")
            subj = obj
        return ["<q> " + " ".join(parts)]


def _unmark(t: str) -> str:
    """'<s> X <r> Y <o> Z' -> '( X Y Z )' for every pattern in a template."""
    import re

    return re.sub(r"<s> (\S+) <r> (\S+) <o> (\S+)", r"( \1 \2 \3 )", t)


class TriplesUnmarkedEncoder(TriplesEncoder):
    """Ablation of encoding C: role markers replaced by plain brackets.

        ( rex isa dog )   <q> ( rex r1 ?x1 ) ( ?x1 r2 ? )

    Subject-relation-object order, true/false value nodes and fact-shaped chain
    patterns are kept; only the <s>/<r>/<o> role tags are removed. Tests whether the
    role tags explain triples' reliable acquisition of transitive queries in the
    context regime (README, "Grid slice 1").
    """

    name = "triples_unmarked"
    FACT_TEMPLATES = {k: [_unmark(t) for t in v] for k, v in TriplesEncoder.FACT_TEMPLATES.items()}
    QUERY_TEMPLATES = {k: [_unmark(t) for t in v] for k, v in TriplesEncoder.QUERY_TEMPLATES.items()}

    def chain_templates(self, k: int) -> list[str]:
        return [_unmark(t) for t in super().chain_templates(k)]
