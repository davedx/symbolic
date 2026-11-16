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
