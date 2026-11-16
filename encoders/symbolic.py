"""Encoding B: typed symbolic S-expressions.

    (isa rex dog)  (sub dog canine)  (has mammal fur)  (not (has penguin fly))
    (val rex color red)  (mentor rex fido)
    ? (has rex fur)  -> true        ? (val rex color ?x) -> red
    ? (r2 (r1 rex))  -> entity      ? (above rex fido)   -> true

Relations are heads of their own forms; chains are nested functional terms.
"""

from __future__ import annotations

import re

from .base import Encoder


class SymbolicEncoder(Encoder):
    name = "symbolic"
    TRUE = "true"
    FALSE = "false"

    FACT_TEMPLATES = {
        "isa": ["( isa {e:entity} {c:category} )"],
        "sub": ["( sub {c:category} {p:category} )"],
        "prop_true": ["( has {n:node} {P:bprop} )"],
        "prop_false": ["( not ( has {n:node} {P:bprop} ) )"],
        "val": ["( val {n:node} {A:cprop} {v:value} )"],
        "rel": ["( {r:frel} {x:entity} {y:entity} )", "( {r:trel} {x:entity} {y:entity} )"],
    }

    QUERY_TEMPLATES = {
        "prop": ["? ( has {e:entity} {P:bprop} )", "? ( val {e:entity} {P:cprop} ?x )"],
        "isa": ["? ( isa {e:entity} {c:category} )"],
        "trans": ["? ( {r:trel} {a:entity} {b:entity} )"],
    }

    def chain_templates(self, k: int) -> list[str]:
        inner = "{e:entity}"
        for i in range(1, k + 1):
            inner = f"( {{r{i}:frel}} {inner} )"
        return [f"? {inner}"]

    def detok(self, tokens: list[str]) -> str:
        return " ".join(tokens).replace("( ", "(").replace(" )", ")")

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[()]|[^\s()]+", text)
