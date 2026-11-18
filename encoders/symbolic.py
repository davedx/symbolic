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


class SymbolicFlatEncoder(SymbolicEncoder):
    """Ablation of encoding B: chain queries as a sequence of fact-shaped steps.

        ? ( r1 rex ?x1 ) ( r2 ?x1 ? )   instead of   ? ( r2 ( r1 rex ) )

    Everything else is identical to SymbolicEncoder. The triples encoding writes chains
    this way (one fact-shaped pattern per hop, in execution order, joined by variables),
    and it is the only encoding whose chains length-generalize in the context regime.
    This variant separates that query decomposition from triples' other differences
    (role markers, argument order). See README, "Grid slice 1".
    """

    name = "symbolic_flat"

    def chain_templates(self, k: int) -> list[str]:
        parts = []
        subj = "{e:entity}"
        for i in range(1, k + 1):
            obj = "?" if i == k else f"?x{i}"
            parts.append(f"( {{r{i}:frel}} {subj} {obj} )")
            subj = obj
        return ["? " + " ".join(parts)]


class SymbolicPathEncoder(SymbolicEncoder):
    """Ablation of encoding B: nested chain queries without nested brackets.

        ? ( r3 r2 r1 rex )   instead of   ? ( r3 ( r2 ( r1 rex ) ) )

    Same outermost-first order and the same relation and entity tokens as
    SymbolicEncoder, but the query ends with a single ")" whatever the chain length.
    The nested form ends with one ")" per hop, so a 4-6 hop query ends in a run of
    closers never seen in training (max 3 hops); nested prose, which ends with one "?",
    length-generalizes when paraphrases are fixed. This variant tests whether the
    run of closing brackets is what caps nested symbolic chains at 3 hops (README).
    """

    name = "symbolic_path"

    def chain_templates(self, k: int) -> list[str]:
        rels = " ".join(f"{{r{i}:frel}}" for i in range(k, 0, -1))  # outermost first
        return [f"? ( {rels} {{e:entity}} )"]
