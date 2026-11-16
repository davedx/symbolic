"""Encoding A: templated natural-language prose with paraphrase variety.

Several templates per fact and query kind; a ``variant`` index picks one. Names are
uninflected (no plurals), so every name is one word-level token.
"""

from __future__ import annotations

import re

from .base import Encoder


class ProseEncoder(Encoder):
    name = "prose"
    TRUE = "yes"
    FALSE = "no"

    FACT_TEMPLATES = {
        "isa": [
            "{e:entity} is a {c:category} .",
            "{e:entity} belongs to the {c:category} group .",
            "{e:entity} is one of the {c:category} .",
        ],
        "sub": [
            "every {c:category} is a {p:category} .",
            "{c:category} is a kind of {p:category} .",
            "the {c:category} group is part of the {p:category} group .",
        ],
        "prop_true": [
            "every {n:category} is {P:bprop} .",
            "all members of the {n:category} group are {P:bprop} .",
            "anything that is a {n:category} is {P:bprop} .",
            "{n:entity} is {P:bprop} .",
            "it is true that {n:entity} is {P:bprop} .",
            "{n:entity} is certainly {P:bprop} .",
        ],
        "prop_false": [
            "no {n:category} is {P:bprop} .",
            "members of the {n:category} group are never {P:bprop} .",
            "anything that is a {n:category} is not {P:bprop} .",
            "{n:entity} is not {P:bprop} .",
            "it is false that {n:entity} is {P:bprop} .",
            "{n:entity} is never {P:bprop} .",
        ],
        "val": [
            "every {n:category} has {A:cprop} {v:value} .",
            "the {A:cprop} of every {n:category} is {v:value} .",
            "members of the {n:category} group have {A:cprop} {v:value} .",
            "the {A:cprop} of {n:entity} is {v:value} .",
            "{n:entity} has {A:cprop} {v:value} .",
            "{v:value} is the {A:cprop} of {n:entity} .",
        ],
        "rel": [
            "the {r:frel} of {x:entity} is {y:entity} .",
            "{y:entity} is the {r:frel} of {x:entity} .",
            "{x:entity} has {y:entity} as its {r:frel} .",
            "{x:entity} is {r:trel} {y:entity} .",
            "{x:entity} lies {r:trel} {y:entity} .",
            "{x:entity} stands {r:trel} {y:entity} .",
        ],
    }

    QUERY_TEMPLATES = {
        "prop": [
            "is {e:entity} {P:bprop} ?",
            "is it true that {e:entity} is {P:bprop} ?",
            "would you say {e:entity} is {P:bprop} ?",
            "what is the {P:cprop} of {e:entity} ?",
            "which {P:cprop} does {e:entity} have ?",
            "what {P:cprop} does {e:entity} have ?",
        ],
        "isa": [
            "is {e:entity} a {c:category} ?",
            "does {e:entity} belong to the {c:category} group ?",
            "is {e:entity} one of the {c:category} ?",
        ],
        "trans": [
            "is {a:entity} {r:trel} {b:entity} ?",
            "does {a:entity} lie {r:trel} {b:entity} ?",
            "does {a:entity} stand {r:trel} {b:entity} ?",
        ],
    }

    def chain_templates(self, k: int) -> list[str]:
        rels_out = " ".join(f"the {{r{i}:frel}} of" for i in range(k, 0, -1))  # outermost first
        follow = " then ".join(f"{{r{i}:frel}}" for i in range(1, k + 1))
        return [
            f"what is {rels_out} {{e:entity}} ?",
            f"start at {{e:entity}} and follow {follow} . who do you reach ?",
            f"who is {rels_out} {{e:entity}} ?",
        ]

    def detok(self, tokens: list[str]) -> str:
        return re.sub(r" ([.?,])", r"\1", " ".join(tokens))

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[^\s.?,]+|[.?,]", text)
