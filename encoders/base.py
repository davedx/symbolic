"""Template-based encoders shared machinery.

Each encoding is a table of *templates*: whitespace-separated token strings where slots
look like ``{name:type}``. The same table drives rendering and parsing, so every
encoding is invertible by construction and the tests can check that each rendering
parses back to exactly one fact or query.

Fact kinds (bindings -> fact tuple):
    isa        e, c            ("isa", e, c)
    sub        c, p            ("sub", c, p)
    prop_true  n, P            ("prop", n, P, True)      n is an entity or category
    prop_false n, P            ("prop", n, P, False)
    val        n, A, v         ("prop", n, A, v)         categorical
    rel        x, r, y         ("rel", r, x, y)

Query kinds (bindings -> (form, args)):
    prop   e, P           ("prop", (e, P))       P boolean or categorical
    isa    e, c           ("isa", (e, c))
    chain  e, r1..rk      ("chain", (e, (r1, ..., rk)))
    trans  a, r, b        ("trans", (a, r, b))

Slot types: entity, category, node (entity|category), bprop (boolean or individual
property), cprop (categorical property), value, frel (non-transitive relation),
trel (transitive relation).

Names are single tokens in every encoding (word-level tokenization), so all three
encodings agree on what an atomic symbol is and differ only in surface structure.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from world.kb import PROP_CAT, REL_TRANSITIVE, World

SLOT_RE = re.compile(r"^\{(\w+):(\w+)\}$")

PAD, BOS, EOS, CTX, QUERY, PROOF, ANS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|ctx|>", "<|query|>", "<|proof|>", "<|ans|>",
)
SPECIALS = [PAD, BOS, EOS, CTX, QUERY, PROOF, ANS]

TOKEN_TYPES = ("special", "literal", "answer", "entity", "category", "property", "value", "relation")


def stable_int(*parts) -> int:
    """Deterministic (cross-process, cross-machine) hash of the given parts."""
    h = hashlib.sha256(json.dumps(parts, default=str).encode()).digest()
    return int.from_bytes(h[:8], "little")


class ParseError(ValueError):
    pass


class Template:
    def __init__(self, kind: str, text: str):
        self.kind = kind
        self.text = text
        self.tokens = text.split()
        self.slots: dict[int, tuple[str, str]] = {}
        for i, tok in enumerate(self.tokens):
            m = SLOT_RE.match(tok)
            if m:
                self.slots[i] = (m.group(1), m.group(2))

    @property
    def literals(self) -> list[str]:
        return [t for i, t in enumerate(self.tokens) if i not in self.slots]

    def accepts(self, bindings: dict, typer: "Typer") -> bool:
        names = {name for name, _ in self.slots.values()}
        if names != set(bindings):
            return False
        return all(typer.is_a(bindings[name], typ) for name, typ in self.slots.values())

    def render(self, bindings: dict, lexicon: dict[str, str]) -> list[str]:
        out = []
        for i, tok in enumerate(self.tokens):
            if i in self.slots:
                out.append(lexicon[bindings[self.slots[i][0]]])
            else:
                out.append(tok)
        return out

    def match(self, toks: list[str], typer: "Typer") -> dict | None:
        if len(toks) != len(self.tokens):
            return None
        b: dict[str, str] = {}
        for i, (tt, tok) in enumerate(zip(self.tokens, toks)):
            if i in self.slots:
                name, typ = self.slots[i]
                ident = typer.name2id.get(tok)
                if ident is None or not typer.is_a(ident, typ):
                    return None
                if name in b and b[name] != ident:
                    return None
                b[name] = ident
            elif tt != tok:
                return None
        return b


class Typer:
    """Classifies world ids into slot types."""

    def __init__(self, world: World):
        self.w = world
        self.name2id = {v: k for k, v in world.lexicon.items()}
        self.values = {v for p in world.properties.values() for v in p.get("values", [])}

    def is_a(self, ident: str, typ: str) -> bool:
        w = self.w
        if typ == "entity":
            return ident in w.entity_leaf
        if typ == "category":
            return ident in w.cat_level
        if typ == "node":
            return ident in w.entity_leaf or ident in w.cat_level
        if typ == "bprop":
            return ident in w.properties and w.properties[ident]["kind"] != PROP_CAT
        if typ == "cprop":
            return ident in w.properties and w.properties[ident]["kind"] == PROP_CAT
        if typ == "value":
            return ident in self.values
        if typ == "frel":
            return ident in w.relations and w.relations[ident]["kind"] != REL_TRANSITIVE
        if typ == "trel":
            return ident in w.relations and w.relations[ident]["kind"] == REL_TRANSITIVE
        raise ValueError(typ)

    def token_type(self, ident: str) -> str:
        w = self.w
        if ident in w.entity_leaf:
            return "entity"
        if ident in w.cat_level:
            return "category"
        if ident in w.properties:
            return "property"
        if ident in self.values:
            return "value"
        if ident in w.relations:
            return "relation"
        raise ValueError(ident)


# ------------------------------------------------------------------ fact/query <-> kinds
def fact_to_kind(world: World, fact: tuple) -> tuple[str, dict]:
    t = fact[0]
    if t == "isa":
        return "isa", {"e": fact[1], "c": fact[2]}
    if t == "sub":
        return "sub", {"c": fact[1], "p": fact[2]}
    if t == "prop":
        _, n, p, v = fact
        if world.properties[p]["kind"] == PROP_CAT:
            return "val", {"n": n, "A": p, "v": v}
        return ("prop_true" if v else "prop_false"), {"n": n, "P": p}
    if t == "rel":
        return "rel", {"x": fact[2], "r": fact[1], "y": fact[3]}
    raise ValueError(fact)


def kind_to_fact(kind: str, b: dict) -> tuple:
    if kind == "isa":
        return ("isa", b["e"], b["c"])
    if kind == "sub":
        return ("sub", b["c"], b["p"])
    if kind == "prop_true":
        return ("prop", b["n"], b["P"], True)
    if kind == "prop_false":
        return ("prop", b["n"], b["P"], False)
    if kind == "val":
        return ("prop", b["n"], b["A"], b["v"])
    if kind == "rel":
        return ("rel", b["r"], b["x"], b["y"])
    raise ValueError(kind)


def query_to_kind(form: str, args: tuple) -> tuple[str, dict]:
    if form == "prop":
        return "prop", {"e": args[0], "P": args[1]}
    if form == "isa":
        return "isa", {"e": args[0], "c": args[1]}
    if form == "chain":
        e, rels = args
        return f"chain{len(rels)}", {"e": e, **{f"r{i + 1}": r for i, r in enumerate(rels)}}
    if form == "trans":
        return "trans", {"a": args[0], "r": args[1], "b": args[2]}
    raise ValueError(form)


def kind_to_query(kind: str, b: dict) -> tuple[str, tuple]:
    if kind == "prop":
        return "prop", (b["e"], b["P"])
    if kind == "isa":
        return "isa", (b["e"], b["c"])
    if kind.startswith("chain"):
        k = int(kind[5:])
        return "chain", (b["e"], tuple(b[f"r{i + 1}"] for i in range(k)))
    if kind == "trans":
        return "trans", (b["a"], b["r"], b["b"])
    raise ValueError(kind)


# ----------------------------------------------------------------------------- encoder
class Encoder:
    """Subclasses define: name, FACT_TEMPLATES, QUERY_TEMPLATES, chain_templates(k),
    TRUE/FALSE answer words, LITERAL_TYPES (optional finer typing), detok/tokenize."""

    name: str = "base"
    FACT_TEMPLATES: dict[str, list[str]] = {}
    QUERY_TEMPLATES: dict[str, list[str]] = {}
    TRUE: str = "true"
    FALSE: str = "false"

    def __init__(self, world: World, max_chain: int | None = None):
        self.world = world
        self.typer = Typer(world)
        self.max_chain = max_chain or world.config.queries.max_hops
        self.fact_templates = {k: [Template(k, t) for t in ts] for k, ts in self.FACT_TEMPLATES.items()}
        self.query_templates = {k: [Template(k, t) for t in ts] for k, ts in self.QUERY_TEMPLATES.items()}
        for k in range(1, self.max_chain + 1):
            self.query_templates[f"chain{k}"] = [Template(f"chain{k}", t) for t in self.chain_templates(k)]

    # subclasses -------------------------------------------------------------
    def chain_templates(self, k: int) -> list[str]:
        raise NotImplementedError

    def detok(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    # rendering --------------------------------------------------------------
    def _candidates(self, table, kind, bindings):
        cands = [t for t in table[kind] if t.accepts(bindings, self.typer)]
        if not cands:
            raise ValueError(f"{self.name}: no template for {kind} {bindings}")
        return cands

    def n_fact_variants(self, fact: tuple) -> int:
        kind, b = fact_to_kind(self.world, fact)
        return len(self._candidates(self.fact_templates, kind, b))

    def n_query_variants(self, form: str, args: tuple) -> int:
        kind, b = query_to_kind(form, args)
        return len(self._candidates(self.query_templates, kind, b))

    def fact(self, fact: tuple, variant: int = 0) -> list[str]:
        kind, b = fact_to_kind(self.world, fact)
        cands = self._candidates(self.fact_templates, kind, b)
        return cands[variant % len(cands)].render(b, self.world.lexicon)

    def query(self, form: str, args: tuple, variant: int = 0) -> list[str]:
        kind, b = query_to_kind(form, args)
        cands = self._candidates(self.query_templates, kind, b)
        return cands[variant % len(cands)].render(b, self.world.lexicon)

    def answer(self, answer) -> list[str]:
        if answer is True:
            return [self.TRUE]
        if answer is False:
            return [self.FALSE]
        return [self.world.lexicon[answer]]

    # parsing ----------------------------------------------------------------
    def _parse(self, toks, table):
        hits = []
        for kind, temps in table.items():
            for t in temps:
                b = t.match(toks, self.typer)
                if b is not None:
                    hits.append((kind, b))
        if len(hits) != 1:
            raise ParseError(f"{self.name}: {len(hits)} parses for {' '.join(toks)!r}")
        return hits[0]

    def parse_fact(self, toks: list[str]) -> tuple:
        kind, b = self._parse(toks, self.fact_templates)
        fact = kind_to_fact(kind, b)
        if fact[0] == "prop" and self.world.properties[fact[2]]["kind"] == PROP_CAT:
            if fact[3] not in self.world.properties[fact[2]]["values"]:
                raise ParseError("value does not belong to attribute")
        return fact

    def parse_query(self, toks: list[str]) -> tuple[str, tuple]:
        return kind_to_query(*self._parse(toks, self.query_templates))

    def parse_answer(self, toks: list[str]):
        if toks == [self.TRUE]:
            return True
        if toks == [self.FALSE]:
            return False
        if len(toks) == 1 and toks[0] in self.typer.name2id:
            return self.typer.name2id[toks[0]]
        raise ParseError(f"bad answer {toks}")

    # vocabulary -------------------------------------------------------------
    @cached_property
    def vocab(self) -> "Vocab":
        tokens: list[str] = list(SPECIALS)
        types: list[str] = ["special"] * len(SPECIALS)
        seen = set(tokens)

        def add(tok, typ):
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
                types.append(typ)

        add(self.TRUE, "answer")
        add(self.FALSE, "answer")
        for table in (self.fact_templates, self.query_templates):
            for temps in table.values():
                for t in temps:
                    for lit in t.literals:
                        add(lit, "literal")
        for ident, name in self.world.lexicon.items():
            add(name, self.typer.token_type(ident))
        return Vocab(tokens, types)


@dataclass
class Vocab:
    tokens: list[str]
    types: list[str]

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.tokens)}
        if len(self.index) != len(self.tokens):
            raise ValueError("duplicate tokens in vocab")

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.index[t] for t in toks]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.index[PAD]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tokens": self.tokens, "types": self.types}))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        d = json.loads(Path(path).read_text())
        return cls(d["tokens"], d["types"])
