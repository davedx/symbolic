"""Training/eval examples in a common format for every encoding.

Sequence layouts (token strings; the trainer maps them through ``Encoder.vocab``):

    KB document   <|bos|> f1 f2 ... fn <|eos|>                               loss on all but <|bos|>
    query         <|bos|> [<|ctx|> c1 ... cm] <|query|> q <|ans|> a <|eos|>   loss on a, <|eos|>
      proof=after   ... <|ans|> a <|proof|> p1 ... pk <|eos|>                 loss on a, proof, <|eos|>
      proof=before  ... <|query|> q <|proof|> p1 ... pk <|ans|> a <|eos|>     loss on proof, a, <|eos|>

``prompt_len`` is the length of the prefix given to the model at evaluation time
(up to and including <|ans|>, or <|proof|> when the proof comes first).

Regimes:
    weights  facts-in-weights: train on KB documents, then on queries with no context
    context  facts-in-context: every query carries the facts it needs plus distractors

Renaming (context regime, ``rename``): with a fixed world every query has a constant
answer, so a model can memorize the world through the train queries and ignore the
context (the M pilot did exactly that; see README). ``rename="nodes"`` gives every
example a fresh, consistent renaming of its entities and categories (drawn from the
world's own names, so the vocabulary is unchanged), applied to context, query, proof
and answer alike. The only way to answer is then to read the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from world.kb import World
from world.queries import Query

from .base import ANS, BOS, CTX, EOS, PROOF, QUERY, Encoder, stable_int
from .prose import ProseEncoder
from .symbolic import SymbolicEncoder, SymbolicFlatEncoder
from .triples import TriplesEncoder, TriplesUnmarkedEncoder

ENCODERS = {
    "prose": ProseEncoder,
    "symbolic": SymbolicEncoder,
    "triples": TriplesEncoder,
    "symbolic_flat": SymbolicFlatEncoder,  # ablation: decomposed chain queries
    "triples_unmarked": TriplesUnmarkedEncoder,  # ablation: no role markers
}


def make_encoder(name: str, world: World) -> Encoder:
    return ENCODERS[name](world)


@dataclass
class DataConfig:
    encoding: str = "symbolic"
    regime: str = "weights"  # weights | context
    proof: str = "none"  # none | before | after
    n_distractors: int = 6  # context regime: extra facts beyond those the proof needs
    distractor_ramp_epochs: int = 0  # context regime, training only: distractors grow linearly from 0 to
    # n_distractors over this many epochs (a curriculum for in-context lookup; evaluation always uses
    # the full n_distractors because the evaluator sets this to 0)
    distractor_ramp_start: int = 0  # distractors at epoch 0 of the ramp. With 0 distractors a 1-hop chain is
    # solvable positionally (copy the only fact's object), a shortcut that does not survive distractors;
    # starting at 2 forces lookup by matching from the start (README, "Diagnostics")
    related_fraction: float = 0.5  # share of distractors that mention a node of the proof
    paraphrase: str = "resample"  # resample: new template per epoch for train items; fixed: always variant 0
    kb_facts_per_doc: int = 16  # weights regime: facts per KB document
    rename: str = "none"  # context regime: none | entities | nodes (entities + categories), per example
    ctx_loss: bool = False  # context regime: also put loss on the context facts (LM objective on the context)
    seed: int = 0


@dataclass
class Example:
    tokens: list[str]
    loss_mask: list[int]
    prompt_len: int
    qid: str | None = None
    answer: list[str] = field(default_factory=list)
    n_facts: int = 0  # KB facts in a KB document, or context facts in a context example
    n_queries: int = 0


def _variant(cfg: DataConfig, key: str, epoch: int) -> int:
    if cfg.paraphrase == "fixed":
        return 0
    return stable_int(cfg.seed, key, epoch) % 1_000_003


# ------------------------------------------------------------------------ KB documents
def kb_documents(enc: Encoder, cfg: DataConfig, epoch: int = 0) -> list[Example]:
    """The whole KB, shuffled per epoch and chunked into documents."""
    facts = enc.world.facts
    rng = np.random.default_rng(stable_int(cfg.seed, "kb", epoch))
    order = rng.permutation(len(facts))
    docs = []
    for start in range(0, len(order), cfg.kb_facts_per_doc):
        chunk = [facts[i] for i in order[start : start + cfg.kb_facts_per_doc]]
        toks = [BOS]
        for f in chunk:
            toks += enc.fact(f, _variant(cfg, repr(f), epoch))
        toks.append(EOS)
        mask = [0] + [1] * (len(toks) - 1)
        docs.append(Example(toks, mask, prompt_len=1, n_facts=len(chunk)))
    return docs


# ------------------------------------------------------------------------- contexts
class ContextSampler:
    """Facts-in-context: needed facts (proof + overridden default) plus distractors."""

    def __init__(self, world: World):
        self.w = world
        self.by_node: dict[str, list[tuple]] = {}
        for f in world.facts:
            for x in _nodes(f):
                self.by_node.setdefault(x, []).append(f)

    def needed(self, q: Query) -> list[tuple]:
        out = [f for _, f in q.proof]
        if q.overridden is not None:
            out.append(q.overridden)
        return list(dict.fromkeys(out))  # a chain can traverse the same edge twice

    def sample(self, q: Query, cfg: DataConfig, epoch: int = 0) -> list[tuple]:
        rng = np.random.default_rng(stable_int(cfg.seed, "ctx", q.qid, epoch))
        needed = self.needed(q)
        exclude = set(needed)
        related = []
        seen = set()
        for f in needed:
            for x in _nodes(f):
                for g in self.by_node.get(x, ()):
                    if g not in exclude and g not in seen:
                        seen.add(g)
                        related.append(g)
        n_dist = n_distractors_at(cfg, epoch)
        n_rel = min(len(related), round(n_dist * cfg.related_fraction))
        picks = [related[i] for i in rng.choice(len(related), size=n_rel, replace=False)] if n_rel else []
        chosen = set(picks) | exclude
        facts = self.w.facts
        while len(picks) < n_dist:
            f = facts[int(rng.integers(len(facts)))]
            if f not in chosen:
                chosen.add(f)
                picks.append(f)
        ctx = needed + picks
        return [ctx[i] for i in rng.permutation(len(ctx))]


def _is_entity(x) -> bool:
    return isinstance(x, str) and len(x) > 1 and x[0] == "E" and x[1:].isdigit()


def _is_category(x) -> bool:
    return isinstance(x, str) and len(x) > 1 and x[0] == "C" and x[1:].isdigit()


def _map(x, m: dict):
    if isinstance(x, (tuple, list)):
        return type(x)(_map(y, m) for y in x)
    return m.get(x, x) if isinstance(x, str) else x


def _collect(x, out: list) -> None:
    if isinstance(x, (tuple, list)):
        for y in x:
            _collect(y, out)
    elif _is_entity(x) or _is_category(x):
        out.append(x)


def rename_example(world: World, q: Query, ctx: list[tuple], cfg: "DataConfig", epoch: int):
    """A fresh, consistent renaming of an example's nodes (see module docstring).

    Each entity (and, for ``nodes``, each category) mentioned anywhere in the example
    maps to a distinct random entity (category) of the world. Deterministic in
    (seed, qid, epoch).
    """
    rng = np.random.default_rng(stable_int(cfg.seed, "rename", q.qid, epoch))
    found: list = []
    _collect([q.args, q.answer, [f for _, f in q.proof], q.overridden, ctx], found)
    found = list(dict.fromkeys(found))
    m: dict[str, str] = {}
    for kind, pool, is_kind in (("E", world.entities, _is_entity), ("C", world.categories, _is_category)):
        if kind == "C" and cfg.rename != "nodes":
            continue
        xs = [x for x in found if is_kind(x)]
        if xs:
            picks = rng.choice(len(pool), size=len(xs), replace=False)
            m.update({x: pool[int(i)] for x, i in zip(xs, picks)})
    q2 = replace(
        q,
        args=_map(q.args, m),
        answer=_map(q.answer, m),
        proof=[(r, _map(f, m)) for r, f in q.proof],
        overridden=_map(q.overridden, m),
    )
    return q2, [_map(f, m) for f in ctx]


def n_distractors_at(cfg: "DataConfig", epoch: int) -> int:
    if cfg.distractor_ramp_epochs <= 0:
        return cfg.n_distractors
    a = min(cfg.distractor_ramp_start, cfg.n_distractors)
    return round(a + (cfg.n_distractors - a) * min(1.0, epoch / cfg.distractor_ramp_epochs))


def _nodes(f: tuple) -> list[str]:
    if f[0] in ("isa", "sub"):
        return [f[1], f[2]]
    if f[0] == "prop":
        return [f[1]]
    return [f[2], f[3]]


# -------------------------------------------------------------------- query examples
def query_example(
    enc: Encoder,
    q: Query,
    cfg: DataConfig,
    epoch: int = 0,
    ctx_sampler: ContextSampler | None = None,
) -> Example:
    ctx_toks: list[str] = []
    n_ctx = 0
    if cfg.regime == "context":
        sampler = ctx_sampler or ContextSampler(enc.world)
        ctx = sampler.sample(q, cfg, epoch)
        if cfg.rename != "none":
            q, ctx = rename_example(enc.world, q, ctx, cfg, epoch)
        n_ctx = len(ctx)
        ctx_toks = [CTX]
        for f in ctx:
            ctx_toks += enc.fact(f, _variant(cfg, q.qid + repr(f), epoch))
    elif cfg.regime != "weights":
        raise ValueError(cfg.regime)

    qtoks = enc.query(q.form, q.args, _variant(cfg, q.qid, epoch))
    ans = enc.answer(q.answer)
    proof: list[str] = []
    if cfg.proof != "none":
        for i, (_, f) in enumerate(q.proof):
            proof += enc.fact(f, _variant(cfg, q.qid + "proof" + str(i), epoch))

    prefix = [BOS] + ctx_toks + [QUERY] + qtoks
    if cfg.proof == "before":
        prefix.append(PROOF)
        target = proof + [ANS] + ans + [EOS]
        # the model generates the proof, then <|ans|>, then the answer
    elif cfg.proof == "after":
        prefix.append(ANS)
        target = ans + [PROOF] + proof + [EOS]
    elif cfg.proof == "none":
        prefix.append(ANS)
        target = ans + [EOS]
    else:
        raise ValueError(cfg.proof)
    toks = prefix + target
    mask = [0] * len(prefix) + [1] * len(target)
    if cfg.ctx_loss and ctx_toks:
        # loss on every context token after <|ctx|> (positions 2 .. 1 + len(ctx_toks) - 1)
        for i in range(2, 1 + len(ctx_toks)):
            mask[i] = 1
    return Example(toks, mask, prompt_len=len(prefix), qid=q.qid, answer=ans, n_facts=n_ctx, n_queries=1)


def extract_answer(toks: list[str], proof_mode: str) -> list[str]:
    """Pull the answer tokens out of a generated continuation (after prompt_len)."""
    if proof_mode == "before":
        if ANS not in toks:
            return []
        toks = toks[toks.index(ANS) + 1 :]
    out = []
    for t in toks:
        if t in (EOS, PROOF, ANS):
            break
        out.append(t)
    return out
