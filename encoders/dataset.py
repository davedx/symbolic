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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world.kb import World
from world.queries import Query

from .base import ANS, BOS, CTX, EOS, PROOF, QUERY, Encoder, stable_int
from .prose import ProseEncoder
from .symbolic import SymbolicEncoder
from .triples import TriplesEncoder

ENCODERS = {"prose": ProseEncoder, "symbolic": SymbolicEncoder, "triples": TriplesEncoder}


def make_encoder(name: str, world: World) -> Encoder:
    return ENCODERS[name](world)


@dataclass
class DataConfig:
    encoding: str = "symbolic"
    regime: str = "weights"  # weights | context
    proof: str = "none"  # none | before | after
    n_distractors: int = 6  # context regime: extra facts beyond those the proof needs
    related_fraction: float = 0.5  # share of distractors that mention a node of the proof
    paraphrase: str = "resample"  # resample: new template per epoch for train items; fixed: always variant 0
    kb_facts_per_doc: int = 16  # weights regime: facts per KB document
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
        n_rel = min(len(related), round(cfg.n_distractors * cfg.related_fraction))
        picks = [related[i] for i in rng.choice(len(related), size=n_rel, replace=False)] if n_rel else []
        chosen = set(picks) | exclude
        facts = self.w.facts
        while len(picks) < cfg.n_distractors:
            f = facts[int(rng.integers(len(facts)))]
            if f not in chosen:
                chosen.add(f)
                picks.append(f)
        ctx = needed + picks
        return [ctx[i] for i in rng.permutation(len(ctx))]


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
