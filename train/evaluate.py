"""Exact-match evaluation per split x task type x hop count.

Answers are single tokens. With proof = none/after the answer is the first generated
token, so one teacher-forced forward pass over the prompt suffices (argmax at the
last prompt position). With proof = before the model must first generate its proof,
so we decode greedily (left-padded batches + key padding mask) until <|ans|>, then
read the next token.

Evaluation always renders queries with a fixed data seed (0) and epoch 0, so every
run (any training seed) is scored on identical token sequences.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import mlx.core as mx
import numpy as np

from encoders.base import ANS, EOS, Encoder
from encoders.dataset import ContextSampler, DataConfig, extract_answer, query_example
from world.build import Dataset
from world.queries import Query
from world.splits import SPLITS, TRAIN

from .model import Transformer, padding_mask

EVAL_SEED = 1234


def eval_sets(ds: Dataset, train_qs: list[Query], max_per_split: int) -> dict[str, list[Query]]:
    """Fixed (seeded) subsample per split. 'train' uses the queries actually trained on
    (so the shuffled-label control reports memorization of its own labels)."""
    out = {}
    for s in SPLITS:
        qs = train_qs if s == TRAIN else ds.split(s)
        if max_per_split and len(qs) > max_per_split:
            rng = np.random.default_rng(EVAL_SEED)
            qs = [qs[i] for i in sorted(rng.choice(len(qs), size=max_per_split, replace=False))]
        out[s] = qs
    return out


class Evaluator:
    def __init__(self, enc: Encoder, dc: DataConfig, batch_size: int = 256):
        self.enc = enc
        self.dc = dataclasses.replace(dc, seed=0, distractor_ramp_epochs=0)  # eval: full distractors
        self.batch_size = batch_size
        self.sampler = ContextSampler(enc.world) if dc.regime == "context" else None
        self._cache: dict[str, tuple] = {}

    def _prompt(self, q: Query):
        if q.qid not in self._cache:
            ex = query_example(self.enc, q, self.dc, epoch=0, ctx_sampler=self.sampler)
            ids = self.enc.vocab.encode(ex.tokens)
            gold = self.enc.vocab.encode(ex.answer)
            self._cache[q.qid] = (ids[: ex.prompt_len], gold)
        return self._cache[q.qid]

    def predict(self, model: Transformer, queries: list[Query]) -> list[dict]:
        if self.dc.proof == "before":
            return self._predict_generate(model, queries)
        return self._predict_single(model, queries)

    def _rows(self, queries, preds):
        rows = []
        for q, pred in zip(queries, preds):
            _, gold = self._prompt(q)
            rows.append(
                {
                    "qid": q.qid,
                    "split": q.split,
                    "task_type": q.task_type,
                    "form": q.form,
                    "hops": q.hops,
                    "gold": self.enc.vocab.decode(gold),
                    "pred": self.enc.vocab.decode(pred),
                    "correct": list(pred) == list(gold),
                }
            )
        return rows

    def _predict_single(self, model, queries):
        pad = self.enc.vocab.pad_id
        preds = []
        order = sorted(range(len(queries)), key=lambda i: len(self._prompt(queries[i])[0]))
        out = [None] * len(queries)
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            prompts = [self._prompt(queries[i])[0] for i in idx]
            T = max(len(p) for p in prompts)
            arr = np.full((len(idx), T), pad, dtype=np.int32)
            for j, p in enumerate(prompts):
                arr[j, : len(p)] = p
            logits = model(mx.array(arr))
            last = mx.array([len(p) - 1 for p in prompts])
            picked = logits[mx.arange(len(idx)), last]
            am = np.array(mx.argmax(picked, axis=-1))
            for j, i in enumerate(idx):
                out[i] = [int(am[j])]
        preds = out
        return self._rows(queries, preds)

    def _predict_generate(self, model, queries, max_new: int = 96):
        pad = self.enc.vocab.pad_id
        ans_id = self.enc.vocab.index[ANS]
        eos_id = self.enc.vocab.index[EOS]
        out = [None] * len(queries)
        order = sorted(range(len(queries)), key=lambda i: len(self._prompt(queries[i])[0]))
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            seqs = [list(self._prompt(queries[i])[0]) for i in idx]
            done = [False] * len(idx)
            gen = [[] for _ in idx]
            for _ in range(max_new):
                T = max(len(x) for x in seqs)
                arr = np.full((len(idx), T), pad, dtype=np.int32)
                padm = np.ones((len(idx), T), dtype=bool)
                for j, x in enumerate(seqs):  # left padding
                    arr[j, T - len(x) :] = x
                    padm[j, T - len(x) :] = False
                logits = model(mx.array(arr), mask=padding_mask(mx.array(padm)))
                nxt = np.array(mx.argmax(logits[:, -1], axis=-1))
                for j in range(len(idx)):
                    if done[j]:
                        continue
                    t = int(nxt[j])
                    seqs[j].append(t)
                    gen[j].append(t)
                    # stop once the answer token after <|ans|> is produced, or at <|eos|>
                    if t == eos_id or (len(gen[j]) >= 2 and gen[j][-2] == ans_id):
                        done[j] = True
                if all(done):
                    break
            for j, i in enumerate(idx):
                toks = self.enc.vocab.decode(gen[j])
                out[i] = self.enc.vocab.encode(extract_answer(toks, "before"))
        return self._rows(queries, out)


def summarize(rows: list[dict]) -> dict:
    """Accuracy per split, per split x task type, per split x task type x hops, and
    per split x form x hops (separates entity-valued chains from boolean trans queries,
    which share the `relation` task type)."""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        for key in (
            (r["split"],),
            (r["split"], r["task_type"]),
            (r["split"], r["task_type"], r["hops"]),
            (r["split"], f"form:{r['form']}"),
            (r["split"], f"form:{r['form']}", r["hops"]),
        ):
            acc[key][0] += r["correct"]
            acc[key][1] += 1
    out: dict = {}
    for key, (c, n) in sorted(acc.items(), key=lambda kv: tuple(map(str, kv[0]))):
        out["/".join(map(str, key))] = {"acc": c / n, "n": n}
    return out


def kb_recall(model: Transformer, enc: Encoder, facts: list[tuple], batch_size: int = 256) -> dict:
    """Storage diagnostic (facts-in-weights): cloze the *last slot* of each KB fact.

    Each fact is rendered with variant 0 as a KB document of one fact; the prompt is
    everything before the fact's last name token, and the target is that token (the
    object of a relation, the parent of a sub/isa edge, the value of a categorical
    statement; boolean property statements are skipped because their last
    name is the property itself). Accuracy per fact kind tells whether the KB is stored at all,
    separately from whether queries can retrieve it.
    """
    from encoders.base import BOS

    names = set(enc.world.lexicon.values())
    pad = enc.vocab.pad_id
    items = []
    for f in facts:
        toks = [BOS] + enc.fact(f, 0)
        last = max(i for i, t in enumerate(toks) if t in names)
        if f[0] == "prop":
            if enc.world.properties[f[2]]["kind"] != "categorical":
                continue  # a boolean statement's last name is its property: not a well-posed cloze
            kind = "prop:categorical"
        else:
            kind = f[0] if f[0] != "rel" else f"rel:{enc.world.relations[f[1]]['kind']}"
        items.append((enc.vocab.encode(toks[:last]), enc.vocab.index[toks[last]], kind))
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        T = max(len(items[i][0]) for i in idx)
        arr = np.full((len(idx), T), pad, dtype=np.int32)
        for j, i in enumerate(idx):
            arr[j, : len(items[i][0])] = items[i][0]
        logits = model(mx.array(arr))
        last = mx.array([len(items[i][0]) - 1 for i in idx])
        am = np.array(mx.argmax(logits[mx.arange(len(idx)), last], axis=-1))
        for j, i in enumerate(idx):
            ok = int(am[j]) == items[i][1]
            for key in ("all", items[i][2]):
                hits[key][0] += ok
                hits[key][1] += 1
    return {k: {"acc": c / n, "n": n} for k, (c, n) in sorted(hits.items())}
