"""Deterministic, resumable training stream.

The stream is a sequence of *epochs*; each epoch's batch plan is a pure function of
(seed, stage, epoch), so a run can resume from (stage, epoch, batch index).

facts-in-weights:
    stage 1  KB documents only (paraphrases resampled each epoch), until stage1_frac
             of the budget is used
    stage 2  each epoch = one pass over the train queries (no context) mixed with
             kb_per_query_epoch passes over the KB documents
facts-in-context:
    stage 2 only; each epoch = one pass over the train queries, each with its context

Batches are homogeneous in padded length (multiples of ``bucket``) and hold at most
``max_batch_tokens`` padded tokens, which keeps the number of compiled shapes small.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from encoders.base import Encoder, stable_int
from encoders.dataset import ContextSampler, DataConfig, Example, kb_documents, query_example
from world.build import Dataset, build_dataset
from world.config import WorldConfig
from world.queries import Query
from world.splits import TRAIN

from .config import TrainConfig


def load_dataset(world_config: str | Path, cache_dir: str | Path = "data") -> Dataset:
    """Build the world from its config, caching it under data/world_<fingerprint>."""
    cfg = WorldConfig.from_yaml(world_config)
    path = Path(cache_dir) / f"world_{cfg.fingerprint()}"
    if (path / "queries.jsonl").exists():
        return Dataset.load(path)
    ds = build_dataset(cfg)
    ds.save(path)
    return ds


def answer_kind(q: Query) -> str:
    if isinstance(q.answer, bool):
        return "bool"
    return "entity" if q.form == "chain" else "value"


def train_queries(ds: Dataset, tc: TrainConfig) -> list[Query]:
    """Train split, optionally subsampled and/or with shuffled labels (controls)."""
    qs = ds.split(TRAIN)
    if tc.train_forms:
        qs = [q for q in qs if q.form in tc.train_forms]
    if tc.train_max_hops:
        qs = [q for q in qs if q.hops <= tc.train_max_hops]
    if tc.train_subset:
        rng = np.random.default_rng(stable_int(tc.seed, "subset"))
        qs = [qs[i] for i in sorted(rng.choice(len(qs), size=tc.train_subset, replace=False))]
    if tc.label_shuffle:
        rng = np.random.default_rng(stable_int(tc.seed, "shuffle"))
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, q in enumerate(qs):
            groups[(q.form, answer_kind(q))].append(i)
        qs = list(qs)
        for key in sorted(groups):
            idx = groups[key]
            perm = rng.permutation(len(idx))
            answers = [qs[idx[j]].answer for j in perm]
            for i, a in zip(idx, answers):
                qs[i] = dataclasses.replace(qs[i], answer=a)
    return qs


@dataclass
class Batch:
    inputs: mx.array  # (B, T) int32
    targets: mx.array  # (B, T) int32
    weights: mx.array  # (B, T) float32, 1 on loss positions
    n_tokens: int  # non-pad input tokens
    n_kb_facts: int
    n_queries: int
    n_ctx_facts: int


@dataclass
class StreamState:
    stage: int = 1
    epoch: int = 0
    batch: int = 0  # index of the next batch within the epoch plan


class TrainStream:
    def __init__(self, enc: Encoder, queries: list[Query], dc: DataConfig, tc: TrainConfig):
        self.enc = enc
        self.queries = queries
        self.dc = dataclasses.replace(dc, seed=tc.seed)
        self.tc = tc
        self.sampler = ContextSampler(enc.world) if dc.regime == "context" else None
        self.state = StreamState(stage=1 if self._has_stage1() else 2)
        self._plan: list[list[Example]] | None = None
        self._plan_key = None
        self.pad = enc.vocab.pad_id

    def _has_stage1(self) -> bool:
        return self.dc.regime == "weights" and self.tc.stage1_frac > 0

    # ---------------------------------------------------------------- epoch plans
    def _examples(self, stage: int, epoch: int) -> list[Example]:
        if stage == 1:
            return kb_documents(self.enc, self.dc, epoch=epoch)
        exs = [query_example(self.enc, q, self.dc, epoch, self.sampler) for q in self.queries]
        if self.dc.regime == "weights" and self.tc.kb_per_query_epoch > 0:
            whole, frac = divmod(self.tc.kb_per_query_epoch, 1.0)
            for k in range(int(whole)):
                exs += kb_documents(self.enc, self.dc, epoch=100_000 + epoch * 100 + k)
            if frac > 0:
                docs = kb_documents(self.enc, self.dc, epoch=100_000 + epoch * 100 + 99)
                exs += docs[: round(frac * len(docs))]
        return exs

    def _build_plan(self, stage: int, epoch: int) -> list[list[Example]]:
        rng = np.random.default_rng(stable_int(self.tc.seed, "plan", stage, epoch))
        exs = self._examples(stage, epoch)
        buckets: dict[int, list[Example]] = defaultdict(list)
        for ex in exs:
            buckets[self.padded_len(len(ex.tokens) - 1)].append(ex)
        batches = []
        for T in sorted(buckets):
            items = buckets[T]
            order = rng.permutation(len(items))
            bs = max(1, min(self.tc.max_batch_size, self.tc.max_batch_tokens // T))
            for s in range(0, len(items), bs):
                batches.append([items[i] for i in order[s : s + bs]])
        return [batches[i] for i in rng.permutation(len(batches))]

    def padded_len(self, n: int) -> int:
        b = self.tc.bucket
        return b * ((n + b - 1) // b)

    def plan(self) -> list[list[Example]]:
        key = (self.state.stage, self.state.epoch)
        if self._plan_key != key:
            self._plan = self._build_plan(*key)
            self._plan_key = key
        return self._plan

    # ------------------------------------------------------------------ iteration
    def end_stage1(self) -> None:
        if self.state.stage == 1:
            self.state = StreamState(stage=2, epoch=0, batch=0)

    def next_batch(self) -> Batch:
        plan = self.plan()
        if self.state.batch >= len(plan):
            self.state = StreamState(self.state.stage, self.state.epoch + 1, 0)
            plan = self.plan()
        exs = plan[self.state.batch]
        self.state.batch += 1
        return self.collate(exs)

    def collate(self, exs: list[Example]) -> Batch:
        T = self.padded_len(max(len(e.tokens) - 1 for e in exs))
        B = len(exs)
        inp = np.full((B, T), self.pad, dtype=np.int32)
        tgt = np.full((B, T), self.pad, dtype=np.int32)
        w = np.zeros((B, T), dtype=np.float32)
        n_tok = 0
        for i, e in enumerate(exs):
            ids = self.enc.vocab.encode(e.tokens)
            n = len(ids) - 1
            inp[i, :n] = ids[:-1]
            tgt[i, :n] = ids[1:]
            w[i, :n] = e.loss_mask[1:]
            n_tok += n
        return Batch(
            mx.array(inp),
            mx.array(tgt),
            mx.array(w),
            n_tokens=n_tok,
            n_kb_facts=sum(e.n_facts for e in exs if e.qid is None),
            n_queries=sum(e.n_queries for e in exs),
            n_ctx_facts=sum(e.n_facts for e in exs if e.qid is not None),
        )
