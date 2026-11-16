"""Per-encoding statistics and side-by-side samples.

    uv run python -m encoders.report --data data/world_default
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from world.build import Dataset, render

from .dataset import ENCODERS, ContextSampler, DataConfig, kb_documents, make_encoder, query_example


def mean_over_variants(lengths_fn, n_fn, items) -> float:
    total, count = 0.0, 0
    for it in items:
        n = n_fn(it)
        total += sum(lengths_fn(it, v) for v in range(n)) / n
        count += 1
    return total / max(count, 1)


def encoding_stats(ds: Dataset, name: str, max_queries: int = 5000) -> dict:
    w = ds.world
    enc = make_encoder(name, w)
    rng = np.random.default_rng(0)
    qs = ds.queries if len(ds.queries) <= max_queries else [ds.queries[i] for i in rng.choice(len(ds.queries), max_queries, replace=False)]
    fact_len = mean_over_variants(lambda f, v: len(enc.fact(f, v)), enc.n_fact_variants, w.facts)
    query_len = mean_over_variants(
        lambda q, v: len(enc.query(q.form, q.args, v)), lambda q: enc.n_query_variants(q.form, q.args), qs
    )
    by_form = {}
    for form in ("prop", "isa", "chain", "trans"):
        sub = [q for q in qs if q.form == form]
        if sub:
            by_form[form] = round(
                mean_over_variants(
                    lambda q, v: len(enc.query(q.form, q.args, v)), lambda q: enc.n_query_variants(q.form, q.args), sub
                ),
                2,
            )
    sampler = ContextSampler(w)
    ctx_cfg = DataConfig(encoding=name, regime="context")
    w_cfg = DataConfig(encoding=name, regime="weights")
    ctx_len = np.mean([len(query_example(enc, q, ctx_cfg, ctx_sampler=sampler).tokens) for q in qs[:2000]])
    w_len = np.mean([len(query_example(enc, q, w_cfg).tokens) for q in qs[:2000]])
    kb_tokens = sum(len(d.tokens) for d in kb_documents(enc, w_cfg))
    types = Counter(enc.vocab.types)
    return {
        "vocab_size": len(enc.vocab),
        "vocab_by_type": dict(types),
        "templates_fact": sum(len(t) for t in enc.fact_templates.values()),
        "templates_query": sum(len(t) for k, t in enc.query_templates.items() if not k.startswith("chain"))
        + len(enc.query_templates["chain1"]),
        "mean_tokens_per_fact": round(fact_len, 2),
        "mean_tokens_per_query": round(query_len, 2),
        "mean_tokens_per_query_by_form": by_form,
        "mean_tokens_query_example_weights": round(float(w_len), 2),
        "mean_tokens_query_example_context": round(float(ctx_len), 2),
        "kb_tokens_per_epoch": kb_tokens,
    }


def samples(ds: Dataset, n_per_form: int = 1, seed: int = 0) -> str:
    w = ds.world
    encs = {name: make_encoder(name, w) for name in ENCODERS}
    rng = np.random.default_rng(seed)
    lines = []
    picks = []
    for task in ("direct", "inherited", "relation", "exception"):
        for form in ("prop", "isa", "chain", "trans"):
            qs = [q for q in ds.queries if q.task_type == task and q.form == form and q.split == "train"]
            if qs:
                picks += [qs[i] for i in rng.choice(len(qs), size=min(n_per_form, len(qs)), replace=False)]
    for q in picks:
        lines.append(render(ds, q))
        for name, enc in encs.items():
            for v in range(enc.n_query_variants(q.form, q.args)):
                lines.append(f"   {name:9s} v{v}: {enc.detok(enc.query(q.form, q.args, v))}  ->  {enc.detok(enc.answer(q.answer))}")
        lines.append("")

    lines.append("=== KB facts (all variants) ===")
    kinds_seen = set()
    for f in w.facts:
        kind = (f[0], type(f[3]).__name__ if f[0] == "prop" else "", f[1] in w.entity_leaf if f[0] == "prop" else "")
        if f[0] == "rel":
            kind = ("rel", w.relations[f[1]]["kind"])
        if kind in kinds_seen:
            continue
        kinds_seen.add(kind)
        lines.append(str(f))
        for name, enc in encs.items():
            for v in range(enc.n_fact_variants(f)):
                lines.append(f"   {name:9s} v{v}: {enc.detok(enc.fact(f, v))}")

    q = next(q for q in ds.queries if q.task_type == "exception" and q.hops >= 2 and q.split == "train")
    lines.append("\n=== facts-in-context example (exception query; context = proof + overridden default + 6 distractors) ===")
    lines.append(render(ds, q))
    sampler = ContextSampler(w)
    for name, enc in encs.items():
        ex = query_example(enc, q, DataConfig(encoding=name, regime="context"), ctx_sampler=sampler)
        lines.append(f"   {name}: {enc.detok(ex.tokens)}")
    lines.append("\n=== proof-as-target examples (proof=after / proof=before), symbolic ===")
    enc = encs["symbolic"]
    for mode in ("after", "before"):
        ex = query_example(enc, q, DataConfig(encoding="symbolic", proof=mode))
        lines.append(f"   {mode}: {enc.detok(ex.tokens)}   [prompt = first {ex.prompt_len} tokens]")
    lines.append("\n=== KB document (weights regime, first 300 characters) ===")
    for name, enc in encs.items():
        doc = kb_documents(enc, DataConfig(encoding=name))[0]
        lines.append(f"   {name}: {enc.detok(doc.tokens)[:300]} ...")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_default")
    ap.add_argument("--no-samples", action="store_true")
    args = ap.parse_args()
    ds = Dataset.load(args.data)
    stats = {name: encoding_stats(ds, name) for name in ENCODERS}
    print(json.dumps(stats, indent=1))
    Path(args.data, "encoding_stats.json").write_text(json.dumps(stats, indent=1))
    if not args.no_samples:
        print(samples(ds))


if __name__ == "__main__":
    main()
