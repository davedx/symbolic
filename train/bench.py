"""Throughput benchmark and wall-time estimate for the sweep grid.

For each (encoding, regime, size, dtype) it builds one epoch's real batch plan, times
the compiled train step once per distinct batch shape (after warmup), and weights the
per-shape times by how often each shape occurs. That gives tokens/sec for the real
mix of KB documents, queries and contexts. It also times one periodic eval and one
final eval, then estimates wall time per run and for the whole grid.

    uv run python -m train.bench --sizes S M L --dtypes float32 bfloat16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from encoders.dataset import DataConfig, make_encoder

from .config import RunConfig
from .data import TrainStream, load_dataset, train_queries
from .evaluate import Evaluator, eval_sets
from .model import ModelConfig, Transformer, count_params
from .trainer import make_optimizer


def time_step(model, opt, batch, reps: int) -> float:
    def loss_fn(model, x, y, w):
        ce = nn.losses.cross_entropy(model(x), y, reduction="none")
        return (ce * w).sum() / mx.maximum(w.sum(), 1.0)

    lag = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, w):
        loss, grads = lag(model, x, y, w)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    for _ in range(3):
        mx.eval(step(batch.inputs, batch.targets, batch.weights), model.parameters(), opt.state)
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(step(batch.inputs, batch.targets, batch.weights), model.parameters(), opt.state)
    return (time.perf_counter() - t) / reps


def bench_one(ds, base: RunConfig, encoding, regime, size, dtype, reps, eval_n) -> dict:
    dc = DataConfig(**{**vars(base.data), "encoding": encoding, "regime": regime})
    tc = base.train
    enc = make_encoder(encoding, ds.world)
    qs = train_queries(ds, tc)
    stream = TrainStream(enc, qs, dc, tc)
    stream.end_stage1()
    plan = stream.plan()  # stage 2: queries (+ KB docs in weights regime)
    shapes = Counter()
    sample = {}
    tokens = 0
    items = 0
    for exs in plan:
        b = stream.collate(exs)
        key = tuple(b.inputs.shape)
        shapes[key] += 1
        sample.setdefault(key, b)
        tokens += b.n_tokens
        items += b.n_kb_facts + b.n_queries
    mx.random.seed(0)
    model = Transformer(ModelConfig(vocab_size=len(enc.vocab), size=size, compute_dtype=dtype))
    opt = make_optimizer(tc)
    epoch_time = 0.0
    for key, n in shapes.items():
        epoch_time += n * time_step(model, opt, sample[key], reps)
    ev = Evaluator(enc, dc, tc.eval_batch_size)
    sets = eval_sets(ds, qs, eval_n)
    n_eval = sum(len(v) for v in sets.values())
    ev.predict(model, sets["train"][:64])  # warm
    t = time.perf_counter()
    for v in sets.values():
        ev.predict(model, v)
    eval_time = time.perf_counter() - t
    return {
        "encoding": encoding,
        "regime": regime,
        "size": size,
        "dtype": dtype,
        "n_params": count_params(model),
        "tokens_per_sec": tokens / epoch_time,
        "items_per_sec": items / epoch_time,
        "tokens_per_item": tokens / items,
        "n_shapes": len(shapes),
        "eval_queries_per_sec": n_eval / eval_time,
    }


def estimate(r: dict, base: RunConfig, final_eval_queries: int, budget_tokens: int, budget_items: int) -> dict:
    tc = base.train
    periodic = tc.n_evals * 5 * tc.eval_max_per_split / r["eval_queries_per_sec"]
    final = final_eval_queries / r["eval_queries_per_sec"]
    return {
        "tokens_budget_hours": (budget_tokens / r["tokens_per_sec"] + periodic + final) / 3600,
        "items_budget_hours": (budget_items / r["items_per_sec"] + periodic + final) / 3600,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/runs/base.yaml")
    ap.add_argument("--encodings", nargs="+", default=["prose", "symbolic", "triples"])
    ap.add_argument("--regimes", nargs="+", default=["weights", "context"])
    ap.add_argument("--sizes", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--budget-tokens", type=float, default=None, help="default: config budget")
    ap.add_argument("--budget-items", type=float, default=None, help="default: same wall-equivalent as symbolic tokens")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/bench.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)  # runs/ is gitignored, so absent in a fresh clone

    def save(extra=None):
        out.write_text(json.dumps({"results": results, **(extra or {})}, indent=1))

    base = RunConfig.from_yaml(args.config)
    ds = load_dataset(base.world)
    final_q = sum(1 for q in ds.queries)
    budget_tokens = int(args.budget_tokens or base.train.budget)
    results = []
    for size in args.sizes:
        for dtype in args.dtypes:
            for regime in args.regimes:
                for enc in args.encodings:
                    r = bench_one(ds, base, enc, regime, size, dtype, args.reps, args.eval_n)
                    results.append(r)
                    save()  # keep partial results if the run is interrupted
                    print(
                        f"{size} {dtype:8s} {regime:7s} {enc:8s} params {r['n_params'] / 1e6:5.2f}M  "
                        f"{r['tokens_per_sec'] / 1e3:7.1f}k tok/s  {r['items_per_sec']:8.0f} items/s  "
                        f"tok/item {r['tokens_per_item']:5.1f}  eval {r['eval_queries_per_sec']:6.0f} q/s",
                        flush=True,
                    )

    # items budget: by default the items that symbolic consumes under the token budget
    sym = [r for r in results if r["encoding"] == "symbolic"]
    tok_per_item = {r["regime"]: r["tokens_per_item"] for r in sym}
    print("\nEstimated wall time per run and for the grid (3 encodings x sizes x seeds):")
    grid = defaultdict(float)
    for r in results:
        bi = int(args.budget_items or budget_tokens / tok_per_item[r["regime"]])
        est = estimate(r, base, final_q, budget_tokens, bi)
        r.update(est, budget_items=bi, budget_tokens=budget_tokens)
        for mode in ("tokens", "items"):
            grid[(r["dtype"], r["regime"], mode)] += est[f"{mode}_budget_hours"] * args.seeds
    for r in results:
        print(
            f"  {r['size']} {r['dtype']:8s} {r['regime']:7s} {r['encoding']:8s} "
            f"tokens-budget {r['tokens_budget_hours'] * 60:6.1f} min  items-budget {r['items_budget_hours'] * 60:6.1f} min"
        )
    for (dtype, regime, mode), h in sorted(grid.items()):
        print(f"  GRID {dtype:8s} {regime:7s} {mode}-matched: {h:6.1f} h  ({args.seeds} seeds)")
    save({"grid_hours": {"/".join(k): v for k, v in grid.items()}})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
