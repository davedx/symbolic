"""Pre-run sanity checks (Phase 3):

1. overfit   200 train queries, no KB documents: train accuracy must reach >= 99%.
2. shuffled  train answers permuted within (form, answer kind); the run must not
             generalize: held-out accuracy must stay near the majority-class baseline,
             while a matched run with true labels must clearly beat it.

    uv run python -m train.checks overfit  [--encodings ...] [overrides...]
    uv run python -m train.checks shuffled [--encodings ...] [overrides...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from world.splits import SPLITS, TRAIN

from .config import RunConfig
from .data import answer_kind, load_dataset
from .trainer import Trainer

OVERFIT = {
    "train.train_subset": 200,
    "train.stage1_frac": 0.0,
    "train.kb_per_query_epoch": 0.0,
    "train.budget": 1_500_000,
    "train.max_batch_tokens": 512,
    "train.warmup_frac": 0.05,
    "train.weight_decay": 0.0,
    "train.n_evals": 5,
    "train.eval_max_per_split": 200,
    "train.final_eval_max_per_split": 1000,
}

SHUFFLED = {
    "train.budget": 8_000_000,
    "train.n_evals": 4,
    "train.eval_max_per_split": 500,
    "train.final_eval_max_per_split": 2000,
}


def majority_baseline(ds) -> dict[str, float]:
    """Accuracy of answering the most common train answer for each (form, answer kind)."""
    train = ds.split(TRAIN)
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for q in train:
        counts[(q.form, answer_kind(q))][str(q.answer)] += 1
    out = {}
    for s in SPLITS:
        qs = ds.split(s)
        hit = sum(str(q.answer) == counts[(q.form, answer_kind(q))].most_common(1)[0][0] for q in qs)
        out[s] = hit / len(qs)
    return out


def run(cfg_path: str, overrides: dict, extra: list[str], runs_dir: str) -> dict:
    ov = [f"{k}={v}" for k, v in overrides.items()] + extra
    cfg = RunConfig.from_yaml(cfg_path, ov)
    return Trainer(cfg, runs_dir=runs_dir).train()


def top(res: dict) -> dict[str, float]:
    return {k: round(v["acc"], 3) for k, v in res["accuracy"].items() if "/" not in k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["overfit", "shuffled"])
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--config", default="configs/runs/base.yaml")
    ap.add_argument("--encodings", nargs="+", default=["prose", "symbolic", "triples"])
    ap.add_argument("--runs-dir", default="runs/checks")
    args = ap.parse_args()

    report = {}
    ok = True
    if args.check == "overfit":
        for enc in args.encodings:
            res = run(args.config, OVERFIT, [f"data.encoding={enc}", *args.overrides], args.runs_dir)
            acc = res["accuracy"]["train"]["acc"]
            passed = acc >= 0.99
            ok &= passed
            report[enc] = {"train_acc": acc, "pass": passed, "all": top(res)}
            print(f"OVERFIT {enc}: train acc {acc:.4f} -> {'PASS' if passed else 'FAIL'}")
    else:
        ds = load_dataset(RunConfig.from_yaml(args.config, args.overrides).world)
        base = majority_baseline(ds)
        print("majority baseline:", {k: round(v, 3) for k, v in base.items()})
        for enc in args.encodings:
            true = run(args.config, SHUFFLED, [f"data.encoding={enc}", *args.overrides], args.runs_dir)
            shuf = run(
                args.config, {**SHUFFLED, "train.label_shuffle": True}, [f"data.encoding={enc}", *args.overrides], args.runs_dir
            )
            t, s = top(true), top(shuf)
            # shuffled must stay within 5 points of the majority baseline on every test split
            near_chance = all(s[k] <= base[k] + 0.05 for k in base if k != TRAIN)
            # and true labels must beat it clearly somewhere
            true_better = any(t[k] >= base[k] + 0.15 for k in base if k != TRAIN)
            passed = near_chance and true_better
            ok &= passed
            report[enc] = {"true_labels": t, "shuffled": s, "majority": base, "pass": passed}
            print(f"SHUFFLED {enc}: true {t}\n          shuffled {s}\n          -> {'PASS' if passed else 'FAIL'}")
    out = Path(args.runs_dir) / f"check_{args.check}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
