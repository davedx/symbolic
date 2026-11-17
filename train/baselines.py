"""Non-learning baselines per split x task type (Phase 4 groundwork).

    uv run python -m train.baselines [--world configs/world/default.yaml]

task_majority       most common train answer per (form, task type)
predicate_prior     most common train answer per predicate (property, or relation path
                    for chains, relation for trans); falls back to task_majority
category_prior      most common train answer per (entity's level-k category, predicate):
                    what a model learns from query statistics alone, without reading the KB
                    (k = the held-out category level); falls back to predicate_prior

``--run <dir>`` also stratifies a run's predictions by whether the gold answer agrees
with predicate_prior: the "disagree" stratum is the part of a split that cannot be
answered from train statistics alone.

A split is only informative about reasoning if a model beats these. In particular
test_heldout_category is built so that category_prior has no data for its pairs, and
its answers may disagree with predicate_prior (see README).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from .config import RunConfig
from .data import load_dataset


def _pred_key(q):
    if q.form == "prop":
        return ("prop", q.args[1])
    if q.form == "chain":
        return ("chain", tuple(q.args[1]))
    if q.form == "trans":
        return ("trans", q.args[1])
    return (q.form,)


def compute(ds, level: int) -> dict:
    w = ds.world

    def cat_at(e):
        cs = [c for c in w.entity_categories(e) if w.cat_level[c] == level]
        return cs[0] if cs else None

    task, pred, cat = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for q in ds.queries:
        if q.split != "train":
            continue
        a = repr(q.answer)
        task[(q.form, q.task_type)][a] += 1
        pred[_pred_key(q)][a] += 1
        cat[(cat_at(q.args[0]), _pred_key(q))][a] += 1

    def top(tab, k):
        c = tab.get(k)
        return c.most_common(1)[0][0] if c else None

    hits = defaultdict(lambda: Counter())
    for q in ds.queries:
        a = repr(q.answer)
        tm = top(task, (q.form, q.task_type))
        pp = top(pred, _pred_key(q)) or tm
        cp = top(cat, (cat_at(q.args[0]), _pred_key(q))) or pp
        for key in (q.split, f"{q.split}/{q.task_type}"):
            h = hits[key]
            h["n"] += 1
            h["task_majority"] += tm == a
            h["predicate_prior"] += pp == a
            h["category_prior"] += cp == a
    return {
        k: {"n": h["n"], **{m: round(h[m] / h["n"], 4) for m in ("task_majority", "predicate_prior", "category_prior")}}
        for k, h in sorted(hits.items())
    }


def prior_table(ds) -> dict:
    tab = defaultdict(Counter)
    for q in ds.queries:
        if q.split == "train":
            tab[_pred_key(q)][repr(q.answer)] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def stratify_run(ds, run_dir) -> dict:
    """Model accuracy per split x task type x (gold agrees / disagrees with predicate_prior)."""
    qs = {q.qid: q for q in ds.queries}
    prior = prior_table(ds)
    hits = defaultdict(lambda: [0, 0])
    with open(f"{run_dir}/predictions.jsonl") as f:
        for line in f:
            p = json.loads(line)
            q = qs[p["qid"]]
            pp = prior.get(_pred_key(q))
            stratum = "agree" if pp == repr(q.answer) else "disagree"
            for key in (f"{p['split']}/{stratum}", f"{p['split']}/{p['task_type']}/{stratum}"):
                hits[key][0] += p["correct"]
                hits[key][1] += 1
    return {k: {"acc": round(c / n, 4), "n": n} for k, (c, n) in sorted(hits.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="configs/world/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", default=None, help="stratify this run's predictions by prior agreement")
    args = ap.parse_args()
    if args.run:
        from .config import RunConfig

        ds = load_dataset(RunConfig.from_yaml(f"{args.run}/config.yaml").world, "data")
        for k, v in stratify_run(ds, args.run).items():
            print(f"{k:50} {v['acc']:.3f}  n={v['n']}")
        return
    ds = load_dataset(args.world, "data")
    res = compute(ds, ds.world.config.splits.heldout_category_level)
    print(f"{'split':42} {'n':>6} {'task_maj':>9} {'pred_prior':>10} {'cat_prior':>9}")
    for k, v in res.items():
        print(f"{k:42} {v['n']:6d} {v['task_majority']:9.3f} {v['predicate_prior']:10.3f} {v['category_prior']:9.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
