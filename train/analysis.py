"""Seed-level reporting for bimodal skill acquisition (Phase 4/5).

    uv run python -m train.analysis "runs/grid_ctx/*" ["runs/depth_ctx/*" ...]

Skill acquisition in these runs is bimodal across seeds (a skill is either learned,
often in a sudden jump, or not at all), so mean accuracy is misleading. For every
condition (run id without its ``_s<seed>`` suffix) this reports:

- **success fraction** per skill: seeds whose final test_iid accuracy passes the bar
  (isa > .9, trans > .9, 1-hop chain > .5);
- **acquisition point**: tokens at the first periodic eval where the skill's *train*
  accuracy passes the bar (chain: all-hop train accuracy > .5);
- **still rising**: the skill's train accuracy rose by > .05 over the last 20% of
  training, i.e. the budget may have cut the run off mid-transition;
- per-hop chain accuracy (1-3 hops from test_iid, 4-6 from test_length) and the
  longest chain length solved at >= 50%, per seed;
- Fisher exact tests for success fractions between groups, via ``fisher(a, b)``.

All conditions compared this way must share lr / weight decay / budget; the report
prints them so this can be checked.
"""

from __future__ import annotations

import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SKILLS = {
    # skill: (final test metric, bar, train metric tracked in eval.jsonl, bar)
    "isa": ("test_iid/form:isa", 0.9, "train/form:isa", 0.9),
    "trans": ("test_iid/form:trans", 0.9, "train/form:trans", 0.9),
    "chain": ("test_iid/form:chain/1", 0.5, "train/form:chain", 0.5),
}
HOPS = [(1, "test_iid"), (2, "test_iid"), (3, "test_iid"), (4, "test_length"), (5, "test_length"), (6, "test_length")]


def load_run(d: Path) -> dict | None:
    if not (d / "results.json").exists():
        return None
    res = json.loads((d / "results.json").read_text())
    acc = {k: v["acc"] for k, v in res["accuracy"].items()}
    evals = [json.loads(line) for line in open(d / "eval.jsonl")]
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    out = {"name": d.name, "acc": acc, "evals": evals, "cfg": cfg, "skills": {}}
    periodic = [e for e in evals if not e.get("final")]
    for skill, (test_key, bar, train_key, tbar) in SKILLS.items():
        ok = acc.get(test_key, 0.0) > bar
        first = next((e["tokens"] for e in periodic if e["acc"].get(train_key, 0.0) > tbar), None)
        tail = [e for e in periodic if e["progress"] >= 0.8]
        rising = bool(tail) and (tail[-1]["acc"].get(train_key, 0) - tail[0]["acc"].get(train_key, 0)) > 0.05
        out["skills"][skill] = {"ok": ok, "acquired_at": first, "still_rising": rising}
    out["hops"] = {h: acc.get(f"{split}/form:chain/{h}") for h, split in HOPS}
    solved = [h for h, a in out["hops"].items() if a is not None and a >= 0.5]
    out["longest_chain"] = max(solved) if solved else 0
    return out


def condition(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


def fisher(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Two-sided Fisher exact p for successes/trials a=(k1, n1) vs b=(k2, n2)."""
    k1, n1 = a
    k2, n2 = b
    K, N = k1 + k2, n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)

    lo, hi = max(0, K - n2), min(K, n1)
    obs = p(k1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def report(patterns: list[str]) -> dict:
    runs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            r = load_run(Path(d))
            if r:
                runs.append(r)
    groups: dict[str, list] = defaultdict(list)
    for r in runs:
        groups[condition(r["name"])].append(r)
    summary = {}
    for cond, rs in sorted(groups.items()):
        t = rs[0]["cfg"]["train"]
        print(f"\n== {cond}  (n={len(rs)}; lr {t['lr']}, wd {t['weight_decay']}, budget {t['budget'] / 1e6:.0f}M)")
        s = {"n": len(rs)}
        for skill in SKILLS:
            k = sum(r["skills"][skill]["ok"] for r in rs)
            at = [r["skills"][skill]["acquired_at"] for r in rs]
            rising = sum(r["skills"][skill]["still_rising"] for r in rs if not r["skills"][skill]["ok"])
            at_s = ", ".join("-" if a is None else f"{a / 1e6:.0f}M" for a in at)
            flag = f"  [{rising} failed seed(s) still rising at end]" if rising else ""
            print(f"  {skill:6} {k}/{len(rs)} seeds   acquired at: {at_s}{flag}")
            s[skill] = (k, len(rs))
        print("  chain accuracy by hops (per seed):")
        for r in rs:
            hops = " ".join("  -  " if a is None else f"{a:.2f}" for a in r["hops"].values())
            print(f"    {r['name'][-3:]}  {hops}   longest >=.5: {r['longest_chain']}")
        s["longest_chain"] = [r["longest_chain"] for r in rs]
        summary[cond] = s
    return summary


if __name__ == "__main__":
    report(sys.argv[1:] or ["runs/grid_ctx/*"])
