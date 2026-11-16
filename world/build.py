"""Build a world + query splits and write them to disk.

    uv run python -m world.build --config configs/world/default.yaml [--out data/world_default]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import WorldConfig
from .generator import generate_world
from .kb import World
from .queries import Query, build_pool
from .splits import SPLITS, HeldOut, assign_splits, choose_heldout


@dataclass
class Dataset:
    world: World
    queries: list[Query]  # only queries assigned to a split
    heldout: HeldOut
    dropped: dict

    def split(self, name: str) -> list[Query]:
        return [q for q in self.queries if q.split == name]

    def save(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.world.save(out / "world.json")
        with open(out / "queries.jsonl", "w") as f:
            for q in self.queries:
                f.write(json.dumps(q.to_dict()) + "\n")
        (out / "heldout.json").write_text(json.dumps(self.heldout.to_dict(), indent=1))
        (out / "stats.json").write_text(json.dumps(stats(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        world = World.load(path / "world.json")
        with open(path / "queries.jsonl") as f:
            queries = [Query.from_dict(json.loads(line)) for line in f]
        held = HeldOut.from_dict(json.loads((path / "heldout.json").read_text()))
        dropped = json.loads((path / "stats.json").read_text()).get("dropped", {})
        return cls(world, queries, held, dropped)


def build_dataset(cfg: WorldConfig) -> Dataset:
    world = generate_world(cfg)
    pool = build_pool(world)
    held = choose_heldout(world)
    dropped = assign_splits(world, pool, held)
    return Dataset(world, [q for q in pool if q.split is not None], held, dropped)


def stats(ds: Dataset) -> dict:
    w = ds.world
    fact_types = Counter(f[0] for f in w.facts)
    prop_facts_on = Counter("entity" if f[1] in w.entity_leaf else "category" for f in w.facts if f[0] == "prop")
    levels = Counter(w.cat_level[c] for c in w.categories)
    leaf_levels = Counter(w.cat_level[c] for c in w.leaves)
    out = {
        "fingerprint": w.config.fingerprint(),
        "n_entities": len(w.entities),
        "n_categories": len(w.categories),
        "categories_per_level": dict(sorted(levels.items())),
        "leaves_per_level": dict(sorted(leaf_levels.items())),
        "n_properties": dict(Counter(m["kind"] for m in w.properties.values())),
        "n_relations": dict(Counter(m["kind"] for m in w.relations.values())),
        "kb_facts": dict(fact_types),
        "kb_prop_facts_on": dict(prop_facts_on),
        "kb_total": len(w.facts),
        "dropped": ds.dropped,
        "splits": {},
    }
    for s in SPLITS:
        qs = ds.split(s)
        by_task = Counter(q.task_type for q in qs)
        by_hops = Counter(q.hops for q in qs)
        by_task_hops = Counter(f"{q.task_type}/{q.hops}" for q in qs)
        answers = {}
        for t in sorted(by_task):
            a = Counter(str(q.answer) for q in qs if q.task_type == t and isinstance(q.answer, bool))
            answers[t] = dict(a)
        out["splits"][s] = {
            "n": len(qs),
            "by_form": dict(Counter(q.form for q in qs)),
            "by_task": dict(sorted(by_task.items())),
            "by_hops": dict(sorted(by_hops.items())),
            "by_task_hops": dict(sorted(by_task_hops.items())),
            "bool_answers_by_task": answers,
        }
    return out


def render(ds: Dataset, q: Query) -> str:
    """Human-readable rendering using lexicon names (for eyeballing only)."""
    lex = ds.world.lexicon

    def n(x):
        if isinstance(x, tuple):
            return "∘".join(lex.get(i, i) for i in x)
        return lex.get(x, str(x)) if isinstance(x, str) else str(x)

    proof = "; ".join(f"{rule}:{'(' + ' '.join(n(t) for t in fact) + ')'}" for rule, fact in q.proof)
    return (
        f"[{q.split} | {q.task_type} | {q.hops} hops] {q.form}({', '.join(n(a) for a in q.args)}) "
        f"= {n(q.answer)}\n      proof: {proof}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=2, help="samples to print per split x task type")
    args = ap.parse_args()
    cfg = WorldConfig.from_yaml(args.config)
    ds = build_dataset(cfg)
    out = args.out or f"data/world_{Path(args.config).stem}"
    ds.save(out)
    print(json.dumps(stats(ds), indent=1))
    import numpy as np

    rng = np.random.default_rng(0)
    for s in SPLITS:
        for t in ("direct", "inherited", "relation", "exception"):
            qs = [q for q in ds.split(s) if q.task_type == t]
            for i in rng.choice(len(qs), size=min(args.samples, len(qs)), replace=False) if qs else []:
                print(render(ds, qs[i]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
