# Project: Representations of Meaning — Small-Model Prototype

You are helping me build a research prototype on my **M5 Max Mac Studio (128GB, Apple silicon)**. Use **MLX** for all model training. Work in this directory. Use `uv` for environment and dependency management (Python 3.11+). Dependencies: `mlx`, `numpy`, `pytest`, `pyyaml`, `scikit-learn` (for probes), `matplotlib` (for plots). Avoid PyTorch unless strictly needed.

## Research question

Does the **encoding of knowledge in the training data** (natural-language prose vs. typed symbolic tokens vs. graph triples, and later continuous binding vectors) change:
1. **Sample efficiency and generalization** — especially compositional and multi-hop generalization beyond training distribution;
2. **The geometry of the model's internal representations** — are categories/properties linearly readable, and do different encodings converge to similar latent geometry (testing the "Platonic representation hypothesis" from a new angle)?

The idea underneath: "language at the edges, structure in the middle." Later phases will add structural primitives (binding operations, hyperbolic category embeddings) to the model itself.

## Guiding principle

Trustworthiness comes from **experimental design, not model size**. Every result must be reproducible from a seed, compared against baselines, run across multiple seeds, and evaluated on held-out splits that separate memorization from generalization. Prefer small, correct, well-tested code over clever code. When in doubt, ask me rather than guess on design decisions that affect validity.

---

## Phase 1 — Synthetic world + ground-truth simulator (`world/`)

Build a deterministic, seeded generator of a synthetic world:

- **Entities:** ~1,000 (configurable), with invented names (pronounceable nonsense words, no real-world semantics leaking in).
- **Category taxonomy:** a tree, depth 4–5, ~50 categories. Each entity belongs to exactly one leaf category (and transitively to its ancestors).
- **Properties:** ~100 boolean/categorical properties attached at various taxonomy levels and **inherited** downward, with a configurable rate of **exceptions** (an entity or subcategory overrides an inherited property, like "penguins don't fly").
- **Relations:** ~10 types between entities, including at least: one transitive (e.g. `ancestor_of`-like), one symmetric, one pair of inverses, and some plain relations.
- **Simulator / oracle:** a reasoner that answers any query with ground truth, including the derivation path (proof trace).

**Query/task types:**
1. Direct property lookup (1 hop, fact stated directly).
2. Inherited property (multi-hop through taxonomy).
3. Relation composition (follow k relation edges; uses transitivity/inverse/symmetry).
4. Exceptions & negation (inherited default overridden; "does X have P?" → false).

Each query has: the question, the answer, the hop count, task type, and proof trace.

**Splits (critical — design these carefully and document them):**
- `train`
- `test_iid` — same distribution as train, unseen query instances.
- `test_heldout_combo` — held-out entity×property combinations (the underlying facts are derivable from training data, but the specific combination was never asked).
- `test_length` — train on chains ≤3 hops, test on 4–6 hops.
- `test_heldout_category` — some category×relation combinations never appear in training queries.

The **knowledge base** (the facts themselves) is shared training material; splits apply to **queries**. Make explicit, and test, which facts must appear in training for held-out queries to be answerable in principle.

**Tests (`pytest`):** determinism under a fixed seed; inheritance correctness; exceptions override correctly; transitivity/symmetry/inverse closure correctness; no leakage between splits (no test query appears in train); every held-out query is derivable from training facts; hop counts are correct.

## Phase 2 — Encoders (`encoders/`)

Render the *same* knowledge base and queries in three encodings, each with its own tokenizer/vocab:

- **A. Prose:** templated natural language with paraphrase variety (multiple templates per fact type, randomized). Simple word-level or small BPE tokenizer trained on the generated corpus.
- **B. Typed symbolic:** S-expression style, e.g. `(isa rex dog) (has mammal fur) ? (has rex fur)` → `true`. Small typed vocab: entity tokens, category tokens, relation tokens, property tokens, logical operators, structural tokens.
- **C. Graph triples:** facts as `<s> <r> <o>` triples with role markers; queries as triple patterns.

All encodings share a common training-example format: `[knowledge context or none] [query] → [answer]` plus optionally the proof trace as a target (make this a config flag; default: answer only).

Two training regimes, configurable:
- **Facts-in-weights:** model first trains on the knowledge base as text, then on queries without context (tests storage + retrieval).
- **Facts-in-context:** relevant facts provided in the context window (tests reasoning separately from storage).

Report per encoding: vocab size, mean tokens per fact, mean tokens per query. Print a few samples of each for me to eyeball. Add tests that decoding round-trips and that all three encodings of a given query have the same answer.

## Phase 3 — MLX training (`train/`)

- A small **nanoGPT-style decoder-only transformer in MLX** (pre-LN, RoPE, GELU/SwiGLU, weight tying). Configurable: layers 4–6, width 256–384, heads 4–8. Target sizes: ~2M, ~8M, ~20M params.
- Training loop: AdamW, cosine schedule with warmup, bf16 where it helps, gradient clipping, checkpointing, resume.
- Logging to CSV/JSONL (no external services): step, loss, tokens seen, facts seen, wall time, periodic eval accuracy per split and task type.
- **Config files (YAML)** for every run; each run writes its config, git hash (if a repo), seed, and results into `runs/<run_id>/`.
- **Matching across encodings:** support budgets by (a) equal tokens/compute and (b) equal number of facts/queries seen. Both are needed because prose is longer.

**Before any real runs:**
1. Overfit a tiny subset (e.g. 200 examples) to ~100% — proves the pipeline works.
2. Shuffled-label control — must fail to generalize.
3. Measure tokens/sec at each model size and report estimated wall time for the full grid.

## Phase 4 — Evaluation and baselines (`eval/`)

- **Metrics:** exact-match accuracy per split × task type × hop count. Mean ± std across seeds.
- **Baselines:**
  - Lookup table: memorizes all training query→answer pairs; answers majority/unknown otherwise. Shows what pure memorization achieves.
  - Oracle (the simulator): upper bound.
  - Majority-class baseline per task type.
- **Representation analysis:**
  - Linear probes (logistic regression) on residual-stream activations at each layer: can we read off an entity's category (at each taxonomy level) and its properties? Use held-out entities for probe testing.
  - **CKA** between models trained on different encodings, computed on activations for the *same* set of queries (aligned by query ID, taking the representation at the final query token and at entity-mention positions). Also CKA across seeds of the same encoding as a reference.
  - Check whether taxonomy hierarchy shows up geometrically (e.g. are parent–child category difference vectors approximately orthogonal to sibling distinctions?).
- **Plots:** accuracy vs. hop count per encoding; learning curves (accuracy vs. facts seen); probe accuracy by layer; CKA heatmap.

## Phase 5 — The sweep

Grid: 3 encodings × 3 model sizes × 3 seeds (× 2 budget-matching modes if time allows). Write a sweep runner that can run unattended, skip completed runs, and resume after interruption. Then a single `report.py` that aggregates `runs/` into a markdown report with tables and plots.

## Phase 6 (later — do NOT build yet, but keep the architecture open for it)

- **Encoding D — continuous binding vectors (VSA / holographic reduced representations):** facts as `role ⊛ filler` bound vectors, bundled; model predicts vectors with cosine/MSE loss; decode via nearest-neighbour cleanup memory.
- **Structured-core variants:** explicit binding operation as a layer; hyperbolic (Poincaré) embeddings for categories; typed-token embeddings.
- Comparison of an explicit structured core against the emergent middle-layer representations of the baseline transformer.

Keep the model/data interfaces clean enough that these can be added as new modules without rewriting phases 1–5.

---

## Repo layout (suggested)

```
pyproject.toml
README.md            # how to run everything, design decisions, split definitions
configs/             # YAML configs
world/               # generator, simulator, splits
encoders/            # prose, symbolic, triples (+ tokenizers)
train/               # MLX model, training loop
eval/                # metrics, baselines, probes, CKA
sweep.py
report.py
tests/
runs/                # gitignored outputs
```

## How to work

- Initialize a git repo and commit after each phase passes its tests.
- Work phase by phase. At the end of each phase, stop and show me: what was built, test results, sample outputs, and any design decisions you made that affect validity. Wait for my go-ahead before the next phase.
- Phase 1 and 2 must be fully tested before writing training code.
- Keep everything deterministic from seeds. Document every non-obvious choice in the README.
- If any MLX API is uncertain, check the installed version's docs/source rather than guessing.

Start with Phase 1.
