# Representations of Meaning — small-model prototype

Does the **encoding of knowledge** in training data (prose vs typed symbolic tokens vs
graph triples, later continuous binding vectors) change (1) sample efficiency and
compositional / multi-hop generalization and (2) the geometry of internal
representations? See `semantic-representations-prototype-prompt.md` for the full brief.

Status: **Phases 1 (world), 2 (encoders) and 3 (MLX training) done.** Phases 4–5 to follow.

## Quick start

```bash
uv sync                                    # Python 3.12, mlx, numpy, pytest, ...
uv run pytest                              # full test suite (~3 s)
uv run python -m world.build --config configs/world/default.yaml   # -> data/world_default/
uv run python -m encoders.report --data data/world_default         # encoding stats + side-by-side samples
uv run python -m train.run configs/runs/smoke.yaml                 # whole pipeline on the tiny world (~10 s)
uv run python -m train.run configs/runs/base.yaml data.encoding=prose model.size=M train.seed=1
uv run python -m train.checks overfit                              # pre-run check 1
uv run python -m train.checks shuffled                             # pre-run check 2
uv run python -m train.bench                                       # tokens/sec per size + grid wall-time estimate
```

`world.build` writes `world.json` (KB + lexicon), `queries.jsonl` (every query with its
split, answer, hop count, proof trace), `heldout.json` and `stats.json`, and prints
stats plus sample queries rendered with their nonsense names.

## Layout

```
world/      config.py     dataclass config, loaded from YAML (unknown keys rejected)
            names.py      nonsense-name grammar + dictionary blocklist (world/data/)
            kb.py         World container, fact tuples, indexes, (de)serialization
            generator.py  seeded generator: taxonomy, entities, properties, exceptions, relations
            oracle.py     ground-truth simulator: answer + proof trace + task type + hops
            closure.py    independent forward-chaining reasoner (cross-checks the oracle)
            proofcheck.py replays proof traces against the KB
            queries.py    Query object, query pool construction
            splits.py     held-out selection and split routing
            build.py      CLI, Dataset save/load, stats
encoders/   base.py       template engine (render + parse from one table), Vocab, fact/query kinds
            prose.py      A: templated English, 3 paraphrases per fact/query kind
            symbolic.py   B: typed S-expressions
            triples.py    C: <s> <r> <o> edges, queries as triple patterns
            dataset.py    common example format, regimes, context sampling, proof targets
            report.py     vocab size, tokens per fact/query, samples
train/      model.py      MLX decoder-only transformer (pre-LN, RoPE, SwiGLU/GELU, tied embeddings)
            config.py     RunConfig = world + data + model + train sections (YAML + dotted overrides)
            data.py       deterministic, resumable batch stream; regimes; controls
            evaluate.py   exact-match eval (single forward pass, or greedy decoding for proof=before)
            trainer.py    AdamW + warmup/cosine, grad clipping, compiled step, checkpoint/resume, logs
            run.py        CLI for one run
            checks.py     overfit and shuffled-label checks
            bench.py      throughput benchmark + wall-time estimate
configs/runs/   base.yaml (default run), smoke.yaml (tiny world, XS model)
configs/world/  default.yaml (1000 entities), tiny.yaml (120 entities, for tests)
tests/          test_world.py, test_oracle.py, test_splits.py, test_encoders.py, test_train.py
```

## The world

Everything is generated from `seed`. Each component (taxonomy, entities, properties,
exceptions, relations, names, queries, splits) has its own RNG stream
`default_rng([seed, stream])`, so changing one part of the config does not reshuffle
the others.

**Names.** Every entity, category, property, value and relation gets a 2–3 syllable
nonsense name from a fixed grammar (`ONSET VOWEL [CODA]`). Any string the grammar can
produce that is also in the macOS dictionary (`/usr/share/dict/words`, 9.5k matches)
is blocklisted. The blocklist is committed (`world/data/name_blocklist.txt`), so
generation does not depend on the machine. Internally everything uses opaque ids
(`E0042`, `C07`, `P013`, `V005`, `R3`). Names only matter to the encoders.

**Taxonomy.** A forest of 2 top-level categories (the implicit root is *not* a
category), grown breadth-first. Each node has 2–3 children. At levels ≥ 3 a node is a
leaf with p = 0.5, and level 5 is the maximum. Trees are resampled until the size is
50 ± 6. Default world: 50 categories, levels {1: 2, 2: 6, 3: 16, 4: 14, 5: 12}, with
leaves at levels 3, 4 and 5. Each entity is in exactly one leaf; each leaf gets at
least 5 entities.

**Properties (100).**
- 70 *inherited boolean* and 10 *inherited categorical* (3–5 values) properties.
  Each is attached to 1–3 mutually unrelated categories (none is an ancestor of
  another) with a default value. The first attachment is always a leaf or a parent of
  a leaf, so every property has short inheritance queries available for training.
- *Category exceptions*: each descendant of an attachment overrides the inherited
  value with p = 0.10. Subtrees are walked top-down, so overrides can nest (exceptions
  to exceptions).
- *Entity exceptions*: each entity overrides each inherited property it has with
  p = 0.05.
- An override always changes the value (no redundant restatements; tested).
- 20 *individual* boolean properties are stated only on entities. Each entity has 3
  of them, each true or false.
- **No closed-world assumption for properties.** A property has a *domain* (the
  entities under its attachments). Inside the domain its value, true or false, is
  always derivable from a stated fact. Outside the domain it is undefined and never
  queried.
- Resolution: the most specific statement wins (entity > deepest category).

**Relations (10).** All are random, so they can only be answered by retrieving
facts, not by finding a regularity.
- 6 *functional*: every entity has exactly one object (≠ itself).
- 1 *symmetric*: a random perfect matching. Each pair is stated once, in a random
  direction.
- 1 *inverse pair* `r`/`r⁻¹`: `r` is a random derangement. Each edge is stated once,
  as either `r(a,b)` or `r⁻¹(b,a)` (50/50).
- 1 *transitive*: a strict order generated by a forest of direct links. Entities are
  split into trees of 25; each new node links to one of the last 3 nodes, which gives
  deep trees. Only direct links are stated.

Default KB: 13,496 facts (1000 `isa`, 48 `sub`, 3,988 `prop` [206 on categories,
3,782 on entities], 8,460 `rel`).

## Queries, task types, hop counts

| form | args | answer | task type |
|---|---|---|---|
| `prop`  | entity, property | bool / value | `direct` (individual property), `inherited`, or `exception` (the deciding statement overrides a different, more general one) |
| `isa`   | entity, category | bool | `direct` if 1 hop, else `inherited` |
| `chain` | entity, (r1..rk) | entity | `relation`: r_k(…r_1(e)), over functional, symmetric and inverse relations |
| `trans` | a, r, b | bool | `relation`: does a r⁺ b hold? |

**Hops = number of KB facts in the proof** (`len(proof)`, tested against an
independent recomputation). So:

- an individual property, or an entity-level exception, takes 1 hop;
- a property inherited from the leaf takes 2 hops (`isa`, `prop`), and each level
  above the leaf adds one;
- `isa` at the leaf level takes 1 hop;
- a k-step chain or k-link order path takes k hops.

**Proof traces** are lists of `(rule, fact)` with rules `stated, isa, sub, inherit,
override, rel, symmetric, inverse, trans`. Exceptions also record the `overridden`
fact (the more general default). Facts-in-context mode in Phase 2 will use it as the
"tempting distractor".

**Negatives that are not stated** come from two explicit structural axioms. They are
recorded in `rules` and should be stated in the encoders' documentation:
- `disjoint`: the taxonomy is a tree and each entity is in exactly one leaf, so an
  entity is in exactly one category per level. Negative `isa` queries use a category
  at a level no deeper than the entity's leaf. Hop count = steps to reach that level.
- `asymmetric`: transitive relations are strict orders. The only `trans` negatives
  are reversed true pairs (b r⁺ a where a r⁺ b), with hops = the path length, so
  positives and negatives are balanced per hop count. Incomparable pairs are never
  asked.

**Query pool.**
- All defined `prop` queries.
- `isa`: the true ancestor plus 1 random same-level negative, for every entity and
  level.
- `chain`: 2,000 sampled per hop count (1–6). Immediate back-and-forth steps
  (`s` then `s`; `r` then `r⁻¹`) are excluded, because they trivially return to the
  start.
- `trans`: up to 1,000 positives per hop count plus their reversed negatives.
- Each (form, task, hops) stratum is capped at 5,000 queries.

**Validation.** The oracle walks up from the entity. `closure.py` forward-chains the
whole KB to a fixpoint (top-down property propagation, naive symmetric / inverse /
transitive closure). The two agree on every query in both the tiny and the default
world. `proofcheck.py` replays every proof: facts are in the KB, steps connect, no
more specific statement is skipped, and the answer follows.

## Splits

The **KB is shared training material for every split**. Splits partition **queries**.
A query's identity is `(form, args)`; `qid` is its string form and is the cross-encoding
alignment key.

`touches` lists every (entity, predicate) application a query makes. For chains this
means *every step*, including intermediate entities.

Routing (`world/splits.py`). Each query is tested against three held-out axes:

| axis / split | definition |
|---|---|
| `test_length` | hops ∈ [4, 6] (train has ≤ 3) |
| `test_heldout_category` | touches a held-out (category, predicate) pair: an entity anywhere under the category has the predicate applied to it. Default: 3 relations and 5 inherited properties, each paired with a random level-2 category |
| `test_heldout_combo` | touches a pair in the grid E_h × P_h: 20% of entities × 30% of predicates (properties + relations), i.e. about 6% of (entity, predicate) pairs |

- A query on **more than one axis is dropped**, so each test split isolates exactly
  one kind of generalization.
- Queries on no axis with hops ≤ 3 go to `test_iid` (10%) or `train`.
- Queries with hops > 6 are dropped.

**What each test split measures.**
- `test_iid`: unseen query instances from the training distribution. Sub-parts of a
  test chain (e.g. `r1(e)`) may be trained.
- `test_heldout_combo`: an entity and a predicate that each appear in training, but
  are **never applied to each other in any training query, at any chain position**.
  The fact itself is in the KB, so the model must use KB knowledge rather than
  query supervision. This is the systematic counterpart of `test_iid`. For a
  single-predicate query the two look alike at the instance level; the difference is
  that the exclusion is a systematic grid, applied across all query forms and chain
  positions.
- `test_length`: longer derivations than any trained, built from the same facts and
  rules. Because hops count facts, the inheritance queries here are mostly about
  properties attached 2–4 levels above the leaf.
- `test_heldout_category`: a predicate never queried for any entity in a whole
  subtree, although it is queried elsewhere and those entities are queried with other
  predicates.

**Guarantees (all tested in `tests/test_splits.py`):**
- query keys are disjoint across splits;
- no train query touches a held-out pair;
- hop ranges are respected;
- every test query sits on exactly the axis its split names;
- every proof fact (and overridden default) of every held-out query is in the KB, i.e.
  derivable from training material;
- every entity, predicate, rule and (relation, rule) combination used in a test query
  (e.g. "symmetric relation traversed in reverse") also occurs in some train query;
- held-out entities and predicates each appear in train, just never together.

`isa` is not a combo or category predicate, so `isa` queries appear only in train,
`test_iid` and `test_length`.

## Default dataset (seed 0)

| split | n | prop | isa | chain | trans |
|---|---|---|---|---|---|
| train | 23,938 | 9,247 | 5,434 | 4,511 | 4,746 |
| test_iid | 2,663 | 1,022 | 566 | 535 | 540 |
| test_heldout_combo | 1,053 | 606 | 0 | 447 | 0 |
| test_length | 16,541 | 6,788 | 2,148 | 3,827 | 3,778 |
| test_heldout_category | 680 | 230 | 0 | 450 | 0 |

Answer balance and full per task × hop counts are in `stats.json`. Baselines in
Phase 4 (majority per task type, lookup table) control for imbalance.

## Tests

`uv run pytest`: 193 tests (~20 s).
- **Determinism**: same seed gives byte-identical output, a different seed gives a
  different world, save/load round-trips, and RNG streams are independent.
- **Names**: unique, follow the grammar, not dictionary words.
- **Taxonomy**: it is a tree and levels are consistent.
- **Properties**: inheritance, exceptions (category, entity, nested) and override
  semantics.
- **Relations**: functional / symmetric / inverse / transitive structure and closure.
- **Oracle**: matches the independent closure on every query, proofs are valid,
  task types are consistent, and hop counts are recomputed independently.
- **Mini world**: a hand-built world (the "penguins don't fly, except robo-penguin"
  case) with hand-checked answers.
- **Splits**: no leakage, derivability and coverage, as listed above.

- **Encoders**:
  - text, tokens and ids round-trip, and every rendering parses back to exactly one
    fact or query, for every variant (all facts and queries of the tiny world, and a
    4,000-item sample of the default world);
  - all three encodings decode to the same answer;
  - golden renderings on the mini world;
  - vocabulary covers the corpus with no dead template words, and names never
    collide with template words;
  - example layout and loss masks are correct for every regime and proof mode;
  - KB documents segment back into exactly the KB;
  - each facts-in-context example contains the proof facts and the overridden
    default, and the independent reasoner, given only that context, derives the
    gold answer.

The suite was mutation-checked by injecting six bugs: oracle ignores overrides,
wrong symmetric step, split leakage, incomplete chain touches, off-by-one isa hops,
redundant exceptions. Each one makes at least two tests fail.
Phase 2 had a separate set of six injected bugs: an ambiguous template, swapped
relation arguments, a context missing the overridden default, broken tokenization,
false distractor facts, and a malformed proof layout. Each one fails at least one test.

## Encodings (Phase 2)

All three render the *same* facts and queries. Each encoding is a table of templates
with typed slots, and one table drives both rendering and parsing, so every encoding
is invertible. The tests check that every rendering parses back to exactly one
fact or query.

| | prose (A) | symbolic (B) | triples (C) |
|---|---|---|---|
| fact | `no foglupon is flurbror.` | `(not (has foglupon flurbror))` | `<s> foglupon <r> flurbror <o> false` |
| fact | `the skulka of plufek is resdre.` | `(skulka plufek resdre)` | `<s> plufek <r> skulka <o> resdre` |
| query | `what is the r2 of the r1 of rex?` | `? (r2 (r1 rex))` | `<q> <s> rex <r> r1 <o> ?x1 <s> ?x1 <r> r2 <o> ?` |
| answer | `yes` / `no` / name | `true` / `false` / name | `true` / `false` / name |
| templates (fact / query kinds) | 30 / 15 | 7 / 5 | 7 / 5 |
| vocab size | 1,258 | 1,220 | 1,223 |
| mean tokens per fact | 6.50 | 5.44 | 6.00 |
| mean tokens per query | 7.97 | 7.21 | 9.76 |
| mean tokens, chain query | 14.51 | 11.58 | 20.15 |
| KB tokens per epoch | 89,256 | 75,159 | 82,664 |
| query example, facts-in-weights | 12.9 | 12.2 | 14.6 |
| query example, facts-in-context | 74.0 | 61.2 | 70.1 |

(Default world. `encoders.report` writes these numbers to `data/<world>/encoding_stats.json`.)

**Design decisions.**

- **Word-level tokens; every name is one token in every encoding.**
  - The encodings therefore differ in *structure*, not in how symbols are segmented.
  - Consequence: prose is only about 20% longer per fact than symbolic. Budget
    matching in Phase 3 still handles this (equal tokens vs equal facts).
  - A BPE tokenizer would split nonsense names into subwords. That is a second
    manipulation, which can be added later as its own encoding.
- **Symbolic** keeps types explicit through its heads: `isa`, `sub`, `has`, `not`,
  `val`, plus one head per relation. Chains are nested functional terms.
- **Triples** is uniform: entities and categories are untyped nodes, booleans are the
  nodes `true` / `false`, and chains are conjunctive patterns with join variables.
- **Prose paraphrases (3 per kind)** include object-first phrasings, e.g. `y is the r
  of x` and `v is the A of e`. So prose sees relation facts in both word orders, while
  symbolic and triples are always subject-first. This is a real property of natural
  language, but it may help reverse lookups (symmetric and inverse relations) in the
  facts-in-weights regime. Treat it as a known confound when interpreting results.
- **Paraphrase sampling.**
  - Training items get a new template each epoch (`paraphrase: resample`).
  - Evaluation uses epoch 0, which gives a fixed but varied template per query (a
    stable hash of the qid).
  - Every template is seen in training; no template is held out.
- **Common example format.** The special tokens are shared; each encoding has its own
  vocabulary.
  ```
  <|bos|> [<|ctx|> facts…] <|query|> query <|ans|> answer <|eos|>
  ```
  Loss is on the answer and `<|eos|>`. Answers are a single token in every encoding.
- **Proof as target** (`proof: none | after | before`, default `none`):
  - `after` is an auxiliary target that comes after the answer;
  - `before` is a scratchpad: the model writes the proof facts, then `<|ans|>`, then
    the answer.
- **Facts-in-weights.**
  - The KB is shuffled each epoch and split into documents of 16 facts, with loss on
    every token. The paraphrase is resampled per epoch.
  - The query phase has no context. Scheduling and mixing are Phase 3 decisions.
- **Facts-in-context.**
  - Context = the proof facts + the overridden default (so exception queries contain
    the conflicting general rule) + 6 distractors, shuffled.
  - Half of the distractors share a node with the needed facts; the rest are random
    KB facts.
  - Tested: the independent reasoner, run on the context alone, reproduces every
    answer.

## Training (Phase 3)

**Model** (`train/model.py`): a nanoGPT-style decoder in MLX.
- Pre-LayerNorm blocks (no biases) and rotary position embeddings.
- SwiGLU MLP by default (hidden size = 8/3·d, rounded up to a multiple of 64), or
  GELU (4·d).
- Tied input/output embeddings.
- Init: N(0, 0.02); residual output projections are scaled by 1/√(2L).

| size | layers | width | heads | params (vocab ≈1.25k) |
|---|---|---|---|---|
| S | 4 | 192 | 4 | 2.0M |
| M | 6 | 320 | 5 | 8.0M |
| L | 6 | 512 | 8 | 19.9M |

The brief asked for widths 256–384 and 4–6 layers. ~2M and ~20M are not reachable
inside both ranges at once, so S uses width 192 and L uses width 512. Layers and
heads stay within the brief.

`compute_dtype` gives mixed precision:
- weights stay float32, and each matmul casts weights and activations to bf16/fp16;
- norms, softmax and the loss stay float32.

On the M1, bf16 gives only a small speed-up and is sometimes slower (no native
bf16), so the default is float32. Re-measure on the M5 with `train.bench`.

**Optimization** (`train/trainer.py`):
- AdamW with β = (0.9, 0.95) and bias correction on (MLX's default is off). Weight
  decay 0.1 applies to matrices only, not to the norm gains.
- Gradients are clipped at 1.0; the step is compiled with `mx.compile`.
- LR schedule: linear warmup, then cosine decay to 10% of the peak. The schedule is a
  function of **budget progress**, not of step count. Progress = consumed / budget:
  - `budget_kind: tokens` counts non-pad tokens processed (equal compute);
  - `budget_kind: items` counts KB facts + queries seen (equal facts/queries).

  So token-matched and item-matched runs follow the same schedule shape however
  many steps each one takes.

**Data stream** (`train/data.py`):
- **Facts-in-weights** has two stages:
  1. KB documents only, for the first `stage1_frac` (10%) of the budget;
  2. each epoch is one pass over the train queries, *mixed with one pass over the
     KB* (`kb_per_query_epoch: 1.0`).

  The brief says "first KB, then queries". Mixing the KB back in during stage 2
  stops the model forgetting facts that are never queried. The held-out-combo split
  depends on exactly those facts, since they are never asked in training. Pure
  sequential training is `kb_per_query_epoch: 0`.
- **Facts-in-context** has no KB documents. Every query carries its context.
- **Batches**: each batch has a single padded length (a multiple of 16) and holds at
  most 8,192 padded tokens, which keeps the number of compiled shapes around 10.
- **Determinism**: each epoch's batch plan is a pure function of (seed, stage,
  epoch). Checkpoints store the model, the optimizer, the counters and the stream
  position (stage, epoch, batch). A resumed run reproduces an uninterrupted one
  (tested).
- **What `train.seed` controls**: model init, data order, paraphrase sampling and
  context sampling. The **world is fixed** (`configs/world/default.yaml`, seed 0),
  so the seed sweep measures training variance on one world.
- **Evaluation** always renders queries with data seed 0 and epoch 0, so every run is
  scored on identical token sequences.

**Evaluation** (`train/evaluate.py`): exact match on the single answer token.
- `proof: none / after`: one forward pass per query; take the argmax at the last
  prompt position.
- `proof: before`: greedy decoding (left-padded, key-padding mask) until `<|ans|>`,
  then read the next token.
- **Periodic evals**: 20 per run, evenly spaced in progress, each on a fixed
  1,000-query subsample per split. They log accuracy per split and per
  split × task type.
- **Final eval**: covers full splits. It writes `results.json` (accuracy per
  split × task type × hops) and `predictions.jsonl` (one row per query, for Phase 4).

**Run directory** `runs/<run_id>/`:
- `config.yaml` (resolved config) and `meta.json` (git hash + dirty flag, seed, world
  fingerprint, parameter count, vocab size, MLX version, machine);
- `log.jsonl`: step, stage, loss, grad norm, lr, progress, tokens, KB facts / queries /
  context facts seen, wall time;
- `eval.jsonl`, `results.json`, `predictions.jsonl`, `ckpt/`.

A run that already has `results.json` is skipped. A run with only a checkpoint
resumes; log rows written after that checkpoint are dropped, and that part is redone.

**Controls**:
- `label_shuffle`: train answers are permuted within each (form, answer kind), so the
  answer marginals are preserved.
- `train_subset`: train on only N queries (used by the overfit check).

**Storage diagnostic** (`kb_recall` in `results.json`): cloze accuracy on the last name
slot of 4,000 KB facts:
- for relations, the object; for `isa` / `sub`, the parent; for categorical
  statements, the value;
- boolean property statements are skipped (their last name is the property, which is
  ambiguous).

This separates "the fact is not stored" from "the fact is stored but queries cannot
retrieve it".

### Memory safeguards (`train/memguard.py`)

Running four jobs at once once used ~200 GB and crashed the 128 GB Mac Studio. Cause:
MLX caches freed GPU buffers. Its default cache limit equals its memory limit, which
defaults to 1.5× the GPU's recommended working set, i.e. **~173 GB per process** on
the M5 Max. Evaluation uses many batch shapes, and a buffer freed for one shape is
rarely reused for another, so the cache only grows.
- Measured on a 6M-token S context run: live memory peaked at 1.5 GB, but the cache
  reached 51 GB. Each periodic eval added several GB, and the final full eval added
  about 28 GB.
- Peak live memory for L in the context regime is 4.2 GB.

Safeguards, all on by default for `train.run`, `train.checks`, `train.probes` and
`train.bench`:

| safeguard | default | env var |
|---|---|---|
| MLX cache limit per process | 6 GB | `SYMBOLIC_CACHE_LIMIT_GB` |
| MLX memory limit per process | 16 GB | `SYMBOLIC_MEM_LIMIT_GB` |
| concurrent jobs (machine-wide file locks in `~/.cache/symbolic/slots`; a job waits for a slot) | 2 | `SYMBOLIC_MAX_JOBS` |
| start only with this much free memory on top of the per-process limit | 16 GB | `SYMBOLIC_RESERVE_GB` |
| watchdog: stop the job if its live memory exceeds the limit, or if system free memory drops below the reserve | on | — |

- When the watchdog fires, it saves a checkpoint and exits with `MemoryGuardError`,
  so the run can be resumed.
- File locks are released by the OS when a process dies, so a crash never leaves a
  slot stuck.
- `SYMBOLIC_MEMGUARD=0` turns off slots and admission. The caps and the watchdog
  stay on. The test suite sets it so tests never queue behind real jobs.
- Cost: a 2 GB cache cap slowed S by 15%. 6 GB matches uncapped speed (312k vs 313k
  tokens/s).
- Each process stays at about 7.5 GB (S) to 10 GB (L). The log records `mlx_gb` per
  step, and `results.json` records `peak_memory_gb`.

### Pre-run checks (M1 Pro, size S)

| check | prose | symbolic | triples |
|---|---|---|---|
| overfit 200 queries (train acc, 1.5M tokens) | 1.000 ✅ | 1.000 ✅ | 1.000 ✅ |
| shuffled labels: test_iid / combo / length / category | .373 / .275 / .393 / .140 ✅ | .385 / .288 / .376 / .151 ✅ | .396 / .296 / .429 / .150 ✅ |
| same budget, true labels | .595 / .386 / .453 / .168 | .597 / .403 / .423 / .084 | .560 / .375 / .412 / .118 |

The majority-class baseline is .397 / .298 / .423 / .151. With shuffled labels every
test split stays within 5 points of it. The shuffled-check runs use 8M tokens; their
outputs are in `runs/checks/`.

### Pilot findings (size S, symbolic; `runs/pilot/`)

**Facts-in-weights, 60M tokens (16 min on the M1).**
- Train accuracy reaches 1.0 by about 20M tokens.
- Held-out accuracy plateaus early:
  - test_iid 0.71, held-out combo 0.47, length 0.41, held-out category 0.12;
  - inherited properties generalize (0.94 on both test_iid and combo);
  - entity-level facts mostly do not: held-out combo relation 0.03, direct 0.58
    (chance ≈ 0.5).
- KB recall is 1.00 for relation and taxonomy facts. So the facts are **stored but
  not extractable** by queries that were never trained. Even 1-hop held-out-combo
  relation queries score 12/80. The same pattern shows for all three encodings in the
  8M-token runs. This is the knowledge-storage-vs-extraction gap described in the
  literature, and it is exactly where the encodings (e.g. prose paraphrase
  augmentation) might differ at larger scale.

**Facts-in-context, 30M tokens (5.5 min).**
- Still improving at the end of the budget: test_iid 0.73, train 0.77.
- Transitive queries are near-perfect and **length-generalize**: 4–6 hops at ≈1.0,
  although training only goes to 3.
- Entity-valued chains are near 0, even on train. The context holds confusable
  distractors (same subject, different relation), so a 1-hop chain needs a two-key
  lookup that S has not acquired in 30M tokens.

(The M pilot below shows that the transitive result is **not** in-context reasoning.)

### Pilot findings (size M, symbolic, 200M tokens, M5 Max; `runs/pilot/`)

Each run takes about 20 min in fp32.

**Facts-in-weights.** Train accuracy reaches 1.0 at about 30M tokens. Every held-out
split is then **flat from 30M to 200M** (iid 0.71, combo 0.46, length 0.40, category
0.16), so there is no late transition at M. Entity-valued `chain` queries score 0.00
on test_iid, even at 1 hop, while KB recall is 1.00. The storage-vs-extraction gap is
total, and the 4× larger model does not close it.

**Facts-in-context: the regime is confounded (the model ignores the context).**
- `chain` queries score 1.00 on train and 0.00 on test_iid at every hop count, even
  though a 1-hop chain is a single lookup in the given context.
- Counterfactual probe on 300 train 1-hop chains: rename the answer entity in the
  context to a fresh entity. The model still outputs the **original** answer 100% of the
  time, and the new one 0%. On test_iid only 4% of predictions are even an entity that
  appears in the context.
- `trans` queries whose answer is true: with the proof edges removed from the context,
  the model still says true 97–99% of the time (train, iid, length). So the near-perfect
  transitive scores and their length generalization come from the weights.
- Cause: the world is fixed, so every query's answer is a constant. The model memorizes
  the world through the train queries instead of reading the context. As built, the
  regime does not test "reasoning separately from storage". Fixed by per-example
  renaming (`data.rename`); see "Diagnostics" below.

**Throughput** (`train.bench`, M5 Max): tokens/s for S/M/L is about
490k/180k/108k (weights) and 610k/225k/134k (context) in fp32. bf16 is 1.3–1.4×
faster (unlike the M1). bf16 has not yet been checked for an accuracy effect.

### Diagnostics (M5 Max; `runs/{ctrl,depth,early,ctx,ladder}/`)

All results below come from one seed on the default world. Tools:
- `train.probes` (context, edit, lens), with results in `<run>/probe_*.json`;
- `train.baselines`.

#### Facts-in-weights: why stored facts are not extracted

- **Without the KB** (M, no KB documents, 60M tokens) the query results are almost
  unchanged.
  - The KB adds 12–16 points on boolean direct properties of unseen
    (entity, property) pairs: iid 0.44 → 0.60, combo 0.51 → 0.63.
  - It adds nothing for entity-valued chains.
- **Edit probe** (M pilot, 200 functional facts whose 1-hop query is trained):
  - Fine-tuning on an edited KB, with the other train queries rehearsed, changes KB
    cloze completely (new object 1.00).
  - The edited facts' trained queries still give the old object 1.00 of the time.
  - Editing in query format does not reach the KB cloze (new object ≤ 0.08).
  - So queries and KB statements are answered from two independent stores.
- **Logit lens** (final norm + unembedding at each layer):
  - For untrained 1-hop chains the stored object is decodable at the query's
    **subject position** (0.93 at the last layer of M) but never at `<|ans|>` (0.00).
  - Train queries get their answer at `<|ans|>` from layer 1 on, a separate
    memorized route.
- **Depth does not help.** At ~8M params, with 3×448, 6×320, 12×224 and 24×160:
  - untrained 1-hop chain accuracy is 0.00–0.06 at every depth;
  - KB retrieval always completes in the **last 2–3 layers**: layer 4 of 6, 11 of 12,
    21 of 24.
  - Extra depth is not used to retrieve earlier. Next-token prediction of the object
    only needs the fact at the output, so nothing pushes retrieval earlier.
- **No early extraction window.** Dense evals after a KB-only stage show train chains
  memorized within ~4M query tokens. Untrained chains jump to ~0.04 and stay flat.
- **KB-only stage fraction** (M, 100M tokens): untrained 1-hop chain accuracy is
  0.020 / **0.127** / 0.054 at 10% / 30% / 60% (n = 204).
  - In the 30% run, trained queries' `<|ans|>` lens profile tracks KB retrieval layer
    by layer, instead of the early memorized route.
  - Some of the retrieved object reaches `<|ans|>` for untrained queries (0.18 at the
    last layer).
  - The effect is not monotone, and it **did not replicate**. Over 3 seeds the
    30% − 10% difference was +0.108 / 0.000 / −0.466 (`runs/early/*_s{1,2}`).
- **Extraction from weights is bistable across seeds.** Untrained 1-hop chain
  accuracy at M, 100M tokens:

  | KB-only stage | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | 10% | 0.020 | 0.025 | **0.475** |
  | 30% | 0.127 | 0.025 | 0.010 |

  - The seed-2 / 10% run also gets 0.263 on held-out combo.
  - That run's extraction appeared as soon as train chains were memorized (0.088 at
    20M tokens) and kept growing until ~65M. The other runs stay ≤ 0.06 throughout.
  - The same seed with the 30% schedule did not extract. So the outcome depends on
    the whole training trajectory, not on the seed's initialization alone.
  - Extraction stays limited to 1 hop; 2-hop untrained chains are 0 in every run.
  - Implication: per-condition means over 2–3 seeds are misleading here. Grid cells
    need more seeds, and should report the fraction of runs that extract alongside
    the mean.
- **Encoding** (M, 100M, KB-only stage 30%): untrained 1-hop chain accuracy is
  symbolic 0.127, triples 0.005, prose 0.000. Prose's paraphrase variety does not
  produce extraction at this scale.
- Summary: entity-valued facts in the weights are stored, and are even retrieved
  inside the query, but are not routed to the answer. Every lever tried so far
  (size, depth, budget, encoding) leaves this at or near floor. The only exception is
  one unreplicated schedule effect.

#### Facts-in-context: renaming, then a lookup curriculum

- **Renaming** (`data.rename=nodes`) removes memorization: train ≈ test (0.56 vs
  0.55). But S (60M) and M (100M) then plateau. isa and trans stay at chance and
  1-hop chains at ≤ 0.08, even though 1-hop isa reduces to "does this category name
  appear in the context?". Loss on the context tokens (`data.ctx_loss`) does not help.
- **RoPE works.** Permuting earlier tokens changes the last-position logits by ~30%
  in both random and trained models.
- **Learnability ladder** (S, 15M tokens, 1-hop chains only; train / untrained
  accuracy):

  | distractors | train | untrained |
  |---|---|---|
  | 0 | 0.94 | 0.97 |
  | 2 | 0.71 (rising) | 0.73 |
  | 6 | 0.11 (rising slowly) | 0.13 |

  - When the lookup is learned, it generalizes fully to unseen entities.
  - The slow-to-emerge skill is content-addressed matching among confusable
    distractors.
- **Distractor curriculum** (`data.distractor_ramp_epochs=10`: 0 → 6 distractors over
  the first 10 epochs; evaluation always uses 6). S, 60M tokens, test_iid:

  | | isa | trans | prop | length trans / prop (4–6 hops) |
  |---|---|---|---|---|
  | no curriculum | 0.51 | 0.49 | 0.86 | 0.50 / 0.63 |
  | curriculum | **0.99** | **0.95** | **0.98** | **0.93 / 0.94** |

  - The jump happens between 27M and 33M tokens.
  - Counterfactual probe: removing the proof edges drops "true" from 0.99 to
    0.04–0.06 on train, iid and length. This is real multi-hop reasoning over the
    given context, generalizing from 1–3 hops to 4–6.
  - Entity-valued chains still fail (0.05). The probe's chain-based PASS criterion
    therefore still fails.
  - Why chains fail under the curriculum: 90% of chain predictions are not even
    entities from the context, so no copy-by-matching mechanism has formed.
    - In the chain-only ladder model (2 distractors), errors rarely pick a
      same-subject fact (~4%).
    - So conjunctive (subject AND relation) matching is not the bottleneck; forming
      the copy circuit at all is.
  - **Starting the ramp at 2 distractors** (`data.distractor_ramp_start=2`), meant to
    prevent a positional-copy shortcut, failed completely. Everything returned to the
    plateau: isa 0.49, trans 0.51, chains 0.00.
  - **Seed replication** (S, 60M; a run "broke the plateau" if iid isa and trans
    both exceed 0.9):

    | ramp | seed 0 | seed 1 | seed 2 | broke |
    |---|---|---|---|---|
    | from 0 | **broke** | plateau | plateau | 1/3 |
    | from 2 | plateau | plateau | plateau | 0/3 |

    - Escaping the plateau is stochastic, and 1/3 vs 0/3 does not separate the two
      ramps.
    - The from-0 runs that stayed on the boolean plateau learned *more* chain
      lookup: 1-hop 0.14 and 0.28.
    - The seed-2 chain skill is real context reading. Renaming the answer in the
      context is followed 0.31–0.33 of the time (0.55–0.58 when originally
      correct), and the old answer is never kept.
    - So different runs acquire different lookup skills first.
  - **150M tokens** (same configuration, 3 seeds):
    - **0/3** broke the boolean plateau. That includes seed 0, which broke at 60M
      (the cosine schedule differs, so trajectories differ).
    - Instead, chain lookup emerged:

      | seed | 1-hop | 2-hop | train-chain curve |
      |---|---|---|---|
      | 0 | 0.58 | 0.16 | steady climb |
      | 1 | 0.49 | 0.38 | late jump 0.05 → 0.42, still rising at the end |
      | 2 | 0.14 | 0.05 | slow |

    - Across all 7 curriculum runs at S, **no run acquired both** the boolean lookup
      (isa/trans) and the chain lookup.
    - Hypothesis: the two in-context skills compete for S's capacity (4 layers × 4
      heads). Test: M (6 × 5 heads), same curriculum, 150M, 3 seeds
      (`runs/ctx/symbolic_context_M_tokens150M_rn-nodes_ramp10_s*`).
  - **M falsifies the capacity hypothesis.** 0/3 M runs learned anything in
    context: isa/trans at chance and chains 0.00–0.04 for all 150M tokens. The
    larger model is worse, not better.
  - Together with the trajectory dependence (the same seed lands differently under
    different schedules), this points at optimization; lr 1e-3 was never tuned.
  - Next: an lr sweep, everything else fixed (`runs/lr/`):
    - S, 60M: lr 3e-4 and 3e-3, 3 seeds each;
    - M, 150M: lr 3e-4, 2 seeds.
  - **The learning rate is the lever.** Boolean success = iid isa and trans > 0.9;
    chain success = iid 1-hop chain > 0.5.

    | size, budget | lr | boolean | chain | 1-hop chain per seed |
    |---|---|---|---|---|
    | S, 60M | 3e-3 | 0/3 | 0/3 | 0.00 / 0.00 / 0.00 |
    | S, 60M | 1e-3 | 1/3 | 0/3 | 0.05 / 0.14 / 0.28 |
    | S, 60M | **3e-4** | 1/3 | **2/3** | 0.75 / 0.70 / 0.46 |
    | M, 150M | 1e-3 | 0/3 | 0/3 | ≤ 0.04 |
    | M, 150M | **3e-4** | 0/2 | **2/2** | 0.64 / 0.81 (2-hop 0.25 / 0.69) |

    - S seed 0 at 3e-4 acquired **both** skills: isa 0.99, trans 0.97, chain 1-hop
      0.75 / 2-hop 0.63, length trans 0.91. The "mutually exclusive skills"
      pattern was an artifact of lr 1e-3 being too high, and M's failure at 1e-3 is
      the usual need for a lower lr at larger size.
    - Booleans remain the less reliable skill. Plausibly, a 2-way answer gives a
      weak gradient on the plateau (guessing already scores 0.5), while a ~1000-way
      chain answer gives a strong one.
    - Next: lr 1e-4 (S 60M × 3 seeds, M 150M × 2) to see whether the trend
      continues.

#### Held-out category split

- It contains only **5** (category, property) pairs, so its effective sample size is
  about 5.
- Its inherited answers are mostly the opposite of each property's train prior:
  - the predicate-prior baseline scores 0.21, against 0.82 for "always true";
  - the weights model follows the prior (0.14).
- It needs more pairs, and reporting split by agreement with the prior, before it
  can be interpreted.
- **World v2** (`configs/world/v2.yaml`) holds out 20 property pairs (23 pairs in
  total), giving 576 inherited queries. The predicate-prior baseline still scores only
  0.318. This is structural: a held-out pair is only informative if the category's
  value can differ from the property's value elsewhere.
- **Prior-agreement strata** (`uv run python -m train.baselines --run <dir>`) report
  accuracy separately for queries whose gold answer agrees or disagrees with the
  predicate prior. On held-out category, inherited:

  | | agree | disagree (n = 93) |
  |---|---|---|
  | weights (M pilot) | 0.44 | **0.054** |
  | context (curriculum, S) | 1.00 | **1.00** |

  The "disagree" stratum is the clean test of reading the KB or the context.
