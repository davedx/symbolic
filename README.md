# Representations of Meaning — small-model prototype

Does the **encoding of knowledge** in training data (prose vs typed symbolic tokens vs
graph triples, later continuous binding vectors) change (1) sample efficiency and
compositional / multi-hop generalization and (2) the geometry of internal
representations? See `semantic-representations-prototype-prompt.md` for the full brief.

Status: **Phase 1 (synthetic world + oracle + splits) and Phase 2 (encoders) done.** Phases 3–5 to follow.

## Quick start

```bash
uv sync                                    # Python 3.12, mlx, numpy, pytest, ...
uv run pytest                              # full test suite (~3 s)
uv run python -m world.build --config configs/world/default.yaml   # -> data/world_default/
uv run python -m encoders.report --data data/world_default         # encoding stats + side-by-side samples
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
configs/world/  default.yaml (1000 entities), tiny.yaml (120 entities, for tests)
tests/          test_world.py, test_oracle.py, test_splits.py, test_encoders.py
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

`uv run pytest`: 173 tests on the tiny and the default world.
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
