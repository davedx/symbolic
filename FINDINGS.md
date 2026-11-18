# Representations of Meaning: Findings and Conclusions

This document summarizes what the prototype found. The full experimental log,
design decisions, and every intermediate result are in `README.md`. Per-run
summaries (configs, final results, learning curves, probe outputs) for all 121 runs
are archived in `results/`.

## 1. The question

Does the **format** in which knowledge is written change how well a small transformer
learns to store it and reason with it? Three encodings of the same facts were
compared:

| encoding | example fact | example query |
|---|---|---|
| prose | "the mentor of rex is fido" (3 paraphrases per fact) | "who is the boss of the mentor of rex ?" |
| symbolic | `( mentor rex fido )` | `? ( boss ( mentor rex ) )` |
| triples | `<s> rex <r> mentor <o> fido` | `<q> <s> rex <r> mentor <o> ?x1 <s> ?x1 <r> boss <o> ?` |

The models were trained in two regimes, as the brief specified:

- **Facts in weights.** The model first learns the knowledge base (KB), then learns
  to answer queries from memory.
- **Facts in context.** Each query comes with the relevant facts plus distractors in
  the prompt.

## 2. Setup (short)

- **World.** A synthetic, seeded world with a category taxonomy, inherited
  properties with exceptions, and relations (functional, symmetric, inverse,
  transitive).
  - Default: 1,000 entities, 50 categories, 100 properties, 10 relations.
  - "v2": the same, but with 20 held-out (category, property) pairs instead of 5.
  - A ground-truth oracle answers every query and records its proof and hop count.
- **Splits.**
  - train (up to 3 hops);
  - `test_iid`;
  - `test_heldout_combo` (unseen entity × predicate pairs);
  - `test_heldout_category` (unseen category × predicate pairs);
  - `test_length` (4–6 hops).
- **Models.** Decoder-only transformers in MLX (pre-LN, RoPE, SwiGLU, tied
  embeddings): S ≈ 2M, M ≈ 8M, L ≈ 20M parameters. Trained on an M5 Max.
- **Skills measured.**
  - `isa`: is X a kind of C (taxonomy, up to 3 hops);
  - `trans`: does a transitive relation hold (up to 6 hops);
  - `prop`: inherited properties and exceptions;
  - `chain`: entity-valued relation composition, e.g. "the boss of the mentor
    of rex".

## 3. Headline conclusions

1. **How a question is written matters more than how facts are written.**
   - In the facts-in-context regime, writing a multi-hop query as fact-shaped steps
     in execution order, joined by placeholders, let symbolic and triples models
     solve chains longer than any seen in training. Nested queries did not.
   - Step-by-step vs nested, seeds solving ≥ 4 hops: symbolic 4/5 vs 0/5,
     triples 5/5 vs 1/5 (Fisher p = .048 each).
   - The effect is not universal: with consistent phrasing, nested prose generalized
     and step-by-step prose did not help.
2. **Fact format has no reliable effect once question format and phrasing are
   controlled.** Triples' apparent advantage came mostly from its step-by-step chain
   queries. Pooled over question formats, trans success was 7/10 for triples vs
   4/10 for symbolic (p = .37).
3. **Surface variability hurts a lot.** Prose's poor results came from paraphrasing:
   facts and queries each used one of three phrasings, chosen independently. With a
   single fixed phrasing, prose performed like the other formats.
4. **Facts in weights: stored but not extracted.**
   - Models memorize the KB perfectly (recall 1.00) and memorize their training
     queries perfectly.
   - But they almost never answer an *untrained* query about a stored entity-valued
     fact. The KB and the query answers live in two separate stores.
   - Model size, depth, schedule, learning rate and encoding did not fix this at
     this scale. It is rare, run-dependent, and limited to 1 hop.
5. **The depth-vs-length explanation was not supported.**
   - Nested chains did not get longer with more layers.
   - Step-by-step generalization *needed* depth (6–8 layers) rather than being
     independent of it.
6. **Skill acquisition is bimodal and run-dependent.**
   - Identical configurations succeed in some seeds and fail in others, often via a
     sudden jump late in training.
   - The learning rate was the single most important setting.
   - Results must be reported as the fraction of seeds that acquire a skill, not as
     mean accuracy.

## 4. Facts in weights

### 4.1 Storage without extraction

M, symbolic, 200M tokens:

- Train accuracy 1.00; KB recall (cloze on KB statements) 1.00.
- Untrained 1-hop chain queries about stored facts: **0.000** (0/204).
- Held-out accuracy plateaus by about 30M tokens and does not move through 200M.

### 4.2 Why: two separate stores

- **Edit probe** (200 facts whose 1-hop query was trained):
  - Fine-tuning on an edited KB changed KB recall completely (new object 1.00). The
    edited facts' trained queries still gave the **old** answer 1.00 of the time.
  - Editing in query format did not reach the KB (≤ .08).
- **Logit lens.**
  - For untrained queries, the stored object is retrieved inside the query at the
    subject position (0.93 at the last layer) but never reaches the answer position
    (0.00).
  - Trained queries get their answer at the answer position from layer 1 onward:
    a separate, memorized route.
- **Retrieval is always late.** At ~8M parameters with 3, 6, 12 and 24 layers, KB
  retrieval always completes in the last 2–3 layers. Extraction stays at
  0.00–0.06. Extra depth does not leave room to route the fact; retrieval moves to
  the end of the network.

### 4.3 What the KB does contribute

- A no-KB control (queries only) matched the full model on everything except
  boolean direct properties.
- On those, the KB added 12–16 points (iid .44 → .60; held-out combo .51 → .63;
  isa .77 → .87).
- It added nothing for entity-valued chains.

### 4.4 Levers tried

Untrained 1-hop chain accuracy, M:

| lever | result |
|---|---|
| depth (3/6/12/24 layers) | .00–.06 |
| encoding (KB-only stage 30%) | symbolic .127, triples .005, prose .000 |
| KB-only stage 10% / 30% / 60% (seed 0) | .020 / .127 / .054 |
| seeds at stage 10% | .020, .025, **.475** |
| seeds at stage 30% | .127, .025, .010 |
| lr 3e-4 / 1e-4 | .049, .162 / .020, .025 |

- Extraction happened once (0.475, with 0.263 on held-out combo) and did not
  replicate with the same seed under a different schedule. It is a trajectory-level
  event, not a property of the configuration.
- On the held-out category split, queries whose answer contradicts the property's
  usual train value score **.054** in the weights regime (vs 1.00 in context): the
  model falls back on query statistics instead of the stored rule.

**Conclusion:** at this scale the weights regime cannot compare encodings on
entity-valued facts. It is usable only for boolean and property queries.

## 5. Facts in context

### 5.1 Making the regime valid

The first version had several flaws. Each was diagnosed and fixed:

| problem | evidence | fix |
|---|---|---|
| Model memorized the fixed world and ignored the context | train chains 1.00, iid 0.00. With the answer renamed in the context, the old answer was kept 100% of the time. With proof edges removed, trans still said "true" 97–99% | Per-example consistent renaming of entities and categories (`data.rename=nodes`) |
| After renaming: plateau at ~.55, isa/trans at chance | RoPE verified (order sensitivity ~30%). Ladder at S: 1-hop chains learned with 0 distractors (.94 train / .97 untrained), 2 (.71 / .73), barely with 6 (.11) | Distractor curriculum: 0 → 6 over 10 epochs; evaluation always uses 6 |
| Success rare and seed-dependent; M learned nothing | lr sweep below | lr 1e-3 → **1e-4** |
| Held-out category split had an effective n of 5 | answers mostly opposed each property's prior | world v2 (20 pairs), plus reporting split by agreement with the prior |

**Learning-rate sweep** (renamed contexts plus curriculum). "Boolean" = isa and trans
> .9; "chain" = 1-hop chain > .5:

| size, budget | lr | boolean | chain |
|---|---|---|---|
| S, 60M | 3e-3 | 0/3 | 0/3 |
| S, 60M | 1e-3 | 1/3 | 0/3 |
| S, 60M | 3e-4 | 1/3 | 2/3 |
| S, 60M | 1e-4 | 1/3 | 3/3 |
| M, 150M | 1e-3 | 0/3 | 0/3 |
| M, 150M | **1e-4** | **2/2** | **2/2** |

Adopted configuration: M, lr 1e-4, weight decay 0.1, 150M tokens, renaming,
10-epoch distractor curriculum, world v2.

Counterfactual probes on the final models confirm the answers come from the context:

- Renaming a chain's answer in the context is followed .79–.85 of the time, and the
  old answer is never kept.
- Removing the proof edges of a true trans query drops "true" from .99 to .36
  (triples) and from .98 to .59 (symbolic). Symbolic leans more on a shortcut cue
  for trans.

### 5.2 Fact format × question format (5 seeds per cell)

| facts \ questions | nested | step-by-step |
|---|---|---|
| prose (paraphrased) | isa 0/5 · trans 0/5 · ≥4 hops 0/5 | isa 0/5 · trans 0/5 · ≥4 hops 0/5 |
| prose (fixed phrasing; 3 seeds) | ≥4 hops **3/3** (longest 5, 6, 4) | ≥4 hops 1/3 (3, 5, 3) |
| symbolic | isa 4/5 · trans 2/5 · ≥4 hops 0/5 (all exactly 3) | isa 2/5 · trans 2/5 · ≥4 hops 4/5 |
| triples | isa 3/5 · trans 2/5 · ≥4 hops 1/5 | isa 5/5 · trans 5/5 · ≥4 hops 5/5 |

"≥4 hops" counts the seeds whose longest chain solved at ≥ 50% exceeds the training
maximum of 3. All cells learn 1-hop chains (5/5).

- **Question format** is the main effect for symbolic and triples (p = .048 each).
- **Fact format** has no reliable effect. The triples 5/5 on trans appears only with
  step-by-step chain queries. Trans queries are identical in both question formats,
  so the chain-query format seems to shape which in-context skills form overall.
- **Role markers** in triples matter little. `triples_unmarked` (brackets instead of
  `<s>/<r>/<o>`) scored isa 5/5, trans 4/5, ≥4 hops 4/5.
- **Within prose**, execution-order phrasing ("start at e and follow r1 then r2")
  beat nested phrasing on 4–6 hops in all 5 seeds (sign test p ≈ .03). The gain was
  small; per-hop placeholders matter more than order alone.
- **Prior-disagree stratum.** Held-out category, inherited properties, answer
  opposite to the train prior: triples .77–.94, symbolic .70–.88, prose .71–.81.
  Weights regime: .054.

### 5.3 Testing the depth-vs-length hypothesis

**Hypothesis.** Nested queries must resolve every hop within one forward pass, so the
longest solvable chain should grow with depth. Step-by-step queries spread hops over
positions and should be largely independent of depth.

Parameter-matched (~8M) symbolic models, seeds with longest chain ≥ 4:

| layers | nested | step-by-step |
|---|---|---|
| 2 | 0/3 (3, 3, 1) | 1/3 (3, 3, 4) |
| 4 | 1/3 (2, 5, 3) | 0/3 (1, 2, 1) |
| 6 | 0/5 (3, 3, 3, 3, 3) | 4/5 (4, 3, 6, 6, 6) |
| 8 | 0/3 (3, 3, 3) | 3/3 (5, 6, 6) |

- **Not supported.**
  - Nested chains do not lengthen with depth.
  - Step-by-step generalization appears at 6–8 layers and mostly fails at 2–4.
- The 4-layer step-by-step result is worse than 2 layers, with weak 1-hop accuracy
  too. That points to an optimization failure at that shape (lr was tuned at 6
  layers) rather than a capacity limit.
- **Bracket test.** Nested symbolic and triples end a k-hop query with k closing
  brackets. `symbolic_path` (`? ( r3 r2 r1 e )`, one bracket pair) reached 4 hops in
  2/3 seeds, with the same cliff in every seed (4 hops ≈ .5, 5–6 hops ≈ .05). The
  bracket run explains at most one hop.
- **The mechanism remains open.**
  - It is not a depth budget.
  - It is not nesting as such: fixed-phrasing nested prose generalizes.
  - A positional explanation (relations at offsets never seen in training) fits
    symbolic but not prose.
  - Settling it needs mechanistic probing, not more ablations.

## 6. Methodological lessons

1. **A fixed world lets models memorize instead of read.** Any "facts in context"
   setup over a fixed world needs per-example renaming, or an equivalent, plus
   counterfactual probes that confirm the context is used.
2. **Untuned hyperparameters can masquerade as findings.** Size effects, "competing
   skills", and much of the seed variance were artifacts of lr 1e-3.
3. **Skill acquisition is bimodal.**
   - Report seed fractions and acquisition points, not means.
   - Three seeds are often too few. Two of this project's interim claims (a
     schedule effect in the weights regime, and a fact-format effect on trans) did
     not survive more seeds.
4. **Budgets truncate transitions.**
   - Acquisitions often happened at 110–140M of 150M tokens.
   - Several failed seeds were still improving at the end (e.g. symbolic trans;
     prose and step-by-step at 2–4 layers on trans).
   - Some failure counts may fall with a longer budget.
5. **Encodings differ in more than fact format.** Query syntax, paraphrase rate,
   answer tokens and sequence length all co-vary. A fair encoding comparison has to
   separate them with ablations, as here.
6. **Baselines must match the split's structure.** Majority baselines per task hid
   that the held-out category split was anti-correlated with property priors;
   prior-agreement strata fixed that.

## 7. Sanity checks performed

- **Prose booleans.** Each prose seed answered one constant to 91–100% of isa/trans
  queries, on balanced test sets (majority .50–.52): a fallback, not slow learning.
- **Answer position.** It is comparable across all formats: the prompt ends at
  `<|ans|>`, there is a single answer token, and every example has 2 loss positions.
- **No leakage.** In ~1,400 chain queries per format (all 8 formats), the start
  entity is the only entity in the query. Intermediates appear only as placeholders.
- **Length test sets.** Identical across all runs (1,388 / 1,262 / 1,177 chain
  queries at 4 / 5 / 6 hops); trans is exactly 50/50.
- **Pre-run checks.** Overfit to 200 queries reaches 1.00. With shuffled labels,
  every test split stays within 5 points of the majority baseline.

## 8. Limitations

- One synthetic world (default and v2 variants); a single world seed.
- Small models (2–20M parameters); most conclusions rest on size M.
- **Learning rate.** Tuned on symbolic at the 6-layer shape and held fixed across
  formats and depths. The per-format check (prose 3e-5 / 3e-4, symbolic 3e-5) did
  not change the ranking, but per-depth tuning was not done.
- **Budgets.** Tokens-matched only; prose saw about 16% fewer queries than symbolic.
  Items-matched budgets were not run.
- 3–5 seeds per condition. With bimodal outcomes, only large differences reach
  significance.
- 150M-token budgets truncate some transitions (§6.4).
- **Not carried out.** Proof-trace targets (`data.proof=before|after`) and the brief's
  Phase 5 representational analyses (CKA, probing across encodings).

## 9. Open questions and next steps

1. **Mechanism of the question-format effect.** Logit-lens and attention analysis by
   layer and position on nested-symbolic, fixed-phrasing nested prose, and
   step-by-step models. Where is each hop's intermediate entity resolved, and why
   does nested prose generalize while nested symbolic stops at 3?
2. **Per-depth learning-rate tuning** at 2 and 4 layers, to confirm whether
   "step-by-step needs depth" is real or an optimization artifact.
3. **Longer budgets** (≈ 300M tokens) for conditions with still-rising failed seeds.
4. **Weights regime restricted to boolean and property queries**, using
   prior-disagree strata as the main metric.
5. **Extraction in the weights regime.** The single successful run shows it is
   reachable at this scale. Training-dynamics interventions are the natural next
   lever; capacity is not.

## 10. Reproducing

- Code: `world/`, `encoders/`, `train/`. 272 tests: `uv run pytest`.
- Every run is a YAML config plus dotted overrides, deterministic from seeds:
  `uv run python -m train.run configs/runs/base.yaml <overrides>`.
- Seed-level reports: `uv run python -m train.analysis "<run glob>"`. These work on
  the archived `results/` without retraining.
- The exact launch commands are in `results/launch_commands.txt`. Queue them with
  `scripts/run_chain.sh`.
- Training is not bit-reproducible on GPU: identical runs diverge within 50 steps.
  Re-runs reproduce success fractions, not each seed's outcome.
- Probes: `uv run python -m train.probes {context,edit,lens} <run>`.
- Baselines and prior strata: `uv run python -m train.baselines [--run <run>]`.
- Memory safeguards (`train/memguard.py`) cap each process's MLX cache and memory
  and allow 2 concurrent jobs. These limits are needed on a 128 GB machine; see
  README.
