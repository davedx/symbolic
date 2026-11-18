# Archived run summaries

Every training run behind `FINDINGS.md` and `README.md`, without checkpoints or
per-query predictions. The original `runs/` directory, with checkpoints (3.3 GB) and
predictions (0.8 GB), was deleted after this archive was made.

Each run directory keeps:
- `config.yaml`: the exact config;
- `meta.json`: git hash, world fingerprint, versions;
- `results.json`: final accuracy per split × task × hops and per form, KB recall,
  counters;
- `eval.jsonl`: the periodic learning curves;
- `probe_*.json`: probe outputs, where probes were run.

The layout mirrors the old `runs/`:

| directory | contents |
|---|---|
| `pilot/` | M pilots, 200M tokens (weights and context, before the fixes) |
| `ctrl/` | no-KB controls |
| `depth/` | weights-regime depth sweep (3 / 12 / 24 layers) |
| `early/` | weights-regime schedule, seed and lr runs, and the encodings at KB-stage 30% |
| `ctx/`, `ladder/`, `lr/` | context-regime fixes: renaming, curriculum, distractor ladder, lr sweep |
| `grid_ctx/`, `grid_ctx_lr/` | context-regime encoding comparison: 3×2 grid, ablations, seeds, per-encoding lr, fixed paraphrase |
| `depth_ctx/` | context-regime depth sweep (2 / 4 / 8 layers; 6 layers = `grid_ctx/*_M_*`) |
| `bench.json` | throughput benchmark |
| `launch_commands.txt` | the exact commands used (from the job-queue logs) |

Seed-level tables can be regenerated without retraining, for example:

    uv run python -m train.analysis "results/grid_ctx/*_context_M_150M_v2_s*"

Re-running a config reproduces the setup and the statistics, but not each seed's
outcome bit for bit: GPU training is not deterministic (two identical runs diverge
within 50 steps), and skill acquisition is bimodal across seeds.
