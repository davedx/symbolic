"""Training loop: AdamW + warmup/cosine schedule over *budget progress*, grad clipping,
compiled step, periodic eval, checkpoint/resume, JSONL logs.

Run directory (runs/<run_id>/):
    config.yaml        resolved run config
    meta.json          git hash, seed, versions, param count, vocab size, world fingerprint
    log.jsonl          training rows (step, loss, lr, tokens, facts/queries seen, wall time, ...)
    eval.jsonl         periodic eval rows (accuracy per split and split x task type)
    results.json       final summary: accuracy per split x task type x hops (+ counters)
    predictions.jsonl  final per-query predictions (for Phase 4 analysis)
    ckpt/              latest checkpoint (model, optimizer, trainer state)

Progress = consumed budget / budget, where consumed is non-pad tokens (budget_kind
tokens) or KB facts + queries seen (budget_kind items). The LR schedule, stage switch
and eval times are functions of progress, so token- and item-matched runs share one
schedule shape regardless of how many steps they take.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from encoders.dataset import make_encoder

from .config import RunConfig
from .data import StreamState, TrainStream, load_dataset, train_queries
from .evaluate import Evaluator, eval_sets, summarize
from .model import Transformer, count_params


@dataclass
class Counters:
    step: int = 0
    tokens: int = 0
    kb_facts: int = 0
    queries: int = 0
    ctx_facts: int = 0
    wall: float = 0.0
    next_eval: int = 1  # index of the next periodic eval

    @property
    def items(self) -> int:
        return self.kb_facts + self.queries


def lr_at(progress: float, tc) -> float:
    if progress < tc.warmup_frac:
        return tc.lr * progress / tc.warmup_frac
    p = min(1.0, (progress - tc.warmup_frac) / max(1e-9, 1 - tc.warmup_frac))
    lo = tc.lr * tc.min_lr_ratio
    return lo + 0.5 * (tc.lr - lo) * (1 + math.cos(math.pi * p))


def make_optimizer(tc) -> optim.Optimizer:
    def adamw(wd):
        return optim.AdamW(
            learning_rate=tc.lr, betas=[tc.beta1, tc.beta2], weight_decay=wd, bias_correction=True
        )

    # matrices get weight decay; norm gains (1-D) do not
    return optim.MultiOptimizer([adamw(tc.weight_decay), adamw(0.0)], [lambda path, w: w.ndim >= 2])


def git_info() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return {"git_hash": h.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"git_hash": None, "git_dirty": None}


class Trainer:
    def __init__(self, cfg: RunConfig, runs_dir: str | Path = "runs", data_dir: str | Path = "data"):
        self.cfg = cfg
        self.run_dir = Path(runs_dir) / cfg.resolved_run_id()
        self.ds = load_dataset(cfg.world, data_dir)
        self.enc = make_encoder(cfg.data.encoding, self.ds.world)
        tc = cfg.train
        if tc.label_shuffle and cfg.data.proof != "none":
            raise ValueError("label_shuffle requires proof=none (proofs would contradict the labels)")
        cfg.model.vocab_size = len(self.enc.vocab)
        self.train_qs = train_queries(self.ds, tc)
        self.stream = TrainStream(self.enc, self.train_qs, cfg.data, tc)
        mx.random.seed(tc.seed)
        self.model = Transformer(cfg.model)
        self.opt = make_optimizer(tc)
        self.opt.init(self.model.trainable_parameters())
        self.c = Counters()
        self.evaluator = Evaluator(self.enc, cfg.data, tc.eval_batch_size)
        self.periodic_sets = eval_sets(self.ds, self.train_qs, tc.eval_max_per_split)

    # ------------------------------------------------------------------ helpers
    @property
    def consumed(self) -> int:
        return self.c.tokens if self.cfg.train.budget_kind == "tokens" else self.c.items

    @property
    def progress(self) -> float:
        return self.consumed / self.cfg.train.budget

    @property
    def done(self) -> bool:
        return (self.run_dir / "results.json").exists()

    def _append(self, name: str, row: dict) -> None:
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(row) + "\n")

    # --------------------------------------------------------------- checkpoints
    def save_checkpoint(self) -> None:
        ck = self.run_dir / "ckpt"
        tmp = self.run_dir / "ckpt.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(tmp / "model.safetensors"))
        mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(self.opt.state)))
        state = {"counters": asdict(self.c), "stream": asdict(self.stream.state)}
        (tmp / "state.json").write_text(json.dumps(state))
        if ck.exists():
            for p in ck.iterdir():
                p.unlink()
            ck.rmdir()
        tmp.rename(ck)

    def load_checkpoint(self) -> bool:
        ck = self.run_dir / "ckpt"
        if not (ck / "state.json").exists():
            return False
        self.model.load_weights(str(ck / "model.safetensors"))
        self.opt.state = tree_unflatten(list(mx.load(str(ck / "optimizer.safetensors")).items()))
        state = json.loads((ck / "state.json").read_text())
        self.c = Counters(**state["counters"])
        self.stream.state = StreamState(**state["stream"])
        mx.eval(self.model.parameters(), self.opt.state)
        # drop log rows written after the checkpoint (they will be regenerated)
        for name in ("log.jsonl", "eval.jsonl"):
            p = self.run_dir / name
            if p.exists():
                rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                keep = [r for r in rows if r["step"] <= self.c.step]
                p.write_text("".join(json.dumps(r) + "\n" for r in keep))
        return True

    # -------------------------------------------------------------------- train
    def _write_meta(self) -> None:
        meta = {
            **git_info(),
            "run_id": self.cfg.resolved_run_id(),
            "seed": self.cfg.train.seed,
            "world_fingerprint": self.ds.world.config.fingerprint(),
            "n_params": count_params(self.model),
            "vocab_size": len(self.enc.vocab),
            "n_train_queries": len(self.train_qs),
            "mlx_version": mx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def train(self, resume: bool = True, verbose: bool = True) -> dict | None:
        if self.done:
            if verbose:
                print(f"{self.run_dir} already complete; skipping")
            return json.loads((self.run_dir / "results.json").read_text())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        resumed = resume and self.load_checkpoint()
        if not resumed:
            for name in ("log.jsonl", "eval.jsonl"):
                (self.run_dir / name).unlink(missing_ok=True)
            self.cfg.save(self.run_dir / "config.yaml")
            self._write_meta()
        elif verbose:
            print(f"resumed {self.run_dir} at step {self.c.step} (progress {self.progress:.3f})")

        tc = self.cfg.train
        model, opt = self.model, self.opt

        def loss_fn(model, x, y, w):
            logits = model(x)
            ce = nn.losses.cross_entropy(logits, y, reduction="none")
            return (ce * w).sum() / mx.maximum(w.sum(), 1.0)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step(x, y, w):
            loss, grads = loss_and_grad(model, x, y, w)
            grads, gnorm = optim.clip_grad_norm(grads, tc.grad_clip)
            opt.update(model, grads)
            return loss, gnorm

        if tc.compile:
            state = [model.state, opt.state]
            step = partial(mx.compile, inputs=state, outputs=state)(step)

        t_last = time.perf_counter()
        tok_since = 0
        while self.progress < 1.0:
            if self.stream.state.stage == 1 and self.progress >= tc.stage1_frac:
                self.stream.end_stage1()
            b = self.stream.next_batch()
            lr = lr_at(self.progress, tc)
            opt.learning_rate = lr
            loss, gnorm = step(b.inputs, b.targets, b.weights)
            mx.eval(loss, gnorm, model.parameters(), opt.state)

            self.c.step += 1
            self.c.tokens += b.n_tokens
            self.c.kb_facts += b.n_kb_facts
            self.c.queries += b.n_queries
            self.c.ctx_facts += b.n_ctx_facts
            now = time.perf_counter()
            self.c.wall += now - t_last
            tok_since += b.n_tokens
            dt_since = now - t_last
            t_last = now

            if self.c.step % tc.log_every == 0 or self.c.step == 1:
                row = {
                    "step": self.c.step,
                    "stage": self.stream.state.stage,
                    "epoch": self.stream.state.epoch,
                    "loss": float(loss),
                    "grad_norm": float(gnorm),
                    "lr": lr,
                    "progress": self.progress,
                    "tokens": self.c.tokens,
                    "kb_facts_seen": self.c.kb_facts,
                    "queries_seen": self.c.queries,
                    "ctx_facts_seen": self.c.ctx_facts,
                    "items_seen": self.c.items,
                    "wall_time": self.c.wall,
                    "batch_shape": list(b.inputs.shape),
                }
                self._append("log.jsonl", row)
                if verbose and self.c.step % (tc.log_every * 10) == 0:
                    print(
                        f"step {self.c.step} stage {row['stage']} loss {row['loss']:.4f} lr {lr:.2e} "
                        f"progress {self.progress:.3f} tok {self.c.tokens} wall {self.c.wall:.0f}s"
                    )

            if tc.n_evals and self.progress >= self.c.next_eval / tc.n_evals and self.progress < 1.0:
                self.periodic_eval(verbose)
                self.c.next_eval = math.floor(self.progress * tc.n_evals) + 1
                t_last = time.perf_counter()  # exclude eval time from wall time
            if self.c.step % tc.ckpt_every == 0:
                self.save_checkpoint()
                t_last = time.perf_counter()

        self.save_checkpoint()
        return self.final_eval(verbose)

    # --------------------------------------------------------------------- eval
    def periodic_eval(self, verbose: bool = True) -> dict:
        rows = []
        for split, qs in self.periodic_sets.items():
            rows += self.evaluator.predict(self.model, qs)
        summ = summarize(rows)
        row = {
            "step": self.c.step,
            "progress": self.progress,
            "tokens": self.c.tokens,
            "items_seen": self.c.items,
            "kb_facts_seen": self.c.kb_facts,
            "queries_seen": self.c.queries,
            "wall_time": self.c.wall,
            "acc": {k: v["acc"] for k, v in summ.items() if k.count("/") <= 1},
        }
        self._append("eval.jsonl", row)
        if verbose:
            top = {k: round(v, 3) for k, v in row["acc"].items() if "/" not in k}
            print(f"eval @ step {self.c.step} progress {self.progress:.2f}: {top}")
        return row

    def final_eval(self, verbose: bool = True) -> dict:
        sets = eval_sets(self.ds, self.train_qs, self.cfg.train.final_eval_max_per_split)
        rows = []
        for split, qs in sets.items():
            rows += self.evaluator.predict(self.model, qs)
        with open(self.run_dir / "predictions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        results = {
            "run_id": self.cfg.resolved_run_id(),
            "accuracy": summarize(rows),
            "counters": asdict(self.c),
            "n_params": count_params(self.model),
        }
        self._append(
            "eval.jsonl",
            {
                "step": self.c.step,
                "progress": self.progress,
                "tokens": self.c.tokens,
                "items_seen": self.c.items,
                "kb_facts_seen": self.c.kb_facts,
                "queries_seen": self.c.queries,
                "wall_time": self.c.wall,
                "final": True,
                "acc": {k: v["acc"] for k, v in results["accuracy"].items() if k.count("/") <= 1},
            },
        )
        (self.run_dir / "results.json").write_text(json.dumps(results, indent=1))
        if verbose:
            top = {k: round(v["acc"], 3) for k, v in results["accuracy"].items() if "/" not in k}
            print(f"final: {top}")
        return results
