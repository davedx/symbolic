"""Run configuration: one YAML file per run (world + data + model + train sections)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from encoders.dataset import DataConfig

from .model import ModelConfig


@dataclass
class TrainConfig:
    seed: int = 0  # model init, data order, paraphrase/context sampling
    budget_kind: str = "tokens"  # tokens: non-pad tokens processed | items: KB facts + queries seen
    budget: int = 20_000_000
    max_batch_tokens: int = 8192  # padded tokens per batch
    max_batch_size: int = 512
    bucket: int = 16  # sequence lengths are padded up to a multiple of this
    lr: float = 1e-3
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    warmup_frac: float = 0.02  # linear warmup over this fraction of the budget
    weight_decay: float = 0.1  # applied to matrices only (not norms)
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # facts-in-weights schedule
    stage1_frac: float = 0.1  # first fraction of the budget: KB documents only
    kb_per_query_epoch: float = 1.0  # stage 2: KB passes mixed into each pass over the train queries
    # controls
    label_shuffle: bool = False  # permute train answers within (form, answer kind)
    train_subset: int = 0  # >0: use only this many train queries (overfit test)
    train_forms: list = field(default_factory=list)  # non-empty: train only on these query forms (diagnostics)
    train_max_hops: int = 0  # >0: train only on queries with at most this many hops (diagnostics)
    # evaluation / logging
    n_evals: int = 20  # periodic evals, evenly spaced in budget progress
    eval_max_per_split: int = 1000  # periodic evals use a fixed subsample per split
    final_eval_max_per_split: int = 0  # 0 = full splits
    eval_batch_size: int = 256
    log_every: int = 50
    ckpt_every: int = 1000
    compile: bool = True


@dataclass
class RunConfig:
    run_id: str | None = None
    world: str = "configs/world/default.yaml"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        d, m, t = self.data, self.model, self.train
        size = m.size or f"L{m.n_layers}d{m.d_model}"
        parts = [d.encoding, d.regime, size, f"{t.budget_kind}{_short(t.budget)}"]
        if d.regime == "context" and d.rename != "none":
            parts.append(f"rn-{d.rename}")
        if d.regime == "context" and d.ctx_loss:
            parts.append("ctxloss")
        if d.proof != "none":
            parts.append(f"proof-{d.proof}")
        if t.label_shuffle:
            parts.append("shuffled")
        if t.train_subset:
            parts.append(f"subset{t.train_subset}")
        if t.train_forms:
            parts.append("only-" + "+".join(t.train_forms))
        if t.train_max_hops:
            parts.append(f"maxhop{t.train_max_hops}")
        if d.regime == "context" and d.n_distractors != DataConfig().n_distractors:
            parts.append(f"dist{d.n_distractors}")
        if d.regime == "context" and d.distractor_ramp_epochs:
            parts.append(f"ramp{d.distractor_ramp_epochs}")
        parts.append(f"s{t.seed}")
        return "_".join(parts)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return _build(cls, d or {})

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "RunConfig":
        d = yaml.safe_load(Path(path).read_text()) or {}
        for ov in overrides or []:
            key, value = ov.split("=", 1)
            _set_path(d, key.split("."), yaml.safe_load(value))
        return cls.from_dict(d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _short(n: int) -> str:
    for unit, div in (("G", 10**9), ("M", 10**6), ("k", 10**3)):
        if n >= div and n % (div // 10) == 0:
            v = n / div
            return f"{v:g}{unit}"
    return str(n)


def _set_path(d: dict, keys: list[str], value) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _build(cls, d: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    defaults = cls()
    kwargs = {}
    for name, value in d.items():
        default = getattr(defaults, name)
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)
