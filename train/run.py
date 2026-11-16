"""Train one run from a YAML config, with optional dotted overrides.

    uv run python -m train.run configs/runs/base.yaml data.encoding=prose model.size=M train.seed=1
"""

from __future__ import annotations

import argparse

from .config import RunConfig
from .trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. train.seed=1")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = RunConfig.from_yaml(args.config, args.overrides)
    Trainer(cfg, runs_dir=args.runs_dir).train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
