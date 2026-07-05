#!/usr/bin/env python
"""Generate Fig. 2-style derandomization summary data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shallow_rm_numerics.config import load_yaml
from shallow_rm_numerics.derandomization import run_cluster_heisenberg_derandomization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fig2_derandomization.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = run_cluster_heisenberg_derandomization(
        n_values=list(cfg.get("n_values", [10, 20])),
        methods=tuple(cfg.get("methods", ["single_qubit", "two_qubit_alternating"])),
        target_hits=int(cfg.get("target_hits", 100)),
        error=float(cfg.get("error", 0.9)),
        seed=int(cfg.get("seed", 0)),
        output_dir=Path(cfg.get("output_dir", "data/processed")),
        save_bases=bool(cfg.get("save_bases", True)),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
