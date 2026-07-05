#!/usr/bin/env python
"""Generate Fig. 5-style DMRG shadow-kernel numerical data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shallow_rm_numerics.config import load_yaml
from shallow_rm_numerics.ml_shadow import run_ml_kernel_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fig5_ml.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = run_ml_kernel_simulation(
        system_sizes=list(cfg.get("system_sizes", [12])),
        block_sizes=list(cfg.get("block_sizes", [1, 2, 3])),
        num_data=int(cfg.get("num_data", 40)),
        parameter_min=float(cfg.get("parameter_min", 0.0)),
        parameter_max=float(cfg.get("parameter_max", 2.0)),
        num_unitaries=int(cfg.get("num_unitaries", 30)),
        shots_per_unitary=int(cfg.get("shots_per_unitary", 1)),
        coupling_even=float(cfg.get("coupling_even", 1.0)),
        delta=float(cfg.get("delta", 0.0)),
        boundary_penalty=float(cfg.get("boundary_penalty", 0.1)),
        max_bond_dim=int(cfg.get("max_bond_dim", 30)),
        dmrg_sweeps=int(cfg.get("dmrg_sweeps", 5)),
        dmrg_krylov=int(cfg.get("dmrg_krylov", 5)),
        exclude_identical_unitaries=bool(cfg.get("exclude_identical_unitaries", False)),
        seed=int(cfg.get("seed", 0)),
        output_dir=Path(cfg.get("output_dir", "data/processed")),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
