#!/usr/bin/env python
"""Generate Fig. 3(b)-style purity-estimation data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shallow_rm_numerics.config import load_yaml
from shallow_rm_numerics.purity import run_purity_simulation_qiskit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fig3_purity.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = run_purity_simulation_qiskit(
        num_qubits=int(cfg.get("num_qubits", 24)),
        depth=int(cfg.get("depth", 12)),
        block_sizes=list(cfg.get("block_sizes", [1, 2, 3])),
        shots_list=list(cfg.get("shots_list", [2000, 3000, 4000])),
        num_unitaries=int(cfg.get("num_unitaries", 500)),
        seed=int(cfg.get("seed", 0)),
        max_circuits_per_job=int(cfg.get("max_circuits_per_job", 50)),
        exact_pair_limit=int(cfg.get("exact_pair_limit", 2_000_000)),
        pair_samples=int(cfg.get("pair_samples", 500_000)),
        output_dir=Path(cfg.get("output_dir", "data/processed")),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
