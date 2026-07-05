#!/usr/bin/env python
"""Generate Fig. 3(a)-style fidelity-estimation data."""

from __future__ import annotations

import argparse
from pathlib import Path

from shallow_rm_numerics.config import load_yaml
from shallow_rm_numerics.fidelity import run_fidelity_simulation_qiskit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fig3_fidelity.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = run_fidelity_simulation_qiskit(
        num_qubits=int(cfg.get("num_qubits", 48)),
        depth=int(cfg.get("depth", 8)),
        block_sizes=list(cfg.get("block_sizes", [1, 2, 3])),
        shots_list=list(cfg.get("shots_list", [1, 5, 10, 20, 40, 100])),
        num_unitaries=int(cfg.get("num_unitaries", 800)),
        seed=int(cfg.get("seed", 0)),
        max_circuits_per_job=int(cfg.get("max_circuits_per_job", 50)),
        output_dir=Path(cfg.get("output_dir", "data/processed")),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
