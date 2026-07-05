# Shallow randomized measurement numerics

This repository contains numerical code for the simulations associated with
"Shallow randomized measurement in noisy quantum devices". It is an initial
reproducibility release focused on the numerical parts of the manuscript:

- derandomized single-qubit and two-qubit block measurement selection,
- multi-shot fidelity and purity simulations,
- DMRG-based shadow-kernel machine-learning simulations.

The Fig. 4 cloud-hardware experiment is not included in this numerical release.

## Installation

```bash
conda env create -f environment.yml
conda activate shallow-rm-numerics
python -m pip install -e .
```

## Run the simulations

```bash
python scripts/run_fig2_derandomization.py --config configs/fig2_derandomization.yaml
python scripts/run_fig3_fidelity.py --config configs/fig3_fidelity.yaml
python scripts/run_fig3_purity.py --config configs/fig3_purity.yaml
python scripts/run_fig5_ml.py --config configs/fig5_ml.yaml
```

The scripts write processed output files to `data/processed/`. The Fig. 3
scripts require Qiskit/Aer. The Fig. 5 script uses the DMRG/MPS workflow.

The configuration files contain manuscript-scale parameters. For a quick local
check, reduce values such as `n_values`, `num_unitaries`, `shots_list`,
`system_sizes`, and `num_data`.

## Repository structure

```text
src/shallow_rm_numerics/   Core numerical routines
scripts/                   Entry-point scripts for each numerical task
configs/                   YAML configuration files
DATA_AVAILABILITY.md       Suggested manuscript wording
```

## Notes

- Random seeds are specified in the YAML files.
- The release is intended to regenerate numerical data, not to rerun cloud
  hardware jobs.
- The Fig. 5 workflow uses DMRG only.
- Generated `.csv` and `.npz` files should be archived with the code release if
  they are used for the submitted manuscript.

## License

The code is released under the MIT License.
