# FedEpitope

FedEpitope is a PyTorch-based project for peptide epitope prediction using federated learning. It fine-tunes the ESM-2 protein language model with LoRA adapters on client-specific epitope datasets and evaluates the resulting global model on held-out validation and test data.

## Overview

The repository implements:
- Federated training with weighted FedAvg aggregation
- Local training baselines
- Centralized and ablation-style experiments
- Differential privacy and personalized FL variants
- Evaluation scripts for validation and client-wise analysis

The core idea is to train a shared epitope classifier without directly sharing each client's data.

## Key Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- scikit-learn
- pandas
- Opacus (differential privacy)

## Repository Structure

- `federated_train.py` – main federated training pipeline (non-IID, weighted FedAvg)
- `iid_federated_train.py` – IID partitioning experiment
- `baseline.py` – centralized baseline training
- `local_main.py` – single-client/local training example
- `personalised_fl_v2.py` – personalized FL variants (loads a federated checkpoint)
- `differential_privacy.py` – differential privacy training variant (Opacus)
- `ablation.py` – LoRA rank ablation and Dirichlet non-IID robustness experiments
- `eval_global_per_client.py` – evaluates global federated checkpoints per client
- `netbce_validation.py` – external validation on NetBCE independent test set
- `preprocess_v2.py` – data preprocessing helper
- `data/` – train/validation/test CSV files for multiple clients
- `results/` – experiment outputs, metrics, and saved weights

## Requirements

```bash
pip install -r requirements.txt
```

A GPU is recommended for training, but CPU execution may work for smaller experiments.

## Data Format

The training scripts expect CSV files with at least the following columns:
- `sequence` – peptide or protein sequence string
- `label` – binary label (`0` or `1`)

Expected files are located under `data/`, including client-specific train/validation splits, IID re-partitioned splits, and a central test set.

## Fixed Defaults

Unless overridden, all experiments use:
- Backbone: ESM-2 (`facebook/esm2_t12_35M_UR50D`)
- LoRA rank: 2 (Pareto-optimal per the rank ablation in Table 5)
- 4 dataloader workers

---

## Reproducing Paper Results

### Table 1: Centralized Baseline (Seeds 42, 43, 44)
```bash
python baseline.py --seed 42
python baseline.py --seed 43
python baseline.py --seed 44
```

### Table 2 / Figure 2: Federated Training — Non-IID (main result, r=2)
```bash
python federated_train.py --seed 42 --lora_rank 2 --results_dir results/r2_seed42
python federated_train.py --seed 43 --lora_rank 2 --results_dir results/r2_seed43
python federated_train.py --seed 44 --lora_rank 2 --results_dir results/r2_seed44
```

### Figure 2: Federated Training — IID Comparison
```bash
python iid_federated_train.py --seed 42 --lora_rank 2 --results_dir results/iid_seed42
python iid_federated_train.py --seed 43 --lora_rank 2 --results_dir results/iid_seed43
python iid_federated_train.py --seed 44 --lora_rank 2 --results_dir results/iid_seed44
```

### Figure 3 / Table 3: Personalized FL Variants (pFL-LFT, pFL-CLF-Only, pFL-FedProx)
Requires the federated checkpoint from the Table 2 / Figure 2 run above.
```bash
python personalised_fl_v2.py --seed 42 --fed_checkpoint results/r2_seed42/best_global_weights.pt --results_dir results/pfl_seed42
python personalised_fl_v2.py --seed 43 --fed_checkpoint results/r2_seed43/best_global_weights.pt --results_dir results/pfl_seed43
python personalised_fl_v2.py --seed 44 --fed_checkpoint results/r2_seed44/best_global_weights.pt --results_dir results/pfl_seed44
```

### Table 4: Differential Privacy (ε = 1, 5, 10, No-DP)
Runs all epsilon values by default; use `--epsilon` to run a single value (use `inf` for No-DP).
```bash
python differential_privacy.py --seed 42 --results_dir results/dp_seed42
python differential_privacy.py --seed 43 --results_dir results/dp_seed43
python differential_privacy.py --seed 44 --results_dir results/dp_seed44
```

### Table 5: LoRA Rank Ablation (r ∈ {4, 8, 16}, seed 42)
```bash
python ablation.py --seed 42 --ranks 4,8,16 --results_dir results/rank_ablation_seed42
```

### Dirichlet Non-IID Robustness (α ∈ {0.3, 0.5, 1.0, 5.0}, K ∈ {3, 5, 10}, seeds 42–44)
```bash
python ablation.py
```


---

## Evaluation

### Per-client evaluation of the global federated model
Single command — evaluates the best global checkpoints for seeds 42, 43, and 44 on each client's pathogen-specific test set:
```bash
python eval_global_per_client.py
```
Outputs:
- `results/r2_seed42/global_per_client_test.csv`
- `results/r2_seed43/global_per_client_test.csv`
- `results/r2_seed44/global_per_client_test.csv`

### External validation on NetBCE
```bash
python netbce_validation.py --lora_rank 2 --results_dir results/netbce
```
Use `--rebuild_netbce` to re-parse and re-filter the NetBCE FASTA file even if `data/netbce_independent_test.csv` already exists.

---

## Notes for Reproduction

- All scripts assume the repository root as the working directory.
- All scripts use consistent seeding across Python, NumPy, and PyTorch RNG states, with validation-based checkpoint selection.
- Required checkpoints (`*.pt`) are excluded from version control by default via `.gitignore`, except for the best global federated weights per seed needed for personalized FL and per-client evaluation. See `.gitignore` for the explicit allow-list.
- Metrics reported: AUC-ROC, AUC-PR, F1, and MCC.

## License

This project is intended for research and educational use.