# FedEpitope

FedEpitope is a PyTorch-based repository for peptide epitope prediction using federated learning. It fine-tunes the ESM-2 protein language model with LoRA adapters on client-specific epitope datasets and evaluates the resulting models on validation and test data.

## Overview

The repository contains training and evaluation scripts for several experiment setups:
- federated training with weighted FedAvg aggregation
- IID federated training for comparison
- centralized baseline training
- local single-client training
- personalized federated learning variants
- differential privacy experiments with Opacus
- ablation studies over LoRA rank and data partitioning
- external validation on a NetBCE test set

## Project Structure

- [federated_train.py](federated_train.py) – main federated training pipeline
- [iid_federated_train.py](iid_federated_train.py) – IID federated training comparison
- [baseline.py](baseline.py) – centralized baseline training
- [local_main.py](local_main.py) – simple local/single-client training example
- [personalised_fl_v2.py](personalised_fl_v2.py) – personalized federated learning variants
- [differential_privacy.py](differential_privacy.py) – DP training experiments
- [ablation.py](ablation.py) – LoRA rank and robustness ablation studies
- [eval_global_per_client.py](eval_global_per_client.py) – evaluate global checkpoints per client
- [netbce_validation.py](netbce_validation.py) – external NetBCE validation
- [preprocess_v2.py](preprocess_v2.py) – preprocessing helper
- [data/](data/) – train/validation/test CSV files for multiple clients
- [results/](results/) – experiment outputs, metrics, and saved weights

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A GPU is recommended for model training, though CPU execution may work for smaller experiments.

## Data Format

The training scripts expect CSV files with at least these columns:
- `sequence`: peptide or protein sequence string
- `label`: binary label (`0` or `1`)

The repository includes client-specific train/validation splits, IID re-partitioned splits, and a central test set under [data/](data/).

## Default Model Setup

The main scripts use:
- backbone: ESM-2 (`facebook/esm2_t12_35M_UR50D`)
- fine-tuning: LoRA via PEFT
- default LoRA rank: 2
- default dataloader workers: 4

## Running the Experiments

### 1. Centralized baseline

```bash
python baseline.py --seed 42
python baseline.py --seed 43
python baseline.py --seed 44
```

### 2. Federated training (non-IID)

```bash
python federated_train.py --seed 42 --lora_rank 2 --results_dir results/r2_seed42
python federated_train.py --seed 43 --lora_rank 2 --results_dir results/r2_seed43
python federated_train.py --seed 44 --lora_rank 2 --results_dir results/r2_seed44
```

### 3. IID federated comparison

```bash
python iid_federated_train.py --seed 42 --lora_rank 2 --results_dir results/iid_seed42
python iid_federated_train.py --seed 43 --lora_rank 2 --results_dir results/iid_seed43
python iid_federated_train.py --seed 44 --lora_rank 2 --results_dir results/iid_seed44
```

### 4. Personalized FL variants

These commands use a federated checkpoint from the runs above.

```bash
python personalised_fl_v2.py --seed 42 --fed_checkpoint results/r2_seed42/best_global_weights.pt --results_dir results/pfl_seed42
python personalised_fl_v2.py --seed 43 --fed_checkpoint results/r2_seed43/best_global_weights.pt --results_dir results/pfl_seed43
python personalised_fl_v2.py --seed 44 --fed_checkpoint results/r2_seed44/best_global_weights.pt --results_dir results/pfl_seed44
```

### 5. Differential privacy experiments

```bash
python differential_privacy.py --seed 42 --results_dir results/dp_seed42
python differential_privacy.py --seed 43 --results_dir results/dp_seed43
python differential_privacy.py --seed 44 --results_dir results/dp_seed44
```

Use `--epsilon` to run a single epsilon setting, and `inf` for the No-DP baseline.

### 6. Ablation studies

```bash
python ablation.py --seed 42 --ranks 4,8,16 --results_dir results/rank_ablation_seed42
```

The script also supports Dirichlet-based non-IID experiments with its default settings.

## Evaluation

### Per-client evaluation

```bash
python eval_global_per_client.py
```

This writes per-client evaluation CSV files under the corresponding seed result directories.

### NetBCE external validation

```bash
python netbce_validation.py --lora_rank 2 --results_dir results/netbce
```

Use `--rebuild_netbce` if you want to regenerate the NetBCE-derived CSV from the source FASTA data.

## Notes

- Run the scripts from the repository root.
- The code uses consistent seeding for Python, NumPy, and PyTorch.
- Checkpoint files such as `*.pt` are ignored by Git by default, so any weights you want to reuse should be kept outside version control or explicitly copied.
- Reported metrics include AUC-ROC, AUC-PR, F1, and MCC.

## License

This project is intended for research and educational use.
