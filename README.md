
# FedEpitope

FedEpitope is a PyTorch-based repository for peptide epitope prediction using federated learning. It fine-tunes the ESM-2 protein language model with LoRA adapters on client-specific epitope datasets and evaluates the resulting models on in-distribution and external benchmark data.

## Overview

The repository contains training and evaluation scripts for the following experiment settings:

- Federated training with weighted FedAvg aggregation
- IID federated training comparison
- Centralised baseline training
- Local single-client training
- Personalised federated learning variants
- Differential privacy experiments using Opacus
- LoRA rank ablation studies
- Dirichlet non-IID partitioning experiments
- External validation on the de-duplicated NetBCE benchmark
- Official pretrained NetBCE baseline evaluation

## Project Structure

- [federated_train.py](federated_train.py) – main non-IID federated training pipeline
- [iid_federated_train.py](iid_federated_train.py) – IID federated training comparison
- [baseline.py](baseline.py) – centralised LoRA baseline training
- [local_main.py](local_main.py) – local/single-client training example
- [personalised_fl_v2.py](personalised_fl_v2.py) – personalised federated learning variants
- [differential_privacy.py](differential_privacy.py) – differential privacy experiments
- [ablation.py](ablation.py) – LoRA rank ablation studies
- [dirichlet_experiment.py](dirichlet_experiment.py) – Dirichlet non-IID partitioning experiments
- [eval_global_per_client.py](eval_global_per_client.py) – evaluation of global checkpoints on each client
- [netbce_validation.py](netbce_validation.py) – external NetBCE validation for FedEpitope and centralised LoRA
- [evaluate_official_netbce.py](evaluate_official_netbce.py) – official pretrained NetBCE model evaluation
- [preprocess_v2.py](preprocess_v2.py) – preprocessing helper
- [data/](data/) – train, validation, test, and external benchmark CSV files
- [results/](results/) – experiment outputs, metrics, and saved weights

## Requirements

Install the main FedEpitope dependencies with:

```bash
pip install -r requirements.txt
````

A GPU is recommended for training ESM-2 LoRA models. CPU execution may work for small evaluation scripts but is not recommended for full training.

## Data Format

The training and evaluation scripts expect CSV files with at least the following columns:

```text
sequence
label
```

where:

* `sequence` is the peptide or protein sequence string.
* `label` is the binary class label, with `1` for epitope and `0` for non-epitope.

The repository includes client-specific train/validation splits, IID re-partitioned splits, a central held-out test set, and a de-duplicated NetBCE external test set under [data/](data/).

## Default Model Setup

The main FedEpitope experiments use:

* Backbone: ESM-2 `facebook/esm2_t12_35M_UR50D`
* Fine-tuning method: LoRA via PEFT
* Default LoRA rank: 2
* Default dataloader workers: 4
* Evaluation metrics: AUC-ROC, AUC-PR, F1, and MCC

## Running the Experiments

Run all scripts from the repository root.

### 1. Centralised baseline

```bash
python baseline.py --seed 42
python baseline.py --seed 43
python baseline.py --seed 44
```

### 2. Federated training, non-IID clients

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

### 4. Personalised federated learning variants

These commands use federated checkpoints from the non-IID FedEpitope runs.

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

Use `--epsilon` to run a single privacy budget. Use `inf` for the non-private baseline.

Example:

```bash
python differential_privacy.py --seed 42 --epsilon 8 --results_dir results/dp_eps8_seed42
```

### 6. LoRA rank ablation

```bash
python ablation.py --seed 42 --ranks 4,8,16 --results_dir results/rank_ablation_seed42
```

### 7. Per-client evaluation

```bash
python eval_global_per_client.py
```

This writes per-client evaluation metrics under the corresponding result directories.

## NetBCE External Validation

The NetBCE validation uses the de-duplicated external benchmark file:

```text
data/netbce_independent_test.csv
```

This file contains NetBCE benchmark sequences after removing overlaps with IEDB.

### 1. Evaluate FedEpitope and centralised LoRA on NetBCE

```bash
python netbce_validation.py --lora_rank 2 --results_dir results/netbce
```

This produces summary and detailed NetBCE validation files under:

```text
results/netbce/
```

Use `--rebuild_netbce` only if you intentionally want to regenerate the NetBCE-derived CSV from the source FASTA data.

```bash
python netbce_validation.py --lora_rank 2 --results_dir results/netbce --rebuild_netbce
```

## Official NetBCE Baseline Evaluation

The official NetBCE model uses an older TensorFlow/Keras stack. Therefore, it should be evaluated in a separate conda environment rather than the main FedEpitope environment.

### 1. Create the official NetBCE environment

```bash
conda create -n netbce-official python=3.7 -y
conda activate netbce-official

pip install pandas numpy scipy plotly dominate scikit-learn
pip install keras==2.3.1
pip install tensorflow==1.15
pip install protobuf==3.20
pip install h5py==2.10.0
```

### 2. Download the official NetBCE repository

From the parent directory of this repository:

```bash
cd ..
wget https://github.com/bsml320/NetBCE/archive/refs/heads/main.zip -O NetBCE-main.zip
unzip NetBCE-main.zip
mv NetBCE-main NetBCE
cd FedEpitope
```

The expected directory structure is:

```text
parent_directory/
├── FedEpitope/
└── NetBCE/
```

### 3. Run official NetBCE on the de-duplicated NetBCE test set

Run from the FedEpitope repository root:

```bash
conda activate netbce-official

CUDA_VISIBLE_DEVICES="" python evaluate_official_netbce.py \
  --test_csv data/netbce_independent_test.csv \
  --netbce_repo ../NetBCE \
  --results_dir results/netbce_official
```

The script evaluates the official pretrained NetBCE model on the same de-duplicated NetBCE test sequences and labels used for FedEpitope and centralised LoRA.

It automatically passes the exact peptide lengths present in `data/netbce_independent_test.csv`, avoiding NetBCE's default candidate peptide lengths.

The outputs are saved under:

```text
results/netbce_official/
```

Important output files:

```text
results/netbce_official/official_netbce_metrics.csv
results/netbce_official/official_netbce_scores.csv
results/netbce_official/NetBCE_predictions.tsv
```

The AUC-ROC value in `official_netbce_metrics.csv` can be used as the official NetBCE row in the independent benchmark validation table.

## Notes

* Run all scripts from the repository root.
* Use the main FedEpitope environment for PyTorch/ESM-2/LoRA experiments.
* Use the separate `netbce-official` environment only for official NetBCE evaluation.
* Checkpoint files such as `*.pt` are ignored by Git by default.
* Metrics reported across experiments include AUC-ROC, AUC-PR, F1, and MCC.
* For FedEpitope and centralised LoRA, reported results are typically mean ± standard deviation over three seeds.
* For the official NetBCE model, the pretrained model is evaluated once on the de-duplicated NetBCE test set.

## License

This project is intended for research and educational use.

```
```
