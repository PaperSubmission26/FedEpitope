# FedEpitope

**FedEpitope: Towards Generalisable B-Cell Epitope Prediction with Personalised Federated Protein Language Models**

FedEpitope is a federated learning framework for peptide-level linear B-cell epitope prediction using ESM-2 and Low-Rank Adaptation (LoRA).

This repository contains the implementation, evaluation scripts, and lightweight reproducibility materials for FedEpitope.

## Key Features

- ESM-2 with parameter-efficient LoRA adaptation
- Federated Averaging (FedAvg)
- FedProx
- Personalised federated learning
- Record-level differential privacy
- Homology-aware data splitting
- IID and biologically heterogeneous non-IID settings
- Controlled Dirichlet heterogeneity experiments
- Strict overlap-filtered evaluation
- External evaluation on BCPreds and NetBCE

## Installation

```bash
git clone https://github.com/PaperSubmission26/FedEpitope.git
cd FedEpitope

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-enabled PyTorch environment is recommended for model training.

## Repository Structure

```text
FedEpitope/
├── README.md
├── requirements.txt
├── data/
├── results/
└── scripts/
```

## Main Scripts

| Task | Script |
|---|---|
| Centralised and local baselines | `scripts/baseline.py` |
| FedAvg | `scripts/federated_train.py` |
| FedProx | `scripts/fedprox_train.py` |
| IID federated training | `scripts/federated_train_iid.py` |
| Personalised federated learning | `scripts/personalised_fl.py` |
| Differential privacy | `scripts/differential_privacy.py` |
| Federated client construction | `scripts/make_federated_clients.py` |
| IID client construction | `scripts/create_iid_clients.py` |
| Personalisation split construction | `scripts/create_pfl_splits.py` |
| Dirichlet partition construction | `scripts/create_dirichlet_partitions.py` |
| Dirichlet federated training | `scripts/dirichlet_train.py` |
| Homology-aware splitting | `scripts/make_homology_split.py` |
| Substring-overlap audit | `scripts/check_substring_overlap.py` |
| Strict-test construction | `scripts/create_strict_test.py` |
| Strict-test evaluation | `scripts/evaluate_strict_test.py` |
| BCPreds audit | `scripts/audit_external_bcpreds.py` |
| BCPreds evaluation | `scripts/evaluate_external_bcpreds.py` |
| NetBCE common-subset evaluation | `scripts/evaluate_netbce_common_subset.py` |
| NetBCE external evaluation | `scripts/run_netbce_external_updated.py` |

## Experimental Configuration

The main federated experiments use:

```text
Model:                  facebook/esm2_t12_35M_UR50D
Federated clients:      5
Communication rounds:   20
Local epochs:           3
Batch size:             64
Learning rate:          2e-4
Optimiser:              AdamW
Maximum peptide length: 30
LoRA rank:              2
LoRA alpha:             4
LoRA dropout:           0.1
FedProx mu:             0.001
Seeds:                  42, 43, 44
```

### Differential Privacy

```text
Target epsilon:  {1, 5, 10}
Delta:           1e-5
Clipping norm:   1
```

Differential privacy is applied at the record level, with one peptide record as the protected unit.

### Dirichlet Heterogeneity

```text
Number of clients K:  {3, 5, 10}
Dirichlet alpha:      {0.1, 0.5, 1.0, 10.0}
```

### LoRA-Rank Analysis

```text
r = {2, 4, 8, 16}
```

## Data

The primary dataset is derived from experimentally assayed peptide records obtained from the **Immune Epitope Database (IEDB)**.

The main experimental setting uses homology-aware data splitting and five biologically heterogeneous, organism-partitioned federated clients.

Lightweight audit and evaluation files included in this repository are:

```text
data/global_test_strict.csv
data/train_test_substring_overlaps.csv
data/train_validation_substring_overlaps.csv
```

Large generated datasets, client partitions, split metadata, model checkpoints, and training logs are not stored directly in this repository.

## Results

Compact result summaries are provided in:

```text
results/baseline_summary_3seeds.csv
results/fedavg_all_seeds.csv
results/fedavg_summary_3seeds.csv
```

Mean global-test performance across three random seeds:

| Model | AUC-ROC | AP | F1 | MCC |
|---|---:|---:|---:|---:|
| Centralised LoRA | 0.8023 | 0.3781 | 0.3719 | 0.3046 |
| FedAvg | 0.7534 | 0.3187 | 0.3535 | 0.2635 |
| FedProx | 0.7578 | 0.3110 | 0.3423 | 0.2586 |

For AUC-ROC, the corresponding mean ± standard deviation values are:

```text
Centralised LoRA:  0.8023 ± 0.0124
FedAvg:             0.7534 ± 0.0069
FedProx:            0.7578 ± 0.0130
```

## Strict-Test Evaluation

Residual peptide substring overlap between the training and test partitions is evaluated separately from the homology-aware split.

The strict test set removes identified train-test substring overlaps without retraining or reselecting model checkpoints.

Relevant scripts:

```text
scripts/check_substring_overlap.py
scripts/create_strict_test.py
scripts/evaluate_strict_test.py
```

## External Evaluation

Generalisation is additionally evaluated on the independent:

- **BCPreds** benchmark
- **NetBCE** benchmark

Relevant scripts:

```text
scripts/audit_external_bcpreds.py
scripts/evaluate_external_bcpreds.py
scripts/evaluate_netbce_common_subset.py
scripts/run_netbce_external_updated.py
```

Third-party benchmark repositories and software are not redistributed here and should be obtained from their original sources.

## Reproducibility

The main experiments use the random seeds:

```text
42
43
44
```

Model selection is performed using validation AUC-ROC, with the selected checkpoint evaluated on the corresponding held-out test set.

Generated checkpoints, per-round logs, large intermediate datasets, third-party repositories, and derived experiment directories are excluded from GitHub to keep the repository lightweight.

## Availability

Source code and lightweight reproducibility materials are available at:

```text
https://github.com/PaperSubmission26/FedEpitope
```
