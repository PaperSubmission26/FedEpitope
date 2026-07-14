# FedEpitope

<p align="center">
  <b>Federated peptide epitope prediction with ESM-2 and parameter-efficient LoRA adaptation</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#main-results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#reproducing-the-experiments">Reproduction</a> •
  <a href="#external-netbce-evaluation">NetBCE</a>
</p>

## Overview

FedEpitope is a PyTorch research codebase for binary peptide epitope prediction under federated data heterogeneity. It adapts the ESM-2 protein language model with LoRA, trains client models locally, and aggregates the trainable parameters with sample-size-weighted FedAvg.

The repository includes:

* non-IID federated training over five pathogen/domain clients;
* an IID federated comparison;
* random, single-client, frozen-backbone, and centralised baselines;
* personalised federated learning variants;
* record-level differential privacy experiments with Opacus;
* LoRA-rank and Dirichlet-heterogeneity studies;
* per-client evaluation of global federated checkpoints; and
* external comparison with the NetBCE benchmark and official NetBCE model.

### Method at a glance

1. Each client receives the current global ESM-2 LoRA model.
2. Each client performs local optimization on its private training split.
3. The server aggregates trainable parameters using client training-set sizes.
4. The best global round is selected using weighted client validation AUC-ROC.
5. The held-out central test set is evaluated once after model selection.

The five non-IID clients are:

| Client | Domain      |
| -----: | ----------- |
|      1 | Coronavirus |
|      2 | Parasite    |
|      3 | Human/Self  |
|      4 | Flavivirus  |
|      5 | Bacteria    |

## Main Results

The table below summarizes the committed central-test results over seeds `42`, `43`, and `44`. Values are mean ± sample standard deviation.

| Training setting           |             AUC-ROC |              AUC-PR |                  F1 |                 MCC |
| -------------------------- | ------------------: | ------------------: | ------------------: | ------------------: |
| FedEpitope, non-IID FedAvg |     0.7380 ± 0.0038 |     0.3993 ± 0.0038 |     0.4301 ± 0.0177 |     0.2795 ± 0.0243 |
| IID FedAvg                 |     0.7978 ± 0.0018 |     0.4806 ± 0.0059 | **0.4944 ± 0.0017** |     0.3647 ± 0.0015 |
| Centralised LoRA           | **0.8009 ± 0.0013** | **0.4857 ± 0.0016** |     0.4937 ± 0.0013 | **0.3653 ± 0.0013** |

Per-seed metrics and auxiliary experiment results are available under [`results/`](results/).

## Default Configuration

| Component                 | Default                            |
| ------------------------- | ---------------------------------- |
| Backbone                  | `facebook/esm2_t12_35M_UR50D`      |
| Task                      | Binary sequence classification     |
| LoRA target modules       | `query`, `value`                   |
| LoRA rank                 | `2`                                |
| LoRA alpha                | `2 × rank`                         |
| LoRA dropout              | `0.1`                              |
| Federated rounds          | `20`                               |
| Local epochs per round    | `3`                                |
| Batch size                | `64`                               |
| Learning rate             | `2e-4`                             |
| Maximum tokenized length  | `30`                               |
| Aggregation               | Sample-size-weighted FedAvg        |
| Selection metric          | Weighted client validation AUC-ROC |
| Test threshold for F1/MCC | `0.5`                              |
| Reported metrics          | AUC-ROC, AUC-PR, F1, MCC           |

## Repository Structure

```text
FedEpitope/
├── data/                         # Client, central, IID, and external CSV files
├── results/                      # Committed metrics and experiment summaries
├── federated_train.py            # Main non-IID FedAvg experiment
├── iid_federated_train.py        # IID FedAvg comparison
├── baseline.py                   # Random/local/frozen/centralised baselines
├── personalised_fl_v2.py         # Personalised FL variants
├── differential_privacy.py       # Opacus DP experiments
├── ablation.py                   # LoRA-rank ablation
├── dirichlet_experiment.py       # Synthetic non-IID robustness grid
├── eval_global_per_client.py     # Global model on each client test split
├── netbce_validation.py          # FedEpitope/centralised external validation
├── evaluate_official_netbce.py   # Official legacy NetBCE evaluation wrapper
├── local_main.py                 # Hard-coded local-training demonstration
├── check_leakage.py              # Sequence-overlap diagnostic utility
├── filter_test.py                # Test-set filtering utility
├── preprocess_v2.py              # Data preprocessing helper
├── verify_esm2.py                # ESM-2 installation/device smoke test
└── requirements.txt              # Author environment snapshot
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/PaperSubmission26/FedEpitope.git
cd FedEpitope
```

### 2. Create an isolated environment

```bash
conda create -n fedepitope python=3.10 -y
conda activate fedepitope
python -m pip install --upgrade pip
```

### 3. Install the runtime dependencies

Install the PyTorch build appropriate for your CPU/CUDA environment, then install the core packages used by the scripts:

```bash
python -m pip install torch==2.5.1
python -m pip install \
  transformers==5.5.0 \
  peft==0.18.1 \
  accelerate==1.13.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  scikit-learn==1.7.2 \
  matplotlib==3.10.8 \
  opacus==1.5.4
```

> **Environment note:** the committed `requirements.txt` is a full author-environment snapshot rather than a minimal portable dependency file. It includes CUDA-specific packages and a machine-local package reference. The minimal installation above is therefore recommended for a clean environment.

### 4. Verify ESM-2

```bash
python verify_esm2.py
```

The first execution downloads `facebook/esm2_t12_35M_UR50D` unless it is already available in the Hugging Face cache.

## Data

All main scripts are intended to be run from the repository root. Input CSV files require at least:

```text
sequence,label
SIINFEKL,1
AAAAAAAA,0
```

* `sequence`: peptide/amino-acid sequence;
* `label`: binary target, where `1` denotes epitope and `0` denotes non-epitope.

The main file layout is:

```text
data/
├── client1_train.csv ... client5_train.csv
├── client1_val.csv   ... client5_val.csv
├── client1_test.csv  ... client5_test.csv
├── iid_client1_train.csv ... iid_client5_train.csv
├── iid_client1_val.csv   ... iid_client5_val.csv
├── central_val.csv
├── central_test.csv
├── external_validation.csv
└── netbce_independent_test.csv
```

## Quick Start

Run one non-IID FedEpitope experiment:

```bash
python federated_train.py \
  --seed 42 \
  --lora_rank 2 \
  --results_dir results/r2_seed42
```

Important outputs are:

```text
results/r2_seed42/
├── best_global_weights.pt
├── global_weights_final.pt
├── federated_val_rounds.csv
├── client_round_logs.csv
└── federated_test_results.csv
```

`best_global_weights.pt` is selected using validation AUC-ROC. `global_weights_final.pt` stores the model after the last communication round; it is not necessarily the selected model.

## Reproducing the Experiments

### Execution order

Model checkpoints are excluded from version control. A fresh clone therefore contains committed metric CSVs but not the `*.pt` files required by downstream evaluations.

```text
federated_train.py ──┬──> personalised_fl_v2.py
                     ├──> eval_global_per_client.py
                     └──┐
                        ├──> netbce_validation.py
baseline.py ────────────┘
```

Run `federated_train.py` before personalised or per-client evaluation. Run both `federated_train.py` and `baseline.py` before the complete FedEpitope-versus-centralised NetBCE comparison.

### 1. Non-IID FedEpitope, three seeds

```bash
for seed in 42 43 44; do
  python federated_train.py \
    --seed "${seed}" \
    --lora_rank 2 \
    --results_dir "results/r2_seed${seed}"
done
```

### 2. Baselines, three seeds

`baseline.py` evaluates a random predictor, five single-client LoRA models, frozen ESM-2, and centralised LoRA on the central test set.

```bash
for seed in 42 43 44; do
  python baseline.py \
    --seed "${seed}" \
    --lora_rank 2 \
    --results_dir "results/baseline_seed${seed}"
done
```

Use a different `--results_dir` for every seed. Otherwise, all runs use the script's default `results/baseline_seed42` directory and later runs overwrite earlier outputs.

### 3. IID FedAvg comparison

```bash
for seed in 42 43 44; do
  python iid_federated_train.py \
    --seed "${seed}" \
    --lora_rank 2 \
    --results_dir "results/iid_seed${seed}"
done
```

### 4. Personalised federated learning

The script evaluates three variants: local fine-tuning, classifier-only personalization, and FedProx-based personalization.

```bash
for seed in 42 43 44; do
  python personalised_fl_v2.py \
    --seed "${seed}" \
    --lora_rank 2 \
    --fed_checkpoint "results/r2_seed${seed}/best_global_weights.pt" \
    --results_dir "results/pfl_seed${seed}"
done
```

Main outputs:

```text
pfl_per_client.csv
pfl_results.csv
```

### 5. Evaluate each global checkpoint on all client test sets

This script uses hard-coded seeds `42`, `43`, and `44` and expects the standard checkpoint paths under `results/r2_seed*/`.

```bash
python eval_global_per_client.py
```

It writes `global_per_client_test.csv` inside each corresponding seed directory.

### 6. Differential privacy

Without `--epsilon`, the script runs cumulative privacy budgets `1`, `5`, `10`, and the non-private setting (`inf`). It uses `δ = 1e-5` and clipping norm `1.0`.

```bash
for seed in 42 43 44; do
  python differential_privacy.py \
    --seed "${seed}" \
    --lora_rank 2 \
    --results_dir "results/dp_seed${seed}"
done
```

Run one privacy budget only:

```bash
python differential_privacy.py \
  --seed 42 \
  --epsilon 5 \
  --results_dir results/dp_eps5_seed42
```

Run the non-private branch only:

```bash
python differential_privacy.py \
  --seed 42 \
  --epsilon inf \
  --results_dir results/dp_nodp_seed42
```

The final summary is stored in `dp_results.csv`.

### 7. LoRA-rank ablation

The standard rank-2 result comes from `federated_train.py`. The ablation script defaults to ranks `4`, `8`, and `16`.

```bash
python ablation.py \
  --seed 42 \
  --ranks 4,8,16 \
  --results_dir results/rank_ablation_seed42
```

The final table is stored in `ablation_lora_rank.csv`.

### 8. Dirichlet non-IID robustness

The default grid evaluates:

* concentration parameters `α ∈ {0.3, 0.5, 1.0, 5.0}`;
* client counts `K ∈ {3, 5, 10}`; and
* seeds `{42, 43, 44}`.

This is a 36-run experiment and uses one local epoch per round for feasibility.

```bash
python dirichlet_experiment.py \
  --lora_rank 2 \
  --results_dir results/dirichlet \
  --figures_dir figures
```

Outputs include:

```text
results/dirichlet/dirichlet_raw.csv
results/dirichlet/dirichlet_results.csv
results/dirichlet/dirichlet_partition_stats.csv
figures/fig_dirichlet.png
figures/fig_dirichlet.pdf
```

## External NetBCE Evaluation

### FedEpitope and centralised LoRA checkpoints

The cached, overlap-filtered external test set is included at:

```text
data/netbce_independent_test.csv
```

After generating all three federated and centralised checkpoints, run:

```bash
python netbce_validation.py \
  --lora_rank 2 \
  --results_dir results/netbce
```

The script searches for:

```text
results/r2_seed{42,43,44}/best_global_weights.pt
results/baseline_seed{42,43,44}/centralised_weights.pt
```

Missing checkpoints are skipped, so generate all six checkpoints for the complete comparison.

The command produces detailed and aggregated CSV files under `results/netbce/` and ROC/bar plots under `figures/`.

#### Rebuilding the NetBCE-derived CSV

Normally, use the committed `data/netbce_independent_test.csv`. The `--rebuild_netbce` path currently expects the official NetBCE repository at `/tmp/NetBCE`, including:

```text
/tmp/NetBCE/data/testing dataset.txt
```

Then run:

```bash
python netbce_validation.py \
  --lora_rank 2 \
  --results_dir results/netbce \
  --rebuild_netbce
```

Edit `NETBCE_REPO` in `netbce_validation.py` when using another checkout location.

<details>
<summary><b>Official legacy NetBCE baseline</b></summary>

The official NetBCE implementation uses Python 3.7, Keras 2.3.1, and TensorFlow 1.15. Keep it in a separate environment from FedEpitope.

```bash
conda create -n netbce-official python=3.7 -y
conda activate netbce-official
pip install pandas numpy scipy plotly dominate scikit-learn
pip install keras==2.3.1 tensorflow==1.15 protobuf==3.20 h5py==2.10.0
```

Clone NetBCE beside FedEpitope:

```bash
cd ..
git clone https://github.com/bsml320/NetBCE.git
cd FedEpitope
```

Run the wrapper on the same de-duplicated external test set:

```bash
CUDA_VISIBLE_DEVICES="" python evaluate_official_netbce.py \
  --test_csv data/netbce_independent_test.csv \
  --netbce_repo ../NetBCE \
  --results_dir results/netbce_official
```

Expected outputs:

```text
results/netbce_official/official_netbce_metrics.csv
results/netbce_official/official_netbce_scores.csv
results/netbce_official/NetBCE_predictions.tsv
```

</details>

## Evaluation Protocol

* Random seeds: `42`, `43`, and `44` for the main reported comparisons.
* Checkpoint selection: validation AUC-ROC only.
* Central test usage: once per trained model after selection.
* Classification threshold: `0.5` for F1 and MCC.
* Multi-seed reporting: mean ± sample standard deviation.
* Model files: `*.pt`, `*.pth`, and `*.ckpt` are excluded by `.gitignore`.
* Committed `results/`: metric tables and summaries, not downloadable trained checkpoints.

## Common Issues

### A downstream script cannot find `best_global_weights.pt`

Run the matching non-IID training command first and confirm that its `--results_dir` matches the downstream path.

### Baseline results from multiple seeds are identical or missing

Pass a seed-specific `--results_dir`. The default output directory in `baseline.py` is always `results/baseline_seed42`, regardless of the value of `--seed`.

### CUDA out-of-memory

Reduce `--batch_size`, set `--num_workers 0`, or run on a GPU with more memory. ESM-2 weights are downloaded and instantiated separately in several experiment loops, so baseline and ablation runs are more demanding than a single training run.

### Hugging Face download fails on a compute node

Download/cache `facebook/esm2_t12_35M_UR50D` on a node with internet access and reuse the same Hugging Face cache on the compute node.

## Citation

Citation metadata will be added after the accompanying paper is publicly available.

## License

No open-source license file is currently included in this repository. Add the intended `LICENSE` file before public release and update this section accordingly.

## Acknowledgements

This project uses ESM-2 through Hugging Face Transformers, LoRA through PEFT, and differential privacy utilities from Opacus. The external baseline evaluation uses the official NetBCE implementation.
