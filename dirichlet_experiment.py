"""
dirichlet_experiment.py
========================
Dirichlet non-IID partitioning robustness experiment for FedEpitope (Figure 6).

Aligned with federated_train.py, baseline.py, personalised_fl_v2.py, and
differential_privacy.py:
  - LoRA rank r=2, alpha=4 (default; matches main experiments)
  - Val-based checkpoint selection; central test evaluated ONCE per run
  - set_seed / seed_worker / seeded per-client DataLoader generators
  - print_gpu_memory after every federated run
  - argparse CLI (--lora_rank, --results_dir, --figures_dir, ...)
  - Error handling with traceback saved to file
  - lora_rank recorded in output CSVs
  - Resume support: skips already-completed (alpha, K, seed) combos

Experiment grid:
  alpha  : 0.3, 0.5, 1.0, 5.0
  K      : 3, 5, 10
  seeds  : 42, 43, 44
  Total  : 36 runs
"""

import os
import copy
import time
import random
import warnings
import argparse
import math
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

NUM_ROUNDS          = 20
LOCAL_EPOCHS        = 1      # kept at 1 for feasibility across 36 runs
BATCH_SIZE          = 64
EVAL_BATCH_SIZE     = 128
LR                  = 2e-4
MAX_LEN             = 30
NUM_WORKERS         = 4

MIN_CLIENT_SAMPLES  = 200
MIN_POSITIVE_RATE   = 0.05
MAX_POSITIVE_RATE   = 0.95
MAX_PARTITION_TRIES = 100

ALPHA_VALUES  = [0.3, 0.5, 1.0, 5.0]
CLIENT_COUNTS = [3, 5, 10]
SEEDS         = [42, 43, 44]


# ============================================================
# Utility — identical to federated_train.py
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_seconds(seconds):
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()      / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()       / 1024 ** 3
        max_alloc = torch.cuda.max_memory_allocated()  / 1024 ** 3
        print(
            f"GPU memory | allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"max allocated: {max_alloc:.2f} GB"
        )
    else:
        print("GPU memory | CUDA not available")


# ============================================================
# Progress tracker
# ============================================================

class ProgressTracker:
    def __init__(self, total_runs: int):
        self.total      = total_runs
        self.completed  = 0
        self.skipped    = 0
        self.start_time = time.time()
        self.run_times  = []
        self._run_start = None

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

    def start_run(self, label: str) -> None:
        self._run_start = time.time()
        done = self.completed + self.skipped
        pct  = 100 * done / self.total
        bar  = chr(9608) * int(pct / 5) + chr(9617) * (20 - int(pct / 5))
        elapsed = time.time() - self.start_time
        eta_str = (
            self._fmt(np.mean(self.run_times) * (self.total - done))
            if self.run_times else "estimating..."
        )
        print(f"\n{'─' * 62}")
        print(f"  Progress  [{bar}] {pct:5.1f}%")
        print(
            f"  Runs      {done}/{self.total} done "
            f"({self.completed} trained, {self.skipped} skipped)"
        )
        print(f"  Elapsed   {self._fmt(elapsed)}")
        print(f"  ETA       {eta_str}")
        print(f"  Running   {label}")
        print(f"{'─' * 62}")

    def finish_run(self, auc=None, skipped: bool = False) -> None:
        t = time.time() - self._run_start
        if skipped:
            self.skipped += 1
            print("  Skipped")
        else:
            self.completed += 1
            self.run_times.append(t)
            msg = f"Test AUC: {auc:.4f}" if auc is not None else ""
            print(f"  Done in {self._fmt(t)} | {msg}")
        print_gpu_memory()

    def final_summary(self) -> None:
        t = time.time() - self.start_time
        print(f"\n{'=' * 62}")
        print("  EXPERIMENT COMPLETE")
        print(f"  Total time : {self._fmt(t)}")
        print(f"  Trained    : {self.completed} runs")
        print(f"  Skipped    : {self.skipped} runs")
        print(f"{'=' * 62}")


# ============================================================
# Dataset
# ============================================================

class EpitopeDataset(Dataset):
    def __init__(self, df_or_path, tokenizer, max_length: int = MAX_LEN):
        if isinstance(df_or_path, (str, Path)):
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path.reset_index(drop=True)

        missing = {"sequence", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.sequences  = df["sequence"].astype(str).tolist()
        self.labels     = df["label"].astype(int).tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Model — lora_rank parameterised, matches federated_train.py
# ============================================================

def build_model(lora_rank: int):
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t12_35M_UR50D", num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,    # matches federated_train.py
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    return get_peft_model(model, lora_config)


def print_model_summary(lora_rank: int):
    tmp       = build_model(lora_rank)
    total     = sum(p.numel() for p in tmp.parameters())
    trainable = sum(p.numel() for p in tmp.parameters() if p.requires_grad)
    lora_p    = sum(p.numel() for n, p in tmp.named_parameters()
                    if p.requires_grad and "lora" in n.lower())
    print(
        f"  LoRA rank         : r={lora_rank}, alpha={2 * lora_rank}"
    )
    print(
        f"  Trainable params  : {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )
    print(
        f"  LoRA params       : {lora_p:,} | "
        f"Head: {trainable - lora_p:,}"
    )
    print(f"  Transmitted/round : {trainable * 4 / 1e6:.3f} MB per client")
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Weight utilities — identical to federated_train.py
# ============================================================

def get_trainable_weights(model, to_cpu: bool = True):
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            w = param.detach().clone()
            weights[name] = w.cpu() if to_cpu else w
    return weights


def set_trainable_weights(model, weights):
    for name, param in model.named_parameters():
        if param.requires_grad and name in weights:
            param.data.copy_(weights[name].to(param.device))
    return model


def federated_averaging(client_weights_list, client_sizes):
    total       = float(sum(client_sizes))
    proportions = [s / total for s in client_sizes]
    averaged    = {}
    for key in client_weights_list[0].keys():
        averaged[key] = sum(
            proportions[i] * client_weights_list[i][key].float()
            for i in range(len(client_weights_list))
        )
    return averaged


# ============================================================
# Data preparation
# ============================================================

def read_required_csvs(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{p} must contain 'sequence' and 'label' columns.")
        dfs.append(df[["sequence", "label"]].copy())
    return dfs


def prepare_train_pool(data_dir: Path):
    train_files = [data_dir / f"client{i + 1}_train.csv" for i in range(5)]
    pool = pd.concat(read_required_csvs(train_files), ignore_index=True)
    pool = pool.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    pool["label"] = pool["label"].astype(int)
    return pool


def prepare_central_val(data_dir: Path):
    central_val_path = data_dir / "central_val.csv"
    if central_val_path.exists():
        return central_val_path
    val_files = [data_dir / f"client{i + 1}_val.csv" for i in range(5)]
    val = pd.concat(read_required_csvs(val_files), ignore_index=True)
    val = val.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    val["label"] = val["label"].astype(int)
    val.to_csv(central_val_path, index=False)
    print(f"  Created {central_val_path} from client val files.")
    return central_val_path


def compute_global_class_weights(train_pool, device):
    pos = int(train_pool["label"].sum())
    neg = int(len(train_pool) - pos)
    if pos == 0 or neg == 0:
        print("  WARNING: training pool has only one class; using unweighted CE loss.")
        return None
    return torch.tensor([1.0, neg / pos], dtype=torch.float, device=device)


# ============================================================
# Dirichlet partitioning
# ============================================================

def dirichlet_partition(train_df, num_clients: int, alpha: float, seed: int):
    rng     = np.random.default_rng(seed)
    labels  = train_df["label"].values
    classes = sorted(np.unique(labels).tolist())

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        counts      = np.floor(proportions * len(class_idx)).astype(int)
        remainder   = len(class_idx) - counts.sum()
        fractional  = proportions * len(class_idx) - counts
        for j in np.argsort(-fractional)[:remainder]:
            counts[j] += 1
        ptr = 0
        for client_id, cnt in enumerate(counts):
            client_indices[client_id].extend(class_idx[ptr:ptr + cnt].tolist())
            ptr += cnt

    client_dfs, partition_stats = {}, []
    for client_id in range(num_clients):
        idx = client_indices[client_id]
        if len(idx) == 0:
            continue
        cdf = train_df.iloc[idx].sample(
            frac=1.0, random_state=seed + client_id
        ).reset_index(drop=True)
        pos = int(cdf["label"].sum())
        client_dfs[client_id] = cdf
        partition_stats.append({
            "client_id":   client_id + 1,
            "num_samples": len(cdf),
            "num_pos":     pos,
            "num_neg":     len(cdf) - pos,
            "pos_rate":    round(pos / len(cdf), 4),
        })
    return client_dfs, partition_stats


def is_valid_partition(client_dfs, required_clients: int):
    if len(client_dfs) < required_clients:
        return False, f"only {len(client_dfs)}/{required_clients} clients have data"
    for cid, cdf in client_dfs.items():
        n        = len(cdf)
        pos_rate = float(cdf["label"].mean())
        if n < MIN_CLIENT_SAMPLES:
            return False, f"client {cid + 1} has {n} samples (min={MIN_CLIENT_SAMPLES})"
        if pos_rate < MIN_POSITIVE_RATE:
            return False, f"client {cid + 1} pos={pos_rate:.1%} below {MIN_POSITIVE_RATE:.0%}"
        if pos_rate > MAX_POSITIVE_RATE:
            return False, f"client {cid + 1} pos={pos_rate:.1%} above {MAX_POSITIVE_RATE:.0%}"
    return True, "OK"


def find_valid_dirichlet_partition(train_pool, num_clients: int,
                                   alpha: float, base_seed: int):
    last_reason = None
    for attempt in range(MAX_PARTITION_TRIES):
        partition_seed            = base_seed + 10000 * attempt
        client_dfs, part_stats   = dirichlet_partition(
            train_pool, num_clients, alpha, partition_seed
        )
        valid, reason = is_valid_partition(client_dfs, num_clients)
        if valid:
            return client_dfs, part_stats, partition_seed, "OK"
        last_reason = reason
    return (None, None, None,
            f"no valid partition after {MAX_PARTITION_TRIES} tries; "
            f"last: {last_reason}")


# ============================================================
# Local training — identical discipline to federated_train.py
# ============================================================

def train_local(model, loader, device, class_weights):
    model.train()
    criterion = (
        torch.nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None
        else torch.nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    for _ in range(LOCAL_EPOCHS):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                ).logits,
                batch["labels"].to(device, non_blocking=True),
            )
            loss.backward()
            optimizer.step()


# ============================================================
# Evaluation
# ============================================================

def evaluate_auc(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                ).logits,
                dim=1,
            )[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["labels"].numpy())
    all_probs  = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)
    if np.isnan(all_probs).any():
        print("  WARNING: NaN in predictions; returning AUC=0.5")
        return 0.5
    if len(np.unique(all_labels)) < 2:
        print("  WARNING: single-class labels; returning AUC=0.5")
        return 0.5
    return float(roc_auc_score(all_labels, all_probs))


# ============================================================
# Single federated run
#
# Val-based checkpoint selection — central test evaluated ONCE.
# ============================================================

def run_federated(train_loaders, val_loader, test_loader,
                  device, class_weights, client_sizes, lora_rank: int):
    global_model = build_model(lora_rank).to(device)

    best_val_auc   = -1.0
    best_weights   = None
    selected_round = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start    = time.time()
        client_weights = []

        for cid in range(len(train_loaders)):
            local_model = copy.deepcopy(global_model)
            train_local(local_model, train_loaders[cid], device, class_weights)
            client_weights.append(get_trainable_weights(local_model, to_cpu=True))
            del local_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        set_trainable_weights(
            global_model, federated_averaging(client_weights, client_sizes)
        )

        # Val-based checkpoint selection — NOT test
        val_auc = evaluate_auc(global_model, val_loader, device)
        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_weights   = get_trainable_weights(global_model, to_cpu=True)
            selected_round = round_num

        elapsed_r = time.time() - round_start
        eta_r     = elapsed_r * (NUM_ROUNDS - round_num)
        bar       = chr(9608) * int(20 * round_num / NUM_ROUNDS) + \
                    chr(9617) * (20 - int(20 * round_num / NUM_ROUNDS))
        eta_str   = format_seconds(eta_r)

        print(
            f"  Round [{bar}] {round_num:02d}/{NUM_ROUNDS} | "
            f"Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f} "
            f"@ round {selected_round:02d} | ETA: {eta_str}     ",
            end="\r",
        )
        if round_num == NUM_ROUNDS:
            print()

    if best_weights is None:
        raise RuntimeError("No best validation checkpoint was saved.")

    # Load best val checkpoint → evaluate central test ONCE
    set_trainable_weights(global_model, best_weights)
    test_auc = evaluate_auc(global_model, test_loader, device)

    del global_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_auc, best_val_auc, selected_round


# ============================================================
# Figure
# ============================================================

def generate_figure(results_df, figures_dir: Path):
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7.5,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "#dddddd",
        "axes.linewidth":     0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linewidth":     0.35,
        "grid.alpha":         0.4,
        "grid.color":         "#bbbbbb",
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })

    colors    = {3: "#0072B2", 5: "#009E73", 10: "#D55E00"}
    markers   = {3: "^",       5: "o",        10: "s"}
    linewidths = {3: 1.4,      5: 2.0,        10: 1.4}

    agg = (
        results_df.groupby(["alpha", "K"])["test_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    for K in CLIENT_COUNTS:
        subset = agg[agg["K"] == K].sort_values("alpha")
        if subset.empty:
            continue
        a, m, s = (subset["alpha"].values, subset["mean"].values,
                   subset["std"].values)
        ax.plot(a, m, color=colors[K], marker=markers[K], markersize=5.5,
                linewidth=linewidths[K], label=f"$K={K}$ clients", zorder=3)
        ax.fill_between(a, m - s, m + s, alpha=0.12, color=colors[K], zorder=2)




    ax.set_xlabel(r"Dirichlet concentration parameter $\alpha$")
    ax.set_ylabel("AUC-ROC on central test set")
    ax.set_xscale("log")
    ax.set_xticks(ALPHA_VALUES)
    ax.set_xticklabels([str(a) for a in ALPHA_VALUES])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.legend(loc="lower right", framealpha=0.95, fontsize=7.2, handlelength=2.2)
    ax.set_axisbelow(True)

    ymin = ax.get_ylim()[0]
    ax.annotate(
        "", xy=(3.5, ymin + 0.006), xytext=(0.12, ymin + 0.006),
        arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.8),
    )
    ax.text(0.55, ymin + 0.010, "increasing heterogeneity",
            fontsize=6.2, color="#888888", ha="center")

    fig.text(
        0.99, 0.01,
        f"mean \u00b1 std over {len(SEEDS)} seeds; selected by validation AUC",
        ha="right", va="bottom", fontsize=6.2, color="#aaaaaa",
    )

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        out = figures_dir / f"fig_dirichlet{ext}"
        plt.savefig(out)
        print(f"  Saved {out}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main(args):
    data_dir    = Path("data")
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dirichlet").mkdir(parents=True, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(12345)   # global seed before main loop; per-run seed set inside loop
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    total_runs = len(ALPHA_VALUES) * len(CLIENT_COUNTS) * len(SEEDS)

    print(f"{'=' * 62}")
    print("  FedEpitope — Dirichlet Non-IID Robustness (Figure 6)")
    print(f"{'=' * 62}")
    print(f"  Device              : {device}")
    print(f"  Alpha values        : {ALPHA_VALUES}")
    print(f"  K values            : {CLIENT_COUNTS}")
    print(f"  Seeds               : {SEEDS}")
    print(f"  Total runs          : {total_runs}")
    print(f"  Rounds              : {NUM_ROUNDS}")
    print(f"  Local epochs        : {LOCAL_EPOCHS}  (kept at 1 for feasibility)")
    print(f"  Min samples/client  : {MIN_CLIENT_SAMPLES}")
    print(f"  Valid pos rate      : [{MIN_POSITIVE_RATE:.0%}, {MAX_POSITIVE_RATE:.0%}]")
    print(f"  Max partition tries : {MAX_PARTITION_TRIES}")
    print(f"  Results dir         : {results_dir}")
    print(f"  Figures dir         : {figures_dir}")
    print(f"  Test-set rule       : val-based checkpoint selection; "
          "test evaluated ONCE per run")
    print(f"{'=' * 62}\n")

    print_model_summary(args.lora_rank)
    print()
    print_gpu_memory()
    print()

    # ── Data setup ──────────────────────────────────────────────────────────
    train_pool        = prepare_train_pool(data_dir)
    central_val_path  = prepare_central_val(data_dir)
    central_test_path = data_dir / "central_test.csv"

    if not central_test_path.exists():
        raise FileNotFoundError(f"Missing required test file: {central_test_path}")

    print(
        f"  Training pool   : {len(train_pool):,} sequences "
        f"({train_pool['label'].mean():.1%} positive)"
    )
    val_df  = pd.read_csv(central_val_path)
    test_df = pd.read_csv(central_test_path)
    print(f"  Central val     : {len(val_df):,} samples "
          f"({val_df['label'].mean():.1%} positive)")
    print(f"  Central test    : {len(test_df):,} samples "
          f"({test_df['label'].mean():.1%} positive)\n")

    val_loader = DataLoader(
        EpitopeDataset(val_df, tokenizer), batch_size=EVAL_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        EpitopeDataset(test_df, tokenizer), batch_size=EVAL_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker,
    )

    global_class_weights = compute_global_class_weights(train_pool, device)
    print(
        f"  Global CE weights : "
        + (f"neg={global_class_weights[0].item():.4f}, "
           f"pos={global_class_weights[1].item():.4f}"
           if global_class_weights is not None else "None (balanced)")
    )
    print()

    # ── Resume support ───────────────────────────────────────────────────────
    raw_path   = results_dir / "dirichlet_raw.csv"
    stats_path = results_dir / "dirichlet_partition_stats.csv"

    if raw_path.exists():
        raw_df       = pd.read_csv(raw_path)
        raw_results  = raw_df.to_dict("records")
        done         = set(zip(raw_df["alpha"], raw_df["K"], raw_df["seed"]))
        tracker      = ProgressTracker(total_runs)
        tracker.completed = int((raw_df["skipped"] == False).sum())
        tracker.skipped   = int((raw_df["skipped"] == True).sum())
        print(
            f"  Resuming — {tracker.completed} trained, "
            f"{tracker.skipped} skipped already recorded.\n"
        )
    else:
        raw_results = []
        done        = set()
        tracker     = ProgressTracker(total_runs)

    all_stats = (
        pd.read_csv(stats_path).to_dict("records")
        if stats_path.exists() else []
    )

    # ── Main experiment loop ─────────────────────────────────────────────────
    for K in CLIENT_COUNTS:
        for alpha in ALPHA_VALUES:
            for seed in SEEDS:
                if (alpha, K, seed) in done:
                    continue

                label = f"alpha={alpha}  K={K}  seed={seed}"
                tracker.start_run(label)
                set_seed(seed)

                client_dfs, part_stats, partition_seed, reason = \
                    find_valid_dirichlet_partition(
                        train_pool, num_clients=K, alpha=alpha, base_seed=seed
                    )

                if client_dfs is None:
                    print(f"  Invalid partition: {reason} — skipping")
                    raw_results.append({
                        "alpha": alpha, "K": K, "seed": seed,
                        "lora_rank":       args.lora_rank,
                        "partition_seed":  None,
                        "test_auc":        None,
                        "best_val_auc":    None,
                        "selected_round":  None,
                        "skipped":         True,
                        "reason":          reason,
                    })
                    pd.DataFrame(raw_results).to_csv(raw_path, index=False)
                    tracker.finish_run(skipped=True)
                    continue

                for stat in part_stats:
                    all_stats.append({
                        "alpha": alpha, "K": K, "seed": seed,
                        "partition_seed": partition_seed, **stat,
                    })
                pd.DataFrame(all_stats).to_csv(stats_path, index=False)

                print("  Clients:")
                train_loaders = []
                client_sizes  = []

                for cid in sorted(client_dfs.keys()):
                    cdf = client_dfs[cid]
                    pos = int(cdf["label"].sum())
                    neg = len(cdf) - pos
                    gen = torch.Generator()
                    gen.manual_seed(seed + 1000 * K + cid)
                    dl = DataLoader(
                        EpitopeDataset(cdf, tokenizer),
                        batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=(device.type == "cuda"),
                        worker_init_fn=seed_worker, generator=gen,
                    )
                    train_loaders.append(dl)
                    client_sizes.append(len(cdf))
                    print(
                        f"    C{cid + 1}: {len(cdf):>7,} seqs  "
                        f"pos={pos / len(cdf):.1%}  neg={neg / len(cdf):.1%}"
                    )

                test_auc, best_val_auc, selected_round = run_federated(
                    train_loaders=train_loaders,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    class_weights=global_class_weights,
                    client_sizes=client_sizes,
                    lora_rank=args.lora_rank,
                )

                raw_results.append({
                    "alpha":          alpha,
                    "K":              K,
                    "seed":           seed,
                    "lora_rank":      args.lora_rank,
                    "partition_seed": partition_seed,
                    "test_auc":       round(test_auc, 4),
                    "best_val_auc":   round(best_val_auc, 4),
                    "selected_round": int(selected_round),
                    "skipped":        False,
                    "reason":         "OK",
                })
                pd.DataFrame(raw_results).to_csv(raw_path, index=False)
                print(
                    f"  Selected round {selected_round} | "
                    f"Best val AUC: {best_val_auc:.4f} | "
                    f"Test AUC: {test_auc:.4f}"
                )
                tracker.finish_run(test_auc)

    # ── Aggregate results ────────────────────────────────────────────────────
    raw_df   = pd.read_csv(raw_path)
    valid_df = raw_df[raw_df["skipped"] == False].copy()

    if len(valid_df) == 0:
        print(
            "\nNo valid runs completed. Consider relaxing MIN_CLIENT_SAMPLES, "
            "MIN_POSITIVE_RATE, MAX_POSITIVE_RATE, or MAX_PARTITION_TRIES."
        )
        return

    agg = (
        valid_df.groupby(["alpha", "K"])["test_auc"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg["mean"]   = agg["mean"].round(4)
    agg["std"]    = agg["std"].fillna(0).round(4)
    agg["result"] = agg.apply(
        lambda r: f"{r['mean']:.4f} +- {r['std']:.4f}", axis=1
    )
    agg.to_csv(results_dir / "dirichlet_results.csv", index=False)

    tracker.final_summary()

    pivot = agg.pivot(index="alpha", columns="K", values="result")
    pivot.columns = [f"K={k}" for k in pivot.columns]
    print("\nCentral test AUC-ROC (mean +- std, val-based selection):")
    print(pivot.to_string())

    skipped_df = raw_df[raw_df["skipped"] == True]
    if len(skipped_df) > 0:
        print(f"\nSkipped runs ({len(skipped_df)}):")
        for _, row in skipped_df.iterrows():
            print(
                f"  alpha={row['alpha']}  K={row['K']}  "
                f"seed={row['seed']}  — {row.get('reason', '')}"
            )

    print("\nGenerating figure...")
    generate_figure(valid_df, figures_dir)

    print("\nDone. Saved:")
    print(f"   {results_dir / 'dirichlet_results.csv'}")
    print(f"   {results_dir / 'dirichlet_raw.csv'}")
    print(f"   {results_dir / 'dirichlet_partition_stats.csv'}")
    print(f"   {figures_dir / 'fig_dirichlet.pdf'}")
    print(f"   {figures_dir / 'fig_dirichlet.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank",    type=int,   default=2,
                        help="LoRA rank (default=2, matches main experiments)")
    parser.add_argument("--results_dir",  type=str,   default="results/dirichlet",
                        help="Directory for CSV outputs")
    parser.add_argument("--figures_dir",  type=str,   default="figures",
                        help="Directory for figure outputs")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as exc:
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        err_path = Path(args.results_dir) / "error_traceback.txt"
        with open(err_path, "w") as f:
            f.write(f"Dirichlet experiment failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {err_path}")
        raise
