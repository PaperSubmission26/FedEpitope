import os
import time
import math
import argparse
import warnings
import traceback
from datetime import timedelta

import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MAX_LEN     = 30
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
NETBCE_REPO = "/tmp/NetBCE"
TEST_FILE   = os.path.join(NETBCE_REPO, "data", "testing dataset.txt")

SEEDS = [42, 43, 44]


# ============================================================
# Utility
# ============================================================

def format_seconds(seconds):
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()     / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()      / 1024 ** 3
        max_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(
            f"GPU memory | allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | max allocated: {max_alloc:.2f} GB"
        )
    else:
        print("GPU memory | CUDA not available")


# ============================================================
# Step 1: Parse NetBCE FASTA
# ============================================================

def parse_netbce_fasta(filepath):
    records = []
    current_label = None
    current_seq   = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq and current_label is not None:
                    seq = "".join(current_seq).replace("-", "").upper()
                    records.append({"sequence": seq, "label": current_label})
                    current_seq = []
                parts = line.split("_")
                current_label = int(parts[-1])
            else:
                current_seq.append(line)

    if current_seq and current_label is not None:
        seq = "".join(current_seq).replace("-", "").upper()
        records.append({"sequence": seq, "label": current_label})

    df = pd.DataFrame(records)
    print(
        f"  Parsed {len(df)} sequences — "
        f"{df['label'].sum()} positive, "
        f"{(df['label'] == 0).sum()} negative"
    )
    return df


# ============================================================
# Step 2: Filter
# ============================================================

def filter_sequences(df):
    before = len(df)
    df = df[df["sequence"].apply(
        lambda s: isinstance(s, str) and len(s) > 0
        and set(s).issubset(STANDARD_AA)
    )].copy()
    df = df[df["sequence"].str.len().between(8, 25)]
    df = df.drop_duplicates(subset=["sequence"])
    print(f"  After filtering: {len(df)}/{before} retained")
    return df


# ============================================================
# Step 3: Remove IEDB overlap
# ============================================================

def remove_iedb_overlap(df):
    iedb_files = (
        ["data/central_test.csv"]
        + [f"data/client{i}_train.csv" for i in range(1, 6)]
        + [f"data/client{i}_val.csv"   for i in range(1, 6)]
        + [f"data/client{i}_test.csv"  for i in range(1, 6)]
    )
    iedb_seqs = set()
    for fpath in iedb_files:
        if os.path.exists(fpath):
            try:
                tmp = pd.read_csv(fpath)
                iedb_seqs.update(tmp["sequence"].str.upper().tolist())
            except Exception:
                pass
    before = len(df)
    df = df[~df["sequence"].isin(iedb_seqs)].copy()
    print(
        f"  Removed {before - len(df)} IEDB overlaps "
        f"— {len(df)} independent sequences remain"
    )
    return df


# ============================================================
# Dataset
# ============================================================

class EpitopeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LEN):
        self.sequences  = df["sequence"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
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
# Model
# ============================================================

def build_model(lora_rank: int):
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t12_35M_UR50D", num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    return get_peft_model(model, lora_config)


def load_model(weights_path, lora_rank, device):
    model = build_model(lora_rank).to(device)
    weights = torch.load(weights_path, map_location=device)
    loaded, missing = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n in weights:
                p.data.copy_(weights[n].to(device))
                loaded += 1
            else:
                missing += 1
    if missing > 0:
        print(f"    WARNING: {missing} trainable params not found in checkpoint "
              f"({weights_path})")
    model.eval()
    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits,
                dim=1,
            )[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["labels"].numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    return {
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "auc_pr":  average_precision_score(all_labels, all_probs),
        "f1":      f1_score(all_labels, preds, zero_division=0),
        "mcc":     matthews_corrcoef(all_labels, preds),
        "probs":   all_probs,
        "labels":  all_labels,
    }


# ============================================================
# Main
# ============================================================

def main(args):
    run_start = time.time()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    print("=" * 80)
    print("FedEpitope — NetBCE Independent Validation (3-seed)")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"LoRA rank    : {args.lora_rank}")
    print(f"Seeds        : {SEEDS}")
    print(f"Results dir  : {args.results_dir}")
    print("=" * 80)
    print()
    print_gpu_memory()
    print()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # ── Prepare NetBCE independent test set (once) ─────────────────────────
    netbce_csv = "data/netbce_independent_test.csv"
    if os.path.exists(netbce_csv) and not args.rebuild_netbce:
        print(f"Loading cached NetBCE test set: {netbce_csv}")
        netbce_df = pd.read_csv(netbce_csv)
    else:
        print(f"Loading NetBCE independent test set from:\n  {TEST_FILE}\n")
        netbce_df = parse_netbce_fasta(TEST_FILE)

        print("\nFiltering sequences (length 8-25, standard AA)...")
        netbce_df = filter_sequences(netbce_df)

        print("\nRemoving IEDB overlap (train + val + test, all clients + central)...")
        netbce_df = remove_iedb_overlap(netbce_df)

        netbce_df.to_csv(netbce_csv, index=False)
        print(f"Saved: {netbce_csv}")

    n_pos = int(netbce_df["label"].sum())
    n_neg = int((netbce_df["label"] == 0).sum())
    print(
        f"\nFinal NetBCE test set: {len(netbce_df)} sequences "
        f"| {n_pos} positive | {n_neg} negative "
        f"| {n_pos / len(netbce_df):.1%} positive rate\n"
    )

    iedb_test_df = pd.read_csv("data/central_test.csv")

    netbce_loader = DataLoader(
        EpitopeDataset(netbce_df, tokenizer), batch_size=128,
        shuffle=False, num_workers=0,
    )
    iedb_loader = DataLoader(
        EpitopeDataset(iedb_test_df, tokenizer), batch_size=128,
        shuffle=False, num_workers=0,
    )

    # ── Models to evaluate, per seed ───────────────────────────────────────
    # FedEpitope: results/r2_seed{S}/best_global_weights.pt
    # Centralised LoRA: results/baseline_seed{S}/centralised_weights.pt
    model_specs = {
        "FedEpitope":  "results/r2_seed{seed}/best_global_weights.pt",
        "Centralised": "results/baseline_seed{seed}/centralised_weights.pt",
    }

    all_rows  = []     # per-seed, per-model rows
    roc_cache = {}     # (model_name) -> list of (seed, netbce_metrics, iedb_metrics)
    for name in model_specs:
        roc_cache[name] = []

    total_jobs = len(model_specs) * len(SEEDS)
    job_idx    = 0

    for model_name, path_template in model_specs.items():
        for seed in SEEDS:
            job_idx += 1
            weights_path = path_template.format(seed=seed)

            print(f"\n{'=' * 80}")
            print(f"[{job_idx}/{total_jobs}] {model_name} | seed {seed}")
            print(f"  Checkpoint: {weights_path}")
            print(f"{'=' * 80}")

            if not os.path.exists(weights_path):
                print(f"  SKIPPING — checkpoint not found")
                continue

            job_start = time.time()
            model = load_model(weights_path, args.lora_rank, device)

            nb = evaluate(model, netbce_loader, device)
            ie = evaluate(model, iedb_loader,   device)

            job_time = time.time() - job_start

            print(
                f"  NetBCE independent — AUC-ROC: {nb['auc_roc']:.4f} | "
                f"AUC-PR: {nb['auc_pr']:.4f} | F1: {nb['f1']:.4f} | "
                f"MCC: {nb['mcc']:.4f}"
            )
            print(
                f"  IEDB held-out      — AUC-ROC: {ie['auc_roc']:.4f} | "
                f"AUC-PR: {ie['auc_pr']:.4f} | F1: {ie['f1']:.4f} | "
                f"MCC: {ie['mcc']:.4f}"
            )
            print(f"  Gap (IEDB - NetBCE): {ie['auc_roc'] - nb['auc_roc']:+.4f}")
            print(f"  Job time           : {format_seconds(job_time)}")

            elapsed = time.time() - run_start
            avg_job = elapsed / job_idx
            eta     = avg_job * (total_jobs - job_idx)
            print(f"  Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}")
            print_gpu_memory()

            row = {
                "model":         model_name,
                "seed":          seed,
                "lora_rank":     args.lora_rank,
                "netbce_auc_roc": nb["auc_roc"],
                "netbce_auc_pr":  nb["auc_pr"],
                "netbce_f1":      nb["f1"],
                "netbce_mcc":     nb["mcc"],
                "iedb_auc_roc":   ie["auc_roc"],
                "iedb_auc_pr":    ie["auc_pr"],
                "iedb_f1":        ie["f1"],
                "iedb_mcc":       ie["mcc"],
                "gap_auc":        ie["auc_roc"] - nb["auc_roc"],
            }
            all_rows.append(row)
            roc_cache[model_name].append((seed, nb, ie))

            # Live save after every job
            live_path = os.path.join(args.results_dir, "netbce_validation_live.csv")
            pd.DataFrame(all_rows).to_csv(live_path, index=False)
            print(f"  Live results: {live_path}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_rows:
        print("\nNo models evaluated — no checkpoints found. Exiting.")
        return

    detail_df = pd.DataFrame(all_rows)
    detail_path = os.path.join(args.results_dir, "netbce_validation_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    # ── Aggregate mean ± SD across seeds, per model ────────────────────────
    metric_cols = [
        "netbce_auc_roc", "netbce_auc_pr", "netbce_f1", "netbce_mcc",
        "iedb_auc_roc", "iedb_auc_pr", "iedb_f1", "iedb_mcc", "gap_auc",
    ]
    summary_rows = []
    for model_name in model_specs:
        sub = detail_df[detail_df["model"] == model_name]
        if sub.empty:
            continue
        row = {"model": model_name, "n_seeds": len(sub), "lora_rank": args.lora_rank}
        for col in metric_cols:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"]  = sub[col].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.results_dir, "netbce_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ── ROC figure — uses seed with median NetBCE AUC for each model ───────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["coral", "steelblue"]

    for ax, (dkey, dlabel, dsize) in zip(axes, [
        ("netbce", "NetBCE Independent Benchmark\n(Xu & Zhao, GPB 2022)",
         f"n={len(netbce_df):,}"),
        ("iedb", "IEDB Held-Out Test Set\n(same distribution as training)",
         f"n={len(iedb_test_df):,}"),
    ]):
        for (model_name, runs), color in zip(roc_cache.items(), colors):
            if not runs:
                continue
            # Pick the seed whose NetBCE AUC-ROC is closest to the mean
            netbce_aucs = [nb["auc_roc"] for _, nb, _ in runs]
            mean_auc = np.mean(netbce_aucs)
            best_seed_idx = int(np.argmin([abs(a - mean_auc) for a in netbce_aucs]))
            seed_used, nb_run, ie_run = runs[best_seed_idx]
            m = nb_run if dkey == "netbce" else ie_run
            fpr, tpr, _ = roc_curve(m["labels"], m["probs"])
            ax.plot(
                fpr, tpr, color=color, linewidth=2,
                label=f"{model_name} seed={seed_used} (AUC={m['auc_roc']:.4f})"
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (0.50)")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC — {dlabel}\n{dsize}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "FedEpitope: IEDB vs NetBCE Independent Validation "
        "(representative seed shown)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "figure_netbce_roc.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Bar chart — mean ± SD across seeds ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    names  = list(summary_df["model"])
    x      = np.arange(len(names))
    width  = 0.30

    netbce_means = summary_df["netbce_auc_roc_mean"].values
    netbce_stds  = summary_df["netbce_auc_roc_std"].values
    iedb_means   = summary_df["iedb_auc_roc_mean"].values
    iedb_stds    = summary_df["iedb_auc_roc_std"].values

    bars1 = ax.bar(x - width / 2, netbce_means, width, yerr=netbce_stds,
                   capsize=4, label="NetBCE independent", color="coral", alpha=0.85)
    bars2 = ax.bar(x + width / 2, iedb_means, width, yerr=iedb_stds,
                   capsize=4, label="IEDB held-out", color="steelblue", alpha=0.85)

    
    ax.axhline(y=0.50, color="grey", linestyle=":", linewidth=1.0,
               label="Random (0.50)")

    for bar, mean in zip(list(bars1) + list(bars2),
                          list(netbce_means) + list(iedb_means)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=13)
    ax.set_title(
        "FedEpitope: IEDB vs NetBCE Independent Validation\n"
        f"(mean \u00b1 SD over {len(SEEDS)} seeds)",
        fontsize=13,
    )
    ax.set_ylim(0.40, 0.90)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "figure_netbce_bar.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Final summary ───────────────────────────────────────────────────────
    total_time = time.time() - run_start
    print("\n" + "=" * 80)
    print("NETBCE INDEPENDENT VALIDATION — FINAL SUMMARY (mean ± SD)")
    print(f"Benchmark: NetBCE testing dataset (Xu & Zhao, GPB 2022)")
    print("=" * 80)
    print(
        f"\n{'Model':<14} {'NetBCE AUC':>14} {'NetBCE MCC':>14} "
        f"{'IEDB AUC':>14} {'IEDB MCC':>14} {'Gap':>10}"
    )
    print("-" * 72)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<14} "
            f"{row['netbce_auc_roc_mean']:.4f}\u00b1{row['netbce_auc_roc_std']:.4f}  "
            f"{row['netbce_mcc_mean']:.4f}\u00b1{row['netbce_mcc_std']:.4f}  "
            f"{row['iedb_auc_roc_mean']:.4f}\u00b1{row['iedb_auc_roc_std']:.4f}  "
            f"{row['iedb_mcc_mean']:.4f}\u00b1{row['iedb_mcc_std']:.4f}  "
            f"{row['gap_auc_mean']:>+.4f}"
        )

    print(f"\nTotal run time : {format_seconds(total_time)}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: figures/figure_netbce_roc.png")
    print(f"Saved: figures/figure_netbce_bar.png")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--results_dir", type=str, default="results/netbce")
    parser.add_argument("--rebuild_netbce", action="store_true",
                         help="Re-parse and re-filter NetBCE FASTA even if "
                              "data/netbce_independent_test.csv exists.")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)
        err_path = os.path.join(args.results_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            f.write(f"NetBCE validation failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {err_path}")
        raise
