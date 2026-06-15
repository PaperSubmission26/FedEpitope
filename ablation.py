import os
import copy
import random
import argparse
import warnings
import time
import math
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

NUM_ROUNDS   = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS  = 5
BATCH_SIZE   = 64
LR           = 2e-4
MAX_LEN      = 30

CLIENT_TRAIN_FILES = [f"data/client{i + 1}_train.csv" for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES   = [f"data/client{i + 1}_val.csv"   for i in range(NUM_CLIENTS)]

CLIENT_NAMES = [
    "Coronavirus",
    "Parasite",
    "Human/Self",
    "Flavivirus",
    "Bacteria",
]


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


def check_required_files():
    required = CLIENT_TRAIN_FILES + CLIENT_VAL_FILES + ["data/central_test.csv"]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


# ============================================================
# Dataset — identical to federated_train.py
# ============================================================

class EpitopeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LEN):
        df = pd.read_csv(csv_path)
        if "sequence" not in df.columns:
            raise ValueError(f"{csv_path} missing column: sequence")
        if "label" not in df.columns:
            raise ValueError(f"{csv_path} missing column: label")
        self.sequences  = df["sequence"].astype(str).tolist()
        self.labels     = df["label"].astype(int).tolist()
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


def print_trainable_summary(model, label: str):
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora       = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    )
    classifier = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad
        and any(k in n.lower() for k in ("classifier", "score", "modules_to_save"))
    )
    print(
        f"  [{label}] total={total:,} | trainable={trainable:,} "
        f"| lora={lora:,} | classifier={classifier:,} "
        f"| {100 * trainable / total:.4f}%"
    )
    return trainable, lora, classifier


# ============================================================
# Weight utilities — identical to federated_train.py
# ============================================================

def get_trainable_weights(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n].to(p.device))
    return model


def federated_averaging(client_weights_list, client_sizes):
    total       = sum(client_sizes)
    proportions = [s / total for s in client_sizes]
    averaged    = {}
    for key in client_weights_list[0].keys():
        averaged[key] = sum(
            proportions[i] * client_weights_list[i][key].float()
            for i in range(len(client_weights_list))
        )
    return averaged


# ============================================================
# Evaluation — identical to federated_train.py
# ============================================================

def evaluate(model, loader, device):
    model.eval()
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
    preds      = (all_probs >= 0.5).astype(int)
    if len(np.unique(all_labels)) < 2:
        return dict(auc_roc=np.nan, auc_pr=np.nan, f1=np.nan, mcc=np.nan)
    return {
        "auc_roc": roc_auc_score(all_labels, all_probs),
        "auc_pr":  average_precision_score(all_labels, all_probs),
        "f1":      f1_score(all_labels, preds, zero_division=0),
        "mcc":     matthews_corrcoef(all_labels, preds),
    }


def weighted_average_metrics(metrics_list, sizes):
    averaged = {}
    for key in metrics_list[0].keys():
        num = sum(
            s * m[key] for m, s in zip(metrics_list, sizes)
            if m[key] is not None and not np.isnan(m[key])
        )
        den = sum(
            s for m, s in zip(metrics_list, sizes)
            if m[key] is not None and not np.isnan(m[key])
        )
        averaged[key] = num / den if den > 0 else np.nan
    return averaged


def train_local(model, loader, optimizer, device, class_weights,
                 round_num, client_id, label):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    epoch_losses = []
    for epoch in range(LOCAL_EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits,
                labels,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Invalid loss: {loss.item()} | {label} "
                    f"Round {round_num} Client {client_id + 1} Epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss   = total_loss / max(len(loader), 1)
        epoch_time = time.time() - epoch_start
        epoch_losses.append(avg_loss)
        print(
            f"  [{label} | Round {round_num:02d} | "
            f"Client {client_id + 1} {CLIENT_NAMES[client_id]:<12} | "
            f"Epoch {epoch + 1}/{LOCAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f} | Epoch time: {format_seconds(epoch_time)}"
        )
    return epoch_losses


# ============================================================
# One full federated run for a single LoRA rank
#
# Val-based checkpoint selection — test set never accessed inside.
# Caller evaluates central test ONCE after this returns.
# ============================================================

def run_federated_for_rank(
    lora_rank,
    train_loaders,
    val_loaders,
    client_train_sizes,
    client_val_sizes,
    class_weights_list,
    device,
    args,
    run_start,
    rank_idx,
    total_ranks,
    results_dir,
    live_rows,
):
    label = f"r={lora_rank}"
    print(f"\n{'=' * 80}")
    print(f"RANK ABLATION {rank_idx}/{total_ranks}: {label}")
    print(
        "  Checkpoint selection : best weighted val AUC across clients\n"
        "  Central test set     : NOT accessed in this function"
    )
    print(f"{'=' * 80}\n")

    global_model = build_model(lora_rank).to(device)
    trainable, lora_p, classifier_p = print_trainable_summary(global_model, label)
    print()

    val_log      = []
    round_times  = []
    best_val_auc = -np.inf
    best_round   = 0
    best_weights = None

    rank_start = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()

        print()
        print("=" * 80)
        print(f"[{label}] ROUND {round_num}/{NUM_ROUNDS}")
        print("=" * 80)

        client_weights_list = []

        for client_id in range(NUM_CLIENTS):
            print()
            print("-" * 80)
            print(f"[{label}] Round {round_num:02d} | Client {client_id + 1} {CLIENT_NAMES[client_id]}")
            print("-" * 80)

            local_model = copy.deepcopy(global_model).to(device)
            optimizer   = torch.optim.AdamW(
                [p for p in local_model.parameters() if p.requires_grad], lr=args.lr
            )

            train_local(
                local_model, train_loaders[client_id], optimizer, device,
                class_weights_list[client_id], round_num, client_id, label,
            )

            client_weights_list.append(get_trainable_weights(local_model))
            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        averaged     = federated_averaging(client_weights_list, client_train_sizes)
        global_model = set_trainable_weights(global_model, averaged)

        # Val evaluation for checkpoint selection — NO test access
        client_val_metrics = []
        for client_id in range(NUM_CLIENTS):
            m = evaluate(global_model, val_loaders[client_id], device)
            client_val_metrics.append(m)
            print(
                f"  [Val | {label} | Round {round_num:02d} | "
                f"Client {client_id + 1} {CLIENT_NAMES[client_id]:<12}] "
                f"AUC-ROC: {m['auc_roc']:.4f}"
            )

        val_metrics = weighted_average_metrics(client_val_metrics, client_val_sizes)

        best_tag = ""
        if not np.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_round   = round_num
            best_weights = get_trainable_weights(global_model)
            torch.save(
                best_weights,
                os.path.join(results_dir, f"best_weights_r{lora_rank}.pt"),
            )
            best_tag = " <-- best val checkpoint"

        round_dur = time.time() - round_start
        round_times.append(round_dur)
        elapsed   = time.time() - run_start
        eta_rank  = (sum(round_times) / len(round_times)) * (NUM_ROUNDS - round_num)

        row = {
            "seed":        args.seed,
            "lora_rank":   lora_rank,
            "round":       round_num,
            "val_auc_roc": val_metrics["auc_roc"],
            "val_auc_pr":  val_metrics["auc_pr"],
            "val_f1":      val_metrics["f1"],
            "val_mcc":     val_metrics["mcc"],
            "round_time_s": round_dur,
        }
        val_log.append(row)
        live_rows.append(row)

        print()
        print("=" * 80)
        print(
            f"[{label}] Round {round_num:02d} | "
            f"Weighted val AUC-ROC: {val_metrics['auc_roc']:.4f} | "
            f"AUC-PR: {val_metrics['auc_pr']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"MCC: {val_metrics['mcc']:.4f}"
            f"{best_tag}"
        )
        print("=" * 80)
        print()
        print("Progress monitor")
        print(f"  Rank setting          : {label} ({rank_idx}/{total_ranks})")
        print(f"  Finished round        : {round_num}/{NUM_ROUNDS}")
        print(f"  Round time            : {format_seconds(round_dur)}")
        print(f"  Elapsed (this rank)   : {format_seconds(time.time() - rank_start)}")
        print(f"  Elapsed (total)       : {format_seconds(elapsed)}")
        print(f"  ETA (this rank)       : {format_seconds(eta_rank)}")
        print(f"  Best val AUC so far   : {best_val_auc:.4f} (round {best_round})")
        print_gpu_memory()

        live_path = os.path.join(results_dir, "rank_val_log_live.csv")
        pd.DataFrame(live_rows).to_csv(live_path, index=False)
        print(f"  Live val log          : {live_path}")

    val_log_path = os.path.join(results_dir, f"rank_val_log_r{lora_rank}.csv")
    pd.DataFrame(val_log).to_csv(val_log_path, index=False)

    return best_weights, best_val_auc, best_round, trainable, lora_p, classifier_p


# ============================================================
# Main
# ============================================================

def main(args):
    run_start = time.time()

    set_seed(args.seed)
    check_required_files()
    os.makedirs(args.results_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    ranks = [int(r) for r in args.ranks.split(",")]

    print("=" * 80)
    print("FedEpitope — LoRA Rank Ablation")
    print("=" * 80)
    print(f"Device         : {device}")
    print(f"Seed           : {args.seed}")
    print(f"Ranks to run   : {ranks}")
    print(f"Results dir    : {args.results_dir}")
    print(
        "Test-set rule  : central test evaluated ONCE per rank, "
        "after val-based checkpoint selection."
    )
    print("=" * 80)
    print()
    print_gpu_memory()
    print()

    # ── Build dataloaders once — reused across all ranks ───────────────────
    train_loaders      = []
    val_loaders        = []
    class_weights_list = []
    client_train_sizes = []
    client_val_sizes   = []

    for client_id in range(NUM_CLIENTS):
        gen = torch.Generator()
        gen.manual_seed(args.seed + client_id)

        train_ds = EpitopeDataset(CLIENT_TRAIN_FILES[client_id], tokenizer)
        val_ds   = EpitopeDataset(CLIENT_VAL_FILES[client_id],   tokenizer)

        train_df = pd.read_csv(CLIENT_TRAIN_FILES[client_id])
        pos      = int(train_df["label"].sum())
        neg      = int(len(train_df) - pos)

        train_loaders.append(DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=gen, pin_memory=torch.cuda.is_available(),
        ))
        val_loaders.append(DataLoader(
            val_ds, batch_size=128, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        ))
        class_weights_list.append(
            torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)
        )
        client_train_sizes.append(len(train_ds))
        client_val_sizes.append(len(val_ds))

        print(
            f"Client {client_id + 1} ({CLIENT_NAMES[client_id]:<12}) | "
            f"Train: {len(train_ds):>7,} | Val: {len(val_ds):>6,} | "
            f"Pos rate: {pos / max(len(train_df), 1):.2%}"
        )

    print()

    test_ds     = EpitopeDataset("data/central_test.csv", tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Central test set: {len(test_ds):,} samples\n")

    # ── Run each rank ────────────────────────────────────────────────────
    total_ranks   = len(ranks)
    summary_rows  = []
    live_rows     = []

    for rank_idx, lora_rank in enumerate(ranks, start=1):
        best_weights, best_val_auc, best_round, trainable, lora_p, classifier_p = \
            run_federated_for_rank(
                lora_rank=lora_rank,
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                client_train_sizes=client_train_sizes,
                client_val_sizes=client_val_sizes,
                class_weights_list=class_weights_list,
                device=device,
                args=args,
                run_start=run_start,
                rank_idx=rank_idx,
                total_ranks=total_ranks,
                results_dir=args.results_dir,
                live_rows=live_rows,
            )

        # ── Evaluate central test ONCE per rank ──────────────────────────
        print(f"\n[r={lora_rank}] Loading best val checkpoint (round {best_round})...")
        eval_model = build_model(lora_rank).to(device)
        eval_model = set_trainable_weights(eval_model, best_weights)
        print(f"[r={lora_rank}] Evaluating central test set exactly once...")
        test_metrics = evaluate(eval_model, test_loader, device)
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Communication cost: total trainable params x 4 bytes (fp32)
        comm_mb = trainable * 4 / 1e6
        lora_only_mb = lora_p * 4 / 1e6

        row = {
            "seed":            args.seed,
            "lora_rank":       lora_rank,
            "lora_alpha":      2 * lora_rank,
            "trainable_params": trainable,
            "lora_params":     lora_p,
            "classifier_params": classifier_p,
            "comm_mb_total":   comm_mb,
            "comm_mb_lora_only": lora_only_mb,
            "selected_round":  best_round,
            "best_val_auc":    best_val_auc,
            "test_auc_roc":    test_metrics["auc_roc"],
            "test_auc_pr":     test_metrics["auc_pr"],
            "test_f1":         test_metrics["f1"],
            "test_mcc":        test_metrics["mcc"],
        }
        summary_rows.append(row)

        print(
            f"\n[r={lora_rank}] Test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
            f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f} | "
            f"Total comm: {comm_mb:.3f} MB/round | LoRA-only: {lora_only_mb:.3f} MB/round"
        )

        # Save live summary after every rank
        summary_df_live = pd.DataFrame(summary_rows)
        summary_df_live.to_csv(
            os.path.join(args.results_dir, "ablation_lora_rank_live.csv"), index=False
        )

    # ── Final save and summary ─────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.results_dir, "ablation_lora_rank.csv")
    summary_df.to_csv(summary_path, index=False)

    total_time = time.time() - run_start

    print("\n" + "=" * 80)
    print("LORA RANK ABLATION — FINAL SUMMARY")
    print("=" * 80)
    print(
        f"\n{'r':>4} {'Trainable':>11} {'Comm MB/rnd':>12} "
        f"{'Test AUC-ROC':>13} {'Best Round':>11}"
    )
    print("-" * 55)
    for _, row in summary_df.iterrows():
        print(
            f"{int(row['lora_rank']):>4} {int(row['trainable_params']):>11,} "
            f"{row['comm_mb_total']:>12.3f} {row['test_auc_roc']:>13.4f} "
            f"{int(row['selected_round']):>11}"
        )

    print(f"\nTotal run time   : {format_seconds(total_time)}")
    print(f"Results saved to : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ranks",       type=str, default="4,8,16",
                        help="Comma-separated LoRA ranks to run (r=2 already done separately).")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="results/rank_ablation_seed42")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)
        err_path = os.path.join(args.results_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            f.write(f"Rank ablation failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {err_path}")
        raise