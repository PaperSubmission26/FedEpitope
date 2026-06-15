"""
iid_federated_train.py


This is a SEPARATE copy of federated_train.py with IID partitioning
baked in — federated_train.py is untouched, preserving reproducibility
of the main Non-IID results.

Only difference from federated_train.py: training data is pooled across
all 5 clients and randomly re-split (ignoring organism labels) before
the standard FedAvg loop. Client sizes are preserved so FedAvg weighting
is unchanged. Everything else (seeding, LoRA rank, val-based checkpoint
selection, central-test-once rule, monitoring) is identical.
"""

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

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS = 5
BATCH_SIZE = 64
LR = 2e-4
MAX_LEN = 30

CLIENT_TRAIN_FILES = [f"data/client{i + 1}_train.csv" for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES = [f"data/client{i + 1}_val.csv" for i in range(NUM_CLIENTS)]

CLIENT_NAMES = [
    "Coronavirus",
    "Parasite",
    "Human/Self",
    "Flavivirus",
    "Bacteria",
]


# ============================================================
# Utility functions — identical to federated_train.py
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_seconds(seconds):
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "unknown"

    return str(timedelta(seconds=int(seconds)))


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"GPU memory | allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"max allocated: {max_allocated:.2f} GB"
        )
    else:
        print("GPU memory | CUDA not available")


def check_required_files():
    required_files = CLIENT_TRAIN_FILES + CLIENT_VAL_FILES + ["data/central_test.csv"]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n"
            + "\n".join(missing)
        )


# ============================================================
# IID partitioning
#
# Pools all 5 clients' train and val CSVs, shuffles with `seed`,
# and re-splits into 5 chunks matching the ORIGINAL per-client sizes
# so FedAvg weighting (proportional to client size) is unchanged
# relative to the Non-IID run. Writes the split CSVs under
# results_dir/iid_data/ for inspection/reproducibility.
# ============================================================

def make_iid_splits(seed: int, results_dir: str):
    iid_dir = os.path.join(results_dir, "iid_data")
    os.makedirs(iid_dir, exist_ok=True)

    def pool_and_split(file_list):
        dfs = [pd.read_csv(f) for f in file_list]
        sizes = [len(d) for d in dfs]
        pooled = pd.concat(dfs, ignore_index=True)
        shuffled = pooled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        splits = []
        start = 0
        for size in sizes:
            splits.append(shuffled.iloc[start:start + size].reset_index(drop=True))
            start += size
        return splits

    train_splits = pool_and_split(CLIENT_TRAIN_FILES)
    val_splits = pool_and_split(CLIENT_VAL_FILES)

    iid_train_files = []
    iid_val_files = []
    for i in range(NUM_CLIENTS):
        tr_path = os.path.join(iid_dir, f"iid_client{i + 1}_train.csv")
        va_path = os.path.join(iid_dir, f"iid_client{i + 1}_val.csv")
        train_splits[i].to_csv(tr_path, index=False)
        val_splits[i].to_csv(va_path, index=False)
        iid_train_files.append(tr_path)
        iid_val_files.append(va_path)

    print("IID partition created (organism labels ignored):")
    for i in range(NUM_CLIENTS):
        pos_rate = train_splits[i]["label"].mean()
        print(
            f"  IID client {i + 1}: train={len(train_splits[i]):>7,} | "
            f"val={len(val_splits[i]):>6,} | pos_rate={pos_rate:.2%}"
        )
    print()

    return iid_train_files, iid_val_files


# ============================================================
# Dataset
# ============================================================

class EpitopeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LEN):
        df = pd.read_csv(csv_path)

        if "sequence" not in df.columns:
            raise ValueError(f"{csv_path} is missing required column: sequence")

        if "label" not in df.columns:
            raise ValueError(f"{csv_path} is missing required column: label")

        self.sequences = df["sequence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.csv_path = csv_path

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Model
# ============================================================

def build_model(lora_rank: int = 2):
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t12_35M_UR50D",
        num_labels=2,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, lora_config)
    return model


# ============================================================
# Trainable weight utilities
# ============================================================

def get_trainable_weights(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def set_trainable_weights(model, weights):
    for name, param in model.named_parameters():
        if param.requires_grad and name in weights:
            param.data.copy_(weights[name].to(param.device))

    return model


def print_trainable_summary(model):
    total_params = 0
    trainable_params = 0
    lora_params = 0
    classifier_like_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

            name_lower = name.lower()

            if "lora" in name_lower:
                lora_params += param.numel()

            if (
                "classifier" in name_lower
                or "score" in name_lower
                or "modules_to_save" in name_lower
            ):
                classifier_like_params += param.numel()

    print(f"Total parameters             : {total_params:,}")
    print(f"Trainable parameters         : {trainable_params:,}")
    print(f"LoRA trainable parameters    : {lora_params:,}")
    print(f"Classifier-like parameters   : {classifier_like_params:,}")
    print(f"Trainable percentage         : {100 * trainable_params / total_params:.4f}%")


# ============================================================
# FedAvg
# ============================================================

def federated_averaging(client_weights_list, client_sizes):
    total_size = sum(client_sizes)
    client_proportions = [size / total_size for size in client_sizes]

    averaged_weights = {}

    for key in client_weights_list[0].keys():
        averaged_weights[key] = sum(
            client_proportions[i] * client_weights_list[i][key].float()
            for i in range(len(client_weights_list))
        )

    return averaged_weights


# ============================================================
# Training
# ============================================================

def train_local(
    model,
    loader,
    optimizer,
    device,
    class_weights,
    round_num,
    client_id,
):
    model.train()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    epoch_losses = []

    for epoch in range(LOCAL_EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Invalid loss detected: {loss.item()} | "
                    f"Round {round_num}, Client {client_id + 1}, "
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}"
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        epoch_losses.append(avg_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"[Round {round_num:02d} | Client {client_id + 1} "
            f"{CLIENT_NAMES[client_id]:<12} | Epoch {epoch + 1}/{LOCAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Epoch time: {format_seconds(epoch_time)}"
        )

    return epoch_losses


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(batch["labels"].detach().cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)

    if len(np.unique(all_labels)) < 2:
        auc_roc = np.nan
        auc_pr = np.nan
    else:
        auc_roc = roc_auc_score(all_labels, all_probs)
        auc_pr = average_precision_score(all_labels, all_probs)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1_score(all_labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(all_labels, preds),
    }


def weighted_average_metrics(metrics_list, sizes):
    averaged = {}

    for metric_name in metrics_list[0].keys():
        numerator = 0.0
        denominator = 0.0

        for metrics, size in zip(metrics_list, sizes):
            value = metrics[metric_name]

            if value is None or np.isnan(value):
                continue

            numerator += size * value
            denominator += size

        averaged[metric_name] = numerator / denominator if denominator > 0 else np.nan

    return averaged


def evaluate_global_on_client_validation_sets(model, val_loaders, val_sizes, device):
    client_metrics = []

    for client_id, val_loader in enumerate(val_loaders):
        metrics = evaluate(model, val_loader, device)
        client_metrics.append(metrics)

        print(
            f"[Global validation | Client {client_id + 1} "
            f"{CLIENT_NAMES[client_id]:<12}] "
            f"AUC-ROC: {metrics['auc_roc']:.4f} | "
            f"AUC-PR: {metrics['auc_pr']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"MCC: {metrics['mcc']:.4f}"
        )

    return weighted_average_metrics(client_metrics, val_sizes)


# ============================================================
# Main
# ============================================================

def main(args):
    run_start_time = time.time()
    round_times = []

    set_seed(args.seed)
    check_required_files()

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    print("=" * 80)
    print("FedEpitope IID Convergence Ablation (Figure 2)")
    print("=" * 80)
    print(f"Device                    : {device}")
    print(f"Seed                      : {args.seed}")
    print(f"Partition mode            : IID (random, organism-agnostic)")
    print(f"Rounds                    : {args.num_rounds}")
    print(f"Local epochs              : {LOCAL_EPOCHS}")
    print(f"Batch size                : {args.batch_size}")
    print(f"Learning rate             : {args.lr}")
    print(f"LoRA rank                 : {args.lora_rank}")
    print(f"LoRA alpha                : {2 * args.lora_rank}")
    print(f"Checkpoint selection      : weighted client validation AUC")
    print(f"Central test usage        : only once after training")
    print(f"Results directory         : {args.results_dir}")
    print("=" * 80)
    print()

    print_gpu_memory()
    print()

    # ── Build IID partition (pooled + reshuffled, sizes preserved) ────────
    train_files, val_files = make_iid_splits(args.seed, args.results_dir)

    train_loaders = []
    val_loaders = []
    class_weights_list = []
    client_train_sizes = []
    client_val_sizes = []

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    for client_id in range(NUM_CLIENTS):
        train_dataset = EpitopeDataset(train_files[client_id], tokenizer)
        val_dataset = EpitopeDataset(val_files[client_id], tokenizer)

        train_df = pd.read_csv(train_files[client_id])

        pos = int(train_df["label"].sum())
        neg = int(len(train_df) - pos)

        positive_weight = neg / max(pos, 1)

        class_weights = torch.tensor(
            [1.0, positive_weight],
            dtype=torch.float,
        ).to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        class_weights_list.append(class_weights)
        client_train_sizes.append(len(train_dataset))
        client_val_sizes.append(len(val_dataset))

        print(
            f"Client {client_id + 1} (IID-{client_id + 1:<8}) | "
            f"Train: {len(train_dataset):>6} | "
            f"Val: {len(val_dataset):>6} | "
            f"Train positives: {pos:>5} | "
            f"Train negatives: {neg:>6} | "
            f"Train positive rate: {pos / max(len(train_df), 1):.2%} | "
            f"Positive class weight: {positive_weight:.2f}"
        )

    print()

    global_model = build_model(lora_rank=args.lora_rank).to(device)

    print("Trainable parameter summary:")
    print_trainable_summary(global_model)
    print()

    val_results = {
        "round": [],
        "val_auc_roc": [],
        "val_auc_pr": [],
        "val_f1": [],
        "val_mcc": [],
        "round_time_seconds": [],
        "elapsed_seconds": [],
        "eta_seconds": [],
    }

    client_round_logs = []

    best_val_auc = -np.inf
    best_round = 0
    best_weights = None

    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()

        print()
        print("=" * 80)
        print(f"ROUND {round_num}/{args.num_rounds}")
        print("=" * 80)

        client_weights_list = []

        for client_id in range(NUM_CLIENTS):
            client_start_time = time.time()

            print()
            print("-" * 80)
            print(f"Training client {client_id + 1}: IID-{client_id + 1}")
            print("-" * 80)

            local_model = copy.deepcopy(global_model).to(device)

            optimizer = torch.optim.AdamW(
                [p for p in local_model.parameters() if p.requires_grad],
                lr=args.lr,
            )

            epoch_losses = train_local(
                model=local_model,
                loader=train_loaders[client_id],
                optimizer=optimizer,
                device=device,
                class_weights=class_weights_list[client_id],
                round_num=round_num,
                client_id=client_id,
            )

            local_val_metrics = evaluate(
                local_model,
                val_loaders[client_id],
                device,
            )

            client_time = time.time() - client_start_time

            print(
                f"[Round {round_num:02d} | Client {client_id + 1} "
                f"{CLIENT_NAMES[client_id]:<12} | Local validation] "
                f"AUC-ROC: {local_val_metrics['auc_roc']:.4f} | "
                f"AUC-PR: {local_val_metrics['auc_pr']:.4f} | "
                f"F1: {local_val_metrics['f1']:.4f} | "
                f"MCC: {local_val_metrics['mcc']:.4f} | "
                f"Client time: {format_seconds(client_time)}"
            )

            client_round_logs.append(
                {
                    "round": round_num,
                    "client_id": client_id + 1,
                    "client_name": f"IID-{client_id + 1}",
                    "epoch_1_loss": epoch_losses[0] if len(epoch_losses) > 0 else np.nan,
                    "epoch_2_loss": epoch_losses[1] if len(epoch_losses) > 1 else np.nan,
                    "epoch_3_loss": epoch_losses[2] if len(epoch_losses) > 2 else np.nan,
                    "local_val_auc_roc": local_val_metrics["auc_roc"],
                    "local_val_auc_pr": local_val_metrics["auc_pr"],
                    "local_val_f1": local_val_metrics["f1"],
                    "local_val_mcc": local_val_metrics["mcc"],
                    "client_time_seconds": client_time,
                }
            )

            client_weights_list.append(get_trainable_weights(local_model))

            del local_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()
        print("Aggregating client models using weighted FedAvg...")

        averaged_weights = federated_averaging(
            client_weights_list=client_weights_list,
            client_sizes=client_train_sizes,
        )

        global_model = set_trainable_weights(global_model, averaged_weights)

        print()
        print("Evaluating aggregated global model on client validation sets...")

        val_metrics = evaluate_global_on_client_validation_sets(
            model=global_model,
            val_loaders=val_loaders,
            val_sizes=client_val_sizes,
            device=device,
        )

        if not np.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_round = round_num
            best_weights = get_trainable_weights(global_model)

            torch.save(
                best_weights,
                os.path.join(args.results_dir, "best_global_weights.pt"),
            )

            best_tag = " <-- best validation checkpoint"
        else:
            best_tag = ""

        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        round_times.append(round_duration)

        elapsed_time = round_end_time - run_start_time
        avg_round_time = sum(round_times) / len(round_times)
        remaining_rounds = args.num_rounds - round_num
        eta_seconds = avg_round_time * remaining_rounds

        val_results["round"].append(round_num)
        val_results["val_auc_roc"].append(val_metrics["auc_roc"])
        val_results["val_auc_pr"].append(val_metrics["auc_pr"])
        val_results["val_f1"].append(val_metrics["f1"])
        val_results["val_mcc"].append(val_metrics["mcc"])
        val_results["round_time_seconds"].append(round_duration)
        val_results["elapsed_seconds"].append(elapsed_time)
        val_results["eta_seconds"].append(eta_seconds)

        print()
        print("=" * 80)
        print(
            f"GLOBAL MODEL | Round {round_num:02d} | "
            f"VAL AUC-ROC: {val_metrics['auc_roc']:.4f} | "
            f"VAL AUC-PR: {val_metrics['auc_pr']:.4f} | "
            f"VAL F1: {val_metrics['f1']:.4f} | "
            f"VAL MCC: {val_metrics['mcc']:.4f}"
            f"{best_tag}"
        )
        print("=" * 80)

        print()
        print("Progress monitor")
        print(f"  Finished round        : {round_num}/{args.num_rounds}")
        print(f"  Round time            : {format_seconds(round_duration)}")
        print(f"  Elapsed time          : {format_seconds(elapsed_time)}")
        print(f"  Estimated time left   : {format_seconds(eta_seconds)}")
        print(f"  Best validation AUC   : {best_val_auc:.4f}")
        print(f"  Best round so far     : {best_round}")
        print_gpu_memory()

        live_val_path = os.path.join(args.results_dir, "federated_val_rounds_live.csv")
        live_client_path = os.path.join(args.results_dir, "client_round_logs_live.csv")

        pd.DataFrame(val_results).to_csv(live_val_path, index=False)
        pd.DataFrame(client_round_logs).to_csv(live_client_path, index=False)

        print(f"  Live validation log   : {live_val_path}")
        print(f"  Live client log       : {live_client_path}")

    val_rounds_path = os.path.join(args.results_dir, "federated_val_rounds.csv")
    client_logs_path = os.path.join(args.results_dir, "client_round_logs.csv")

    pd.DataFrame(val_results).to_csv(val_rounds_path, index=False)
    pd.DataFrame(client_round_logs).to_csv(client_logs_path, index=False)

    final_weights_path = os.path.join(args.results_dir, "global_weights_final.pt")

    torch.save(
        get_trainable_weights(global_model),
        final_weights_path,
    )

    if best_weights is None:
        raise RuntimeError("No best validation checkpoint was saved.")

    print()
    print("=" * 80)
    print("Loading best validation checkpoint for final central test evaluation...")
    print("=" * 80)

    global_model = set_trainable_weights(global_model, best_weights)

    test_dataset = EpitopeDataset("data/central_test.csv", tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Central test samples: {len(test_dataset)}")
    print("Evaluating central test set exactly once...")

    test_metrics = evaluate(global_model, test_loader, device)

    total_run_time = time.time() - run_start_time

    test_summary = {
        "seed": args.seed,
        "lora_rank": args.lora_rank,
        "lora_alpha": 2 * args.lora_rank,
        "partition": "IID",
        "selected_round": best_round,
        "best_val_auc_roc": best_val_auc,
        "test_auc_roc": test_metrics["auc_roc"],
        "test_auc_pr": test_metrics["auc_pr"],
        "test_f1": test_metrics["f1"],
        "test_mcc": test_metrics["mcc"],
        "total_run_time_seconds": total_run_time,
    }

    test_results_path = os.path.join(args.results_dir, "federated_test_results.csv")

    pd.DataFrame([test_summary]).to_csv(
        test_results_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("IID federated training complete")
    print("=" * 80)
    print(f"Best validation AUC-ROC : {best_val_auc:.4f}")
    print(f"Selected round          : {best_round}")
    print(f"Final test AUC-ROC      : {test_metrics['auc_roc']:.4f}")
    print(f"Final test AUC-PR       : {test_metrics['auc_pr']:.4f}")
    print(f"Final test F1           : {test_metrics['f1']:.4f}")
    print(f"Final test MCC          : {test_metrics['mcc']:.4f}")
    print(f"Total run time          : {format_seconds(total_run_time)}")
    print()
    print(f"Saved best weights      : {os.path.join(args.results_dir, 'best_global_weights.pt')}")
    print(f"Saved final weights     : {final_weights_path}")
    print(f"Saved validation rounds : {val_rounds_path}")
    print(f"Saved client logs       : {client_logs_path}")
    print(f"Saved final test result : {test_results_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--results_dir", type=str, default="results/iid_seed42")

    args = parser.parse_args()

    try:
        main(args)

    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)

        error_path = os.path.join(args.results_dir, "error_traceback.txt")

        with open(error_path, "w", encoding="utf-8") as f:
            f.write("IID training failed with the following error:\n\n")
            f.write(str(exc))
            f.write("\n\nFull traceback:\n")
            f.write(traceback.format_exc())

        print()
        print("=" * 80)
        print("Training failed")
        print("=" * 80)
        print(f"Error: {exc}")
        print(f"Full traceback saved to: {error_path}")
        print("=" * 80)

        raise