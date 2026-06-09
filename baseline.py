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

BATCH_SIZE   = 64
LR           = 2e-4
MAX_LEN      = 30
EPOCHS       = 5

CLIENT_TRAIN_FILES = [f"data/client{i + 1}_train.csv" for i in range(5)]
CLIENT_VAL_FILES   = [f"data/client{i + 1}_val.csv"   for i in range(5)]

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
        allocated = torch.cuda.memory_allocated()     / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()      / 1024 ** 3
        max_alloc = torch.cuda.max_memory_allocated()  / 1024 ** 3
        print(
            f"GPU memory | allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"max allocated: {max_alloc:.2f} GB"
        )
    else:
        print("GPU memory | CUDA not available")


def check_required_files():
    required = (
        CLIENT_TRAIN_FILES
        + CLIENT_VAL_FILES
        + ["data/central_test.csv"]
    )
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


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

def build_lora_model(lora_rank: int):
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t12_35M_UR50D", num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,       # matches federated_train.py
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    return get_peft_model(model, lora_config)


def build_frozen_model():
    """ESM-2 backbone frozen; only the classification head is trainable."""
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t12_35M_UR50D", num_labels=2
    )
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model


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
    print(f"  [{label}] total={total:,} | trainable={trainable:,} "
          f"| lora={lora:,} | classifier={classifier:,} "
          f"| {100 * trainable / total:.4f}%")


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


# ============================================================
# Training with val-based checkpoint selection
# No test-set access inside this function.
# ============================================================

def train_model(model, train_loader, val_loader, device,
                class_weights, epochs, label, args):
    optimizer  = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion  = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_val_auc  = -np.inf
    best_state    = None

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits,
                labels,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics  = evaluate(model, val_loader, device)
        epoch_time   = time.time() - epoch_start

        print(
            f"  [{label} | Epoch {epoch + 1}/{epochs}] "
            f"Loss: {total_loss / max(len(train_loader), 1):.4f} | "
            f"Val AUC-ROC: {val_metrics['auc_roc']:.4f} | "
            f"Epoch time: {format_seconds(epoch_time)}"
        )

        # Checkpoint selection is done on VALIDATION AUC only.
        # The central test set is never seen inside this function.
        if not np.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state   = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [{label}] Loaded best val checkpoint (AUC-ROC: {best_val_auc:.4f})")

    return model, best_val_auc


# ============================================================
# DataLoader factory — seeded, matching federated_train.py
# ============================================================

def make_loader(dataset, batch_size, shuffle, args, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


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

    # 7 timed models: 5 single-client + 1 frozen + 1 centralised
    # (random predictor is instant and not counted)
    TOTAL_MODELS = 7
    model_times  = []   # wall-clock seconds per completed model
    models_done  = 0    # incremented after each model finishes

    def progress_monitor(label):
        nonlocal models_done
        models_done += 1
        elapsed  = time.time() - run_start
        avg_time = sum(model_times) / len(model_times) if model_times else 0.0
        eta      = avg_time * (TOTAL_MODELS - models_done)
        print()
        print("Progress monitor")
        print(f"  Finished model        : {models_done}/{TOTAL_MODELS}  ({label})")
        print(f"  Model time            : {format_seconds(model_times[-1])}")
        print(f"  Elapsed time          : {format_seconds(elapsed)}")
        print(f"  Estimated time left   : {format_seconds(eta)}")
        print_gpu_memory()
        # Live save after every model so a crash doesn't lose earlier results.
        live_path = os.path.join(args.results_dir, "baseline_results_live.csv")
        live_df   = pd.DataFrame(results).T
        live_df.index.name = "model"
        live_df.insert(0, "seed",      args.seed)
        live_df.insert(1, "lora_rank", args.lora_rank)
        live_df.to_csv(live_path)
        print(f"  Live results log      : {live_path}")

    print("=" * 80)
    print("FedEpitope Baseline Experiments")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Seed          : {args.seed}")
    print(f"LoRA rank     : {args.lora_rank}   alpha: {2 * args.lora_rank}")
    print(f"Epochs        : {args.epochs}")
    print(f"Results dir   : {args.results_dir}")
    print(f"Total models  : {TOTAL_MODELS}  (5 single-client + frozen ESM-2 + centralised LoRA)")
    print(f"Test-set rule : evaluated ONCE per model, after val-based checkpoint selection")
    print("=" * 80)
    print()

    print_gpu_memory()
    print()

    # ── Load central test set once; never used for checkpoint selection ────
    test_dataset = EpitopeDataset("data/central_test.csv", tokenizer)
    test_loader  = make_loader(test_dataset, batch_size=128, shuffle=False, args=args)
    print(f"Central test set: {len(test_dataset):,} samples\n")

    results = {}

    # ── Baseline 1: Random predictor ──────────────────────────────────────
    print("=" * 80)
    print("BASELINE 1: Random predictor")
    print("=" * 80)

    test_df    = pd.read_csv("data/central_test.csv")
    # Uses args.seed so random baseline is reproducible and seed-matched.
    rng        = np.random.default_rng(args.seed)
    rand_probs = rng.uniform(0, 1, len(test_df))
    rand_preds = (rand_probs >= 0.5).astype(int)
    rand_labels = test_df["label"].values

    results["Random"] = {
        "auc_roc": roc_auc_score(rand_labels, rand_probs),
        "auc_pr":  average_precision_score(rand_labels, rand_probs),
        "f1":      f1_score(rand_labels, rand_preds, zero_division=0),
        "mcc":     matthews_corrcoef(rand_labels, rand_preds),
        "best_val_auc": np.nan,
    }
    m = results["Random"]
    print(
        f"  AUC-ROC: {m['auc_roc']:.4f} | AUC-PR: {m['auc_pr']:.4f} | "
        f"F1: {m['f1']:.4f} | MCC: {m['mcc']:.4f}\n"
    )

    # ── Baseline 2: Single-client models (one per client) ─────────────────
    print("=" * 80)
    print("BASELINE 2: Single-client training (each client independently)")
    print("=" * 80)

    for client_id in range(5):
        print(f"\n  Training client {client_id + 1} ({CLIENT_NAMES[client_id]}) independently...")

        model_start = time.time()

        train_df  = pd.read_csv(CLIENT_TRAIN_FILES[client_id])
        pos       = int(train_df["label"].sum())
        neg       = int(len(train_df) - pos)
        cw        = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)

        # Per-client seeded generator so client order doesn't affect others.
        gen = torch.Generator()
        gen.manual_seed(args.seed + client_id)

        train_loader = make_loader(
            EpitopeDataset(CLIENT_TRAIN_FILES[client_id], tokenizer),
            batch_size=args.batch_size, shuffle=True, args=args, generator=gen,
        )
        val_loader = make_loader(
            EpitopeDataset(CLIENT_VAL_FILES[client_id], tokenizer),
            batch_size=128, shuffle=False, args=args,
        )

        model = build_lora_model(args.lora_rank).to(device)
        print_trainable_summary(model, f"Client{client_id + 1}-{CLIENT_NAMES[client_id]}")

        model, best_val = train_model(
            model, train_loader, val_loader, device, cw,
            args.epochs, f"Client{client_id + 1}-{CLIENT_NAMES[client_id]}", args,
        )

        # Central test evaluated ONCE after val-based selection.
        test_metrics = evaluate(model, test_loader, device)
        key = f"Single_Client{client_id + 1}_{CLIENT_NAMES[client_id]}"
        results[key] = {**test_metrics, "best_val_auc": best_val}

        print(
            f"  → Test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
            f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
        )

        model_times.append(time.time() - model_start)
        progress_monitor(f"Client{client_id + 1}-{CLIENT_NAMES[client_id]}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Baseline 3: Frozen ESM-2 + linear head ────────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE 3: Frozen ESM-2 + linear classification head")
    print("=" * 80)

    model_start = time.time()

    # Pool all client training and validation data.
    all_train_df = pd.concat([pd.read_csv(f) for f in CLIENT_TRAIN_FILES], ignore_index=True)
    all_val_df   = pd.concat([pd.read_csv(f) for f in CLIENT_VAL_FILES],   ignore_index=True)

    all_train_path = os.path.join(args.results_dir, "tmp_all_train.csv")
    all_val_path   = os.path.join(args.results_dir, "tmp_all_val.csv")
    all_train_df.to_csv(all_train_path, index=False)
    all_val_df.to_csv(all_val_path,     index=False)

    pos = int(all_train_df["label"].sum())
    neg = int(len(all_train_df) - pos)
    cw  = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)

    gen_frozen = torch.Generator()
    gen_frozen.manual_seed(args.seed + 10)

    frozen_train_loader = make_loader(
        EpitopeDataset(all_train_path, tokenizer),
        batch_size=args.batch_size, shuffle=True, args=args, generator=gen_frozen,
    )
    frozen_val_loader = make_loader(
        EpitopeDataset(all_val_path, tokenizer),
        batch_size=128, shuffle=False, args=args,
    )

    frozen_model = build_frozen_model().to(device)
    trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    print(f"  Trainable params (frozen ESM-2, head only): {trainable:,}")

    frozen_model, best_val = train_model(
        frozen_model, frozen_train_loader, frozen_val_loader, device, cw,
        args.epochs, "Frozen-ESM2", args,
    )

    test_metrics = evaluate(frozen_model, test_loader, device)
    results["Frozen_ESM2"] = {**test_metrics, "best_val_auc": best_val}
    print(
        f"  → Test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
        f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
    )

    model_times.append(time.time() - model_start)
    progress_monitor("Frozen-ESM2")

    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Baseline 4: Centralised LoRA (upper bound) ────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE 4: Centralised ESM-2 + LoRA (upper bound, no privacy constraints)")
    print("=" * 80)

    model_start = time.time()

    gen_central = torch.Generator()
    gen_central.manual_seed(args.seed + 20)

    central_train_loader = make_loader(
        EpitopeDataset(all_train_path, tokenizer),
        batch_size=args.batch_size, shuffle=True, args=args, generator=gen_central,
    )
    central_val_loader = make_loader(
        EpitopeDataset(all_val_path, tokenizer),
        batch_size=128, shuffle=False, args=args,
    )

    central_model = build_lora_model(args.lora_rank).to(device)
    print_trainable_summary(central_model, "Centralised-LoRA")

    central_model, best_val = train_model(
        central_model, central_train_loader, central_val_loader, device, cw,
        args.epochs, "Centralised-LoRA", args,
    )

    test_metrics = evaluate(central_model, test_loader, device)
    results["Centralised_LoRA"] = {**test_metrics, "best_val_auc": best_val}
    print(
        f"  → Test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
        f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
    )

    # Save centralised weights for potential downstream use.
    centralised_weights_path = os.path.join(args.results_dir, "centralised_weights.pt")
    torch.save(
        {n: p.data.clone() for n, p in central_model.named_parameters() if p.requires_grad},
        centralised_weights_path,
    )
    print(f"  Saved centralised weights: {centralised_weights_path}")

    model_times.append(time.time() - model_start)
    progress_monitor("Centralised-LoRA")

    del central_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Cleanup temp files ─────────────────────────────────────────────────
    for p in (all_train_path, all_val_path):
        if os.path.exists(p):
            os.remove(p)

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - run_start

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY — all models on central test set")
    print("=" * 80)
    print(f"{'Model':<45} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<45} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} "
            f"{m['f1']:>8.4f} {m['mcc']:>8.4f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T
    results_df.index.name = "model"
    results_df.insert(0, "seed", args.seed)
    results_df.insert(1, "lora_rank", args.lora_rank)

    results_path = os.path.join(args.results_dir, "baseline_results.csv")
    results_df.to_csv(results_path)

    print(f"\nTotal run time   : {format_seconds(total_time)}")
    print(f"Results saved to : {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--lora_rank",   type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument(
        "--results_dir", type=str, default="results/baseline_seed42"
    )

    args = parser.parse_args()

    try:
        main(args)

    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)
        error_path = os.path.join(args.results_dir, "error_traceback.txt")
        with open(error_path, "w") as f:
            f.write(f"Baseline run failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {error_path}")
        raise