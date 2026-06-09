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

NUM_CLIENTS  = 5
BATCH_SIZE   = 64
LR           = 2e-4
MAX_LEN      = 30

PERSONAL_EPOCHS = 10
PERSONAL_LR_LFT = 1e-5   # pFL-LFT  : LoRA + head
PERSONAL_LR_CLF = 1e-4   # pFL-CLF  : head only
PERSONAL_LR_FPX = 1e-5   # pFL-FedProx
MU              = 0.01

CLIENT_TRAIN_FILES = [f"data/client{i + 1}_train.csv" for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES   = [f"data/client{i + 1}_val.csv"   for i in range(NUM_CLIENTS)]
CLIENT_TEST_FILES  = [f"data/client{i + 1}_test.csv"  for i in range(NUM_CLIENTS)]

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


def check_required_files(fed_checkpoint: str):
    required = (
        CLIENT_TRAIN_FILES
        + CLIENT_VAL_FILES
        + CLIENT_TEST_FILES
        + ["data/central_test.csv", fed_checkpoint]
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


def get_classifier_param_names(model):
    """Return the exact set of classifier parameter names for this model."""
    return {
        n for n, p in model.named_parameters()
        if p.requires_grad
        and any(k in n.lower() for k in ("classifier", "score", "modules_to_save"))
        and "lora" not in n.lower()
    }


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


# ============================================================
# Weight utilities
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
# Standard local training (used in pFL-LFT and pFL-FedProx)
# ============================================================

def train_local_standard(model, loader, optimizer, device,
                          class_weights, epochs, label,
                          run_start, variant_start, client_id,
                          total_clients, variant_num, total_variants):
    """
    Trains for `epochs` epochs. Prints per-epoch loss and timing.
    No val or test access inside this function.
    """
    model.train()
    criterion  = torch.nn.CrossEntropyLoss(weight=class_weights)
    epoch_logs = []

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss  = 0.0

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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss   = total_loss / max(len(loader), 1)
        epoch_time = time.time() - epoch_start
        epoch_logs.append(avg_loss)

        elapsed     = time.time() - run_start
        done_epochs = (
            (variant_num - 1) * total_clients * epochs
            + client_id * epochs
            + (epoch + 1)
        )
        total_epochs = total_variants * total_clients * epochs
        avg_ep_time  = (time.time() - variant_start) / max(done_epochs - (variant_num - 1) * total_clients * epochs, 1)
        eta          = avg_ep_time * (total_epochs - done_epochs)

        print(
            f"  [{label} | Epoch {epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Epoch time: {format_seconds(epoch_time)} | "
            f"Elapsed: {format_seconds(elapsed)} | "
            f"ETA: {format_seconds(eta)}"
        )

    return epoch_logs


def train_local_fedprox(model, loader, optimizer, device,
                         class_weights, epochs, global_weights_cpu,
                         mu, label,
                         run_start, variant_start, client_id,
                         total_clients, variant_num, total_variants):
    """FedProx training with proximal term. No val or test access."""
    model.train()
    criterion         = torch.nn.CrossEntropyLoss(weight=class_weights)
    global_weights_gpu = {n: w.to(device) for n, w in global_weights_cpu.items()}
    epoch_logs        = []

    for epoch in range(epochs):
        epoch_start = time.time()
        total_ce    = 0.0

        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            ce_loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits,
                labels,
            )
            prox_loss = sum(
                ((p - global_weights_gpu[n]) ** 2).sum()
                for n, p in model.named_parameters()
                if p.requires_grad and n in global_weights_gpu
            )
            loss = ce_loss + (mu / 2) * prox_loss
            loss.backward()
            optimizer.step()
            total_ce += ce_loss.item()

        avg_ce     = total_ce / max(len(loader), 1)
        epoch_time = time.time() - epoch_start
        epoch_logs.append(avg_ce)

        elapsed     = time.time() - run_start
        done_epochs = (
            (variant_num - 1) * total_clients * epochs
            + client_id * epochs
            + (epoch + 1)
        )
        total_epochs = total_variants * total_clients * epochs
        avg_ep_time  = (time.time() - variant_start) / max(done_epochs - (variant_num - 1) * total_clients * epochs, 1)
        eta          = avg_ep_time * (total_epochs - done_epochs)

        print(
            f"  [{label} | Epoch {epoch + 1}/{epochs}] "
            f"CE Loss: {avg_ce:.4f} | "
            f"Epoch time: {format_seconds(epoch_time)} | "
            f"Elapsed: {format_seconds(elapsed)} | "
            f"ETA: {format_seconds(eta)}"
        )

    return epoch_logs


# ============================================================
# DataLoader factory — seeded
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
# pFL core: one variant across all clients
#
# Key design decisions (no test-set leakage):
#   - Global federated checkpoint loaded from --fed_checkpoint
#   - Val set used ONLY to pick the best personalised checkpoint
#   - Client test set evaluated ONCE after best checkpoint is loaded
#   - Central test set is NEVER accessed in this script
# ============================================================

def run_personalisation(
    variant_name,
    variant_num,
    total_variants,
    global_weights,
    lora_rank,
    train_loaders,
    val_loaders,
    client_test_loaders,
    device,
    class_weights_list,
    epochs,
    lr,
    freeze_lora,
    use_fedprox,
    args,
    run_start,
    results_dir,
    live_rows,
):
    print(f"\n{'=' * 80}")
    print(f"PERSONALISATION VARIANT {variant_num}/{total_variants}: {variant_name}")
    print(f"  epochs={epochs} | lr={lr} | "
          f"freeze_lora={freeze_lora} | use_fedprox={use_fedprox}")
    print(
        "  Checkpoint selection: best val AUC per client\n"
        "  Test evaluation: client-specific test set, ONCE after best ckpt loaded\n"
        "  Central test set: NOT accessed here"
    )
    print(f"{'=' * 80}\n")

    global_weights_cpu = (
        {n: w.cpu().clone() for n, w in global_weights.items()}
        if use_fedprox else None
    )

    variant_start = time.time()
    client_times  = []
    results       = []

    for client_id in range(NUM_CLIENTS):
        client_start = time.time()

        print(f"\n{'-' * 80}")
        print(
            f"[{variant_name} | Client {client_id + 1}/{NUM_CLIENTS} "
            f"{CLIENT_NAMES[client_id]}]"
        )
        print(f"{'-' * 80}")

        # Build fresh model and load global checkpoint
        personal_model = build_model(lora_rank).to(device)
        personal_model = set_trainable_weights(personal_model, global_weights)

        if freeze_lora:
            # pFL-CLF: freeze LoRA, train head only
            clf_names = get_classifier_param_names(personal_model)
            for n, p in personal_model.named_parameters():
                p.requires_grad = (n in clf_names)
            print_trainable_summary(personal_model,
                                    f"CLF-only client {client_id + 1}")
        else:
            print_trainable_summary(personal_model,
                                    f"Full client {client_id + 1}")

        optimizer = torch.optim.AdamW(
            [p for p in personal_model.parameters() if p.requires_grad],
            lr=lr,
        )

        # Val-based checkpoint selection: track best val AUC during training
        best_val_auc   = -np.inf
        best_state     = None
        val_trajectory = []

        # ── Per-epoch training loop with inline val ──────────────────────
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_list[client_id])
        if use_fedprox:
            global_weights_gpu = {n: w.to(device) for n, w in global_weights_cpu.items()}

        for epoch in range(epochs):
            epoch_start = time.time()
            personal_model.train()
            total_loss = 0.0

            for batch in train_loaders[client_id]:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                optimizer.zero_grad()

                ce_loss = criterion(
                    personal_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits,
                    labels,
                )

                if use_fedprox:
                    prox = sum(
                        ((p - global_weights_gpu[n]) ** 2).sum()
                        for n, p in personal_model.named_parameters()
                        if p.requires_grad and n in global_weights_gpu
                    )
                    loss = ce_loss + (MU / 2) * prox
                else:
                    loss = ce_loss

                loss.backward()
                optimizer.step()
                total_loss += ce_loss.item()

            avg_loss  = total_loss / max(len(train_loaders[client_id]), 1)
            epoch_dur = time.time() - epoch_start

            # Evaluate on val set for checkpoint selection
            val_metrics = evaluate(personal_model, val_loaders[client_id], device)
            val_trajectory.append(val_metrics["auc_roc"])

            elapsed = time.time() - run_start
            done_ep = (variant_num - 1) * NUM_CLIENTS * epochs + client_id * epochs + (epoch + 1)
            total_ep = total_variants * NUM_CLIENTS * epochs
            if done_ep > 0:
                avg_ep = (time.time() - variant_start) / done_ep
                eta    = avg_ep * (total_ep - done_ep)
            else:
                eta = float("inf")

            print(
                f"  [{variant_name} | Client {client_id + 1} "
                f"{CLIENT_NAMES[client_id]:<12} | Epoch {epoch + 1}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Val AUC-ROC: {val_metrics['auc_roc']:.4f} | "
                f"Epoch time: {format_seconds(epoch_dur)} | "
                f"Elapsed: {format_seconds(elapsed)} | "
                f"ETA: {format_seconds(eta)}"
            )

            # Save best val checkpoint — test set never seen here
            if not np.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_state   = copy.deepcopy(personal_model.state_dict())

        # Load best val checkpoint
        if best_state is not None:
            personal_model.load_state_dict(best_state)
        print(
            f"  [{variant_name} | Client {client_id + 1}] "
            f"Best val AUC: {best_val_auc:.4f} (epoch "
            f"{int(np.argmax(val_trajectory)) + 1})"
        )

        # Evaluate on client-specific test set ONCE
        test_metrics = evaluate(
            personal_model, client_test_loaders[client_id], device
        )
        print(
            f"  [{variant_name} | Client {client_id + 1} "
            f"{CLIENT_NAMES[client_id]:<12}] "
            f"Client test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
            f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"MCC: {test_metrics['mcc']:.4f}"
        )

        results.append({
            "variant":      variant_name,
            "client_id":    client_id + 1,
            "client_name":  CLIENT_NAMES[client_id],
            "best_val_auc": best_val_auc,
            "auc_roc":      test_metrics["auc_roc"],
            "auc_pr":       test_metrics["auc_pr"],
            "f1":           test_metrics["f1"],
            "mcc":          test_metrics["mcc"],
        })
        live_rows.append(results[-1])

        client_time = time.time() - client_start
        client_times.append(client_time)

        # Progress monitor
        print()
        print("  Progress monitor")
        print(
            f"    Variant {variant_num}/{total_variants} | "
            f"Client {client_id + 1}/{NUM_CLIENTS} "
            f"({CLIENT_NAMES[client_id]})"
        )
        print(f"    Client time    : {format_seconds(client_time)}")
        print(f"    Elapsed        : {format_seconds(time.time() - run_start)}")
        remaining = (NUM_CLIENTS - client_id - 1) + (total_variants - variant_num) * NUM_CLIENTS
        avg_ct = sum(client_times) / len(client_times)
        print(f"    ETA            : {format_seconds(avg_ct * remaining)}")
        print_gpu_memory()

        # Live save after every client
        live_path = os.path.join(results_dir, "pfl_results_live.csv")
        pd.DataFrame(live_rows).to_csv(live_path, index=False)
        print(f"    Live results   : {live_path}")

        # Save this client's personalised weights
        ckpt_name = (
            f"pfl_{variant_name.split()[0].lower()}_"
            f"client{client_id + 1}.pt"
        )
        torch.save(
            {n: p.data.clone() for n, p in personal_model.named_parameters()
             if p.requires_grad},
            os.path.join(results_dir, ckpt_name),
        )

        del personal_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_auc = float(np.mean([r["auc_roc"] for r in results]))
    print(f"\n  {variant_name} — avg client test AUC-ROC: {avg_auc:.4f}")
    return results, avg_auc


# ============================================================
# Main
# ============================================================

def main(args):
    run_start = time.time()

    set_seed(args.seed)
    check_required_files(args.fed_checkpoint)
    os.makedirs(args.results_dir, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    print("=" * 80)
    print("FedEpitope — Personalised Federated Learning")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Seed                : {args.seed}")
    print(f"LoRA rank           : {args.lora_rank}   alpha: {2 * args.lora_rank}")
    print(f"Federated checkpoint: {args.fed_checkpoint}")
    print(f"pFL epochs          : {PERSONAL_EPOCHS}")
    print(f"Results dir         : {args.results_dir}")
    print(
        "Test-set rule       : client test set evaluated ONCE per client,\n"
        "                      after val-based checkpoint selection.\n"
        "                      Central test set NOT accessed here."
    )
    print("=" * 80)
    print()

    print_gpu_memory()
    print()

    # ── Build data loaders ────────────────────────────────────────────────
    train_loaders      = []
    val_loaders        = []
    client_test_loaders = []
    class_weights_list = []
    client_sizes       = []

    for client_id in range(NUM_CLIENTS):
        gen = torch.Generator()
        gen.manual_seed(args.seed + client_id)

        train_ds = EpitopeDataset(CLIENT_TRAIN_FILES[client_id], tokenizer)
        val_ds   = EpitopeDataset(CLIENT_VAL_FILES[client_id],   tokenizer)
        test_ds  = EpitopeDataset(CLIENT_TEST_FILES[client_id],  tokenizer)

        train_df = pd.read_csv(CLIENT_TRAIN_FILES[client_id])
        pos      = int(train_df["label"].sum())
        neg      = int(len(train_df) - pos)

        train_loaders.append(make_loader(train_ds, BATCH_SIZE, True,  args, gen))
        val_loaders.append(  make_loader(val_ds,   128,        False, args))
        client_test_loaders.append(make_loader(test_ds, 128, False, args))

        class_weights_list.append(
            torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)
        )
        client_sizes.append(len(train_ds))

        print(
            f"Client {client_id + 1} ({CLIENT_NAMES[client_id]:<12}) | "
            f"Train: {len(train_ds):>7,} | Val: {len(val_ds):>6,} | "
            f"Test: {len(test_ds):>6,} | "
            f"Pos rate: {pos / max(len(train_df), 1):.2%}"
        )

    print()

    # ── Load global federated checkpoint ─────────────────────────────────
    print(f"Loading global federated checkpoint: {args.fed_checkpoint}")
    global_weights = torch.load(args.fed_checkpoint, map_location="cpu")

    # Verify the checkpoint fits the model
    ref_model = build_model(args.lora_rank)
    print_trainable_summary(ref_model, f"Reference model (r={args.lora_rank})")
    missing = [k for k in get_trainable_weights(ref_model) if k not in global_weights]
    extra   = [k for k in global_weights if k not in get_trainable_weights(ref_model)]
    if missing:
        print(f"  WARNING: {len(missing)} keys missing from checkpoint")
    if extra:
        print(f"  WARNING: {len(extra)} unexpected keys in checkpoint")
    if not missing and not extra:
        print("  Checkpoint keys verified ✓")
    del ref_model
    print()

    # ── Run three pFL variants ────────────────────────────────────────────
    VARIANTS = [
        dict(
            variant_name="pFL-LFT",
            epochs=PERSONAL_EPOCHS,
            lr=PERSONAL_LR_LFT,
            freeze_lora=False,
            use_fedprox=False,
        ),
        dict(
            variant_name="pFL-CLF-Only",
            epochs=PERSONAL_EPOCHS,
            lr=PERSONAL_LR_CLF,
            freeze_lora=True,
            use_fedprox=False,
        ),
        dict(
            variant_name="pFL-FedProx",
            epochs=PERSONAL_EPOCHS,
            lr=PERSONAL_LR_FPX,
            freeze_lora=False,
            use_fedprox=True,
        ),
    ]
    TOTAL_VARIANTS = len(VARIANTS)

    all_results = {}
    live_rows   = []   # accumulates across all variants for live CSV

    for variant_num, vkwargs in enumerate(VARIANTS, start=1):
        results, avg_auc = run_personalisation(
            variant_num=variant_num,
            total_variants=TOTAL_VARIANTS,
            global_weights=global_weights,
            lora_rank=args.lora_rank,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            client_test_loaders=client_test_loaders,
            device=device,
            class_weights_list=class_weights_list,
            args=args,
            run_start=run_start,
            results_dir=args.results_dir,
            live_rows=live_rows,
            **vkwargs,
        )
        all_results[vkwargs["variant_name"]] = (results, avg_auc)

    # ── Build per-client summary CSV ──────────────────────────────────────
    per_client_rows = []
    for client_id in range(NUM_CLIENTS):
        row = {
            "seed":        args.seed,
            "lora_rank":   args.lora_rank,
            "client_id":   client_id + 1,
            "client_name": CLIENT_NAMES[client_id],
        }
        for vname, (res, _) in all_results.items():
            row[f"{vname}_auc_roc"] = res[client_id]["auc_roc"]
            row[f"{vname}_auc_pr"]  = res[client_id]["auc_pr"]
            row[f"{vname}_f1"]      = res[client_id]["f1"]
            row[f"{vname}_mcc"]     = res[client_id]["mcc"]
            row[f"{vname}_val_auc"] = res[client_id]["best_val_auc"]
        per_client_rows.append(row)

    per_client_df = pd.DataFrame(per_client_rows)
    per_client_path = os.path.join(args.results_dir, "pfl_per_client.csv")
    per_client_df.to_csv(per_client_path, index=False)

    # Full detail CSV
    detail_df   = pd.DataFrame(live_rows)
    detail_path = os.path.join(args.results_dir, "pfl_results.csv")
    detail_df.to_csv(detail_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - run_start
    print("\n" + "=" * 80)
    print("PERSONALISED FL — FINAL SUMMARY")
    print("(All AUC-ROC values on per-client pathogen-specific test sets)")
    print("=" * 80)

    print(f"\n{'Variant':<20} {'Avg AUC-ROC':>12}")
    print("-" * 35)
    for vname, (_, avg) in all_results.items():
        print(f"{vname:<20} {avg:>12.4f}")

    print(f"\nPer-client breakdown:")
    header = f"{'Client':<14}"
    for vname in all_results:
        header += f"  {vname:>12}"
    print(header)
    print("-" * (14 + 14 * TOTAL_VARIANTS))
    for client_id in range(NUM_CLIENTS):
        row_str = f"{CLIENT_NAMES[client_id]:<14}"
        for vname, (res, _) in all_results.items():
            row_str += f"  {res[client_id]['auc_roc']:>12.4f}"
        print(row_str)

    print(f"\nTotal run time   : {format_seconds(total_time)}")
    print(f"Saved results    : {per_client_path}")
    print(f"Saved detail     : {detail_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--lora_rank",      type=int,   default=2)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument(
        "--fed_checkpoint",
        type=str,
        default="results/r2_seed42/best_global_weights.pt",
        help="Path to best_global_weights.pt from federated_train.py",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/pfl_seed42",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)
        err_path = os.path.join(args.results_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            f.write(f"pFL run failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {err_path}")
        raise