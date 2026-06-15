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
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

NUM_ROUNDS     = 20
LOCAL_EPOCHS   = 3
NUM_CLIENTS    = 5
BATCH_SIZE     = 64
LR             = 2e-4
MAX_LEN        = 30
MAX_GRAD_NORM  = 1.0
DELTA          = 1e-5

# Privacy budgets; inf = No-DP baseline
EPSILON_VALUES = [1.0, 5.0, 10.0, float("inf")]

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


def extract_epsilon(result) -> float:
    """
    Safely extract epsilon from RDPAccountant.get_privacy_spent().
    Returns a float regardless of whether Opacus returns a scalar or tuple.
    """
    if isinstance(result, (tuple, list)):
        return float(result[0])
    return float(result)


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
    model = get_peft_model(model, lora_config)
    # ModuleValidator.fix makes the model Opacus-compatible
    # (replaces BatchNorm with GroupNorm etc.)
    model = ModuleValidator.fix(model)
    return model


def print_trainable_summary(model, label: str):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora      = sum(
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
# Opacus wraps the model so parameters gain a '_module.' prefix.
# We strip it when extracting and handle it when setting, so that
# FedAvg aggregation and checkpoint I/O always use clean names.
# ============================================================

def get_trainable_weights(model):
    weights = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            clean = n[len("_module."):] if n.startswith("_module.") else n
            weights[clean] = p.detach().cpu().clone()
    return weights


def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad:
            clean = n[len("_module."):] if n.startswith("_module.") else n
            if clean in weights:
                p.data.copy_(weights[clean].to(p.device))
    return model


# ============================================================
# FedAvg — identical to federated_train.py
# ============================================================

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


# ============================================================
# Noise multiplier pre-computation
#
# KEY FIX: compute the noise_multiplier for the FULL training budget
# (NUM_ROUNDS × LOCAL_EPOCHS effective epochs) per client, so that
# the cumulative privacy guarantee across ALL rounds equals target_epsilon.
#
# We compute one noise_multiplier per client because clients have
# different dataset sizes → different sample_rates → different NMs
# needed to achieve the same target_epsilon.
# The reported epsilon is the worst-case (max) across clients.
# ============================================================

def precompute_noise_multipliers(target_epsilon, client_sizes):
    """
    Returns dict {client_id: noise_multiplier} such that training
    for NUM_ROUNDS * LOCAL_EPOCHS epochs with that NM achieves
    (target_epsilon, DELTA)-DP for that client's dataset size.

    Threat model:
      - Example-level (record-level) local DP via Opacus per-sample clipping
      - Honest-but-curious server: clients add calibrated Gaussian noise to
        LoRA + classifier gradients before transmission
      - Composition via RDP accountant across all rounds
    """
    if target_epsilon == float("inf"):
        return {i: 0.0 for i in range(NUM_CLIENTS)}

    nm_per_client = {}
    for client_id in range(NUM_CLIENTS):
        sample_rate = BATCH_SIZE / client_sizes[client_id]
        nm = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=DELTA,
            sample_rate=sample_rate,
            epochs=NUM_ROUNDS * LOCAL_EPOCHS,   # full training budget
            accountant="rdp",
            epsilon_tolerance=0.01,
        )
        nm_per_client[client_id] = nm
    return nm_per_client


# ============================================================
# One round of local DP training
#
# KEY FIX: takes pre-computed noise_multiplier (constant across rounds)
# and updates a persistent RDPAccountant for cross-round composition.
# No val or test access inside this function.
# ============================================================

def train_local_dp(
    model,
    train_dataset,
    device,
    class_weights,
    noise_multiplier,
    client_id,
    client_size,
    round_num,
    is_dp,
    client_accountant,      # persistent RDPAccountant, updated in-place
    run_start,
    epsilon_label,
    round_times_eps,
):
    # Ensure train mode before DataLoader construction and make_private.
    # evaluate() sets the model to eval mode; without this explicit reset,
    # the deepcopied global_model (which was in eval mode after val evaluation)
    # would reach make_private() still in eval mode, causing Opacus to error.
    model.train()

    # Opacus requires num_workers=0 for DP data loading.
    # No-DP uses the same setup for a fair comparison.
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if is_dp:
        # Use make_private with the pre-computed NM (not make_private_with_epsilon)
        # so the noise level is fixed for the full training budget.
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=MAX_GRAD_NORM,
        )
        if round_num == 1:
            print(
                f"  [{epsilon_label} | Client {client_id + 1} "
                f"{CLIENT_NAMES[client_id]:<12}] "
                f"noise_multiplier = {noise_multiplier:.4f} "
                f"(fixed for all {NUM_ROUNDS} rounds)"
            )

    epoch_losses  = []
    actual_steps  = 0   # count exact optimiser steps for accountant

    for epoch in range(LOCAL_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits,
                labels,
            )
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Invalid loss: {loss.item()} | "
                    f"ε={epsilon_label} Round {round_num} "
                    f"Client {client_id + 1} Epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total_loss  += loss.item()
            actual_steps += 1

        avg_loss   = total_loss / max(len(loader), 1)
        epoch_time = time.time() - epoch_start
        epoch_losses.append(avg_loss)

        elapsed = time.time() - run_start
        done_rounds = len(round_times_eps)
        avg_round = (
            sum(round_times_eps) / done_rounds
            if done_rounds > 0 else float('nan')
        )
        eta = avg_round * (NUM_ROUNDS - round_num)

        print(
            f"  [{epsilon_label} | Round {round_num:02d} | "
            f"Client {client_id + 1} {CLIENT_NAMES[client_id]:<12} | "
            f"Epoch {epoch + 1}/{LOCAL_EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Epoch time: {format_seconds(epoch_time)} | "
            f"Elapsed: {format_seconds(elapsed)} | "
            f"ETA: {format_seconds(eta)}"
        )

    # ── Cross-round DP accounting (KEY FIX) ──────────────────────────────
    # RDPAccountant.step() is called once per optimiser step.
    # RDPAccountant groups consecutive identical (nm, sample_rate) pairs
    # into one history entry, so looping actual_steps times is O(1) storage.
    # Composing across ALL NUM_ROUNDS rounds gives the true cumulative eps.
    if is_dp and client_accountant is not None:
        sample_rate = BATCH_SIZE / client_size
        for _ in range(actual_steps):
            client_accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )
        cumulative_eps = extract_epsilon(
            client_accountant.get_privacy_spent(delta=DELTA)
        )
        print(
            f"  [{epsilon_label} | Round {round_num:02d} | "
            f"Client {client_id + 1} {CLIENT_NAMES[client_id]:<12}] "
            f"Cumulative ε = {cumulative_eps:.4f} "
            f"(δ={DELTA:.0e}, {round_num * LOCAL_EPOCHS} total epochs)"
        )
    else:
        cumulative_eps = float("inf")

    return model, cumulative_eps


# ============================================================
# Full federated run for one epsilon value
#
# Val-based checkpoint selection — test set never accessed here.
# Central test set is evaluated ONCE by the caller after this returns.
# ============================================================

def run_federated_dp(
    train_datasets,
    val_loaders,
    client_val_sizes,
    device,
    class_weights_list,
    client_sizes,
    target_epsilon,
    noise_multipliers,
    args,
    run_start,
    epsilon_idx,
    total_epsilons,
    results_dir,
    label,
    live_rows,
):
    is_dp = target_epsilon != float("inf")

    print(f"\n{'=' * 80}")
    print(f"EPSILON {epsilon_idx}/{total_epsilons}: {label}")
    if is_dp:
        print(
            f"  Threat model  : example-level local DP (Opacus per-sample clipping)\n"
            f"  Mechanism     : Gaussian noise on LoRA + classifier gradients\n"
            f"  Server model  : honest-but-curious\n"
            f"  Accounting    : RDP composition across ALL {NUM_ROUNDS} rounds\n"
            f"  Budget        : target ε={target_epsilon}, δ={DELTA:.0e}\n"
            f"  Clip norm     : {MAX_GRAD_NORM}"
        )
        for cid in range(NUM_CLIENTS):
            print(
                f"  Client {cid + 1} ({CLIENT_NAMES[cid]:<12}) "
                f"sample_rate={BATCH_SIZE / client_sizes[cid]:.5f} | "
                f"NM={noise_multipliers[cid]:.4f}"
            )
    else:
        print(
            "  No-DP baseline — same configuration as DP runs\n"
            "  (num_workers=0, same seed) for a controlled comparison.\n"
            "  Note: may differ slightly from headline FedEpitope result\n"
            "  (federated_train.py) due to num_workers=0 vs 4."
        )
    print(
        f"  Checkpoint selection : best weighted val AUC across clients\n"
        f"  Central test set     : NOT accessed in this function"
    )
    print(f"{'=' * 80}\n")

    # Persistent per-client RDP accountants — composed across ALL rounds
    client_accountants = [RDPAccountant() for _ in range(NUM_CLIENTS)]

    global_model  = build_model(args.lora_rank).to(device)
    print_trainable_summary(global_model, f"Global model ({label})")
    print()

    val_log       = []     # per-round val metrics
    round_times   = []
    best_val_auc  = -np.inf
    best_round    = 0
    best_weights  = None

    epsilon_start = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()

        print()
        print("=" * 80)
        print(f"[{label}] ROUND {round_num}/{NUM_ROUNDS}")
        print("=" * 80)

        client_weights_list_round = []
        round_eps_per_client      = []

        for client_id in range(NUM_CLIENTS):
            print()
            print(f"{'-' * 80}")
            print(
                f"[{label}] Round {round_num:02d} | "
                f"Client {client_id + 1} {CLIENT_NAMES[client_id]}"
            )
            print(f"{'-' * 80}")

            local_model = copy.deepcopy(global_model).to(device)

            local_model, cumulative_eps = train_local_dp(
                model=local_model,
                train_dataset=train_datasets[client_id],
                device=device,
                class_weights=class_weights_list[client_id],
                noise_multiplier=noise_multipliers[client_id],
                client_id=client_id,
                client_size=client_sizes[client_id],
                round_num=round_num,
                is_dp=is_dp,
                client_accountant=client_accountants[client_id],
                run_start=run_start,
                epsilon_label=label,
                round_times_eps=round_times,
            )

            round_eps_per_client.append(cumulative_eps)
            client_weights_list_round.append(get_trainable_weights(local_model))

            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FedAvg
        averaged     = federated_averaging(client_weights_list_round, client_sizes)
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

        # Restore train mode after val evaluation so the deepcopy at the
        # start of the next round's client loop starts in train mode.
        global_model.train()

        val_metrics  = weighted_average_metrics(client_val_metrics, client_val_sizes)
        worst_eps    = (
            max(round_eps_per_client)
            if is_dp else float("inf")
        )

        # Best checkpoint selected by weighted val AUC — test never accessed
        best_tag = ""
        if not np.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_round   = round_num
            best_weights = get_trainable_weights(global_model)
            torch.save(
                best_weights,
                os.path.join(results_dir, f"best_weights_{label}.pt"),
            )
            best_tag = " <-- best val checkpoint"

        round_dur = time.time() - round_start
        round_times.append(round_dur)
        elapsed   = time.time() - run_start
        eta       = (sum(round_times) / len(round_times)) * (NUM_ROUNDS - round_num)

        row = {
            "seed":          args.seed,
            "lora_rank":     args.lora_rank,
            "epsilon_label": label,
            "target_epsilon": target_epsilon,
            "round":         round_num,
            "val_auc_roc":   val_metrics["auc_roc"],
            "val_auc_pr":    val_metrics["auc_pr"],
            "val_f1":        val_metrics["f1"],
            "val_mcc":       val_metrics["mcc"],
            "worst_case_cumulative_eps": worst_eps,
            "round_time_s":  round_dur,
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
        if is_dp:
            print(
                f"  Worst-case cumulative ε after round {round_num}: "
                f"{worst_eps:.4f} (target: {target_epsilon})"
            )
        print("=" * 80)
        print()
        print("Progress monitor")
        print(f"  Epsilon setting       : {label} ({epsilon_idx}/{total_epsilons})")
        print(f"  Finished round        : {round_num}/{NUM_ROUNDS}")
        print(f"  Round time            : {format_seconds(round_dur)}")
        print(f"  Elapsed (this ε)      : {format_seconds(time.time() - epsilon_start)}")
        print(f"  Elapsed (total)       : {format_seconds(elapsed)}")
        print(f"  ETA (this ε)          : {format_seconds(eta)}")
        print(f"  Best val AUC so far   : {best_val_auc:.4f} (round {best_round})")
        print_gpu_memory()

        # Live save after every round
        live_path = os.path.join(results_dir, "dp_val_log_live.csv")
        pd.DataFrame(live_rows).to_csv(live_path, index=False)
        print(f"  Live val log          : {live_path}")

    # Final cumulative epsilon report
    if is_dp:
        print(f"\n  [{label}] Final cumulative ε per client:")
        for cid in range(NUM_CLIENTS):
            final_eps = extract_epsilon(
                client_accountants[cid].get_privacy_spent(delta=DELTA)
            )
            print(
                f"    Client {cid + 1} ({CLIENT_NAMES[cid]:<12}): "
                f"ε = {final_eps:.4f}  "
                f"(over {NUM_ROUNDS * LOCAL_EPOCHS} epochs, "
                f"δ={DELTA:.0e})"
            )
        worst_final = max(
            extract_epsilon(
                client_accountants[cid].get_privacy_spent(delta=DELTA)
            )
            for cid in range(NUM_CLIENTS)
        )
        print(f"    Worst-case ε: {worst_final:.4f}")
    else:
        worst_final = float("inf")

    # Save per-round val log for this epsilon
    val_log_path = os.path.join(results_dir, f"dp_val_log_{label}.csv")
    pd.DataFrame(val_log).to_csv(val_log_path, index=False)

    return best_weights, best_val_auc, best_round, worst_final, val_log


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

    print("=" * 80)
    print("FedEpitope — Differential Privacy Experiments")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Seed                : {args.seed}")
    print(f"LoRA rank           : {args.lora_rank}  alpha: {2 * args.lora_rank}")
    print(f"ε values            : {EPSILON_VALUES}")
    print(f"δ                   : {DELTA:.0e}")
    print(f"Clip norm           : {MAX_GRAD_NORM}")
    print(f"Rounds / local eps  : {NUM_ROUNDS} / {LOCAL_EPOCHS}")
    print(f"Results dir         : {args.results_dir}")
    print()
    print("DP threat model:")
    print("  - Example-level (record-level) local DP via Opacus per-sample clipping")
    print("  - Gaussian noise added to LoRA + classifier gradients before transmission")
    print("  - Honest-but-curious server (motivated by gradient inversion attacks)")
    print("  - RDP composition across ALL rounds (not per-round only)")
    print("  - Noise multiplier fixed for full training budget (NUM_ROUNDS x LOCAL_EPOCHS)")
    print()
    print("Test-set rule: central test set evaluated ONCE per epsilon after training.")
    print("Checkpoint selection: best weighted val AUC across clients.")
    print("=" * 80)
    print()

    print_gpu_memory()
    print()

    # ── Data loading ──────────────────────────────────────────────────────
    train_datasets     = []
    val_loaders        = []
    class_weights_list = []
    client_sizes       = []
    client_val_sizes   = []

    for client_id in range(NUM_CLIENTS):
        train_ds = EpitopeDataset(CLIENT_TRAIN_FILES[client_id], tokenizer)
        val_ds   = EpitopeDataset(CLIENT_VAL_FILES[client_id],   tokenizer)

        train_df = pd.read_csv(CLIENT_TRAIN_FILES[client_id])
        pos      = int(train_df["label"].sum())
        neg      = int(len(train_df) - pos)

        # Val loader: regular (no Opacus), seeded generator
        gen = torch.Generator()
        gen.manual_seed(args.seed + client_id)
        val_loader = DataLoader(
            val_ds,
            batch_size=128,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        train_datasets.append(train_ds)
        val_loaders.append(val_loader)
        class_weights_list.append(
            torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float).to(device)
        )
        client_sizes.append(len(train_ds))
        client_val_sizes.append(len(val_ds))

        print(
            f"Client {client_id + 1} ({CLIENT_NAMES[client_id]:<12}) | "
            f"Train: {len(train_ds):>7,} | Val: {len(val_ds):>6,} | "
            f"Pos rate: {pos / max(len(train_df), 1):.2%}"
        )

    print()

    # Central test set — loaded once, evaluated ONCE per epsilon at the end
    test_ds     = EpitopeDataset("data/central_test.csv", tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Central test set: {len(test_ds):,} samples\n")

    # ── Pre-compute noise multipliers ─────────────────────────────────────
    print("Pre-computing noise multipliers for full training budget "
          f"({NUM_ROUNDS} rounds × {LOCAL_EPOCHS} epochs)...\n")
    nm_by_epsilon = {}
    for eps in EPSILON_VALUES:
        if eps == float("inf"):
            nm_by_epsilon[eps] = {i: 0.0 for i in range(NUM_CLIENTS)}
            print(f"  ε=inf (No-DP): noise_multiplier = 0.0 (no noise)")
        else:
            nm = precompute_noise_multipliers(eps, client_sizes)
            nm_by_epsilon[eps] = nm
            print(f"  ε={eps}: noise_multiplier per client:")
            for cid in range(NUM_CLIENTS):
                print(
                    f"    Client {cid + 1} ({CLIENT_NAMES[cid]:<12}) "
                    f"NM={nm[cid]:.4f}  "
                    f"sample_rate={BATCH_SIZE / client_sizes[cid]:.5f}"
                )
    print()

    # ── Verify weight prefix stripping ───────────────────────────────────
    print("Verifying Opacus weight prefix stripping...")
    _test_model = build_model(args.lora_rank).to(device)
    _test_model.train()
    _test_opt = torch.optim.AdamW(_test_model.parameters(), lr=LR)
    _test_ds  = EpitopeDataset(CLIENT_TRAIN_FILES[0], tokenizer)
    _test_ld  = DataLoader(_test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    _pe = PrivacyEngine()
    _test_model, _test_opt, _ = _pe.make_private(
        module=_test_model, optimizer=_test_opt,
        data_loader=_test_ld, noise_multiplier=1.0,
        max_grad_norm=MAX_GRAD_NORM,
    )
    _w = get_trainable_weights(_test_model)
    assert not any(k.startswith("_module.") for k in _w.keys()), \
        "Prefix stripping failed — weight names still contain _module."
    expected_weights = 2 * 2 * 12 + 4   # (lora_A + lora_B) × (query + value) × 12 layers + 4 head params = 52
    assert len(_w) == expected_weights, \
        f"Expected {expected_weights} weight tensors, got {len(_w)}"
    print(f"  Weight prefix stripped correctly: {len(_w)} tensors ✓\n")
    del _test_model, _test_opt, _pe, _test_ld, _test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run one epsilon at a time ─────────────────────────────────────────
    # Ordered so No-DP runs last (useful as comparison baseline after DP runs)
    ordered_epsilons = sorted(
        [e for e in EPSILON_VALUES if e != float("inf")]
    ) + [float("inf")]
    if args.epsilon is not None:
        ordered_epsilons = [args.epsilon]

    total_epsilons = len(ordered_epsilons)
    summary_rows   = []
    live_rows      = []   # cross-epsilon live log

    for epsilon_idx, target_epsilon in enumerate(ordered_epsilons, start=1):
        label = f"eps={target_epsilon}" if target_epsilon != float("inf") else "No-DP"

        best_weights, best_val_auc, best_round, worst_final_eps, val_log = run_federated_dp(
            train_datasets=train_datasets,
            val_loaders=val_loaders,
            client_val_sizes=client_val_sizes,
            device=device,
            class_weights_list=class_weights_list,
            client_sizes=client_sizes,
            target_epsilon=target_epsilon,
            noise_multipliers=nm_by_epsilon[target_epsilon],
            args=args,
            run_start=run_start,
            epsilon_idx=epsilon_idx,
            total_epsilons=total_epsilons,
            results_dir=args.results_dir,
            label=label,
            live_rows=live_rows,
        )

        # ── Evaluate on central test set ONCE per epsilon ─────────────────
        if best_weights is None:
            raise RuntimeError(f"[{label}] No checkpoint was saved — all val AUC values were NaN.")
        print(f"\n[{label}] Loading best val checkpoint (round {best_round})...")
        eval_model = build_model(args.lora_rank).to(device)
        eval_model = set_trainable_weights(eval_model, best_weights)
        print(f"[{label}] Evaluating central test set exactly once...")
        test_metrics = evaluate(eval_model, test_loader, device)
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {
            "seed":              args.seed,
            "lora_rank":         args.lora_rank,
            "epsilon_label":     label,
            "target_epsilon":    target_epsilon,
            "reported_epsilon":  (
                worst_final_eps
                if target_epsilon != float("inf")
                else float("inf")
            ),
            "delta":             DELTA,
            "best_val_auc":      best_val_auc,
            "best_round":        best_round,
            "test_auc_roc":      test_metrics["auc_roc"],
            "test_auc_pr":       test_metrics["auc_pr"],
            "test_f1":           test_metrics["f1"],
            "test_mcc":          test_metrics["mcc"],
        }
        summary_rows.append(row)

        print(
            f"\n[{label}] Test AUC-ROC: {test_metrics['auc_roc']:.4f} | "
            f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"MCC: {test_metrics['mcc']:.4f}"
        )
        if target_epsilon != float("inf"):
            print(
                f"[{label}] Final worst-case cumulative ε = {worst_final_eps:.4f} "
                f"(target was {target_epsilon}, δ={DELTA:.0e})"
            )

    # ── Save all results ──────────────────────────────────────────────────
    summary_df      = pd.DataFrame(summary_rows)
    summary_path    = os.path.join(args.results_dir, "dp_results.csv")
    summary_df.to_csv(summary_path, index=False)

    live_df   = pd.DataFrame(live_rows)
    live_path = os.path.join(args.results_dir, "dp_val_log_live.csv")
    live_df.to_csv(live_path, index=False)

    total_time = time.time() - run_start

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DIFFERENTIAL PRIVACY — FINAL SUMMARY")
    print("(Cumulative ε across all rounds — not per-round)")
    print("=" * 80)
    no_dp_row = summary_df[summary_df["epsilon_label"] == "No-DP"]
    no_dp_auc = no_dp_row["test_auc_roc"].item() if len(no_dp_row) else np.nan

    print(
        f"\n{'Setting':<12} {'Reported ε':>11} {'Best val AUC':>13} "
        f"{'Test AUC-ROC':>13} {'Δ vs No-DP':>11} "
        f"{'Retention':>10}"
    )
    print("-" * 75)
    for _, row in summary_df.iterrows():
        delta_auc = row["test_auc_roc"] - no_dp_auc if not np.isnan(no_dp_auc) else np.nan
        retention = (
            row["test_auc_roc"] / no_dp_auc * 100
            if (not np.isnan(no_dp_auc) and no_dp_auc > 0) else np.nan
        )
        eps_str = (
            f"{row['reported_epsilon']:.4f}"
            if row["reported_epsilon"] != float("inf") else "∞"
        )
        ret_str = f"{retention:.1f}%" if not np.isnan(retention) else "—"
        print(
            f"{row['epsilon_label']:<12} {eps_str:>11} "
            f"{row['best_val_auc']:>13.4f} "
            f"{row['test_auc_roc']:>13.4f} "
            f"{delta_auc:>+11.4f} "
            f"{ret_str:>10}"
        )

    print(f"\nTotal run time   : {format_seconds(total_time)}")
    print(f"Results saved to : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--lora_rank",   type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="Workers for val/test loaders. "
                             "DP training loaders always use 0 (Opacus requirement).")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Run a single epsilon value instead of all. "
             "Use inf for No-DP. Default: run all.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/dp_seed42",
    )

    args = parser.parse_args()

    # Resolve --epsilon inf from command line
    if args.epsilon is not None:
        try:
            args.epsilon = float(args.epsilon)
        except ValueError:
            args.epsilon = float("inf")

    try:
        main(args)
    except Exception as exc:
        os.makedirs(args.results_dir, exist_ok=True)
        err_path = os.path.join(args.results_dir, "error_traceback.txt")
        with open(err_path, "w") as f:
            f.write(f"DP run failed:\n\n{exc}\n\nFull traceback:\n")
            f.write(traceback.format_exc())
        print(f"\nFailed. Traceback saved to: {err_path}")
        raise