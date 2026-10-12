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

from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

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

BATCH_SIZE = 64
LR = 2e-4
MAX_LEN = 30
EPOCHS = 5


# All paths are relative to homology/
CLIENT_TRAIN_FILES = [
    f"data/federated/client{i}_train.csv"
    for i in range(1, 6)
]

CLIENT_VAL_FILES = [
    f"data/federated/client{i}_val.csv"
    for i in range(1, 6)
]

GLOBAL_VAL_FILE = "data/federated/global_val.csv"
GLOBAL_TEST_FILE = "data/federated/global_test.csv"


CLIENT_NAMES = [
    "Coronavirus",
    "Parasite",
    "Human/Self",
    "Flavivirus",
    "Bacteria",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:

    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Utility
# ============================================================

def format_seconds(seconds):

    if (
        seconds is None
        or math.isnan(seconds)
        or math.isinf(seconds)
    ):
        return "unknown"

    return str(
        timedelta(seconds=int(seconds))
    )


def print_gpu_memory():

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024 ** 3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024 ** 3
        )

        max_alloc = (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        )

        print(
            f"GPU memory | "
            f"allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"max allocated: {max_alloc:.2f} GB"
        )

    else:

        print(
            "GPU memory | CUDA not available"
        )


def check_required_files():

    required = (
        CLIENT_TRAIN_FILES
        + CLIENT_VAL_FILES
        + [
            GLOBAL_VAL_FILE,
            GLOBAL_TEST_FILE,
        ]
    )

    missing = [
        path
        for path in required
        if not os.path.exists(path)
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )


# ============================================================
# Dataset
# ============================================================

class EpitopeDataset(Dataset):

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=MAX_LEN,
    ):

        df = pd.read_csv(csv_path)

        if "sequence" not in df.columns:
            raise ValueError(
                f"{csv_path} missing column: sequence"
            )

        if "label" not in df.columns:
            raise ValueError(
                f"{csv_path} missing column: label"
            )

        self.sequences = (
            df["sequence"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            df["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
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
            "input_ids":
                enc["input_ids"].squeeze(0),

            "attention_mask":
                enc["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long,
                ),
        }


# ============================================================
# Models
# ============================================================

def build_lora_model(lora_rank: int):

    model = (
        EsmForSequenceClassification
        .from_pretrained(
            "facebook/esm2_t12_35M_UR50D",
            num_labels=2,
        )
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "value",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


def build_frozen_model():

    """
    Frozen ESM-2 backbone.
    Only the classification head is trainable.
    """

    model = (
        EsmForSequenceClassification
        .from_pretrained(
            "facebook/esm2_t12_35M_UR50D",
            num_labels=2,
        )
    )

    for name, param in model.named_parameters():

        if "classifier" not in name:
            param.requires_grad = False

    return model


def print_trainable_summary(
    model,
    label: str,
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    lora = sum(
        p.numel()
        for name, p in model.named_parameters()
        if (
            p.requires_grad
            and "lora" in name.lower()
        )
    )

    classifier = sum(
        p.numel()
        for name, p in model.named_parameters()
        if (
            p.requires_grad
            and any(
                x in name.lower()
                for x in (
                    "classifier",
                    "score",
                    "modules_to_save",
                )
            )
        )
    )

    print(
        f"[{label}] "
        f"total={total:,} | "
        f"trainable={trainable:,} | "
        f"lora={lora:,} | "
        f"classifier={classifier:,} | "
        f"{100 * trainable / total:.4f}%"
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            probs = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .cpu()
                .numpy()
            )

            all_probs.extend(probs)

            all_labels.extend(
                batch["labels"].numpy()
            )

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    preds = (
        all_probs >= 0.5
    ).astype(int)

    if len(
        np.unique(all_labels)
    ) < 2:

        return {
            "auc_roc": np.nan,
            "auc_pr": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
        }

    return {

        "auc_roc":
            roc_auc_score(
                all_labels,
                all_probs,
            ),

        "auc_pr":
            average_precision_score(
                all_labels,
                all_probs,
            ),

        "f1":
            f1_score(
                all_labels,
                preds,
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                all_labels,
                preds,
            ),
    }


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_weights,
    epochs,
    label,
    args,
):

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    best_val_auc = -np.inf
    best_state = None

    for epoch in range(epochs):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # Validation only.
        # Test set is NOT used here.
        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"[{label} | "
            f"Epoch {epoch + 1}/{epochs}] "
            f"Loss: "
            f"{total_loss / max(len(train_loader), 1):.4f} | "
            f"Val AUC-ROC: "
            f"{val_metrics['auc_roc']:.4f} | "
            f"Val AUC-PR: "
            f"{val_metrics['auc_pr']:.4f} | "
            f"Epoch time: "
            f"{format_seconds(epoch_time)}"
        )

        if (
            not np.isnan(
                val_metrics["auc_roc"]
            )
            and
            val_metrics["auc_roc"]
            > best_val_auc
        ):

            best_val_auc = (
                val_metrics["auc_roc"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        print(
            f"[{label}] "
            f"Loaded best validation checkpoint "
            f"(AUC-ROC: {best_val_auc:.4f})"
        )

    return (
        model,
        best_val_auc,
    )


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    dataset,
    batch_size,
    shuffle,
    args,
    generator=None,
):

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=args.num_workers,

        worker_init_fn=(
            seed_worker
            if args.num_workers > 0
            else None
        ),

        generator=generator,

        pin_memory=(
            torch.cuda.is_available()
        ),
    )


# ============================================================
# Main
# ============================================================

def main(args):

    run_start = time.time()

    set_seed(args.seed)

    check_required_files()

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "facebook/esm2_t12_35M_UR50D"
        )
    )

    print("=" * 80)
    print(
        "FedEpitope Homology-Aware "
        "Baseline Experiments"
    )
    print("=" * 80)

    print(
        f"Device              : {device}"
    )

    print(
        f"Seed                : {args.seed}"
    )

    print(
        f"LoRA rank           : {args.lora_rank}"
    )

    print(
        f"LoRA alpha          : {2 * args.lora_rank}"
    )

    print(
        f"Epochs              : {args.epochs}"
    )

    print(
        f"Batch size          : {args.batch_size}"
    )

    print(
        f"Learning rate       : {args.lr}"
    )

    print(
        f"Results directory   : {args.results_dir}"
    )

    print(
        f"Global validation   : {GLOBAL_VAL_FILE}"
    )

    print(
        f"Global test         : {GLOBAL_TEST_FILE}"
    )

    print()

    print_gpu_memory()

    print()


    # ========================================================
    # Global homology-disjoint test
    # ========================================================

    test_dataset = EpitopeDataset(
        GLOBAL_TEST_FILE,
        tokenizer,
    )

    test_loader = make_loader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        args=args,
    )

    print(
        "Homology-disjoint global test set:",
        f"{len(test_dataset):,} samples"
    )


    # ========================================================
    # Global validation
    # ========================================================

    global_val_dataset = EpitopeDataset(
        GLOBAL_VAL_FILE,
        tokenizer,
    )

    global_val_loader = make_loader(
        global_val_dataset,
        batch_size=128,
        shuffle=False,
        args=args,
    )

    print(
        "Homology-disjoint global validation set:",
        f"{len(global_val_dataset):,} samples"
    )

    print()


    results = {}


    # ========================================================
    # Random predictor
    # ========================================================

    print("=" * 80)
    print("BASELINE 1: Random predictor")
    print("=" * 80)

    test_df = pd.read_csv(
        GLOBAL_TEST_FILE
    )

    rng = np.random.default_rng(
        args.seed
    )

    rand_probs = rng.uniform(
        0,
        1,
        len(test_df),
    )

    rand_preds = (
        rand_probs >= 0.5
    ).astype(int)

    rand_labels = (
        test_df["label"]
        .astype(int)
        .values
    )

    results["Random"] = {

        "auc_roc":
            roc_auc_score(
                rand_labels,
                rand_probs,
            ),

        "auc_pr":
            average_precision_score(
                rand_labels,
                rand_probs,
            ),

        "f1":
            f1_score(
                rand_labels,
                rand_preds,
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                rand_labels,
                rand_preds,
            ),

        "best_val_auc":
            np.nan,
    }

    m = results["Random"]

    print(
        f"AUC-ROC: {m['auc_roc']:.4f} | "
        f"AUC-PR: {m['auc_pr']:.4f} | "
        f"F1: {m['f1']:.4f} | "
        f"MCC: {m['mcc']:.4f}"
    )


    # ========================================================
    # Single-client baselines
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "BASELINE 2: "
        "Single-client training"
    )

    print("=" * 80)


    for client_id in range(5):

        client_number = (
            client_id + 1
        )

        client_name = (
            CLIENT_NAMES[client_id]
        )

        label = (
            f"Client{client_number}-"
            f"{client_name}"
        )

        print()
        print(
            f"Training {label} independently..."
        )

        train_df = pd.read_csv(
            CLIENT_TRAIN_FILES[
                client_id
            ]
        )

        pos = int(
            train_df["label"].sum()
        )

        neg = int(
            len(train_df) - pos
        )

        class_weights = torch.tensor(
            [
                1.0,
                neg / max(pos, 1),
            ],
            dtype=torch.float,
        ).to(device)

        print(
            f"Training samples: "
            f"{len(train_df):,}"
        )

        print(
            f"Positive: {pos:,} | "
            f"Negative: {neg:,}"
        )

        print(
            f"Positive class weight: "
            f"{class_weights[1].item():.4f}"
        )

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            args.seed + client_id
        )

        train_loader = make_loader(

            EpitopeDataset(
                CLIENT_TRAIN_FILES[
                    client_id
                ],
                tokenizer,
            ),

            batch_size=args.batch_size,

            shuffle=True,

            args=args,

            generator=generator,
        )

        # Local cluster-disjoint
        # validation for this client.
        val_loader = make_loader(

            EpitopeDataset(
                CLIENT_VAL_FILES[
                    client_id
                ],
                tokenizer,
            ),

            batch_size=128,

            shuffle=False,

            args=args,
        )

        model = build_lora_model(
            args.lora_rank
        ).to(device)

        print_trainable_summary(
            model,
            label,
        )

        model, best_val = train_model(

            model,

            train_loader,

            val_loader,

            device,

            class_weights,

            args.epochs,

            label,

            args,
        )

        # Global test is evaluated only
        # after validation-based selection.
        test_metrics = evaluate(
            model,
            test_loader,
            device,
        )

        key = (
            f"Single_Client"
            f"{client_number}_"
            f"{client_name}"
        )

        results[key] = {
            **test_metrics,
            "best_val_auc":
                best_val,
        }

        print(
            f"Global Test → "
            f"AUC-ROC: "
            f"{test_metrics['auc_roc']:.4f} | "
            f"AUC-PR: "
            f"{test_metrics['auc_pr']:.4f} | "
            f"F1: "
            f"{test_metrics['f1']:.4f} | "
            f"MCC: "
            f"{test_metrics['mcc']:.4f}"
        )

        # Save live progress
        live_df = (
            pd.DataFrame(results)
            .T
        )

        live_df.index.name = "model"

        live_df.insert(
            0,
            "seed",
            args.seed,
        )

        live_df.insert(
            1,
            "lora_rank",
            args.lora_rank,
        )

        live_df.to_csv(
            os.path.join(
                args.results_dir,
                "baseline_results_live.csv",
            )
        )

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # Build pooled centralized training set
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "Preparing pooled centralized "
        "training data"
    )

    print("=" * 80)

    all_train_df = pd.concat(

        [
            pd.read_csv(path)
            for path in CLIENT_TRAIN_FILES
        ],

        ignore_index=True,
    )

    all_train_path = os.path.join(
        args.results_dir,
        "tmp_all_train.csv",
    )

    all_train_df.to_csv(
        all_train_path,
        index=False,
    )

    print(
        f"Pooled training samples: "
        f"{len(all_train_df):,}"
    )

    pos = int(
        all_train_df["label"].sum()
    )

    neg = int(
        len(all_train_df) - pos
    )

    class_weights = torch.tensor(
        [
            1.0,
            neg / max(pos, 1),
        ],
        dtype=torch.float,
    ).to(device)

    print(
        f"Pooled positives: {pos:,}"
    )

    print(
        f"Pooled negatives: {neg:,}"
    )

    print(
        f"Positive class weight: "
        f"{class_weights[1].item():.4f}"
    )


    # ========================================================
    # Frozen ESM-2 baseline
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "BASELINE 3: "
        "Frozen ESM-2 + classification head"
    )

    print("=" * 80)

    generator_frozen = (
        torch.Generator()
    )

    generator_frozen.manual_seed(
        args.seed + 10
    )

    frozen_train_loader = make_loader(

        EpitopeDataset(
            all_train_path,
            tokenizer,
        ),

        batch_size=args.batch_size,

        shuffle=True,

        args=args,

        generator=generator_frozen,
    )

    frozen_model = (
        build_frozen_model()
        .to(device)
    )

    print_trainable_summary(
        frozen_model,
        "Frozen-ESM2",
    )

    frozen_model, best_val = (
        train_model(

            frozen_model,

            frozen_train_loader,

            global_val_loader,

            device,

            class_weights,

            args.epochs,

            "Frozen-ESM2",

            args,
        )
    )

    test_metrics = evaluate(
        frozen_model,
        test_loader,
        device,
    )

    results["Frozen_ESM2"] = {
        **test_metrics,
        "best_val_auc":
            best_val,
    }

    print(
        f"Global Test → "
        f"AUC-ROC: "
        f"{test_metrics['auc_roc']:.4f} | "
        f"AUC-PR: "
        f"{test_metrics['auc_pr']:.4f} | "
        f"F1: "
        f"{test_metrics['f1']:.4f} | "
        f"MCC: "
        f"{test_metrics['mcc']:.4f}"
    )

    del frozen_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # Centralized LoRA baseline
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "BASELINE 4: "
        "Centralized ESM-2 + LoRA "
        "(pooled-data baseline)"
    )

    print("=" * 80)

    generator_central = (
        torch.Generator()
    )

    generator_central.manual_seed(
        args.seed + 20
    )

    central_train_loader = make_loader(

        EpitopeDataset(
            all_train_path,
            tokenizer,
        ),

        batch_size=args.batch_size,

        shuffle=True,

        args=args,

        generator=generator_central,
    )

    central_model = (
        build_lora_model(
            args.lora_rank
        )
        .to(device)
    )

    print_trainable_summary(
        central_model,
        "Centralized-LoRA",
    )

    central_model, best_val = (
        train_model(

            central_model,

            central_train_loader,

            global_val_loader,

            device,

            class_weights,

            args.epochs,

            "Centralized-LoRA",

            args,
        )
    )

    test_metrics = evaluate(
        central_model,
        test_loader,
        device,
    )

    results["Centralized_LoRA"] = {
        **test_metrics,
        "best_val_auc":
            best_val,
    }

    print(
        f"Global Test → "
        f"AUC-ROC: "
        f"{test_metrics['auc_roc']:.4f} | "
        f"AUC-PR: "
        f"{test_metrics['auc_pr']:.4f} | "
        f"F1: "
        f"{test_metrics['f1']:.4f} | "
        f"MCC: "
        f"{test_metrics['mcc']:.4f}"
    )


    # ========================================================
    # Save centralized trainable weights
    # ========================================================

    weights_path = os.path.join(
        args.results_dir,
        "centralized_weights.pt",
    )

    torch.save(

        {
            name:
                param.data
                .detach()
                .cpu()
                .clone()

            for name, param
            in central_model.named_parameters()

            if param.requires_grad
        },

        weights_path,
    )

    print(
        f"Saved centralized weights: "
        f"{weights_path}"
    )


    del central_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # Cleanup
    # ========================================================

    if os.path.exists(
        all_train_path
    ):
        os.remove(
            all_train_path
        )


    # ========================================================
    # Final results
    # ========================================================

    total_time = (
        time.time()
        - run_start
    )

    print("\n" + "=" * 80)

    print(
        "RESULTS SUMMARY — "
        "homology-disjoint global test set"
    )

    print("=" * 80)

    print(
        f"{'Model':<45} "
        f"{'AUC-ROC':>8} "
        f"{'AUC-PR':>8} "
        f"{'F1':>8} "
        f"{'MCC':>8} "
        f"{'Best-Val':>9}"
    )

    print("-" * 95)

    for name, metrics in results.items():

        best_val = (
            metrics["best_val_auc"]
        )

        if np.isnan(best_val):
            best_val_text = "NA"
        else:
            best_val_text = (
                f"{best_val:.4f}"
            )

        print(
            f"{name:<45} "
            f"{metrics['auc_roc']:>8.4f} "
            f"{metrics['auc_pr']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['mcc']:>8.4f} "
            f"{best_val_text:>9}"
        )


    # ========================================================
    # Save final CSV
    # ========================================================

    results_df = (
        pd.DataFrame(results)
        .T
    )

    results_df.index.name = "model"

    results_df.insert(
        0,
        "seed",
        args.seed,
    )

    results_df.insert(
        1,
        "lora_rank",
        args.lora_rank,
    )

    results_path = os.path.join(
        args.results_dir,
        "baseline_results.csv",
    )

    results_df.to_csv(
        results_path
    )

    print()

    print(
        f"Total run time: "
        f"{format_seconds(total_time)}"
    )

    print(
        f"Results saved to: "
        f"{results_path}"
    )

    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/baseline_seed42",
    )

    args = parser.parse_args()

    try:

        main(args)

    except Exception as exc:

        os.makedirs(
            args.results_dir,
            exist_ok=True,
        )

        error_path = os.path.join(
            args.results_dir,
            "error_traceback.txt",
        )

        with open(
            error_path,
            "w"
        ) as f:

            f.write(
                "Homology baseline run failed."
                "\n\n"
            )

            f.write(
                str(exc)
            )

            f.write(
                "\n\nFull traceback:\n"
            )

            f.write(
                traceback.format_exc()
            )

        print(
            "\nRun failed."
        )

        print(
            f"Traceback saved to: "
            f"{error_path}"
        )

        raise