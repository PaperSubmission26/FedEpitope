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
# Default configuration
# ============================================================

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS = 5

BATCH_SIZE = 64
LR = 2e-4
MAX_LEN = 30


# ============================================================
# Homology-aware data paths
# ============================================================

CLIENT_TRAIN_FILES = [
    f"data/federated/client{i}_train.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

CLIENT_VAL_FILES = [
    f"data/federated/client{i}_val.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

GLOBAL_VAL_FILE = (
    "data/federated/global_val.csv"
)

GLOBAL_TEST_FILE = (
    "data/federated/global_test.csv"
)


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

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# General utilities
# ============================================================

def format_seconds(seconds):

    if (
        seconds is None
        or math.isnan(seconds)
        or math.isinf(seconds)
    ):
        return "unknown"

    return str(
        timedelta(
            seconds=int(seconds)
        )
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

        max_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        )

        print(
            f"GPU memory | "
            f"allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"max allocated: {max_allocated:.2f} GB"
        )

    else:

        print(
            "GPU memory | CUDA not available"
        )


def check_required_files():

    required_files = (
        CLIENT_TRAIN_FILES
        + CLIENT_VAL_FILES
        + [
            GLOBAL_VAL_FILE,
            GLOBAL_TEST_FILE,
        ]
    )

    missing = [
        path
        for path in required_files
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

            "input_ids":
                encoded["input_ids"]
                .squeeze(0),

            "attention_mask":
                encoded["attention_mask"]
                .squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long,
                ),
        }


# ============================================================
# Model
# ============================================================

def build_model(
    lora_rank: int = 2
):

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

        lora_alpha=(
            2 * lora_rank
        ),

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


# ============================================================
# Trainable parameter utilities
# ============================================================

def get_trainable_weights(
    model
):

    return {

        name:
            param
            .detach()
            .cpu()
            .clone()

        for name, param
        in model.named_parameters()

        if param.requires_grad
    }


def set_trainable_weights(
    model,
    weights
):

    for name, param in (
        model.named_parameters()
    ):

        if (
            param.requires_grad
            and name in weights
        ):

            param.data.copy_(
                weights[name]
                .to(param.device)
            )

    return model


def print_trainable_summary(
    model
):

    total_params = 0
    trainable_params = 0
    lora_params = 0
    classifier_params = 0

    for name, param in (
        model.named_parameters()
    ):

        total_params += (
            param.numel()
        )

        if param.requires_grad:

            trainable_params += (
                param.numel()
            )

            lower = name.lower()

            if "lora" in lower:

                lora_params += (
                    param.numel()
                )

            if (
                "classifier" in lower
                or "score" in lower
                or "modules_to_save" in lower
            ):

                classifier_params += (
                    param.numel()
                )

    print(
        f"Total parameters          : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters      : "
        f"{trainable_params:,}"
    )

    print(
        f"LoRA trainable parameters : "
        f"{lora_params:,}"
    )

    print(
        f"Classifier-like parameters: "
        f"{classifier_params:,}"
    )

    print(
        f"Trainable percentage      : "
        f"{100 * trainable_params / total_params:.4f}%"
    )


# ============================================================
# FedAvg
# ============================================================

def federated_averaging(
    client_weights_list,
    client_sizes,
):

    total_size = sum(
        client_sizes
    )

    proportions = [

        size / total_size

        for size
        in client_sizes
    ]

    averaged_weights = {}

    for key in (
        client_weights_list[0]
        .keys()
    ):

        averaged_weights[key] = sum(

            proportions[i]
            * client_weights_list[i][key]
            .float()

            for i in range(
                len(client_weights_list)
            )
        )

    return averaged_weights


# ============================================================
# Local training
# ============================================================

def train_local(
    model,
    loader,
    optimizer,
    device,
    class_weights,
    local_epochs,
    round_num,
    client_id,
):

    model.train()

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    epoch_losses = []

    for epoch in range(
        local_epochs
    ):

        epoch_start = time.time()

        total_loss = 0.0

        for batch_idx, batch in (
            enumerate(loader)
        ):

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

                attention_mask=
                    attention_mask,

            ).logits

            loss = criterion(
                logits,
                labels,
            )

            if (
                torch.isnan(loss)
                or torch.isinf(loss)
            ):

                raise RuntimeError(

                    f"Invalid loss: "
                    f"{loss.item()} | "
                    f"Round {round_num} | "
                    f"Client {client_id + 1} | "
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx + 1}"
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        avg_loss = (
            total_loss
            / max(
                len(loader),
                1,
            )
        )

        epoch_losses.append(
            avg_loss
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(

            f"[Round {round_num:02d} | "
            f"Client {client_id + 1} "
            f"{CLIENT_NAMES[client_id]:<12} | "
            f"Epoch {epoch + 1}/{local_epochs}] "

            f"Loss: {avg_loss:.4f} | "

            f"Epoch time: "
            f"{format_seconds(epoch_time)}"
        )

    return epoch_losses


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

                attention_mask=
                    attention_mask,

            ).logits

            probs = (
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]
                .detach()
                .cpu()
                .numpy()
            )

            labels = (
                batch["labels"]
                .detach()
                .cpu()
                .numpy()
            )

            all_probs.extend(
                probs
            )

            all_labels.extend(
                labels
            )

    all_probs = np.asarray(
        all_probs
    )

    all_labels = np.asarray(
        all_labels
    )

    preds = (
        all_probs >= 0.5
    ).astype(int)

    if len(
        np.unique(all_labels)
    ) < 2:

        auc_roc = np.nan
        auc_pr = np.nan

    else:

        auc_roc = (
            roc_auc_score(
                all_labels,
                all_probs,
            )
        )

        auc_pr = (
            average_precision_score(
                all_labels,
                all_probs,
            )
        )

    return {

        "auc_roc":
            auc_roc,

        "auc_pr":
            auc_pr,

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
# DataLoader helper
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

        batch_size=
            batch_size,

        shuffle=
            shuffle,

        num_workers=
            args.num_workers,

        worker_init_fn=(
            seed_worker
            if args.num_workers > 0
            else None
        ),

        generator=
            generator,

        pin_memory=
            torch.cuda.is_available(),
    )


# ============================================================
# Main
# ============================================================

def main(args):

    run_start_time = time.time()

    round_times = []

    set_seed(
        args.seed
    )

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
        AutoTokenizer
        .from_pretrained(
            "facebook/esm2_t12_35M_UR50D"
        )
    )


    # ========================================================
    # Run configuration
    # ========================================================

    print("=" * 80)

    print(
        "FedEpitope Homology-Aware "
        "Federated Training"
    )

    print("=" * 80)

    print(
        f"Device               : {device}"
    )

    print(
        f"Seed                 : {args.seed}"
    )

    print(
        f"Federated rounds     : {args.num_rounds}"
    )

    print(
        f"Local epochs         : {args.local_epochs}"
    )

    print(
        f"Batch size           : {args.batch_size}"
    )

    print(
        f"Learning rate        : {args.lr}"
    )

    print(
        f"LoRA rank            : {args.lora_rank}"
    )

    print(
        f"LoRA alpha           : {2 * args.lora_rank}"
    )

    print(
        "Aggregation          : weighted FedAvg"
    )

    print(
        "Checkpoint selection : global homology validation AUROC"
    )

    print(
        "Global test usage    : final evaluation only"
    )

    print(
        f"Global validation    : {GLOBAL_VAL_FILE}"
    )

    print(
        f"Global test          : {GLOBAL_TEST_FILE}"
    )

    print(
        f"Results directory    : {args.results_dir}"
    )

    print("=" * 80)

    print()

    print_gpu_memory()

    print()


    # ========================================================
    # Build client data loaders
    # ========================================================

    train_loaders = []
    local_val_loaders = []

    class_weights_list = []

    client_train_sizes = []
    client_val_sizes = []


    for client_id in range(
        NUM_CLIENTS
    ):

        train_dataset = (
            EpitopeDataset(

                CLIENT_TRAIN_FILES[
                    client_id
                ],

                tokenizer,
            )
        )

        val_dataset = (
            EpitopeDataset(

                CLIENT_VAL_FILES[
                    client_id
                ],

                tokenizer,
            )
        )

        train_df = pd.read_csv(

            CLIENT_TRAIN_FILES[
                client_id
            ]
        )

        pos = int(
            train_df["label"]
            .sum()
        )

        neg = int(
            len(train_df)
            - pos
        )

        positive_weight = (
            neg / max(pos, 1)
        )

        class_weights = (
            torch.tensor(

                [
                    1.0,
                    positive_weight,
                ],

                dtype=torch.float,

            ).to(device)
        )


        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            args.seed
            + client_id
        )


        train_loader = make_loader(

            train_dataset,

            batch_size=
                args.batch_size,

            shuffle=True,

            args=args,

            generator=
                generator,
        )


        val_loader = make_loader(

            val_dataset,

            batch_size=128,

            shuffle=False,

            args=args,
        )


        train_loaders.append(
            train_loader
        )

        local_val_loaders.append(
            val_loader
        )

        class_weights_list.append(
            class_weights
        )

        client_train_sizes.append(
            len(train_dataset)
        )

        client_val_sizes.append(
            len(val_dataset)
        )


        print(

            f"Client {client_id + 1} "
            f"({CLIENT_NAMES[client_id]:<12}) | "

            f"Train: "
            f"{len(train_dataset):>7,} | "

            f"Local val: "
            f"{len(val_dataset):>6,} | "

            f"Positive: "
            f"{pos:>6,} | "

            f"Negative: "
            f"{neg:>7,} | "

            f"Positive rate: "
            f"{pos / len(train_df):.2%} | "

            f"Class weight: "
            f"{positive_weight:.2f}"
        )


    # ========================================================
    # Global validation loader
    # ========================================================

    global_val_dataset = (
        EpitopeDataset(
            GLOBAL_VAL_FILE,
            tokenizer,
        )
    )

    global_val_loader = (
        make_loader(

            global_val_dataset,

            batch_size=128,

            shuffle=False,

            args=args,
        )
    )


    print()

    print(
        "Global homology validation samples:",
        f"{len(global_val_dataset):,}"
    )


    # ========================================================
    # Build initial global model
    # ========================================================

    global_model = (
        build_model(
            args.lora_rank
        )
        .to(device)
    )

    print()

    print(
        "Trainable parameter summary:"
    )

    print_trainable_summary(
        global_model
    )

    print()


    # ========================================================
    # Logs
    # ========================================================

    val_results = {

        "round": [],

        "global_val_auc_roc": [],

        "global_val_auc_pr": [],

        "global_val_f1": [],

        "global_val_mcc": [],

        "round_time_seconds": [],

        "elapsed_seconds": [],

        "eta_seconds": [],
    }


    client_round_logs = []


    best_val_auc = -np.inf

    best_round = 0

    best_weights = None


    # ========================================================
    # Federated rounds
    # ========================================================

    for round_num in range(
        1,
        args.num_rounds + 1
    ):

        round_start_time = (
            time.time()
        )


        print()

        print("=" * 80)

        print(
            f"ROUND "
            f"{round_num}/"
            f"{args.num_rounds}"
        )

        print("=" * 80)


        client_weights_list = []


        # ----------------------------------------------------
        # Local client training
        # ----------------------------------------------------

        for client_id in range(
            NUM_CLIENTS
        ):

            client_start_time = (
                time.time()
            )


            print()

            print("-" * 80)

            print(

                f"Training client "
                f"{client_id + 1}: "
                f"{CLIENT_NAMES[client_id]}"
            )

            print("-" * 80)


            # Each client starts from
            # current global model.
            local_model = (
                copy.deepcopy(
                    global_model
                )
                .to(device)
            )


            optimizer = (
                torch.optim.AdamW(

                    [
                        p
                        for p
                        in local_model.parameters()
                        if p.requires_grad
                    ],

                    lr=args.lr,
                )
            )


            epoch_losses = (
                train_local(

                    model=
                        local_model,

                    loader=
                        train_loaders[
                            client_id
                        ],

                    optimizer=
                        optimizer,

                    device=
                        device,

                    class_weights=
                        class_weights_list[
                            client_id
                        ],

                    local_epochs=
                        args.local_epochs,

                    round_num=
                        round_num,

                    client_id=
                        client_id,
                )
            )


            # -----------------------------------------------
            # Local validation is monitoring only
            # -----------------------------------------------

            local_val_metrics = (
                evaluate(

                    local_model,

                    local_val_loaders[
                        client_id
                    ],

                    device,
                )
            )


            client_time = (
                time.time()
                - client_start_time
            )


            print(

                f"[Round {round_num:02d} | "
                f"Client {client_id + 1} "
                f"{CLIENT_NAMES[client_id]:<12} | "
                f"Local validation] "

                f"AUC-ROC: "
                f"{local_val_metrics['auc_roc']:.4f} | "

                f"AUC-PR: "
                f"{local_val_metrics['auc_pr']:.4f} | "

                f"F1: "
                f"{local_val_metrics['f1']:.4f} | "

                f"MCC: "
                f"{local_val_metrics['mcc']:.4f} | "

                f"Client time: "
                f"{format_seconds(client_time)}"
            )


            log_row = {

                "round":
                    round_num,

                "client_id":
                    client_id + 1,

                "client_name":
                    CLIENT_NAMES[
                        client_id
                    ],

                "local_val_auc_roc":
                    local_val_metrics[
                        "auc_roc"
                    ],

                "local_val_auc_pr":
                    local_val_metrics[
                        "auc_pr"
                    ],

                "local_val_f1":
                    local_val_metrics[
                        "f1"
                    ],

                "local_val_mcc":
                    local_val_metrics[
                        "mcc"
                    ],

                "client_time_seconds":
                    client_time,
            }


            for epoch_index in range(
                args.local_epochs
            ):

                key = (
                    f"epoch_"
                    f"{epoch_index + 1}_loss"
                )

                log_row[key] = (

                    epoch_losses[
                        epoch_index
                    ]

                    if epoch_index
                    < len(epoch_losses)

                    else np.nan
                )


            client_round_logs.append(
                log_row
            )


            client_weights_list.append(

                get_trainable_weights(
                    local_model
                )
            )


            del local_model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        # ----------------------------------------------------
        # Weighted FedAvg
        # ----------------------------------------------------

        print()

        print(
            "Aggregating client models "
            "using weighted FedAvg..."
        )


        averaged_weights = (
            federated_averaging(

                client_weights_list=
                    client_weights_list,

                client_sizes=
                    client_train_sizes,
            )
        )


        global_model = (
            set_trainable_weights(

                global_model,

                averaged_weights,
            )
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Evaluate aggregated model on GLOBAL validation
        # ----------------------------------------------------

        print()

        print(
            "Evaluating aggregated global model "
            "on homology-disjoint GLOBAL validation..."
        )


        global_val_metrics = (
            evaluate(

                global_model,

                global_val_loader,

                device,
            )
        )


        # ----------------------------------------------------
        # Checkpoint selection
        # ----------------------------------------------------

        if (

            not np.isnan(
                global_val_metrics[
                    "auc_roc"
                ]
            )

            and

            global_val_metrics[
                "auc_roc"
            ]
            > best_val_auc

        ):

            best_val_auc = (
                global_val_metrics[
                    "auc_roc"
                ]
            )

            best_round = (
                round_num
            )

            best_weights = (
                get_trainable_weights(
                    global_model
                )
            )


            torch.save(

                best_weights,

                os.path.join(

                    args.results_dir,

                    "best_global_weights.pt",
                )
            )


            best_tag = (
                " <-- best validation checkpoint"
            )

        else:

            best_tag = ""


        # ----------------------------------------------------
        # Runtime monitoring
        # ----------------------------------------------------

        round_end_time = (
            time.time()
        )

        round_duration = (
            round_end_time
            - round_start_time
        )

        round_times.append(
            round_duration
        )

        elapsed_time = (
            round_end_time
            - run_start_time
        )

        avg_round_time = (
            sum(round_times)
            / len(round_times)
        )

        remaining_rounds = (
            args.num_rounds
            - round_num
        )

        eta_seconds = (
            avg_round_time
            * remaining_rounds
        )


        # ----------------------------------------------------
        # Save validation logs
        # ----------------------------------------------------

        val_results[
            "round"
        ].append(
            round_num
        )

        val_results[
            "global_val_auc_roc"
        ].append(

            global_val_metrics[
                "auc_roc"
            ]
        )

        val_results[
            "global_val_auc_pr"
        ].append(

            global_val_metrics[
                "auc_pr"
            ]
        )

        val_results[
            "global_val_f1"
        ].append(

            global_val_metrics[
                "f1"
            ]
        )

        val_results[
            "global_val_mcc"
        ].append(

            global_val_metrics[
                "mcc"
            ]
        )

        val_results[
            "round_time_seconds"
        ].append(
            round_duration
        )

        val_results[
            "elapsed_seconds"
        ].append(
            elapsed_time
        )

        val_results[
            "eta_seconds"
        ].append(
            eta_seconds
        )


        print()

        print("=" * 80)

        print(

            f"GLOBAL MODEL | "
            f"Round {round_num:02d} | "

            f"VAL AUC-ROC: "
            f"{global_val_metrics['auc_roc']:.4f} | "

            f"VAL AUC-PR: "
            f"{global_val_metrics['auc_pr']:.4f} | "

            f"VAL F1: "
            f"{global_val_metrics['f1']:.4f} | "

            f"VAL MCC: "
            f"{global_val_metrics['mcc']:.4f}"

            f"{best_tag}"
        )

        print("=" * 80)


        print()

        print(
            "Progress monitor"
        )

        print(
            f"  Finished round       : "
            f"{round_num}/"
            f"{args.num_rounds}"
        )

        print(
            f"  Round time           : "
            f"{format_seconds(round_duration)}"
        )

        print(
            f"  Elapsed time         : "
            f"{format_seconds(elapsed_time)}"
        )

        print(
            f"  Estimated time left : "
            f"{format_seconds(eta_seconds)}"
        )

        print(
            f"  Best validation AUC : "
            f"{best_val_auc:.4f}"
        )

        print(
            f"  Best round so far   : "
            f"{best_round}"
        )

        print_gpu_memory()


        # ----------------------------------------------------
        # Save live files after every round
        # ----------------------------------------------------

        live_val_path = (
            os.path.join(

                args.results_dir,

                "federated_val_rounds_live.csv",
            )
        )

        live_client_path = (
            os.path.join(

                args.results_dir,

                "client_round_logs_live.csv",
            )
        )


        pd.DataFrame(
            val_results
        ).to_csv(
            live_val_path,
            index=False,
        )


        pd.DataFrame(
            client_round_logs
        ).to_csv(
            live_client_path,
            index=False,
        )


        print(
            f"  Live validation log : "
            f"{live_val_path}"
        )

        print(
            f"  Live client log     : "
            f"{live_client_path}"
        )


    # ========================================================
    # Save final round logs
    # ========================================================

    val_rounds_path = (
        os.path.join(

            args.results_dir,

            "federated_val_rounds.csv",
        )
    )

    client_logs_path = (
        os.path.join(

            args.results_dir,

            "client_round_logs.csv",
        )
    )


    pd.DataFrame(
        val_results
    ).to_csv(
        val_rounds_path,
        index=False,
    )


    pd.DataFrame(
        client_round_logs
    ).to_csv(
        client_logs_path,
        index=False,
    )


    # ========================================================
    # Save final-round weights
    # ========================================================

    final_weights_path = (
        os.path.join(

            args.results_dir,

            "global_weights_final.pt",
        )
    )


    torch.save(

        get_trainable_weights(
            global_model
        ),

        final_weights_path,
    )


    if best_weights is None:

        raise RuntimeError(
            "No best validation checkpoint was saved."
        )


    # ========================================================
    # FINAL TEST
    # ========================================================
    #
    # Test set is loaded only now,
    # AFTER all federated training and checkpoint selection.
    #
    # ========================================================

    print()

    print("=" * 80)

    print(
        "Loading best validation checkpoint "
        "for final homology-disjoint test evaluation"
    )

    print("=" * 80)


    global_model = (
        set_trainable_weights(

            global_model,

            best_weights,
        )
    )


    test_dataset = (
        EpitopeDataset(

            GLOBAL_TEST_FILE,

            tokenizer,
        )
    )


    test_loader = (
        make_loader(

            test_dataset,

            batch_size=128,

            shuffle=False,

            args=args,
        )
    )


    print(
        "Homology-disjoint global test samples:",
        f"{len(test_dataset):,}"
    )

    print(
        "Evaluating global test exactly once..."
    )


    test_metrics = (
        evaluate(

            global_model,

            test_loader,

            device,
        )
    )


    total_run_time = (
        time.time()
        - run_start_time
    )


    # ========================================================
    # Save final test result
    # ========================================================

    test_summary = {

        "seed":
            args.seed,

        "lora_rank":
            args.lora_rank,

        "lora_alpha":
            2 * args.lora_rank,

        "num_rounds":
            args.num_rounds,

        "local_epochs":
            args.local_epochs,

        "selected_round":
            best_round,

        "best_val_auc_roc":
            best_val_auc,

        "test_auc_roc":
            test_metrics[
                "auc_roc"
            ],

        "test_auc_pr":
            test_metrics[
                "auc_pr"
            ],

        "test_f1":
            test_metrics[
                "f1"
            ],

        "test_mcc":
            test_metrics[
                "mcc"
            ],

        "total_run_time_seconds":
            total_run_time,
    }


    test_results_path = (
        os.path.join(

            args.results_dir,

            "federated_test_results.csv",
        )
    )


    pd.DataFrame(
        [test_summary]
    ).to_csv(

        test_results_path,

        index=False,
    )


    # ========================================================
    # Final summary
    # ========================================================

    print()

    print("=" * 80)

    print(
        "Homology-aware federated training complete"
    )

    print("=" * 80)

    print(
        f"Best global validation AUC-ROC : "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Selected federated round       : "
        f"{best_round}"
    )

    print(
        f"Final test AUC-ROC             : "
        f"{test_metrics['auc_roc']:.4f}"
    )

    print(
        f"Final test AUC-PR              : "
        f"{test_metrics['auc_pr']:.4f}"
    )

    print(
        f"Final test F1                  : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Final test MCC                 : "
        f"{test_metrics['mcc']:.4f}"
    )

    print(
        f"Total run time                 : "
        f"{format_seconds(total_run_time)}"
    )

    print()

    print(
        f"Saved best weights      : "
        f"{os.path.join(args.results_dir, 'best_global_weights.pt')}"
    )

    print(
        f"Saved final weights     : "
        f"{final_weights_path}"
    )

    print(
        f"Saved validation rounds : "
        f"{val_rounds_path}"
    )

    print(
        f"Saved client logs       : "
        f"{client_logs_path}"
    )

    print(
        f"Saved final test result : "
        f"{test_results_path}"
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
        "--num_rounds",
        type=int,
        default=NUM_ROUNDS,
    )


    parser.add_argument(
        "--local_epochs",
        type=int,
        default=LOCAL_EPOCHS,
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
        "--lora_rank",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )


    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/fedavg_seed42",
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
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "Homology-aware federated training failed.\n\n"
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


        print()

        print("=" * 80)

        print(
            "Training failed"
        )

        print("=" * 80)

        print(
            f"Error: {exc}"
        )

        print(
            f"Full traceback saved to: "
            f"{error_path}"
        )

        print("=" * 80)

        raise