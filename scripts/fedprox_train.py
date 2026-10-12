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
# DEFAULT CONFIGURATION
# ============================================================

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS = 5

BATCH_SIZE = 64
LR = 2e-4
MAX_LEN = 30
DEFAULT_MU = 0.01


# ============================================================
# HOMOLOGY-AWARE DATA
# ============================================================

CLIENT_TRAIN_FILES = [
    f"data/federated/client{i}_train.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

CLIENT_VAL_FILES = [
    f"data/federated/client{i}_val.csv"
    for i in range(1, NUM_CLIENTS + 1)
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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# UTILITIES
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


def print_gpu_memory():

    if not torch.cuda.is_available():

        print(
            "GPU memory | CUDA not available"
        )

        return

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


# ============================================================
# DATASET
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
                f"{csv_path} missing sequence column"
            )

        if "label" not in df.columns:
            raise ValueError(
                f"{csv_path} missing label column"
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
# MODEL
# ============================================================

def build_model(lora_rank):

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
# TRAINABLE PARAMETERS
# ============================================================

def get_trainable_weights(model):

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
    weights,
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


def get_global_reference(model):

    """
    Fixed reference parameters for FedProx.

    Called immediately after the local model is copied
    from the current global model.

    Only trainable parameters are needed because the
    ESM-2 backbone is frozen.
    """

    return {

        name:
            param
            .detach()
            .clone()

        for name, param
        in model.named_parameters()

        if param.requires_grad
    }


def print_trainable_summary(model):

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
        for name, p
        in model.named_parameters()
        if (
            p.requires_grad
            and "lora" in name.lower()
        )
    )

    print(
        f"Total parameters          : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters      : "
        f"{trainable:,}"
    )

    print(
        f"LoRA trainable parameters : "
        f"{lora:,}"
    )

    print(
        f"Trainable percentage      : "
        f"{100 * trainable / total:.4f}%"
    )


# ============================================================
# FEDERATED AVERAGING
# ============================================================

def weighted_federated_average(
    client_weights,
    client_sizes,
):

    total_size = sum(
        client_sizes
    )

    proportions = [

        size / total_size

        for size in client_sizes
    ]

    averaged = {}

    for key in (
        client_weights[0]
        .keys()
    ):

        averaged[key] = sum(

            proportions[i]
            * client_weights[i][key]
            .float()

            for i in range(
                len(client_weights)
            )
        )

    return averaged


# ============================================================
# FEDPROX LOCAL TRAINING
# ============================================================

def train_local_fedprox(
    model,
    loader,
    optimizer,
    device,
    class_weights,
    local_epochs,
    round_num,
    client_id,
    mu,
    global_reference,
):

    """
    FedProx objective:

        CE_loss
        +
        (mu / 2) * ||w_local - w_global||^2

    Only trainable parameters are included.
    """

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    epoch_logs = []


    for epoch in range(
        local_epochs
    ):

        model.train()

        epoch_start = time.time()

        total_cls_loss = 0.0
        total_prox_loss = 0.0
        total_prox_distance = 0.0
        total_objective = 0.0


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


            classification_loss = (
                criterion(
                    logits,
                    labels,
                )
            )


            # =================================================
            # FEDPROX PENALTY
            # =================================================

            proximal_distance = (
                torch.zeros(
                    (),
                    device=device,
                )
            )


            for name, param in (
                model.named_parameters()
            ):

                if not param.requires_grad:
                    continue

                reference = (
                    global_reference[name]
                )

                proximal_distance = (
                    proximal_distance
                    +
                    torch.sum(
                        (
                            param
                            - reference
                        ) ** 2
                    )
                )


            proximal_loss = (
                0.5
                * mu
                * proximal_distance
            )


            loss = (
                classification_loss
                +
                proximal_loss
            )


            if (
                torch.isnan(loss)
                or torch.isinf(loss)
            ):

                raise RuntimeError(

                    f"Invalid loss | "
                    f"round={round_num} | "
                    f"client={client_id + 1} | "
                    f"epoch={epoch + 1} | "
                    f"batch={batch_idx + 1}"
                )


            loss.backward()

            optimizer.step()


            total_cls_loss += (
                classification_loss.item()
            )

            total_prox_loss += (
                proximal_loss.item()
            )

            total_prox_distance += (
                proximal_distance.item()
            )

            total_objective += (
                loss.item()
            )


        n_batches = max(
            len(loader),
            1,
        )


        avg_cls = (
            total_cls_loss
            / n_batches
        )

        avg_prox = (
            total_prox_loss
            / n_batches
        )

        avg_prox_distance = (
            total_prox_distance
            / n_batches
        )

        avg_total = (
            total_objective
            / n_batches
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

            f"Cls: {avg_cls:.4f} | "

            f"Prox distance: "
            f"{avg_prox_distance:.6f} | "

            f"Prox loss: "
            f"{avg_prox:.6f} | "

            f"Total: "
            f"{avg_total:.4f} | "

            f"Time: "
            f"{format_seconds(epoch_time)}"
        )


        epoch_logs.append({

            "classification_loss":
                avg_cls,

            "proximal_distance":
                avg_prox_distance,

            "proximal_loss":
                avg_prox,

            "total_loss":
                avg_total,
        })


    return epoch_logs


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    probs_all = []
    labels_all = []


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


            probs_all.extend(
                probs
            )

            labels_all.extend(
                labels
            )


    probs_all = np.asarray(
        probs_all
    )

    labels_all = np.asarray(
        labels_all
    )


    preds = (
        probs_all >= 0.5
    ).astype(int)


    if len(
        np.unique(labels_all)
    ) < 2:

        auc_roc = np.nan
        auc_pr = np.nan

    else:

        auc_roc = (
            roc_auc_score(
                labels_all,
                probs_all,
            )
        )

        auc_pr = (
            average_precision_score(
                labels_all,
                probs_all,
            )
        )


    return {

        "auc_roc":
            auc_roc,

        "auc_pr":
            auc_pr,

        "f1":
            f1_score(
                labels_all,
                preds,
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                labels_all,
                preds,
            ),
    }


# ============================================================
# DATALOADER
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
# MAIN
# ============================================================

def main(args):

    run_start = time.time()

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
    # CONFIGURATION
    # ========================================================

    print("=" * 80)

    print(
        "FedEpitope Homology-Aware "
        "FedProx Training"
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
        f"LoRA alpha           : "
        f"{2 * args.lora_rank}"
    )

    print(
        f"FedProx mu           : {args.mu}"
    )

    print(
        "Aggregation          : "
        "weighted parameter averaging"
    )

    print(
        "Local objective      : "
        "weighted CE + "
        "(mu/2)||w_local-w_global||^2"
    )

    print(
        "Checkpoint selection : "
        "global validation AUROC"
    )

    if args.skip_test:

        print(
            "Global test usage    : "
            "DISABLED (mu tuning)"
        )

    else:

        print(
            "Global test usage    : "
            "final evaluation only"
        )

    print(
        f"Results directory    : "
        f"{args.results_dir}"
    )

    print("=" * 80)


    # ========================================================
    # CLIENT DATA
    # ========================================================

    train_loaders = []
    local_val_loaders = []

    class_weights_list = []
    client_train_sizes = []


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
            neg
            / max(pos, 1)
        )


        class_weights = (
            torch.tensor(

                [
                    1.0,
                    positive_weight,
                ],

                dtype=torch.float,

                device=device,
            )
        )


        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            args.seed
            + client_id
        )


        train_loader = (
            make_loader(

                train_dataset,

                args.batch_size,

                True,

                args,

                generator,
            )
        )


        val_loader = (
            make_loader(

                val_dataset,

                128,

                False,

                args,
            )
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


        print(

            f"Client {client_id + 1} "
            f"({CLIENT_NAMES[client_id]:<12}) | "

            f"Train: "
            f"{len(train_dataset):>7,} | "

            f"Local val: "
            f"{len(val_dataset):>6,} | "

            f"Positive rate: "
            f"{pos / len(train_df):.2%} | "

            f"Class weight: "
            f"{positive_weight:.2f}"
        )


    # ========================================================
    # GLOBAL VALIDATION
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

            128,

            False,

            args,
        )
    )


    print()

    print(
        "Global validation samples:",
        f"{len(global_val_dataset):,}"
    )


    # ========================================================
    # IMPORTANT:
    # RESET RNG IMMEDIATELY BEFORE MODEL CREATION
    # ========================================================

    set_seed(
        args.seed
    )


    global_model = (

        build_model(
            args.lora_rank
        )

        .to(device)
    )


    print()

    print_trainable_summary(
        global_model
    )


    # ========================================================
    # LOGGING
    # ========================================================

    val_rows = []
    client_rows = []

    best_val_auc = -np.inf
    best_round = None
    best_weights = None

    round_times = []


    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    for round_num in range(
        1,
        args.num_rounds + 1
    ):

        round_start = (
            time.time()
        )


        print()

        print("=" * 80)

        print(
            f"FEDPROX ROUND "
            f"{round_num}/"
            f"{args.num_rounds}"
        )

        print("=" * 80)


        client_weights = []


        for client_id in range(
            NUM_CLIENTS
        ):

            client_start = (
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


            # =================================================
            # CLIENT STARTS FROM CURRENT GLOBAL MODEL
            # =================================================

            local_model = (

                copy.deepcopy(
                    global_model
                )

                .to(device)
            )


            # =================================================
            # FIXED GLOBAL REFERENCE FOR FEDPROX
            # =================================================

            global_reference = (
                get_global_reference(
                    local_model
                )
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


            epoch_logs = (
                train_local_fedprox(

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

                    mu=
                        args.mu,

                    global_reference=
                        global_reference,
                )
            )


            # =================================================
            # LOCAL VAL = MONITORING ONLY
            # =================================================

            local_metrics = (
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
                - client_start
            )


            print(

                f"[Round {round_num:02d} | "
                f"Client {client_id + 1} | "
                f"Local validation] "

                f"AUC-ROC: "
                f"{local_metrics['auc_roc']:.4f} | "

                f"AUC-PR: "
                f"{local_metrics['auc_pr']:.4f} | "

                f"F1: "
                f"{local_metrics['f1']:.4f} | "

                f"MCC: "
                f"{local_metrics['mcc']:.4f}"
            )


            row = {

                "round":
                    round_num,

                "client_id":
                    client_id + 1,

                "client_name":
                    CLIENT_NAMES[
                        client_id
                    ],

                "local_val_auc_roc":
                    local_metrics[
                        "auc_roc"
                    ],

                "local_val_auc_pr":
                    local_metrics[
                        "auc_pr"
                    ],

                "local_val_f1":
                    local_metrics[
                        "f1"
                    ],

                "local_val_mcc":
                    local_metrics[
                        "mcc"
                    ],

                "client_time_seconds":
                    client_time,
            }


            for epoch_index, log in (
                enumerate(
                    epoch_logs,
                    start=1,
                )
            ):

                row[
                    f"epoch_{epoch_index}_classification_loss"
                ] = log[
                    "classification_loss"
                ]

                row[
                    f"epoch_{epoch_index}_proximal_distance"
                ] = log[
                    "proximal_distance"
                ]

                row[
                    f"epoch_{epoch_index}_proximal_loss"
                ] = log[
                    "proximal_loss"
                ]

                row[
                    f"epoch_{epoch_index}_total_loss"
                ] = log[
                    "total_loss"
                ]


            client_rows.append(
                row
            )


            client_weights.append(

                get_trainable_weights(
                    local_model
                )
            )


            del local_model
            del global_reference


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


        # ====================================================
        # WEIGHTED AGGREGATION
        # ====================================================

        averaged = (
            weighted_federated_average(

                client_weights,

                client_train_sizes,
            )
        )


        global_model = (
            set_trainable_weights(

                global_model,

                averaged,
            )
        )


        # ====================================================
        # GLOBAL VALIDATION
        # ====================================================

        global_metrics = (
            evaluate(

                global_model,

                global_val_loader,

                device,
            )
        )


        if (

            not np.isnan(
                global_metrics[
                    "auc_roc"
                ]
            )

            and

            global_metrics[
                "auc_roc"
            ]
            > best_val_auc

        ):

            best_val_auc = (
                global_metrics[
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


        # ====================================================
        # TIME
        # ====================================================

        round_time = (
            time.time()
            - round_start
        )


        round_times.append(
            round_time
        )


        elapsed = (
            time.time()
            - run_start
        )


        avg_round_time = (
            sum(round_times)
            / len(round_times)
        )


        eta = (

            avg_round_time

            * (
                args.num_rounds
                - round_num
            )
        )


        val_rows.append({

            "round":
                round_num,

            "mu":
                args.mu,

            "global_val_auc_roc":
                global_metrics[
                    "auc_roc"
                ],

            "global_val_auc_pr":
                global_metrics[
                    "auc_pr"
                ],

            "global_val_f1":
                global_metrics[
                    "f1"
                ],

            "global_val_mcc":
                global_metrics[
                    "mcc"
                ],

            "round_time_seconds":
                round_time,

            "elapsed_seconds":
                elapsed,

            "eta_seconds":
                eta,
        })


        print()

        print("=" * 80)


        print(

            f"GLOBAL FEDPROX MODEL | "
            f"Round {round_num:02d} | "

            f"VAL AUC-ROC: "
            f"{global_metrics['auc_roc']:.4f} | "

            f"VAL AUC-PR: "
            f"{global_metrics['auc_pr']:.4f} | "

            f"VAL F1: "
            f"{global_metrics['f1']:.4f} | "

            f"VAL MCC: "
            f"{global_metrics['mcc']:.4f}"

            f"{best_tag}"
        )


        print("=" * 80)


        print(
            f"Round time           : "
            f"{format_seconds(round_time)}"
        )

        print(
            f"Elapsed time         : "
            f"{format_seconds(elapsed)}"
        )

        print(
            f"Estimated time left : "
            f"{format_seconds(eta)}"
        )

        print(
            f"Best validation AUC : "
            f"{best_val_auc:.4f}"
        )

        print(
            f"Best round so far   : "
            f"{best_round}"
        )

        print_gpu_memory()


        # ====================================================
        # LIVE LOGS
        # ====================================================

        pd.DataFrame(
            val_rows
        ).to_csv(

            os.path.join(

                args.results_dir,

                "fedprox_val_rounds_live.csv",
            ),

            index=False,
        )


        pd.DataFrame(
            client_rows
        ).to_csv(

            os.path.join(

                args.results_dir,

                "client_round_logs_live.csv",
            ),

            index=False,
        )


    # ========================================================
    # SAVE FINAL TRAINING LOGS
    # ========================================================

    val_path = os.path.join(

        args.results_dir,

        "fedprox_val_rounds.csv",
    )


    client_path = os.path.join(

        args.results_dir,

        "client_round_logs.csv",
    )


    pd.DataFrame(
        val_rows
    ).to_csv(
        val_path,
        index=False,
    )


    pd.DataFrame(
        client_rows
    ).to_csv(
        client_path,
        index=False,
    )


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
            "No best validation checkpoint was created."
        )


    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    validation_summary = {

        "seed":
            args.seed,

        "method":
            "FedProx",

        "mu":
            args.mu,

        "lora_rank":
            args.lora_rank,

        "num_rounds":
            args.num_rounds,

        "local_epochs":
            args.local_epochs,

        "selected_round":
            best_round,

        "best_val_auc_roc":
            best_val_auc,
    }


    validation_summary_path = (
        os.path.join(

            args.results_dir,

            "fedprox_validation_summary.csv",
        )
    )


    pd.DataFrame(
        [validation_summary]
    ).to_csv(

        validation_summary_path,

        index=False,
    )


    print()

    print("=" * 80)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 80)

    print(
        f"Mu                       : "
        f"{args.mu}"
    )

    print(
        f"Best validation AUROC    : "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Selected round           : "
        f"{best_round}"
    )


    # ========================================================
    # MU TUNING MODE:
    # STOP HERE - NO TEST ACCESS
    # ========================================================

    if args.skip_test:

        total_time = (
            time.time()
            - run_start
        )

        print()

        print(
            "TEST EVALUATION SKIPPED."
        )

        print(
            "This run is safe for "
            "FedProx mu selection."
        )

        print(
            f"Total runtime            : "
            f"{format_seconds(total_time)}"
        )

        print(
            f"Validation summary       : "
            f"{validation_summary_path}"
        )

        print("=" * 80)

        return


    # ========================================================
    # FINAL TEST
    # ========================================================

    global_model = (
        set_trainable_weights(

            global_model,

            best_weights,
        )
    )


    print()

    print("=" * 80)

    print(
        "FINAL HOMOLOGY-DISJOINT TEST"
    )

    print("=" * 80)


    test_dataset = (
        EpitopeDataset(

            GLOBAL_TEST_FILE,

            tokenizer,
        )
    )


    test_loader = (
        make_loader(

            test_dataset,

            128,

            False,

            args,
        )
    )


    test_metrics = (
        evaluate(

            global_model,

            test_loader,

            device,
        )
    )


    total_time = (
        time.time()
        - run_start
    )


    result = {

        "seed":
            args.seed,

        "method":
            "FedProx",

        "mu":
            args.mu,

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
            total_time,
    }


    result_path = os.path.join(

        args.results_dir,

        "fedprox_test_results.csv",
    )


    pd.DataFrame(
        [result]
    ).to_csv(
        result_path,
        index=False,
    )


    print(
        f"Best validation AUROC : "
        f"{best_val_auc:.4f}"
    )

    print(
        f"Selected round        : "
        f"{best_round}"
    )

    print(
        f"Test AUROC            : "
        f"{test_metrics['auc_roc']:.4f}"
    )

    print(
        f"Test AUPRC            : "
        f"{test_metrics['auc_pr']:.4f}"
    )

    print(
        f"Test F1               : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test MCC              : "
        f"{test_metrics['mcc']:.4f}"
    )

    print(
        f"Total runtime         : "
        f"{format_seconds(total_time)}"
    )

    print(
        f"Saved result          : "
        f"{result_path}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
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
        "--mu",
        type=float,
        default=DEFAULT_MU,
    )


    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )


    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/fedprox_seed42",
    )


    parser.add_argument(

        "--skip_test",

        action="store_true",

        help=(
            "Do not access the global test set. "
            "Use this during FedProx mu tuning."
        ),
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
                "FedProx training failed.\n\n"
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

        print(
            f"Training failed: {exc}"
        )

        print(
            f"Traceback saved to: "
            f"{error_path}"
        )

        raise