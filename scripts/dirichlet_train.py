import os
import copy
import time
import random
import argparse
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import federated_train as base


# ============================================================
# Fixed configuration
# ============================================================

NUM_CLIENTS = 5  # overwritten from --num_clients in main()

GLOBAL_VAL_FILE = "data/federated/global_val.csv"
GLOBAL_TEST_FILE = "data/federated/global_test.csv"

EXPECTED_TOTAL_TRAIN = 321368

DEFAULT_ROUNDS = 20
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 2e-4
DEFAULT_LORA_RANK = 2

SUPPORTED_ALPHAS = (
    0.1,
    0.5,
    1.0,
    10.0,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
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
# Utilities
# ============================================================

def alpha_slug(alpha):

    return (
        str(alpha)
        .replace(".", "p")
    )


def format_seconds(seconds):

    if (
        seconds is None
        or not np.isfinite(seconds)
    ):
        return "unknown"

    return str(
        timedelta(
            seconds=int(seconds)
        )
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

    maximum = (
        torch.cuda.max_memory_allocated()
        / 1024 ** 3
    )

    print(
        f"GPU memory | "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"max={maximum:.2f} GB"
    )


def get_positive_weight(positives, negatives):

    # Under extreme Dirichlet skew, a client can legitimately
    # contain only one class. The positive-class weight is unused
    # when no positive samples are present, so use a neutral value
    # rather than dividing by zero.
    if positives <= 0 or negatives <= 0:
        return 1.0

    return negatives / positives


def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    generator=None,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=(
            seed_worker
            if num_workers > 0
            else None
        ),
        generator=generator,
        pin_memory=
            torch.cuda.is_available(),
    )


# ============================================================
# Paths
# ============================================================

def get_client_files(alpha, partition_root):

    directory = os.path.join(
        partition_root,
        f"alpha_{alpha_slug(alpha)}",
    )

    client_files = [
        os.path.join(
            directory,
            f"client{i}_train.csv",
        )
        for i in range(
            1,
            NUM_CLIENTS + 1,
        )
    ]

    return (
        directory,
        client_files,
    )


def check_required_files(alpha, partition_root):

    directory, client_files = (
        get_client_files(alpha, partition_root)
    )

    required = (
        client_files
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

    return (
        directory,
        client_files,
    )


def load_partition_metadata(partition_directory):

    metadata_path = os.path.join(
        partition_directory,
        "partition_metadata.json",
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing partition metadata: {metadata_path}"
        )

    import json

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    stored_k = int(metadata.get("num_clients", -1))
    if stored_k != NUM_CLIENTS:
        raise RuntimeError(
            f"Partition metadata reports K={stored_k}, but "
            f"--num_clients={NUM_CLIENTS}."
        )

    return metadata


# ============================================================
# Validate frozen partition
# ============================================================

def validate_partition(
    client_files,
):

    frames = []
    client_sizes = []
    client_class_counts = []

    print()
    print("=" * 90)
    print(
        "Checking frozen Dirichlet partition"
    )
    print("=" * 90)

    for client_id, path in enumerate(
        client_files,
        start=1,
    ):

        df = pd.read_csv(path)

        if "sequence" not in df.columns:

            raise ValueError(
                f"{path} missing column: sequence"
            )

        if "label" not in df.columns:

            raise ValueError(
                f"{path} missing column: label"
            )

        labels = set(
            df["label"]
            .astype(int)
            .unique()
            .tolist()
        )

        if not labels.issubset(
            {0, 1}
        ):

            raise ValueError(
                f"{path} contains invalid labels: "
                f"{sorted(labels)}"
            )

        positives = int(
            (df["label"] == 1)
            .sum()
        )

        negatives = int(
            (df["label"] == 0)
            .sum()
        )

        if positives <= 0 or negatives <= 0:
            print(
                f"WARNING: Client {client_id} is single-class "
                f"(positive={positives:,}, negative={negatives:,}). "
                "This is permitted for extreme Dirichlet skew."
            )

        frames.append(
            df
        )

        client_sizes.append(
            len(df)
        )

        client_class_counts.append(
            (
                positives,
                negatives,
            )
        )

        print(
            f"Client {client_id} | "
            f"N={len(df):>6,} | "
            f"positive={positives:>6,} | "
            f"negative={negatives:>6,} | "
            f"positive rate="
            f"{positives / len(df):.4%} | "
            f"local positive weight="
            f"{get_positive_weight(positives, negatives):.4f}"
        )

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(pooled) != EXPECTED_TOTAL_TRAIN:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_TRAIN:,} "
            f"training examples; found "
            f"{len(pooled):,}."
        )

    duplicate_count = int(
        pooled["sequence"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:

        raise RuntimeError(
            f"Found {duplicate_count} duplicate "
            f"peptide sequences across clients."
        )

    pooled_positive = int(
        (pooled["label"] == 1)
        .sum()
    )

    pooled_negative = int(
        (pooled["label"] == 0)
        .sum()
    )

    print("-" * 90)

    print(
        f"Pooled samples          : "
        f"{len(pooled):,}"
    )

    print(
        f"Pooled positive         : "
        f"{pooled_positive:,}"
    )

    print(
        f"Pooled negative         : "
        f"{pooled_negative:,}"
    )

    print(
        f"Pooled positive rate    : "
        f"{pooled_positive / len(pooled):.4%}"
    )

    print(
        "Duplicate peptides      : 0"
    )

    print(
        "Partition validation    : PASS"
    )

    return (
        client_sizes,
        client_class_counts,
    )


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

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    epoch_losses = []

    for epoch in range(
        1,
        local_epochs + 1,
    ):

        epoch_start = time.time()

        model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_index, batch in enumerate(
            loader,
            start=1,
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

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss | "
                    f"round={round_num} | "
                    f"client={client_id} | "
                    f"epoch={epoch} | "
                    f"batch={batch_index}"
                )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            num_batches += 1

        mean_loss = (
            total_loss
            / max(
                num_batches,
                1,
            )
        )

        epoch_losses.append(
            mean_loss
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"[Round {round_num:02d} | "
            f"Client {client_id} | "
            f"Epoch {epoch}/{local_epochs}] "
            f"Loss={mean_loss:.4f} | "
            f"time="
            f"{format_seconds(epoch_time)}"
        )

    return epoch_losses


# ============================================================
# Main experiment
# ============================================================

def main(args):

    global NUM_CLIENTS

    NUM_CLIENTS = int(args.num_clients)

    if args.partition_root is None:
        # Preserve the existing K=5 location for backward compatibility.
        args.partition_root = (
            "data/dirichlet"
            if NUM_CLIENTS == 5
            else f"data/dirichlet_k{NUM_CLIENTS}"
        )

    run_start = time.time()

    set_seed(
        args.seed
    )

    (
        partition_directory,
        client_files,
    ) = check_required_files(
        args.alpha,
        args.partition_root,
    )

    partition_metadata = load_partition_metadata(
        partition_directory
    )
    partition_seed = int(
        partition_metadata.get("partition_seed", -1)
    )

    if args.results_dir is None:

        if NUM_CLIENTS == 5:
            args.results_dir = (
                f"results/"
                f"dirichlet_alpha_"
                f"{alpha_slug(args.alpha)}_"
                f"seed{args.seed}"
            )
        else:
            args.results_dir = (
                f"results/"
                f"dirichlet_k{NUM_CLIENTS}_"
                f"alpha_{alpha_slug(args.alpha)}_"
                f"seed{args.seed}"
            )

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)

    print(
        "FedEpitope — Homology-Aware "
        "Dirichlet Label-Skew Experiment"
    )

    print("=" * 90)

    print(
        f"Device                  : "
        f"{device}"
    )

    print(
        f"Dirichlet alpha         : "
        f"{args.alpha}"
    )

    print(
        f"Synthetic clients (K)   : "
        f"{NUM_CLIENTS}"
    )

    print(
        f"Partition               : "
        f"{partition_directory}"
    )

    print(
        f"Partition seed          : "
        f"{partition_seed}"
    )

    print(
        f"Model seed              : "
        f"{args.seed}"
    )

    print(
        f"Federated rounds        : "
        f"{args.num_rounds}"
    )

    print(
        f"Local epochs            : "
        f"{args.local_epochs}"
    )

    print(
        f"Batch size              : "
        f"{args.batch_size}"
    )

    print(
        f"Learning rate           : "
        f"{args.lr}"
    )

    print(
        f"LoRA rank               : "
        f"{args.lora_rank}"
    )

    print(
        f"LoRA alpha              : "
        f"{2 * args.lora_rank}"
    )

    print(
        "Loss                    : "
        "client-specific weighted CE "
        "(same rule as main FedAvg)"
    )

    print(
        "Aggregation             : "
        "sample-size weighted FedAvg"
    )

    print(
        "Checkpoint selection    : "
        "global validation AUROC"
    )

    print(
        "Global test usage       : "
        "final evaluation only"
    )

    print(
        f"Results directory       : "
        f"{args.results_dir}"
    )

    print("=" * 90)

    # ========================================================
    # Validate partition
    # ========================================================

    (
        client_sizes,
        client_class_counts,
    ) = validate_partition(
        client_files
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            "facebook/esm2_t12_35M_UR50D"
        )
    )

    # ========================================================
    # Client loaders + SAME local weighting rule
    # as primary FedAvg
    # ========================================================

    train_loaders = []
    class_weights_list = []

    for client_index, path in enumerate(
        client_files
    ):

        dataset = (
            base.EpitopeDataset(
                path,
                tokenizer,
            )
        )

        (
            positives,
            negatives,
        ) = client_class_counts[
            client_index
        ]

        positive_weight = get_positive_weight(
            positives,
            negatives,
        )

        class_weights = (
            torch.tensor(
                [
                    1.0,
                    positive_weight,
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        generator = (
            torch.Generator()
        )

        # Same pattern used in main FedAvg:
        # seed + client index.
        generator.manual_seed(
            args.seed
            + client_index
        )

        train_loader = make_loader(
            dataset=dataset,
            batch_size=
                args.batch_size,
            shuffle=True,
            num_workers=
                args.num_workers,
            generator=generator,
        )

        train_loaders.append(
            train_loader
        )

        class_weights_list.append(
            class_weights
        )

    # ========================================================
    # Global validation
    # ========================================================

    global_val_dataset = (
        base.EpitopeDataset(
            GLOBAL_VAL_FILE,
            tokenizer,
        )
    )

    global_val_loader = (
        make_loader(
            dataset=
                global_val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=
                args.num_workers,
        )
    )

    print()
    print(
        f"Global validation samples: "
        f"{len(global_val_dataset):,}"
    )

    # ========================================================
    # Model
    # ========================================================

    global_model = (
        base.build_model(
            args.lora_rank
        )
        .to(device)
    )

    print()
    print(
        "Trainable parameter summary:"
    )

    base.print_trainable_summary(
        global_model
    )

    # ========================================================
    # Experiment state
    # ========================================================

    val_rows = []
    client_rows = []
    round_times = []

    best_val_auc = -np.inf
    best_round = None
    best_weights = None

    # ========================================================
    # Federated rounds
    # ========================================================

    for round_num in range(
        1,
        args.num_rounds + 1,
    ):

        round_start = time.time()

        print()
        print("=" * 90)

        print(
            f"ROUND {round_num}/"
            f"{args.num_rounds}"
        )

        print("=" * 90)

        client_weights_list = []

        # ----------------------------------------------------
        # Train each synthetic client
        # ----------------------------------------------------

        for client_index in range(
            NUM_CLIENTS
        ):

            client_id = (
                client_index + 1
            )

            client_start = time.time()

            (
                positives,
                negatives,
            ) = client_class_counts[
                client_index
            ]

            positive_weight = get_positive_weight(
                positives,
                negatives,
            )

            print()
            print("-" * 90)

            print(
                f"Training synthetic "
                f"client {client_id} | "
                f"N={client_sizes[client_index]:,} | "
                f"positive={positives:,} | "
                f"positive weight="
                f"{positive_weight:.4f}"
            )

            print("-" * 90)

            # Each client starts from the current
            # aggregated global model.
            local_model = (
                copy.deepcopy(
                    global_model
                )
                .to(device)
            )

            optimizer = (
                torch.optim.AdamW(
                    [
                        parameter
                        for parameter
                        in local_model.parameters()
                        if parameter.requires_grad
                    ],
                    lr=args.lr,
                )
            )

            epoch_losses = train_local(
                model=
                    local_model,

                loader=
                    train_loaders[
                        client_index
                    ],

                optimizer=
                    optimizer,

                device=
                    device,

                class_weights=
                    class_weights_list[
                        client_index
                    ],

                local_epochs=
                    args.local_epochs,

                round_num=
                    round_num,

                client_id=
                    client_id,
            )

            client_duration = (
                time.time()
                - client_start
            )

            client_weights_list.append(
                base.get_trainable_weights(
                    local_model
                )
            )

            log_row = {
                "alpha":
                    args.alpha,

                "num_clients":
                    NUM_CLIENTS,

                "partition_seed":
                    partition_seed,

                "seed":
                    args.seed,

                "round":
                    round_num,

                "client_id":
                    client_id,

                "client_size":
                    client_sizes[
                        client_index
                    ],

                "positive":
                    positives,

                "negative":
                    negatives,

                "positive_rate":
                    positives
                    / client_sizes[
                        client_index
                    ],

                "positive_class_weight":
                    positive_weight,

                "client_time_seconds":
                    client_duration,
            }

            for epoch_index, loss in enumerate(
                epoch_losses,
                start=1,
            ):

                log_row[
                    f"epoch_{epoch_index}_loss"
                ] = loss

            client_rows.append(
                log_row
            )

            del local_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Weighted FedAvg
        # ----------------------------------------------------

        print()
        print(
            "Aggregating client updates "
            "using weighted FedAvg..."
        )

        averaged_weights = (
            base.federated_averaging(
                client_weights_list,
                client_sizes,
            )
        )

        global_model = (
            base.set_trainable_weights(
                global_model,
                averaged_weights,
            )
        )

        # ----------------------------------------------------
        # GLOBAL VALIDATION ONLY
        # ----------------------------------------------------

        print(
            "Evaluating aggregated model "
            "on global homology validation..."
        )

        val_metrics = (
            base.evaluate(
                global_model,
                global_val_loader,
                device,
            )
        )

        # ----------------------------------------------------
        # Validation checkpoint selection
        # ----------------------------------------------------

        improved = (
            np.isfinite(
                val_metrics[
                    "auc_roc"
                ]
            )
            and
            val_metrics[
                "auc_roc"
            ] > best_val_auc
        )

        if improved:

            best_val_auc = float(
                val_metrics[
                    "auc_roc"
                ]
            )

            best_round = (
                round_num
            )

            best_weights = (
                base.get_trainable_weights(
                    global_model
                )
            )

            torch.save(
                best_weights,
                os.path.join(
                    args.results_dir,
                    "best_global_weights.pt",
                ),
            )

        round_duration = (
            time.time()
            - round_start
        )

        round_times.append(
            round_duration
        )

        elapsed = (
            time.time()
            - run_start
        )

        average_round_time = (
            sum(round_times)
            / len(round_times)
        )

        eta = (
            average_round_time
            * (
                args.num_rounds
                - round_num
            )
        )

        val_rows.append(
            {
                "alpha":
                    args.alpha,

                "num_clients":
                    NUM_CLIENTS,

                "partition_seed":
                    partition_seed,

                "seed":
                    args.seed,

                "round":
                    round_num,

                "global_val_auc_roc":
                    val_metrics[
                        "auc_roc"
                    ],

                "global_val_auc_pr":
                    val_metrics[
                        "auc_pr"
                    ],

                "global_val_f1":
                    val_metrics[
                        "f1"
                    ],

                "global_val_mcc":
                    val_metrics[
                        "mcc"
                    ],

                "round_time_seconds":
                    round_duration,

                "elapsed_seconds":
                    elapsed,

                "eta_seconds":
                    eta,
            }
        )

        best_tag = (
            " <-- best validation checkpoint"
            if improved
            else ""
        )

        print()
        print("=" * 90)

        print(
            f"GLOBAL MODEL | "
            f"alpha={args.alpha} | "
            f"Round {round_num:02d} | "
            f"VAL AUROC="
            f"{val_metrics['auc_roc']:.4f} | "
            f"AUPRC="
            f"{val_metrics['auc_pr']:.4f} | "
            f"F1="
            f"{val_metrics['f1']:.4f} | "
            f"MCC="
            f"{val_metrics['mcc']:.4f}"
            f"{best_tag}"
        )

        print("=" * 90)

        print(
            f"Round time             : "
            f"{format_seconds(round_duration)}"
        )

        print(
            f"Elapsed                : "
            f"{format_seconds(elapsed)}"
        )

        print(
            f"Estimated time left    : "
            f"{format_seconds(eta)}"
        )

        print(
            f"Best validation AUROC  : "
            f"{best_val_auc:.4f}"
        )

        print(
            f"Best round             : "
            f"{best_round}"
        )

        print_gpu_memory()

        # ----------------------------------------------------
        # Live recovery files
        # ----------------------------------------------------

        pd.DataFrame(
            val_rows
        ).to_csv(
            os.path.join(
                args.results_dir,
                "federated_val_rounds_live.csv",
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
    # Save completed training logs
    # ========================================================

    val_path = os.path.join(
        args.results_dir,
        "federated_val_rounds.csv",
    )

    client_log_path = os.path.join(
        args.results_dir,
        "client_round_logs.csv",
    )

    final_weights_path = os.path.join(
        args.results_dir,
        "global_weights_final.pt",
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
        client_log_path,
        index=False,
    )

    torch.save(
        base.get_trainable_weights(
            global_model
        ),
        final_weights_path,
    )

    if (
        best_weights is None
        or best_round is None
    ):

        raise RuntimeError(
            "No global validation "
            "checkpoint was selected."
        )

    # ========================================================
    # FINAL TEST
    #
    # Test data are instantiated only after all
    # model training and validation-based selection.
    # ========================================================

    print()
    print("=" * 90)

    print(
        f"Loading validation-selected "
        f"checkpoint from round "
        f"{best_round}"
    )

    print(
        "Preparing global test for "
        "final evaluation only..."
    )

    print("=" * 90)

    global_model = (
        base.set_trainable_weights(
            global_model,
            best_weights,
        )
    )

    global_test_dataset = (
        base.EpitopeDataset(
            GLOBAL_TEST_FILE,
            tokenizer,
        )
    )

    global_test_loader = (
        make_loader(
            dataset=
                global_test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=
                args.num_workers,
        )
    )

    print(
        f"Global test samples     : "
        f"{len(global_test_dataset):,}"
    )

    print(
        "Evaluating global test exactly once..."
    )

    test_metrics = (
        base.evaluate(
            global_model,
            global_test_loader,
            device,
        )
    )

    total_runtime = (
        time.time()
        - run_start
    )

    # ========================================================
    # Final result
    # ========================================================

    result = {
        "alpha":
            args.alpha,

        "num_clients":
            NUM_CLIENTS,

        "partition_seed":
            partition_seed,

        "seed":
            args.seed,

        "partition_type":
            "equal_size_dirichlet_label_skew",

        "training_samples":
            EXPECTED_TOTAL_TRAIN,

        "class_weighting":
            "client_specific_neg_over_pos; neutral fallback for single-class clients",

        "lora_rank":
            args.lora_rank,

        "lora_alpha":
            2 * args.lora_rank,

        "num_rounds":
            args.num_rounds,

        "local_epochs":
            args.local_epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

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
            total_runtime,
    }

    result_path = os.path.join(
        args.results_dir,
        "federated_test_results.csv",
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        result_path,
        index=False,
    )

    print()
    print("=" * 90)

    print(
        "DIRICHLET EXPERIMENT COMPLETE"
    )

    print("=" * 90)

    print(
        f"Dirichlet alpha        : "
        f"{args.alpha}"
    )

    print(
        f"Model seed             : "
        f"{args.seed}"
    )

    print(
        f"Selected round         : "
        f"{best_round}"
    )

    print(
        f"Best validation AUROC  : "
        f"{best_val_auc:.6f}"
    )

    print(
        f"TEST AUROC             : "
        f"{test_metrics['auc_roc']:.6f}"
    )

    print(
        f"TEST AUPRC             : "
        f"{test_metrics['auc_pr']:.6f}"
    )

    print(
        f"TEST F1                : "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        f"TEST MCC               : "
        f"{test_metrics['mcc']:.6f}"
    )

    print(
        f"Runtime                : "
        f"{format_seconds(total_runtime)}"
    )

    print(
        f"Result                 : "
        f"{result_path}"
    )

    print("=" * 90)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Homology-aware FedEpitope "
            "Dirichlet label-skew sensitivity "
            "experiment."
        )
    )

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        choices=
            SUPPORTED_ALPHAS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=
            DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=
            DEFAULT_LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=
            DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=
            DEFAULT_LR,
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=
            DEFAULT_LORA_RANK,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=5,
        choices=[3, 5, 10],
        help="Number of synthetic Dirichlet clients.",
    )

    parser.add_argument(
        "--partition_root",
        type=str,
        default=None,
        help=(
            "Partition root. Defaults to data/dirichlet for K=5 "
            "and data/dirichlet_k{K} otherwise."
        ),
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    try:

        main(args)

    except Exception as exc:

        if args.results_dir is None:

            if args.num_clients == 5:
                results_dir = (
                    f"results/"
                    f"dirichlet_alpha_"
                    f"{alpha_slug(args.alpha)}_"
                    f"seed{args.seed}"
                )
            else:
                results_dir = (
                    f"results/"
                    f"dirichlet_k{args.num_clients}_"
                    f"alpha_{alpha_slug(args.alpha)}_"
                    f"seed{args.seed}"
                )

        else:

            results_dir = (
                args.results_dir
            )

        os.makedirs(
            results_dir,
            exist_ok=True,
        )

        error_path = os.path.join(
            results_dir,
            "error_traceback.txt",
        )

        with open(
            error_path,
            "w",
            encoding="utf-8",
        ) as handle:

            handle.write(
                f"{exc}\n\n"
            )

            handle.write(
                traceback.format_exc()
            )

        print()
        print("=" * 90)

        print(
            "DIRICHLET EXPERIMENT FAILED"
        )

        print("=" * 90)

        print(
            f"Error: {exc}"
        )

        print(
            f"Traceback saved to: "
            f"{error_path}"
        )

        print("=" * 90)

        raise
