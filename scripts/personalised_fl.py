import os
import time
import random
import argparse
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import federated_train as base


NUM_CLIENTS = 5
CLIENT_NAMES = [
    "Coronavirus",
    "Parasite",
    "Human/Self",
    "Flavivirus",
    "Bacteria",
]

CLIENT_TRAIN_FILES = [
    f"data/federated/client{i}_train.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

CLIENT_VAL_FILES = [
    f"data/personalization/client{i}_val.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

CLIENT_TEST_FILES = [
    f"data/personalization/client{i}_test.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

DEFAULT_EPOCHS = 10

LFT_LR = 1e-5
CLF_LR = 1e-4
FEDPROX_LR = 1e-5
FEDPROX_MU = 0.01

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128


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
    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_seconds(seconds):
    return str(
        timedelta(
            seconds=int(seconds)
        )
    )


# ============================================================
# Data loaders
# ============================================================

def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    seed=None,
):
    generator = None

    if shuffle:
        if seed is None:
            raise ValueError(
                "A seed is required for shuffled loaders."
            )

        generator = torch.Generator()
        generator.manual_seed(seed)

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
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# File and checkpoint checks
# ============================================================

def check_files(checkpoint):
    required = (
        CLIENT_TRAIN_FILES
        + CLIENT_VAL_FILES
        + CLIENT_TEST_FILES
        + [checkpoint]
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


def clone_weight_dict(weights):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in weights.items()
    }


def build_from_global(
    rank,
    global_weights,
    device,
):
    model = (
        base.build_model(rank)
        .to(device)
    )

    model = base.set_trainable_weights(
        model,
        global_weights,
    )

    return model


def verify_checkpoint(
    rank,
    global_weights,
):
    model = base.build_model(rank)

    expected = (
        base.get_trainable_weights(
            model
        )
    )

    expected_keys = set(
        expected.keys()
    )

    actual_keys = set(
        global_weights.keys()
    )

    missing = sorted(
        expected_keys
        - actual_keys
    )

    extra = sorted(
        actual_keys
        - expected_keys
    )

    if missing or extra:
        raise RuntimeError(
            "FedAvg checkpoint does not match current model.\n"
            f"Missing keys: {missing}\n"
            f"Extra keys: {extra}"
        )

    del model


# ============================================================
# PFL parameter configuration
# ============================================================

def configure_variant(
    model,
    variant,
):

    if variant in {
        "pFL-LFT",
        "pFL-FedProx",
    }:
        # base.build_model() already exposes
        # LoRA + classifier parameters as trainable.
        return

    if variant == "pFL-CLF-Only":

        classifier_names = set()

        for name, _ in (
            model.named_parameters()
        ):

            lname = (
                name.lower()
            )

            is_classifier = (
                "classifier" in lname
                or "score" in lname
                or "modules_to_save"
                in lname
            )

            is_lora = (
                "lora" in lname
            )

            if (
                is_classifier
                and not is_lora
            ):
                classifier_names.add(
                    name
                )

        if not classifier_names:
            raise RuntimeError(
                "Could not identify classifier parameters."
            )

        for name, param in (
            model.named_parameters()
        ):
            param.requires_grad = (
                name in classifier_names
            )

        return

    raise ValueError(
        f"Unknown variant: {variant}"
    )


def trainable_counts(model):

    total_trainable = 0
    lora = 0
    classifier = 0

    for name, param in (
        model.named_parameters()
    ):

        if not param.requires_grad:
            continue

        total_trainable += (
            param.numel()
        )

        lname = (
            name.lower()
        )

        if "lora" in lname:
            lora += (
                param.numel()
            )

        elif (
            "classifier" in lname
            or "score" in lname
            or "modules_to_save"
            in lname
        ):
            classifier += (
                param.numel()
            )

    return (
        total_trainable,
        lora,
        classifier,
    )


# ============================================================
# Local checkpoint helpers
# ============================================================

def capture_trainable(model):

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


def restore_trainable(
    model,
    weights,
):

    restored = set()

    for name, param in (
        model.named_parameters()
    ):

        if (
            param.requires_grad
            and name in weights
        ):

            param.data.copy_(
                weights[
                    name
                ].to(
                    param.device
                )
            )

            restored.add(
                name
            )

    missing = (
        set(weights.keys())
        - restored
    )

    if missing:
        raise RuntimeError(
            "Could not restore personalized weights: "
            f"{sorted(missing)}"
        )

    return model


# ============================================================
# One local training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    variant,
    global_reference,
    mu,
):

    model.train()

    ce_sum = 0.0
    objective_sum = 0.0

    for batch in loader:

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

        ce_loss = criterion(
            logits,
            labels,
        )

        if variant == "pFL-FedProx":

            prox = torch.zeros(
                (),
                device=device,
            )

            for name, param in (
                model.named_parameters()
            ):

                if (
                    param.requires_grad
                    and name
                    in global_reference
                ):

                    prox = (
                        prox
                        + torch.sum(
                            (
                                param
                                - global_reference[
                                    name
                                ]
                            ) ** 2
                        )
                    )

            objective = (
                ce_loss
                + (mu / 2.0)
                * prox
            )

        else:

            objective = (
                ce_loss
            )

        objective.backward()
        optimizer.step()

        ce_sum += (
            ce_loss.item()
        )

        objective_sum += (
            objective.item()
        )

    n_batches = max(
        len(loader),
        1,
    )

    return (
        ce_sum / n_batches,
        objective_sum / n_batches,
    )


# ============================================================
# Personalize one client with one method
# ============================================================

def personalize_client(
    variant,
    model,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    device,
    epochs,
    lr,
    global_weights,
    mu,
    client_id,
    client_name,
):

    trainable_params = [
        param
        for param
        in model.parameters()
        if param.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError(
            f"{variant}: no trainable parameters."
        )

    optimizer = (
        torch.optim.AdamW(
            trainable_params,
            lr=lr,
        )
    )

    criterion = (
        torch.nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    global_reference = None

    if variant == "pFL-FedProx":

        global_reference = {
            name:
                tensor
                .detach()
                .to(device)

            for name, tensor
            in global_weights.items()
        }

    best_val_auc = -np.inf
    best_epoch = None
    best_weights = None

    epoch_rows = []

    # --------------------------------------------------------
    # Local training + validation checkpoint selection.
    #
    # Test set is NOT used in this loop.
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        epoch_start = time.time()

        (
            ce_loss,
            objective_loss,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            variant=variant,
            global_reference=
                global_reference,
            mu=mu,
        )

        val_metrics = (
            base.evaluate(
                model,
                val_loader,
                device,
            )
        )

        epoch_rows.append(
            {
                "client_id":
                    client_id,

                "client_name":
                    client_name,

                "variant":
                    variant,

                "epoch":
                    epoch,

                "ce_loss":
                    ce_loss,

                "objective_loss":
                    objective_loss,

                "val_auc_roc":
                    val_metrics[
                        "auc_roc"
                    ],

                "val_auc_pr":
                    val_metrics[
                        "auc_pr"
                    ],

                "val_f1":
                    val_metrics[
                        "f1"
                    ],

                "val_mcc":
                    val_metrics[
                        "mcc"
                    ],

                "epoch_time_seconds":
                    time.time()
                    - epoch_start,
            }
        )

        print(
            f"[{variant} | "
            f"Client {client_id} "
            f"{client_name} | "
            f"Epoch {epoch:02d}/{epochs}] "
            f"CE={ce_loss:.4f} | "
            f"Val AUROC="
            f"{val_metrics['auc_roc']:.4f}"
        )

        auc = (
            val_metrics[
                "auc_roc"
            ]
        )

        if (
            not np.isnan(auc)
            and auc > best_val_auc
        ):

            best_val_auc = auc
            best_epoch = epoch

            best_weights = (
                capture_trainable(
                    model
                )
            )

    if best_weights is None:
        raise RuntimeError(
            f"{variant}, client {client_id}: "
            "no valid checkpoint selected."
        )

    # --------------------------------------------------------
    # Restore best local validation checkpoint
    # --------------------------------------------------------

    model = (
        restore_trainable(
            model,
            best_weights,
        )
    )

    # --------------------------------------------------------
    # Final client test evaluation exactly once
    # --------------------------------------------------------

    test_metrics = (
        base.evaluate(
            model,
            test_loader,
            device,
        )
    )

    return (
        best_epoch,
        best_val_auc,
        test_metrics,
        epoch_rows,
    )


# ============================================================
# Main
# ============================================================

def main(args):

    run_start = time.time()

    set_seed(
        args.seed
    )

    checkpoint = (
        args.fed_checkpoint
        if args.fed_checkpoint
        is not None
        else
        f"results/fedavg_seed"
        f"{args.seed}/"
        f"best_global_weights.pt"
    )

    results_dir = (
        args.results_dir
        if args.results_dir
        is not None
        else
        f"results/pfl_seed"
        f"{args.seed}"
    )

    check_files(
        checkpoint
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        base.AutoTokenizer
        .from_pretrained(
            "facebook/"
            "esm2_t12_35M_UR50D"
        )
    )

    print("=" * 88)
    print(
        "FedEpitope — Homology-Aware "
        "Personalized Federated Learning"
    )
    print("=" * 88)

    print(
        f"Device                 : "
        f"{device}"
    )

    print(
        f"Model seed             : "
        f"{args.seed}"
    )

    print(
        f"FedAvg checkpoint      : "
        f"{checkpoint}"
    )

    print(
        f"Maximum local epochs   : "
        f"{args.personal_epochs}"
    )

    print(
        f"pFL-LFT LR             : "
        f"{args.lft_lr}"
    )

    print(
        f"pFL-CLF-Only LR        : "
        f"{args.clf_lr}"
    )

    print(
        f"pFL-FedProx LR         : "
        f"{args.fedprox_lr}"
    )

    print(
        f"pFL-FedProx mu         : "
        f"{args.mu}"
    )

    print(
        "Local selection        : "
        "PFL validation AUROC"
    )

    print(
        "Local test usage       : "
        "once after local checkpoint selection"
    )

    print(
        "Global test usage      : "
        "NOT ACCESSED"
    )

    print("=" * 88)

    # --------------------------------------------------------
    # Load global FedAvg checkpoint
    # --------------------------------------------------------

    global_weights = (
        torch.load(
            checkpoint,
            map_location="cpu",
        )
    )

    if not isinstance(
        global_weights,
        dict,
    ):
        raise TypeError(
            "Expected FedAvg checkpoint to contain "
            "a dictionary of weights."
        )

    verify_checkpoint(
        args.lora_rank,
        global_weights,
    )

    global_weights = (
        clone_weight_dict(
            global_weights
        )
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_datasets = []
    val_datasets = []
    test_datasets = []
    class_weights = []

    print()
    print(
        "PERSONALIZATION DATA"
    )
    print("=" * 88)

    for i in range(
        NUM_CLIENTS
    ):

        train_df = pd.read_csv(
            CLIENT_TRAIN_FILES[i]
        )

        val_df = pd.read_csv(
            CLIENT_VAL_FILES[i]
        )

        test_df = pd.read_csv(
            CLIENT_TEST_FILES[i]
        )

        train_ds = (
            base.EpitopeDataset(
                CLIENT_TRAIN_FILES[i],
                tokenizer,
            )
        )

        val_ds = (
            base.EpitopeDataset(
                CLIENT_VAL_FILES[i],
                tokenizer,
            )
        )

        test_ds = (
            base.EpitopeDataset(
                CLIENT_TEST_FILES[i],
                tokenizer,
            )
        )

        pos = int(
            train_df[
                "label"
            ].sum()
        )

        neg = (
            len(train_df)
            - pos
        )

        weights = (
            torch.tensor(
                [
                    1.0,
                    neg
                    / max(
                        pos,
                        1,
                    ),
                ],
                dtype=torch.float,
            )
            .to(device)
        )

        train_datasets.append(
            train_ds
        )

        val_datasets.append(
            val_ds
        )

        test_datasets.append(
            test_ds
        )

        class_weights.append(
            weights
        )

        print(
            f"Client {i + 1} "
            f"{CLIENT_NAMES[i]:<12} | "
            f"Train="
            f"{len(train_df):>7,} | "
            f"Val="
            f"{len(val_df):>6,} | "
            f"Test="
            f"{len(test_df):>6,} | "
            f"Val pos="
            f"{val_df['label'].mean():.2%} | "
            f"Test pos="
            f"{test_df['label'].mean():.2%}"
        )

    result_rows = []
    epoch_rows = []

    global_test_lookup = {}

    # ========================================================
    # Baseline: unpersonalized global FedAvg
    # evaluated on exactly the same client-specific test sets
    # ========================================================

    print()
    print(
        "GLOBAL FEDAVG BASELINE "
        "ON PFL TEST SETS"
    )
    print("=" * 88)

    for i in range(
        NUM_CLIENTS
    ):

        client_id = (
            i + 1
        )

        model = (
            build_from_global(
                args.lora_rank,
                global_weights,
                device,
            )
        )

        test_loader = (
            make_loader(
                test_datasets[i],
                EVAL_BATCH_SIZE,
                False,
                args.num_workers,
            )
        )

        metrics = (
            base.evaluate(
                model,
                test_loader,
                device,
            )
        )

        global_test_lookup[
            client_id
        ] = metrics

        result_rows.append(
            {
                "seed":
                    args.seed,

                "variant":
                    "Global-FedAvg",

                "client_id":
                    client_id,

               "client_name":
                    CLIENT_NAMES[i],

                "best_epoch":
                    np.nan,

                "best_val_auc":
                    np.nan,

                "test_auc_roc":
                    metrics[
                        "auc_roc"
                    ],

                "test_auc_pr":
                    metrics[
                        "auc_pr"
                    ],

                "test_f1":
                    metrics[
                        "f1"
                    ],

                "test_mcc":
                    metrics[
                        "mcc"
                    ],

                "delta_auc_roc_vs_global":
                    0.0,

                "delta_auc_pr_vs_global":
                    0.0,

                "delta_f1_vs_global":
                    0.0,

                "delta_mcc_vs_global":
                    0.0,
            }
        )

        print(
            f"Client {client_id} "
            f"{CLIENT_NAMES[i]:<12} | "
            f"AUROC="
            f"{metrics['auc_roc']:.4f} | "
            f"AUPRC="
            f"{metrics['auc_pr']:.4f} | "
            f"F1="
            f"{metrics['f1']:.4f} | "
            f"MCC="
            f"{metrics['mcc']:.4f}"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Three original personalization variants
    # ========================================================

    variants = [
        (
            "pFL-LFT",
            args.lft_lr,
            None,
        ),

        (
            "pFL-CLF-Only",
            args.clf_lr,
            None,
        ),

        (
            "pFL-FedProx",
            args.fedprox_lr,
            args.mu,
        ),
    ]

    for (
        variant,
        lr,
        mu,
    ) in variants:

        print()
        print("=" * 88)

        if mu is None:
            print(
                f"{variant} | "
                f"lr={lr}"
            )
        else:
            print(
                f"{variant} | "
                f"lr={lr} | "
                f"mu={mu}"
            )

        print("=" * 88)

        for i in range(
            NUM_CLIENTS
        ):

            client_id = (
                i + 1
            )

            client_name = (
                CLIENT_NAMES[i]
            )

            model = (
                build_from_global(
                    args.lora_rank,
                    global_weights,
                    device,
                )
            )

            configure_variant(
                model,
                variant,
            )

            (
                total_trainable,
                lora_count,
                clf_count,
            ) = trainable_counts(
                model
            )

            print(
                f"\nClient {client_id}: "
                f"{client_name} | "
                f"trainable="
                f"{total_trainable:,} | "
                f"LoRA="
                f"{lora_count:,} | "
                f"classifier="
                f"{clf_count:,}"
            )

            if (
                variant
                == "pFL-CLF-Only"
            ):

                if (
                    lora_count != 0
                    or clf_count <= 0
                ):
                    raise RuntimeError(
                        "pFL-CLF-Only parameter "
                        "configuration is incorrect."
                    )

            else:

                if (
                    lora_count <= 0
                    or clf_count <= 0
                ):
                    raise RuntimeError(
                        f"{variant} must train both "
                        "LoRA and classifier parameters."
                    )

            # Same batch ordering across methods
            # for same seed/client.
            train_loader = (
                make_loader(
                    train_datasets[i],
                    TRAIN_BATCH_SIZE,
                    True,
                    args.num_workers,
                    seed=(
                        args.seed
                        + i
                    ),
                )
            )

            val_loader = (
                make_loader(
                    val_datasets[i],
                    EVAL_BATCH_SIZE,
                    False,
                    args.num_workers,
                )
            )

            test_loader = (
                make_loader(
                    test_datasets[i],
                    EVAL_BATCH_SIZE,
                    False,
                    args.num_workers,
                )
            )

            (
                best_epoch,
                best_val_auc,
                test_metrics,
                local_epoch_rows,
            ) = personalize_client(
                variant=
                    variant,

                model=
                    model,

                train_loader=
                    train_loader,

                val_loader=
                    val_loader,

                test_loader=
                    test_loader,

                class_weights=
                    class_weights[i],

                device=
                    device,

                epochs=
                    args.personal_epochs,

                lr=
                    lr,

                global_weights=
                    global_weights,

                mu=
                    mu,

                client_id=
                    client_id,

                client_name=
                    client_name,
            )

            for row in (
                local_epoch_rows
            ):
                row["seed"] = (
                    args.seed
                )

            epoch_rows.extend(
                local_epoch_rows
            )

            baseline = (
                global_test_lookup[
                    client_id
                ]
            )

            result = {
                "seed":
                    args.seed,

                "variant":
                    variant,

                "client_id":
                    client_id,

                "client_name":
                    client_name,

                "best_epoch":
                    best_epoch,

                "best_val_auc":
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

                "delta_auc_roc_vs_global":
                    (
                        test_metrics[
                            "auc_roc"
                        ]
                        - baseline[
                            "auc_roc"
                        ]
                    ),

                "delta_auc_pr_vs_global":
                    (
                        test_metrics[
                            "auc_pr"
                        ]
                        - baseline[
                            "auc_pr"
                        ]
                    ),

                "delta_f1_vs_global":
                    (
                        test_metrics[
                            "f1"
                        ]
                        - baseline[
                            "f1"
                        ]
                    ),

                "delta_mcc_vs_global":
                    (
                        test_metrics[
                            "mcc"
                        ]
                        - baseline[
                            "mcc"
                        ]
                    ),
            }

            result_rows.append(
                result
            )

            print(
                f"elected epoch="
                f"{best_epoch} | "
                f"Val AUROC="
                f"{best_val_auc:.4f} | "
                f"TEST AUROC="
                f"{test_metrics['auc_roc']:.4f} | "
                f"AUPRC="
                f"{test_metrics['auc_pr']:.4f} | "
                f"F1="
                f"{test_metrics['f1']:.4f} | "
                f"MCC="
                f"{test_metrics['mcc']:.4f}"
            )

            # Live saving
            pd.DataFrame(
                result_rows
            ).to_csv(
                os.path.join(
                    results_dir,
                    "pfl_results_live.csv",
                ),
                index=False,
            )

            pd.DataFrame(
                epoch_rows
            ).to_csv(
                os.path.join(
                    results_dir,
                    "pfl_val_trajectories_live.csv",
                ),
                index=False,
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # Final outputs
    # ========================================================

    results_df = (
        pd.DataFrame(
            result_rows
        )
    )

    epochs_df = (
        pd.DataFrame(
            epoch_rows
        )
    )

    results_path = (
        os.path.join(
            results_dir,
            "pfl_results.csv",
        )
    )

    epochs_path = (
        os.path.join(
            results_dir,
            "pfl_val_trajectories.csv",
        )
    )

    summary_path = (
        os.path.join(
            results_dir,
            "pfl_macro_summary.csv",
        )
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    epochs_df.to_csv(
        epochs_path,
        index=False,
    )

    summary_rows = []

    for variant in [
        "Global-FedAvg",
        "pFL-LFT",
        "pFL-CLF-Only",
        "pFL-FedProx",
    ]:

        subset = (
            results_df[
                results_df[
                    "variant"
                ]
                == variant
            ]
        )

        summary_rows.append(
            {
                "seed":
                    args.seed,

                "variant":
                    variant,

                "macro_auc_roc":
                    subset[
                        "test_auc_roc"
                    ].mean(),

                "macro_auc_pr":
                    subset[
                        "test_auc_pr"
                    ].mean(),

                "macro_f1":
                    subset[
                        "test_f1"
                    ].mean(),

                "macro_mcc":
                    subset[
                        "test_mcc"
                    ].mean(),
            }
        )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 88)
    print(
        "PERSONALIZED FL — FINAL SUMMARY"
    )
    print("=" * 88)

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['variant']:<16} | "
            f"AUROC="
            f"{row['macro_auc_roc']:.4f} | "
            f"AUPRC="
            f"{row['macro_auc_pr']:.4f} | "
            f"F1="
            f"{row['macro_f1']:.4f} | "
            f"MCC="
            f"{row['macro_mcc']:.4f}"
        )

    print(
        "\nNOTE: these are macro-averages over "
        "client-specific PFL test sets and are "
        "not directly comparable with the "
        "main global_test.csv scores."
    )

    print(
        f"Runtime          : "
        f"{format_seconds(time.time() - run_start)}"
    )

    print(
        f"Detailed results : "
        f"{results_path}"
    )

    print(
        f"Val trajectories : "
        f"{epochs_path}"
    )

    print(
        f"Macro summary    : "
        f"{summary_path}"
    )

    print("=" * 88)


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
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--personal_epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--lft_lr",
        type=float,
        default=LFT_LR,
    )

    parser.add_argument(
        "--clf_lr",
        type=float,
        default=CLF_LR,
    )

    parser.add_argument(
        "--fedprox_lr",
        type=float,
        default=FEDPROX_LR,
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=FEDPROX_MU,
    )

    parser.add_argument(
        "--fed_checkpoint",
        type=str,
        default=None,
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

        results_dir = (
            args.results_dir
            if args.results_dir
            is not None
            else
            f"results/pfl_seed"
            f"{args.seed}"
        )

        os.makedirs(
            results_dir,
            exist_ok=True,
        )

        error_path = (
            os.path.join(
                results_dir,
                "error_traceback.txt",
            )
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
        print("=" * 88)
        print(
            "PFL EXPERIMENT FAILED"
        )
        print("=" * 88)

        print(
            f"Error: {exc}"
        )

        print(
            f"Traceback saved to: "
            f"{error_path}"
        )

        print("=" * 88)

        raise
