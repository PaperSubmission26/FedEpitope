import os
import copy
import time
import math
import random
import argparse
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

import federated_train as base


# ============================================================
# Fixed benchmark configuration
# ============================================================

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

GLOBAL_VAL_FILE = "data/federated/global_val.csv"
GLOBAL_TEST_FILE = "data/federated/global_test.csv"

DEFAULT_ROUNDS = 20
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 2e-4
DEFAULT_LORA_RANK = 2

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

SUPPORTED_EPSILONS = (
    1.0,
    5.0,
    10.0,
)


# ============================================================
# Reproducibility / utilities
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        f"allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB | "
        f"max allocated: {maximum:.2f} GB"
    )


def check_required_files():

    required = (
        CLIENT_TRAIN_FILES
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
# Model / weights
# ============================================================

def build_dp_compatible_model(
    lora_rank,
):

    model = (
        base.build_model(
            lora_rank
        )
    )

    model = (
        ModuleValidator.fix(
            model
        )
    )

    errors = (
        ModuleValidator.validate(
            model,
            strict=False,
        )
    )

    if errors:

        raise RuntimeError(
            "Model remains incompatible "
            "with Opacus:\n"
            + "\n".join(
                str(error)
                for error in errors
            )
        )

    return model


def clean_parameter_name(
    name,
):

    while name.startswith(
        "_module."
    ):

        name = name[
            len("_module.") :
        ]

    return name


def get_trainable_weights(
    model,
):

    weights = {}

    for name, parameter in (
        model.named_parameters()
    ):

        if (
            parameter.requires_grad
        ):

            clean_name = (
                clean_parameter_name(
                    name
                )
            )

            weights[
                clean_name
            ] = (
                parameter
                .detach()
                .cpu()
                .clone()
            )

    return weights


def set_trainable_weights(
    model,
    weights,
):

    found = set()

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        clean_name = (
            clean_parameter_name(
                name
            )
        )

        if clean_name in weights:

            parameter.data.copy_(
                weights[
                    clean_name
                ].to(
                    parameter.device
                )
            )

            found.add(
                clean_name
            )

    missing = (
        set(weights.keys())
        - found
    )

    if missing:

        raise RuntimeError(
            "Could not restore all "
            "trainable parameters: "
            + str(
                sorted(missing)
            )
        )

    return model


def print_trainable_summary(
    model,
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
        for name, p
        in model.named_parameters()
        if (
            p.requires_grad
            and "lora"
            in name.lower()
        )
    )

    classifier = sum(
        p.numel()
        for name, p
        in model.named_parameters()
        if (
            p.requires_grad
            and "lora"
            not in name.lower()
            and any(
                key in name.lower()
                for key in (
                    "classifier",
                    "score",
                    "modules_to_save",
                )
            )
        )
    )

    print(
        f"Model parameters | "
        f"total={total:,} | "
        f"trainable={trainable:,} | "
        f"LoRA={lora:,} | "
        f"classifier={classifier:,} | "
        f"{100.0 * trainable / total:.4f}%"
    )

    if (
        lora <= 0
        or classifier <= 0
    ):

        raise RuntimeError(
            "Expected both LoRA and "
            "classifier parameters "
            "to be trainable."
        )


# ============================================================
# Weighted FedAvg
# ============================================================

def federated_averaging(
    client_weights,
    client_sizes,
):

    if (
        len(client_weights)
        != len(client_sizes)
    ):

        raise ValueError(
            "client_weights/client_sizes "
            "length mismatch."
        )

    total_size = float(
        sum(client_sizes)
    )

    proportions = [
        size / total_size
        for size in client_sizes
    ]

    reference_keys = set(
        client_weights[
            0
        ].keys()
    )

    for client_id, weights in enumerate(
        client_weights[1:],
        start=2,
    ):

        if (
            set(weights.keys())
            != reference_keys
        ):

            raise RuntimeError(
                f"Weight keys differ "
                f"for client {client_id}."
            )

    averaged = {}

    for key in (
        client_weights[
            0
        ].keys()
    ):

        value = torch.zeros_like(
            client_weights[
                0
            ][key],
            dtype=torch.float32,
        )

        for index, weights in enumerate(
            client_weights
        ):

            value += (
                proportions[
                    index
                ]
                * weights[
                    key
                ].float()
            )

        averaged[
            key
        ] = value

    return averaged


# ============================================================
# Privacy accounting
# ============================================================

def steps_per_epoch(
    dataset_size,
    batch_size,
):

    return math.ceil(
        dataset_size
        / batch_size
    )


def sample_rate_for_client(
    dataset_size,
    batch_size,
):

    # Opacus make_private() uses:
    # sample_rate = 1 / len(source_loader)

    return (
        1.0
        / steps_per_epoch(
            dataset_size,
            batch_size,
        )
    )


def calibrate_noise_multiplier(
    target_epsilon,
    delta,
    dataset_size,
    batch_size,
    num_rounds,
    local_epochs,
):

    sample_rate = (
        sample_rate_for_client(
            dataset_size,
            batch_size,
        )
    )

    total_steps = (
        steps_per_epoch(
            dataset_size,
            batch_size,
        )
        * num_rounds
        * local_epochs
    )

    noise_multiplier = (
        get_noise_multiplier(
            target_epsilon=
                target_epsilon,

            target_delta=
                delta,

            sample_rate=
                sample_rate,

            steps=
                total_steps,

            accountant=
                "rdp",

            epsilon_tolerance=
                0.01,
        )
    )

    return (
        float(
            noise_multiplier
        ),
        float(
            sample_rate
        ),
        int(
            total_steps
        ),
    )


def accountant_epsilon(
    accountant,
    delta,
):

    return float(
        accountant.get_epsilon(
            delta=delta
        )
    )


# ============================================================
# Loaders
# ============================================================

def make_eval_loader(
    dataset,
    batch_size,
    num_workers,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=
            torch.cuda.is_available(),
    )


def make_private_source_loader(
    dataset,
    batch_size,
    seed,
):

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=False,
    )


# ============================================================
# One client / one round
# ============================================================

def train_local_dp(
    model,
    dataset,
    device,
    class_weights,
    noise_multiplier,
    max_grad_norm,
    lr,
    batch_size,
    local_epochs,
    accountant,
    sample_rate,
    seed,
    round_num,
    client_id,
):

    model.train()

    source_loader = (
        make_private_source_loader(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
        )
    )

    expected_steps_per_epoch = (
        len(source_loader)
    )

    optimizer = (
        torch.optim.AdamW(
            [
                p
                for p
                in model.parameters()
                if p.requires_grad
            ],
            lr=lr,
        )
    )

    privacy_engine = (
        PrivacyEngine(
            accountant="rdp",
            secure_mode=False,
        )
    )

    (
        model,
        optimizer,
        private_loader,
    ) = privacy_engine.make_private(

        module=model,

        optimizer=optimizer,

        data_loader=
            source_loader,

        noise_multiplier=
            noise_multiplier,

        max_grad_norm=
            max_grad_norm,

        poisson_sampling=True,

        clipping="flat",

        grad_sample_mode="hooks",
    )

    expected_sample_rate = (
        1.0
        / expected_steps_per_epoch
    )

    if not np.isclose(
        sample_rate,
        expected_sample_rate,
        rtol=0.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            f"Sample-rate mismatch "
            f"for client {client_id}: "
            f"{sample_rate} vs "
            f"{expected_sample_rate}"
        )

    epoch_rows = []
    actual_steps_total = 0

    for epoch in range(
        1,
        local_epochs + 1
    ):

        epoch_start = time.time()

        model.train()

        loss_sum = 0.0
        steps_this_epoch = 0

        for batch in private_loader:

            input_ids = (
                batch[
                    "input_ids"
                ].to(device)
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ].to(device)
            )

            labels = (
                batch[
                    "labels"
                ].to(device)
            )

            optimizer.zero_grad()

            logits = model(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,
            ).logits

            # Weighted per-example CE, then ordinary
            # batch mean. This preserves class weighting
            # while keeping the reduction compatible with
            # Opacus mean-reduction semantics.
            losses = (
                torch.nn.functional
                .cross_entropy(
                    logits,
                    labels,
                    weight=
                        class_weights,
                    reduction="none",
                )
            )

            loss = (
                losses.mean()
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite loss | "
                    f"round={round_num}, "
                    f"client={client_id}, "
                    f"epoch={epoch}"
                )

            loss.backward()

            optimizer.step()

            loss_sum += float(
                loss.item()
            )

            steps_this_epoch += 1

        if (
            steps_this_epoch
            != expected_steps_per_epoch
        ):

            raise RuntimeError(
                f"Unexpected step count "
                f"for client {client_id}: "
                f"expected "
                f"{expected_steps_per_epoch}, "
                f"got {steps_this_epoch}"
            )

        actual_steps_total += (
            steps_this_epoch
        )

        # Compose privacy expenditure manually
        # across communication rounds.
        for _ in range(
            steps_this_epoch
        ):

            accountant.step(
                noise_multiplier=
                    noise_multiplier,

                sample_rate=
                    sample_rate,
            )

        cumulative_epsilon = (
            accountant_epsilon(
                accountant,
                DELTA,
            )
        )

        mean_loss = (
            loss_sum
            / max(
                steps_this_epoch,
                1,
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        epoch_rows.append(
            {
                "round":
                    round_num,

                "client_id":
                    client_id,

                "client_name":
                    CLIENT_NAMES[
                        client_id - 1
                    ],

                "epoch":
                    epoch,

                "loss":
                    mean_loss,

                "cumulative_epsilon":
                    cumulative_epsilon,

                "epoch_time_seconds":
                    epoch_time,
            }
        )

        print(
            f"[Round {round_num:02d} | "
            f"Client {client_id} "
            f"{CLIENT_NAMES[client_id - 1]:<12} | "
            f"Epoch {epoch}/{local_epochs}] "
            f"Loss={mean_loss:.4f} | "
            f"cumulative eps="
            f"{cumulative_epsilon:.4f} | "
            f"time="
            f"{format_seconds(epoch_time)}"
        )

    return (
        model,
        epoch_rows,
        actual_steps_total,
    )


# ============================================================
# Main
# ============================================================

def main(args):

    run_start = time.time()

    set_seed(
        args.seed
    )

    check_required_files()

    if (
        args.results_dir
        is None
    ):

        epsilon_label = (
            str(
                args.epsilon
            )
            .replace(
                ".",
                "p",
            )
        )

        args.results_dir = (
            f"results/"
            f"dp_eps"
            f"{epsilon_label}_"
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

    tokenizer = (
        base.AutoTokenizer
        .from_pretrained(
            "facebook/"
            "esm2_t12_35M_UR50D"
        )
    )

    print("=" * 92)

    print(
        "FedEpitope — Homology-Aware "
        "Record-Level DP-SGD"
    )

    print("=" * 92)

    print(
        f"Device                  : "
        f"{device}"
    )

    print(
        f"Seed                    : "
        f"{args.seed}"
    )

    print(
        f"Target epsilon          : "
        f"{args.epsilon}"
    )

    print(
        f"Delta                   : "
        f"{DELTA:.0e}"
    )

    print(
        f"Rounds                  : "
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
        f"LoRA rank / alpha       : "
        f"{args.lora_rank} / "
        f"{2 * args.lora_rank}"
    )

    print(
        f"Max per-sample grad norm: "
        f"{args.max_grad_norm}"
    )

    print(
        "Privacy unit            : "
        "individual training peptide record"
    )

    print(
        "Mechanism               : "
        "client-side DP-SGD"
    )

    print(
        "Accounting              : "
        "RDP across all rounds/local steps"
    )

    print(
        "Aggregation             : "
        "weighted FedAvg"
    )

    print(
        "Checkpoint selection    : "
        "global homology validation AUROC"
    )

    print(
        "Global test usage       : "
        "once after checkpoint selection"
    )

    print(
        f"Results directory       : "
        f"{args.results_dir}"
    )

    print("=" * 92)

    print()

    # ========================================================
    # Data
    # ========================================================

    train_datasets = []
    client_sizes = []
    class_weights = []

    for client_id, path in enumerate(
        CLIENT_TRAIN_FILES,
        start=1,
    ):

        dataset = (
            base.EpitopeDataset(
                path,
                tokenizer,
            )
        )

        frame = pd.read_csv(
            path
        )

        positives = int(
            frame[
                "label"
            ].sum()
        )

        negatives = (
            len(frame)
            - positives
        )

        weights = (
            torch.tensor(
                [
                    1.0,
                    negatives
                    / max(
                        positives,
                        1,
                    ),
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        train_datasets.append(
            dataset
        )

        client_sizes.append(
            len(dataset)
        )

        class_weights.append(
            weights
        )

        print(
            f"Client {client_id} "
            f"{CLIENT_NAMES[client_id - 1]:<12} | "
            f"N={len(dataset):>7,} | "
            f"positive={positives:>6,} | "
            f"rate="
            f"{positives / len(frame):.4%}"
        )

    global_val_dataset = (
        base.EpitopeDataset(
            GLOBAL_VAL_FILE,
            tokenizer,
        )
    )

    global_test_dataset = (
        base.EpitopeDataset(
            GLOBAL_TEST_FILE,
            tokenizer,
        )
    )

    global_val_loader = (
        make_eval_loader(
            global_val_dataset,
            batch_size=128,
            num_workers=
                args.num_workers,
        )
    )

    global_test_loader = (
        make_eval_loader(
            global_test_dataset,
            batch_size=128,
            num_workers=
                args.num_workers,
        )
    )

    print(
        f"\nGlobal validation N     : "
        f"{len(global_val_dataset):,}"
    )

    print(
        f"Global test N           : "
        f"{len(global_test_dataset):,}"
    )

    # ========================================================
    # Privacy calibration
    # ========================================================

    noise_multipliers = []
    sample_rates = []
    planned_steps = []

    print(
        "\nDP calibration for "
        "the full training budget"
    )

    print("-" * 92)

    for client_id, size in enumerate(
        client_sizes,
        start=1,
    ):

        (
            noise_multiplier,
            sample_rate,
            total_steps,
        ) = calibrate_noise_multiplier(

            target_epsilon=
                args.epsilon,

            delta=
                DELTA,

            dataset_size=
                size,

            batch_size=
                args.batch_size,

            num_rounds=
                args.num_rounds,

            local_epochs=
                args.local_epochs,
        )

        noise_multipliers.append(
            noise_multiplier
        )

        sample_rates.append(
            sample_rate
        )

        planned_steps.append(
            total_steps
        )

        print(
            f"Client {client_id} "
            f"{CLIENT_NAMES[client_id - 1]:<12} | "
            f"q={sample_rate:.8f} | "
            f"steps={total_steps:,} | "
            f"noise multiplier="
            f"{noise_multiplier:.6f}"
        )

    # ========================================================
    # Global model
    # ========================================================

    global_model = (
        build_dp_compatible_model(
            args.lora_rank
        )
        .to(device)
    )

    print()

    print_trainable_summary(
        global_model
    )

    initial_keys = set(
        get_trainable_weights(
            global_model
        ).keys()
    )

    accountants = [
        RDPAccountant()
        for _ in range(
            NUM_CLIENTS
        )
    ]

    val_rows = []
    client_epoch_rows = []
    round_times = []

    best_val_auc = -np.inf
    best_round = None
    best_weights = None

    # ========================================================
    # Federated training
    # ========================================================

    for round_num in range(
        1,
        args.num_rounds + 1,
    ):

        round_start = time.time()

        print()
        print("=" * 92)

        print(
            f"ROUND "
            f"{round_num}/"
            f"{args.num_rounds}"
        )

        print("=" * 92)

        client_weight_list = []
        round_epsilons = []

        for client_index in range(
            NUM_CLIENTS
        ):

            client_id = (
                client_index + 1
            )

            print()
            print("-" * 92)

            print(
                f"Client {client_id}: "
                f"{CLIENT_NAMES[client_index]}"
            )

            print("-" * 92)

            local_model = (
                copy.deepcopy(
                    global_model
                )
                .to(device)
            )

            local_seed = (
                args.seed
                * 100000
                + round_num
                * 100
                + client_id
            )

            (
                local_model,
                epoch_log,
                actual_steps,
            ) = train_local_dp(

                model=
                    local_model,

                dataset=
                    train_datasets[
                        client_index
                    ],

                device=
                    device,

                class_weights=
                    class_weights[
                        client_index
                    ],

                noise_multiplier=
                    noise_multipliers[
                        client_index
                    ],

                max_grad_norm=
                    args.max_grad_norm,

                lr=
                    args.lr,

                batch_size=
                    args.batch_size,

                local_epochs=
                    args.local_epochs,

                accountant=
                    accountants[
                        client_index
                    ],

                sample_rate=
                    sample_rates[
                        client_index
                    ],

                seed=
                    local_seed,

                round_num=
                    round_num,

                client_id=
                    client_id,
            )

            client_epoch_rows.extend(
                epoch_log
            )

            expected_round_steps = (
                steps_per_epoch(
                    client_sizes[
                        client_index
                    ],
                    args.batch_size,
                )
                * args.local_epochs
            )

            if (
                actual_steps
                != expected_round_steps
            ):

                raise RuntimeError(
                    f"Client {client_id}: "
                    f"expected "
                    f"{expected_round_steps} "
                    f"steps, got "
                    f"{actual_steps}."
                )

            weights = (
                get_trainable_weights(
                    local_model
                )
            )

            if (
                set(weights.keys())
                != initial_keys
            ):

                raise RuntimeError(
                    f"Client {client_id}: "
                    "trainable parameter keys "
                    "changed under Opacus."
                )

            client_weight_list.append(
                weights
            )

            epsilon_now = (
                accountant_epsilon(
                    accountants[
                        client_index
                    ],
                    DELTA,
                )
            )

            round_epsilons.append(
                epsilon_now
            )

            del local_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ====================================================
        # Weighted FedAvg
        # ====================================================

        averaged = (
            federated_averaging(
                client_weight_list,
                client_sizes,
            )
        )

        global_model = (
            set_trainable_weights(
                global_model,
                averaged,
            )
        )

        # ====================================================
        # Global validation only
        # ====================================================

        val_metrics = (
            base.evaluate(
                global_model,
                global_val_loader,
                device,
            )
        )

        worst_epsilon = max(
            round_epsilons
        )

        round_duration = (
            time.time()
            - round_start
        )

        round_times.append(
            round_duration
        )

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
                get_trainable_weights(
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

        row = {
            "seed":
                args.seed,

            "target_epsilon":
                args.epsilon,

            "delta":
                DELTA,

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

            "worst_case_cumulative_epsilon":
                worst_epsilon,

            "round_time_seconds":
                round_duration,
        }

        for index, epsilon_now in enumerate(
            round_epsilons,
            start=1,
        ):

            row[
                f"client{index}_"
                f"cumulative_epsilon"
            ] = epsilon_now

        val_rows.append(
            row
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

        best_tag = (
            " <-- best validation checkpoint"
            if improved
            else ""
        )

        print()
        print("=" * 92)

        print(
            f"GLOBAL DP MODEL | "
            f"Round {round_num:02d} | "
            f"VAL AUROC="
            f"{val_metrics['auc_roc']:.4f} | "
            f"AUPRC="
            f"{val_metrics['auc_pr']:.4f} | "
            f"F1="
            f"{val_metrics['f1']:.4f} | "
            f"MCC="
            f"{val_metrics['mcc']:.4f} | "
            f"worst eps="
            f"{worst_epsilon:.4f}"
            f"{best_tag}"
        )

        print("=" * 92)

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

        # Live recovery logs
        pd.DataFrame(
            val_rows
        ).to_csv(
            os.path.join(
                args.results_dir,
                "dp_val_rounds_live.csv",
            ),
            index=False,
        )

        pd.DataFrame(
            client_epoch_rows
        ).to_csv(
            os.path.join(
                args.results_dir,
                "dp_client_epoch_log_live.csv",
            ),
            index=False,
        )

    # ========================================================
    # Final privacy accounting
    # ========================================================

    final_epsilons = [
        accountant_epsilon(
            accountant,
            DELTA,
        )
        for accountant
        in accountants
    ]

    print()
    print(
        "Final cumulative privacy expenditure"
    )
    print("-" * 92)

    privacy_rows = []

    for client_index, final_epsilon in enumerate(
        final_epsilons
    ):

        client_id = (
            client_index + 1
        )

        privacy_rows.append(
            {
                "seed":
                    args.seed,

                "client_id":
                    client_id,

                "client_name":
                    CLIENT_NAMES[
                        client_index
                    ],

                "target_epsilon":
                    args.epsilon,

                "reported_epsilon":
                    final_epsilon,

                "delta":
                    DELTA,

                "noise_multiplier":
                    noise_multipliers[
                        client_index
                    ],

                "sample_rate":
                    sample_rates[
                        client_index
                    ],

                "total_optimizer_steps":
                    planned_steps[
                        client_index
                    ],

                "max_grad_norm":
                    args.max_grad_norm,
            }
        )

        print(
            f"Client {client_id} "
            f"{CLIENT_NAMES[client_index]:<12} | "
            f"reported eps="
            f"{final_epsilon:.4f} | "
            f"target="
            f"{args.epsilon:.4f}"
        )

    worst_final_epsilon = max(
        final_epsilons
    )

    if (
        worst_final_epsilon
        > args.epsilon
        + 0.05
    ):

        raise RuntimeError(
            f"Final epsilon "
            f"{worst_final_epsilon:.4f} "
            f"materially exceeds "
            f"target "
            f"{args.epsilon:.4f}."
        )

    # Save finished training logs before test
    pd.DataFrame(
        val_rows
    ).to_csv(
        os.path.join(
            args.results_dir,
            "dp_val_rounds.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        client_epoch_rows
    ).to_csv(
        os.path.join(
            args.results_dir,
            "dp_client_epoch_log.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        privacy_rows
    ).to_csv(
        os.path.join(
            args.results_dir,
            "dp_privacy_accounting.csv",
        ),
        index=False,
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
    # FINAL TEST ONCE
    # ========================================================

    print()
    print("=" * 92)

    print(
        f"Loading best validation "
        f"checkpoint from round "
        f"{best_round} for FINAL TEST"
    )

    print("=" * 92)

    global_model = (
        set_trainable_weights(
            global_model,
            best_weights,
        )
    )

    test_metrics = (
        base.evaluate(
            global_model,
            global_test_loader,
            device,
        )
    )

    summary = {
        "seed":
            args.seed,

        "target_epsilon":
            args.epsilon,

        "reported_worst_case_epsilon":
            worst_final_epsilon,

        "delta":
            DELTA,

        "max_grad_norm":
            args.max_grad_norm,

        "num_rounds":
            args.num_rounds,

        "local_epochs":
            args.local_epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "lora_rank":
            args.lora_rank,

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
            time.time()
            - run_start,
    }

    summary_path = (
        os.path.join(
            args.results_dir,
            "dp_test_results.csv",
        )
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 92)

    print(
        "DP EXPERIMENT COMPLETE"
    )

    print("=" * 92)

    print(
        f"Target epsilon          : "
        f"{args.epsilon:.4f}"
    )

    print(
        f"Reported worst epsilon  : "
        f"{worst_final_epsilon:.4f}"
    )

    print(
        f"Delta                   : "
        f"{DELTA:.0e}"
    )

    print(
        f"Selected round          : "
        f"{best_round}"
    )

    print(
        f"Best validation AUROC   : "
        f"{best_val_auc:.6f}"
    )

    print(
        f"TEST AUROC              : "
        f"{test_metrics['auc_roc']:.6f}"
    )

    print(
        f"TEST AUPRC              : "
        f"{test_metrics['auc_pr']:.6f}"
    )

    print(
        f"TEST F1                 : "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        f"TEST MCC                : "
        f"{test_metrics['mcc']:.6f}"
    )

    print(
        f"Runtime                 : "
        f"{format_seconds(time.time() - run_start)}"
    )

    print(
        f"Result                  : "
        f"{summary_path}"
    )

    print("=" * 92)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Homology-aware record-level "
            "DP-SGD experiment."
        )
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        choices=SUPPORTED_EPSILONS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=DEFAULT_LORA_RANK,
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=DEFAULT_LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=MAX_GRAD_NORM,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
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

        if (
            args.results_dir
            is None
        ):

            epsilon_label = (
                str(
                    args.epsilon
                )
                .replace(
                    ".",
                    "p",
                )
            )

            results_dir = (
                f"results/"
                f"dp_eps"
                f"{epsilon_label}_"
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
        print("=" * 92)

        print(
            "DP EXPERIMENT FAILED"
        )

        print("=" * 92)

        print(
            f"Error: {exc}"
        )

        print(
            f"Traceback saved to: "
            f"{error_path}"
        )

        print("=" * 92)

        raise
