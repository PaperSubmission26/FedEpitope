import os
import copy
import time
import argparse
import traceback

import numpy as np
import pandas as pd
import torch

import federated_train as base


# ============================================================
# IID experiment configuration
# ============================================================

NUM_CLIENTS = 5

CLIENT_TRAIN_FILES = [
    f"data/federated_iid/client{i}_train.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

GLOBAL_VAL_FILE = "data/federated/global_val.csv"
GLOBAL_TEST_FILE = "data/federated/global_test.csv"

CLIENT_NAMES = [
    "IID-1",
    "IID-2",
    "IID-3",
    "IID-4",
    "IID-5",
]

# train_local() in the validated base implementation uses
# base.CLIENT_NAMES only for terminal logging.
# Override those labels so IID clients are not misleadingly
# printed as Coronavirus/Parasite/etc.
base.CLIENT_NAMES = CLIENT_NAMES


# ============================================================
# Required-file check
# ============================================================

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
# Main
# ============================================================

def main(args):

    run_start_time = time.time()
    round_times = []

    base.set_seed(args.seed)

    check_required_files()

    if args.results_dir is None:
        args.results_dir = (
            f"results/fedavg_iid_seed{args.seed}"
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
            "facebook/esm2_t12_35M_UR50D"
        )
    )

    # ========================================================
    # Configuration
    # ========================================================

    print("=" * 80)
    print(
        "FedEpitope Homology-Aware IID "
        "Federated Training"
    )
    print("=" * 80)

    print(f"Device               : {device}")
    print(f"Seed                 : {args.seed}")
    print(f"Federated rounds     : {args.num_rounds}")
    print(f"Local epochs         : {args.local_epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Learning rate        : {args.lr}")
    print(f"LoRA rank            : {args.lora_rank}")
    print(f"LoRA alpha           : {2 * args.lora_rank}")
    print("Aggregation          : weighted FedAvg")
    print(
        "Partition            : fixed stratified IID "
        "(partition seed 42)"
    )
    print(
        "Client sizes         : preserved from natural "
        "non-IID partition"
    )
    print(
        "Checkpoint selection : global homology "
        "validation AUROC"
    )
    print(
        "Global test usage    : final evaluation only"
    )
    print(f"Global validation    : {GLOBAL_VAL_FILE}")
    print(f"Global test          : {GLOBAL_TEST_FILE}")
    print(f"Results directory    : {args.results_dir}")

    print("=" * 80)
    print()

    base.print_gpu_memory()
    print()

    # ========================================================
    # Client training loaders
    # ========================================================

    train_loaders = []
    class_weights_list = []
    client_train_sizes = []

    print("IID CLIENT DATA")
    print("=" * 80)

    for client_id in range(NUM_CLIENTS):

        train_dataset = (
            base.EpitopeDataset(
                CLIENT_TRAIN_FILES[
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
            train_df["label"].sum()
        )

        neg = int(
            len(train_df) - pos
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
            )
            .to(device)
        )

        generator = torch.Generator()

        generator.manual_seed(
            args.seed + client_id
        )

        train_loader = (
            base.make_loader(
                train_dataset,
                batch_size=
                    args.batch_size,
                shuffle=True,
                args=args,
                generator=generator,
            )
        )

        train_loaders.append(
            train_loader
        )

        class_weights_list.append(
            class_weights
        )

        client_train_sizes.append(
            len(train_dataset)
        )

        print(
            f"Client {client_id + 1} "
            f"({CLIENT_NAMES[client_id]}) | "
            f"Train: {len(train_dataset):>7,} | "
            f"Positive: {pos:>6,} | "
            f"Negative: {neg:>7,} | "
            f"Positive rate: "
            f"{pos / len(train_df):.4%} | "
            f"Class weight: "
            f"{positive_weight:.4f}"
        )

    print()
    print(
        "Total IID training samples:",
        f"{sum(client_train_sizes):,}"
    )

    # Expected sizes from natural non-IID experiment
    expected_sizes = [
        139375,
        133124,
        20060,
        14795,
        14014,
    ]

    if client_train_sizes != expected_sizes:
        raise RuntimeError(
            "IID client sizes do not match the "
            "original natural non-IID client sizes.\n"
            f"Expected: {expected_sizes}\n"
            f"Found:    {client_train_sizes}"
        )

    if sum(client_train_sizes) != 321368:
        raise RuntimeError(
            "Unexpected total IID training size: "
            f"{sum(client_train_sizes)}"
        )

    print(
        "Client-size sanity check: PASSED"
    )

    # ========================================================
    # Global validation loader
    # ========================================================

    global_val_dataset = (
        base.EpitopeDataset(
            GLOBAL_VAL_FILE,
            tokenizer,
        )
    )

    global_val_loader = (
        base.make_loader(
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
    # Initial global model
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

        round_start_time = time.time()

        print()
        print("=" * 80)
        print(
            f"ROUND {round_num}/"
            f"{args.num_rounds}"
        )
        print("=" * 80)

        client_weights_list = []

        # ----------------------------------------------------
        # Local training
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

            # Every client starts from the same current
            # aggregated global checkpoint.
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
                base.train_local(
                    model=local_model,
                    loader=
                        train_loaders[
                            client_id
                        ],
                    optimizer=optimizer,
                    device=device,
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

            client_time = (
                time.time()
                - client_start_time
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
                "client_time_seconds":
                    client_time,
            }

            for epoch_index in range(
                args.local_epochs
            ):

                log_row[
                    f"epoch_"
                    f"{epoch_index + 1}_loss"
                ] = (
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
                base.get_trainable_weights(
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
            "Aggregating IID client models "
            "using weighted FedAvg..."
        )

        averaged_weights = (
            base.federated_averaging(
                client_weights_list=
                    client_weights_list,
                client_sizes=
                    client_train_sizes,
            )
        )

        global_model = (
            base.set_trainable_weights(
                global_model,
                averaged_weights,
            )
        )

        # ----------------------------------------------------
        # Global validation
        # ----------------------------------------------------

        print()
        print(
            "Evaluating aggregated global "
            "model on frozen homology-disjoint "
            "GLOBAL validation..."
        )

        global_val_metrics = (
            base.evaluate(
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
            ] > best_val_auc
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
                base.get_trainable_weights(
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
        # Timing
        # ----------------------------------------------------

        round_duration = (
            time.time()
            - round_start_time
        )

        round_times.append(
            round_duration
        )

        elapsed_time = (
            time.time()
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
        # Record round-level metrics
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

        # ----------------------------------------------------
        # Terminal progress
        # ----------------------------------------------------

        print()
        print("=" * 80)

        print(
            f"GLOBAL IID MODEL | "
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
        print("Progress monitor")

        print(
            f"  Finished round       : "
            f"{round_num}/"
            f"{args.num_rounds}"
        )

        print(
            f"  Round time           : "
            f"{base.format_seconds(round_duration)}"
        )

        print(
            f"  Elapsed time         : "
            f"{base.format_seconds(elapsed_time)}"
        )

        print(
            f"  Estimated time left : "
            f"{base.format_seconds(eta_seconds)}"
        )

        print(
            f"  Best validation AUC : "
            f"{best_val_auc:.4f}"
        )

        print(
            f"  Best round so far   : "
            f"{best_round}"
        )

        base.print_gpu_memory()

        # ----------------------------------------------------
        # Live logs after every round
        # ----------------------------------------------------

        pd.DataFrame(
            val_results
        ).to_csv(
            os.path.join(
                args.results_dir,
                "federated_val_rounds_live.csv",
            ),
            index=False,
        )

        pd.DataFrame(
            client_round_logs
        ).to_csv(
            os.path.join(
                args.results_dir,
                "client_round_logs_live.csv",
            ),
            index=False,
        )

    # ========================================================
    # Save completed round logs
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
    # Save final-round global weights
    # ========================================================

    final_weights_path = (
        os.path.join(
            args.results_dir,
            "global_weights_final.pt",
        )
    )

    torch.save(
        base.get_trainable_weights(
            global_model
        ),
        final_weights_path,
    )

    if best_weights is None:
        raise RuntimeError(
            "No best validation checkpoint "
            "was saved."
        )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Loading best validation checkpoint "
        "for final homology-disjoint test"
    )
    print("=" * 80)

    global_model = (
        base.set_trainable_weights(
            global_model,
            best_weights,
        )
    )

    test_dataset = (
        base.EpitopeDataset(
            GLOBAL_TEST_FILE,
            tokenizer,
        )
    )

    test_loader = (
        base.make_loader(
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
        base.evaluate(
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
    # Save test summary
    # ========================================================

    test_summary = {
        "partition":
            "IID",
        "partition_seed":
            42,
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
        "IID federated training complete"
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
        f"{base.format_seconds(total_run_time)}"
    )

    print()
    print(
        f"Figure 2 convergence log       : "
        f"{val_rounds_path}"
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
        default=base.NUM_ROUNDS,
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=base.LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=base.BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=base.LR,
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
        default=None,
    )

    args = parser.parse_args()

    try:
        main(args)

    except Exception as exc:

        if args.results_dir is None:
            args.results_dir = (
                f"results/"
                f"fedavg_iid_seed{args.seed}"
            )

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
                "IID federated training failed.\n\n"
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
        print("Training failed")
        print("=" * 80)
        print(f"Error: {exc}")
        print(
            f"Full traceback saved to: "
            f"{error_path}"
        )
        print("=" * 80)

        raise
