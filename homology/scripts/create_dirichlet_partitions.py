import os
import json
import shutil
import argparse

import numpy as np
import pandas as pd


# ============================================================
# Fixed configuration
# ============================================================

NUM_CLIENTS = 5

SOURCE_FILES = [
    f"data/federated/client{i}_train.csv"
    for i in range(1, NUM_CLIENTS + 1)
]

GLOBAL_VAL_FILE = "data/federated/global_val.csv"
GLOBAL_TEST_FILE = "data/federated/global_test.csv"

DEFAULT_ALPHAS = [
    0.1,
    0.5,
    1.0,
    10.0,
]

DEFAULT_PARTITION_SEED = 42

EXPECTED_TOTAL_TRAIN = 321368

# 321,368 split as evenly as possible across five clients
EQUAL_CAPACITIES = np.asarray(
    [
        64274,
        64274,
        64274,
        64273,
        64273,
    ],
    dtype=np.int64,
)


# ============================================================
# Helpers
# ============================================================

def alpha_slug(alpha):
    return str(alpha).replace(".", "p")


def validate_columns(df, path):

    required = {
        "sequence",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    labels = set(
        df["label"]
        .astype(int)
        .unique()
        .tolist()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{path} contains invalid labels: "
            f"{sorted(labels)}"
        )


def check_required_files():

    required = (
        SOURCE_FILES
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


def load_training_pool():

    frames = []

    print("=" * 90)
    print(
        "Loading strict homology-aware "
        "federated training pool"
    )
    print("=" * 90)

    for client_index, path in enumerate(
        SOURCE_FILES,
        start=1,
    ):

        df = pd.read_csv(path)

        validate_columns(
            df,
            path,
        )

        frames.append(
            df.copy()
        )

        positives = int(
            (df["label"] == 1)
            .sum()
        )

        negatives = int(
            (df["label"] == 0)
            .sum()
        )

        print(
            f"Source client {client_index} | "
            f"N={len(df):>7,} | "
            f"pos={positives:>6,} | "
            f"neg={negatives:>7,} | "
            f"positive rate="
            f"{positives / len(df):.4%}"
        )

    pool = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(pool) != EXPECTED_TOTAL_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_TRAIN:,} "
            f"training examples, found "
            f"{len(pool):,}."
        )

    if (
        pool["sequence"]
        .astype(str)
        .duplicated()
        .any()
    ):
        duplicate_count = int(
            pool["sequence"]
            .astype(str)
            .duplicated()
            .sum()
        )

        raise RuntimeError(
            f"Training pool contains "
            f"{duplicate_count} duplicate "
            f"peptide sequences."
        )

    if (
        int(EQUAL_CAPACITIES.sum())
        != len(pool)
    ):
        raise RuntimeError(
            "Equal client capacities do not "
            "sum to training pool size."
        )

    total_positive = int(
        (pool["label"] == 1)
        .sum()
    )

    total_negative = int(
        (pool["label"] == 0)
        .sum()
    )

    print("-" * 90)

    print(
        f"Total training samples : "
        f"{len(pool):,}"
    )

    print(
        f"Positive               : "
        f"{total_positive:,}"
    )

    print(
        f"Negative               : "
        f"{total_negative:,}"
    )

    print(
        f"Global positive rate   : "
        f"{total_positive / len(pool):.4%}"
    )

    print(
        f"Synthetic capacities   : "
        f"{EQUAL_CAPACITIES.tolist()}"
    )

    print("=" * 90)

    return pool


def check_global_overlap(pool):

    train_sequences = set(
        pool["sequence"]
        .astype(str)
        .tolist()
    )

    val_df = pd.read_csv(
        GLOBAL_VAL_FILE
    )

    test_df = pd.read_csv(
        GLOBAL_TEST_FILE
    )

    validate_columns(
        val_df,
        GLOBAL_VAL_FILE,
    )

    validate_columns(
        test_df,
        GLOBAL_TEST_FILE,
    )

    val_sequences = set(
        val_df["sequence"]
        .astype(str)
        .tolist()
    )

    test_sequences = set(
        test_df["sequence"]
        .astype(str)
        .tolist()
    )

    val_overlap = (
        train_sequences
        & val_sequences
    )

    test_overlap = (
        train_sequences
        & test_sequences
    )

    print()
    print(
        "Frozen benchmark integrity"
    )
    print("-" * 90)

    print(
        "Training / global-val exact "
        "peptide overlap:",
        len(val_overlap),
    )

    print(
        "Training / global-test exact "
        "peptide overlap:",
        len(test_overlap),
    )

    if val_overlap:
        raise RuntimeError(
            "Training pool overlaps "
            "global validation."
        )

    if test_overlap:
        raise RuntimeError(
            "Training pool overlaps "
            "global test."
        )

    print(
        "Exact-overlap check: PASS"
    )

    print(
        "Homology separation remains inherited "
        "from the frozen benchmark."
    )


# ============================================================
# Integer constrained allocation
# ============================================================

def allocate_counts(
    total_items,
    proportions,
    capacities,
):

    proportions = np.asarray(
        proportions,
        dtype=np.float64,
    )

    capacities = np.asarray(
        capacities,
        dtype=np.int64,
    )

    if total_items < 0:
        raise ValueError(
            "total_items cannot be negative."
        )

    if total_items > int(
        capacities.sum()
    ):
        raise ValueError(
            "Not enough total capacity."
        )

    if proportions.sum() <= 0:
        raise ValueError(
            "Invalid proportions."
        )

    proportions = (
        proportions
        / proportions.sum()
    )

    allocation = np.zeros(
        len(capacities),
        dtype=np.int64,
    )

    remaining_capacity = (
        capacities.copy()
    )

    remaining_items = int(
        total_items
    )

    while remaining_items > 0:

        active = np.where(
            remaining_capacity > 0
        )[0]

        if len(active) == 0:
            raise RuntimeError(
                "Allocation ran out of capacity."
            )

        weights = proportions[
            active
        ]

        if weights.sum() <= 0:
            weights = np.ones(
                len(active),
                dtype=np.float64,
            )

        weights = (
            weights
            / weights.sum()
        )

        desired = (
            remaining_items
            * weights
        )

        additions = np.floor(
            desired
        ).astype(np.int64)

        additions = np.minimum(
            additions,
            remaining_capacity[
                active
            ],
        )

        added = int(
            additions.sum()
        )

        if added > 0:

            allocation[
                active
            ] += additions

            remaining_capacity[
                active
            ] -= additions

            remaining_items -= added

            continue

        fractional = (
            desired
            - np.floor(desired)
        )

        order = np.argsort(
            -fractional
        )

        made_progress = False

        for local_idx in order:

            client_idx = int(
                active[
                    local_idx
                ]
            )

            if (
                remaining_capacity[
                    client_idx
                ] <= 0
            ):
                continue

            allocation[
                client_idx
            ] += 1

            remaining_capacity[
                client_idx
            ] -= 1

            remaining_items -= 1

            made_progress = True

            if remaining_items == 0:
                break

        if not made_progress:
            raise RuntimeError(
                "Integer allocation stalled."
            )

    if (
        int(allocation.sum())
        != int(total_items)
    ):
        raise RuntimeError(
            "Allocation total mismatch."
        )

    if np.any(
        allocation > capacities
    ):
        raise RuntimeError(
            "Client capacity exceeded."
        )

    return allocation


# ============================================================
# One Dirichlet partition
# ============================================================

def create_partition(
    pool,
    alpha,
    partition_seed,
    output_root,
):

    print()
    print("=" * 90)
    print(
        f"DIRICHLET PARTITION | "
        f"alpha={alpha}"
    )
    print("=" * 90)

    if alpha <= 0:
        raise ValueError(
            "alpha must be > 0."
        )

    total_positive = int(
        (pool["label"] == 1)
        .sum()
    )

    total_negative = int(
        (pool["label"] == 0)
        .sum()
    )

    # --------------------------------------------------------
    # Standard symmetric Dirichlet label preferences
    # --------------------------------------------------------
    #
    # Draw separate client-preference vectors for positives
    # and negatives.
    #
    # Because synthetic client sizes are almost identical,
    # alpha now has a clean interpretation as label skew.
    # --------------------------------------------------------

    alpha_code = int(
        round(
            alpha * 10000
        )
    )

    rng_positive = (
        np.random.default_rng(
            partition_seed
            + alpha_code
            + 100000
        )
    )

    rng_negative = (
        np.random.default_rng(
            partition_seed
            + alpha_code
            + 200000
        )
    )

    concentration = np.full(
        NUM_CLIENTS,
        alpha,
        dtype=np.float64,
    )

    positive_preference = (
        rng_positive.dirichlet(
            concentration
        )
    )

    negative_preference = (
        rng_negative.dirichlet(
            concentration
        )
    )

    # --------------------------------------------------------
    # Positive allocation first
    # --------------------------------------------------------

    positive_counts = (
        allocate_counts(
            total_items=
                total_positive,

            proportions=
                positive_preference,

            capacities=
                EQUAL_CAPACITIES,
        )
    )

    remaining_capacity = (
        EQUAL_CAPACITIES
        - positive_counts
    )

    # --------------------------------------------------------
    # Negative allocation into remaining capacities
    # --------------------------------------------------------

    negative_counts = (
        allocate_counts(
            total_items=
                total_negative,

            proportions=
                negative_preference,

            capacities=
                remaining_capacity,
        )
    )

    final_sizes = (
        positive_counts
        + negative_counts
    )

    if not np.array_equal(
        final_sizes,
        EQUAL_CAPACITIES,
    ):
        raise RuntimeError(
            f"alpha={alpha}: final client "
            f"sizes do not match target "
            f"capacities."
        )

    if (
        int(positive_counts.sum())
        != total_positive
    ):
        raise RuntimeError(
            "Positive total changed."
        )

    if (
        int(negative_counts.sum())
        != total_negative
    ):
        raise RuntimeError(
            "Negative total changed."
        )

    # --------------------------------------------------------
    # Prepare deterministic shuffled label pools
    # --------------------------------------------------------

    positive_df = (
        pool[
            pool["label"] == 1
        ]
        .copy()
        .reset_index(drop=True)
    )

    negative_df = (
        pool[
            pool["label"] == 0
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Same source ordering for every alpha
    positive_order_rng = (
        np.random.default_rng(
            partition_seed
            + 300001
        )
    )

    negative_order_rng = (
        np.random.default_rng(
            partition_seed
            + 400001
        )
    )

    positive_df = (
        positive_df
        .iloc[
            positive_order_rng
            .permutation(
                len(positive_df)
            )
        ]
        .reset_index(drop=True)
    )

    negative_df = (
        negative_df
        .iloc[
            negative_order_rng
            .permutation(
                len(negative_df)
            )
        ]
        .reset_index(drop=True)
    )

    alpha_dir = os.path.join(
        output_root,
        f"alpha_{alpha_slug(alpha)}",
    )

    if os.path.exists(
        alpha_dir
    ):
        shutil.rmtree(
            alpha_dir
        )

    os.makedirs(
        alpha_dir,
        exist_ok=True,
    )

    pos_start = 0
    neg_start = 0

    client_frames = []
    summary_rows = []

    for client_index in range(
        NUM_CLIENTS
    ):

        pos_count = int(
            positive_counts[
                client_index
            ]
        )

        neg_count = int(
            negative_counts[
                client_index
            ]
        )

        pos_end = (
            pos_start
            + pos_count
        )

        neg_end = (
            neg_start
            + neg_count
        )

        client_positive = (
            positive_df.iloc[
                pos_start:pos_end
            ]
        )

        client_negative = (
            negative_df.iloc[
                neg_start:neg_end
            ]
        )

        client_df = pd.concat(
            [
                client_positive,
                client_negative,
            ],
            ignore_index=True,
        )

        expected_size = int(
            EQUAL_CAPACITIES[
                client_index
            ]
        )

        if (
            len(client_df)
            != expected_size
        ):
            raise RuntimeError(
                f"Client {client_index + 1}: "
                f"expected {expected_size:,}, "
                f"generated "
                f"{len(client_df):,}."
            )

        row_rng = (
            np.random.default_rng(
                partition_seed
                + alpha_code
                + 500001
                + client_index
            )
        )

        client_df = (
            client_df
            .iloc[
                row_rng.permutation(
                    len(client_df)
                )
            ]
            .reset_index(drop=True)
        )

        output_path = os.path.join(
            alpha_dir,
            f"client{client_index + 1}_train.csv",
        )

        client_df.to_csv(
            output_path,
            index=False,
        )

        positive_rate = (
            pos_count
            / expected_size
        )

        summary_rows.append(
            {
                "alpha":
                    alpha,

                "partition_seed":
                    partition_seed,

                "client_id":
                    client_index + 1,

                "client_size":
                    expected_size,

                "positive":
                    pos_count,

                "negative":
                    neg_count,

                "positive_rate":
                    positive_rate,

                "raw_positive_preference":
                    positive_preference[
                        client_index
                    ],

                "raw_negative_preference":
                    negative_preference[
                        client_index
                    ],

                "single_class_client":
                    (
                        pos_count == 0
                        or neg_count == 0
                    ),
            }
        )

        client_frames.append(
            client_df
        )

        print(
            f"Client {client_index + 1} | "
            f"N={expected_size:>6,} | "
            f"pos={pos_count:>6,} | "
            f"neg={neg_count:>6,} | "
            f"positive rate="
            f"{positive_rate:>8.4%}"
            + (
                " | SINGLE CLASS"
                if (
                    pos_count == 0
                    or neg_count == 0
                )
                else ""
            )
        )

        pos_start = pos_end
        neg_start = neg_end

    # ========================================================
    # Reconstruction validation
    # ========================================================

    reconstructed = pd.concat(
        client_frames,
        ignore_index=True,
    )

    if (
        len(reconstructed)
        != len(pool)
    ):
        raise RuntimeError(
            f"alpha={alpha}: "
            f"reconstructed size mismatch."
        )

    reconstructed_sequences = (
        reconstructed[
            "sequence"
        ]
        .astype(str)
    )

    duplicate_count = int(
        reconstructed_sequences
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            f"alpha={alpha}: "
            f"{duplicate_count} duplicate "
            f"peptides detected."
        )

    source_set = set(
        pool["sequence"]
        .astype(str)
        .tolist()
    )

    output_set = set(
        reconstructed_sequences
        .tolist()
    )

    missing = (
        source_set
        - output_set
    )

    extra = (
        output_set
        - source_set
    )

    if missing or extra:
        raise RuntimeError(
            f"alpha={alpha}: reconstruction "
            f"failed | missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    positive_rates = (
        summary_df[
            "positive_rate"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    global_positive_rate = (
        total_positive
        / len(pool)
    )

    rate_std = float(
        np.std(
            positive_rates,
            ddof=0,
        )
    )

    rate_range = float(
        positive_rates.max()
        - positive_rates.min()
    )

    mean_abs_deviation = float(
        np.mean(
            np.abs(
                positive_rates
                - global_positive_rate
            )
        )
    )

    single_class_count = int(
        summary_df[
            "single_class_client"
        ].sum()
    )

    summary_df.to_csv(
        os.path.join(
            alpha_dir,
            "partition_summary.csv",
        ),
        index=False,
    )

    metadata = {
        "alpha":
            float(alpha),

        "partition_seed":
            int(partition_seed),

        "num_clients":
            NUM_CLIENTS,

        "total_samples":
            int(len(pool)),

        "client_capacities":
            [
                int(x)
                for x
                in EQUAL_CAPACITIES
            ],

        "total_positive":
            int(total_positive),

        "total_negative":
            int(total_negative),

        "global_positive_rate":
            float(
                global_positive_rate
            ),

        "positive_preferences":
            [
                float(x)
                for x
                in positive_preference
            ],

        "negative_preferences":
            [
                float(x)
                for x
                in negative_preference
            ],

        "realized_positive_rates":
            [
                float(x)
                for x
                in positive_rates
            ],

        "positive_rate_std":
            rate_std,

        "positive_rate_range":
            rate_range,

        "mean_absolute_rate_deviation":
            mean_abs_deviation,

        "single_class_clients":
            single_class_count,

        "exact_reconstruction":
            True,

        "duplicate_peptides":
            0,
    }

    with open(
        os.path.join(
            alpha_dir,
            "partition_metadata.json",
        ),
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print("-" * 90)

    print(
        f"Positive-rate SD        : "
        f"{rate_std:.6f}"
    )

    print(
        f"Positive-rate range     : "
        f"{rate_range:.6f}"
    )

    print(
        f"Mean abs. deviation     : "
        f"{mean_abs_deviation:.6f}"
    )

    print(
        f"Single-class clients    : "
        f"{single_class_count}"
    )

    print(
        "Exact reconstruction   : PASS"
    )

    print(
        "Duplicate peptides      : 0"
    )

    return summary_df


# ============================================================
# Main
# ============================================================

def main(args):

    check_required_files()

    pool = (
        load_training_pool()
    )

    check_global_overlap(
        pool
    )

    if os.path.exists(
        args.output_root
    ):

        print()
        print(
            f"Replacing previous "
            f"Dirichlet outputs: "
            f"{args.output_root}"
        )

        shutil.rmtree(
            args.output_root
        )

    os.makedirs(
        args.output_root,
        exist_ok=True,
    )

    all_summaries = []

    for alpha in args.alphas:

        summary = (
            create_partition(
                pool=pool,
                alpha=alpha,
                partition_seed=
                    args.partition_seed,
                output_root=
                    args.output_root,
            )
        )

        all_summaries.append(
            summary
        )

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_path = os.path.join(
        args.output_root,
        "all_partition_summary.csv",
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print()
    print("=" * 90)

    print(
        "ALL DIRICHLET PARTITIONS COMPLETE"
    )

    print("=" * 90)

    for alpha in args.alphas:

        subset = (
            combined[
                combined["alpha"]
                == alpha
            ]
        )

        print()
        print(
            f"alpha={alpha}"
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  Client "
                f"{int(row['client_id'])}: "
                f"N="
                f"{int(row['client_size']):,} | "
                f"positive rate="
                f"{row['positive_rate']:.4%}"
            )

        rates = (
            subset[
                "positive_rate"
            ]
            .to_numpy(
                dtype=float
            )
        )

        print(
            f"  SD="
            f"{np.std(rates, ddof=0):.6f} | "
            f"range="
            f"{rates.max() - rates.min():.6f}"
        )

    print()
    print(
        f"Combined summary:\n"
        f"{combined_path}"
    )

    print()
    print(
        "Inspect the realized positive "
        "rates before model training."
    )

    print("=" * 90)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )

    parser.add_argument(
        "--partition_seed",
        type=int,
        default=
            DEFAULT_PARTITION_SEED,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=
            "data/dirichlet",
    )

    args = parser.parse_args()

    main(args)
