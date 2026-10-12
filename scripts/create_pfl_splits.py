from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Configuration
# ============================================================

BASE_SEED = 42
NUM_CLIENTS = 5

# Search many deterministic candidate group splits.
# We select ONE split per client and then freeze it.
N_SEARCH_SEEDS = 1000

CLIENT_NAMES = [
    "Coronavirus",
    "Parasite",
    "Human/Self",
    "Flavivirus",
    "Bacteria",
]

SOURCE_DIR = Path("data/federated")
OUT_DIR = Path("data/personalization")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def positive_rate(df):
    if len(df) == 0:
        return np.nan

    return float(
        df["label"].mean()
    )


def overlap_count(a, b, column):
    return len(
        set(a[column])
        & set(b[column])
    )


def check_required_columns(df, path):

    required = {
        "sequence",
        "label",
        "cluster_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )


def split_score(
    source_df,
    val_df,
    test_df,
):
    """
    Lower is better.

    We care about:
      1. approximately equal sample counts;
      2. val prevalence close to source prevalence;
      3. test prevalence close to source prevalence;
      4. val/test prevalence close to each other.

    All candidate splits are already cluster-disjoint.
    """

    source_rate = positive_rate(
        source_df
    )

    val_rate = positive_rate(
        val_df
    )

    test_rate = positive_rate(
        test_df
    )

    total_n = len(source_df)

    val_fraction = (
        len(val_df)
        / total_n
    )

    test_fraction = (
        len(test_df)
        / total_n
    )

    # Sample-balance penalty
    size_penalty = (
        abs(val_fraction - 0.5)
        + abs(test_fraction - 0.5)
    )

    # Label-distribution penalty
    prevalence_penalty = (
        abs(val_rate - source_rate)
        + abs(test_rate - source_rate)
        + abs(val_rate - test_rate)
    )

    # Give sample balance and prevalence balance
    # comparable influence.
    score = (
        size_penalty
        + 2.0 * prevalence_penalty
    )

    return float(score)


def find_best_split(
    heldout_df,
    client_id,
):
    """
    Search multiple deterministic StratifiedGroupKFold
    random states and keep the candidate giving the best
    combination of:
      - approximately 50/50 sample count;
      - similar class prevalence;
      - strict cluster separation.

    The test set is NOT chosen based on model performance.
    Only dataset composition is used.
    """

    best = None

    y = (
        heldout_df["label"]
        .to_numpy()
    )

    groups = (
        heldout_df["cluster_id"]
        .to_numpy()
    )

    dummy_x = np.zeros(
        len(heldout_df)
    )

    for offset in range(
        N_SEARCH_SEEDS
    ):

        split_seed = (
            BASE_SEED
            + offset
        )

        splitter = (
            StratifiedGroupKFold(
                n_splits=2,
                shuffle=True,
                random_state=split_seed,
            )
        )

        # With n_splits=2 we inspect both orientations.
        folds = list(
            splitter.split(
                dummy_x,
                y,
                groups,
            )
        )

        for orientation, (
            val_idx,
            test_idx,
        ) in enumerate(folds):

            val_df = (
                heldout_df
                .iloc[val_idx]
                .copy()
                .reset_index(drop=True)
            )

            test_df = (
                heldout_df
                .iloc[test_idx]
                .copy()
                .reset_index(drop=True)
            )

            # AUROC requires both classes.
            if (
                val_df["label"].nunique()
                < 2
            ):
                continue

            if (
                test_df["label"].nunique()
                < 2
            ):
                continue

            score = split_score(
                heldout_df,
                val_df,
                test_df,
            )

            candidate = {
                "score":
                    score,
                "split_seed":
                    split_seed,
                "orientation":
                    orientation,
                "val_df":
                    val_df,
                "test_df":
                    test_df,
            }

            if (
                best is None
                or score
                < best["score"]
            ):
                best = candidate

    if best is None:
        raise RuntimeError(
            f"Client {client_id}: "
            "could not construct a valid "
            "two-class cluster-disjoint split."
        )

    return best


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 92)
    print(
        "Creating balanced homology-aware "
        "PFL validation/test splits"
    )
    print("=" * 92)

    print(
        f"Base split seed          : "
        f"{BASE_SEED}"
    )

    print(
        f"Candidate random states  : "
        f"{N_SEARCH_SEEDS}"
    )

    print(
        "Source                   : "
        "existing client*_val.csv"
    )

    print(
        "Grouping                 : "
        "cluster_id"
    )

    print(
        "Target                   : "
        "~50/50 samples with similar "
        "positive prevalence"
    )

    print(
        "IMPORTANT                : "
        "split selection uses only "
        "dataset composition, never model performance"
    )

    print("=" * 92)

    summary_rows = []

    for client_id in range(
        1,
        NUM_CLIENTS + 1
    ):

        client_name = (
            CLIENT_NAMES[
                client_id - 1
            ]
        )

        train_path = (
            SOURCE_DIR
            / f"client{client_id}_train.csv"
        )

        heldout_path = (
            SOURCE_DIR
            / f"client{client_id}_val.csv"
        )

        if not train_path.exists():
            raise FileNotFoundError(
                train_path
            )

        if not heldout_path.exists():
            raise FileNotFoundError(
                heldout_path
            )

        train_df = pd.read_csv(
            train_path
        )

        heldout_df = pd.read_csv(
            heldout_path
        )

        check_required_columns(
            train_df,
            train_path,
        )

        check_required_columns(
            heldout_df,
            heldout_path,
        )

        # ----------------------------------------------------
        # Confirm original local train/held-out separation
        # ----------------------------------------------------

        original_cluster_overlap = (
            overlap_count(
                train_df,
                heldout_df,
                "cluster_id",
            )
        )

        original_peptide_overlap = (
            overlap_count(
                train_df,
                heldout_df,
                "sequence",
            )
        )

        if original_cluster_overlap != 0:
            raise RuntimeError(
                f"Client {client_id}: "
                f"original train/val cluster "
                f"overlap = {original_cluster_overlap}"
            )

        if original_peptide_overlap != 0:
            raise RuntimeError(
                f"Client {client_id}: "
                f"original train/val peptide "
                f"overlap = {original_peptide_overlap}"
            )

        # ----------------------------------------------------
        # Search for best balanced cluster split
        # ----------------------------------------------------

        best = find_best_split(
            heldout_df,
            client_id,
        )

        pfl_val = (
            best["val_df"]
        )

        pfl_test = (
            best["test_df"]
        )

        # ----------------------------------------------------
        # Final overlap checks
        # ----------------------------------------------------

        train_val_cluster_overlap = (
            overlap_count(
                train_df,
                pfl_val,
                "cluster_id",
            )
        )

        train_test_cluster_overlap = (
            overlap_count(
                train_df,
                pfl_test,
                "cluster_id",
            )
        )

        val_test_cluster_overlap = (
            overlap_count(
                pfl_val,
                pfl_test,
                "cluster_id",
            )
        )

        train_val_peptide_overlap = (
            overlap_count(
                train_df,
                pfl_val,
                "sequence",
            )
        )

        train_test_peptide_overlap = (
            overlap_count(
                train_df,
                pfl_test,
                "sequence",
            )
        )

        val_test_peptide_overlap = (
            overlap_count(
                pfl_val,
                pfl_test,
                "sequence",
            )
        )

        overlaps = {
            "train-val cluster":
                train_val_cluster_overlap,

            "train-test cluster":
                train_test_cluster_overlap,

            "val-test cluster":
                val_test_cluster_overlap,

            "train-val peptide":
                train_val_peptide_overlap,

            "train-test peptide":
                train_test_peptide_overlap,

            "val-test peptide":
                val_test_peptide_overlap,
        }

        bad = {
            name: value
            for name, value
            in overlaps.items()
            if value != 0
        }

        if bad:
            raise RuntimeError(
                f"Client {client_id}: "
                f"overlap checks failed: "
                f"{bad}"
            )

        # ----------------------------------------------------
        # Reconstruction check
        # ----------------------------------------------------

        combined = pd.concat(
            [
                pfl_val,
                pfl_test,
            ],
            ignore_index=True,
        )

        if len(combined) != len(
            heldout_df
        ):
            raise RuntimeError(
                f"Client {client_id}: "
                "sample count changed during split."
            )

        if set(
            combined["sequence"]
        ) != set(
            heldout_df["sequence"]
        ):
            raise RuntimeError(
                f"Client {client_id}: "
                "val+test does not reconstruct "
                "the source held-out peptides."
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        val_out = (
            OUT_DIR
            / f"client{client_id}_val.csv"
        )

        test_out = (
            OUT_DIR
            / f"client{client_id}_test.csv"
        )

        pfl_val.to_csv(
            val_out,
            index=False,
        )

        pfl_test.to_csv(
            test_out,
            index=False,
        )

        source_rate = positive_rate(
            heldout_df
        )

        val_rate = positive_rate(
            pfl_val
        )

        test_rate = positive_rate(
            pfl_test
        )

        val_fraction = (
            len(pfl_val)
            / len(heldout_df)
        )

        test_fraction = (
            len(pfl_test)
            / len(heldout_df)
        )

        row = {
            "client_id":
                client_id,

            "client_name":
                client_name,

            "selected_split_seed":
                best[
                    "split_seed"
                ],

            "selected_orientation":
                best[
                    "orientation"
                ],

            "split_score":
                best[
                    "score"
                ],

            "train_n":
                len(train_df),

            "train_positive":
                int(
                    train_df[
                        "label"
                    ].sum()
                ),

            "train_positive_rate":
                positive_rate(
                    train_df
                ),

            "train_clusters":
                train_df[
                    "cluster_id"
                ].nunique(),

            "source_heldout_n":
                len(heldout_df),

            "source_heldout_positive":
                int(
                    heldout_df[
                        "label"
                    ].sum()
                ),

            "source_heldout_positive_rate":
                source_rate,

            "source_heldout_clusters":
                heldout_df[
                    "cluster_id"
                ].nunique(),

            "pfl_val_n":
                len(pfl_val),

            "pfl_val_fraction":
                val_fraction,

            "pfl_val_positive":
                int(
                    pfl_val[
                        "label"
                    ].sum()
                ),

            "pfl_val_positive_rate":
                val_rate,

            "pfl_val_clusters":
                pfl_val[
                    "cluster_id"
                ].nunique(),

            "pfl_test_n":
                len(pfl_test),

            "pfl_test_fraction":
                test_fraction,

            "pfl_test_positive":
                int(
                    pfl_test[
                        "label"
                    ].sum()
                ),

            "pfl_test_positive_rate":
                test_rate,

            "pfl_test_clusters":
                pfl_test[
                    "cluster_id"
                ].nunique(),

            "val_test_prevalence_difference":
                abs(
                    val_rate
                    - test_rate
                ),

            "val_source_prevalence_difference":
                abs(
                    val_rate
                    - source_rate
                ),

            "test_source_prevalence_difference":
                abs(
                    test_rate
                    - source_rate
                ),

            "train_val_cluster_overlap":
                train_val_cluster_overlap,

            "train_test_cluster_overlap":
                train_test_cluster_overlap,

            "val_test_cluster_overlap":
                val_test_cluster_overlap,

            "train_val_peptide_overlap":
                train_val_peptide_overlap,

            "train_test_peptide_overlap":
                train_test_peptide_overlap,

            "val_test_peptide_overlap":
                val_test_peptide_overlap,
        }

        summary_rows.append(
            row
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()
        print("-" * 92)

        print(
            f"Client {client_id}: "
            f"{client_name}"
        )

        print("-" * 92)

        print(
            f"Original held-out    : "
            f"N={len(heldout_df):,} | "
            f"pos="
            f"{int(heldout_df['label'].sum()):,} | "
            f"rate="
            f"{source_rate:.4%} | "
            f"clusters="
            f"{heldout_df['cluster_id'].nunique():,}"
        )

        print(
            f"PFL validation       : "
            f"N={len(pfl_val):,} "
            f"({val_fraction:.2%}) | "
            f"pos="
            f"{int(pfl_val['label'].sum()):,} | "
            f"rate="
            f"{val_rate:.4%} | "
            f"clusters="
            f"{pfl_val['cluster_id'].nunique():,}"
        )

        print(
            f"PFL test             : "
            f"N={len(pfl_test):,} "
            f"({test_fraction:.2%}) | "
            f"pos="
            f"{int(pfl_test['label'].sum()):,} | "
            f"rate="
            f"{test_rate:.4%} | "
            f"clusters="
            f"{pfl_test['cluster_id'].nunique():,}"
        )

        print(
            f"Val-test prevalence Δ: "
            f"{abs(val_rate - test_rate):.4%}"
        )

        print(
            f"Selected search seed : "
            f"{best['split_seed']}"
        )

        print(
            f"Composition score    : "
            f"{best['score']:.6f}"
        )

        print(
            "Cluster overlaps      : "
            f"train-val="
            f"{train_val_cluster_overlap}, "
            f"train-test="
            f"{train_test_cluster_overlap}, "
            f"val-test="
            f"{val_test_cluster_overlap}"
        )

        print(
            "Peptide overlaps      : "
            f"train-val="
            f"{train_val_peptide_overlap}, "
            f"train-test="
            f"{train_test_peptide_overlap}, "
            f"val-test="
            f"{val_test_peptide_overlap}"
        )

        print(
            f"Saved validation     : "
            f"{val_out}"
        )

        print(
            f"Saved test           : "
            f"{test_out}"
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUT_DIR
        / "split_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 92)

    print(
        "PFL SPLIT SEARCH COMPLETE"
    )

    print("=" * 92)

    print(
        "The split was chosen using only sample counts, "
        "labels, and cluster assignments."
    )

    print(
        "No model predictions or performance metrics "
        "were used to choose the split."
    )

    print(
        "All train/validation/test cluster overlaps are zero."
    )

    print(
        "All train/validation/test peptide overlaps are zero."
    )

    print(
        "These files should now be FROZEN and reused for "
        "model seeds 42, 43, and 44."
    )

    print(
        f"Summary: {summary_path}"
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
