import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42

DATA = Path("data/protein_level_split.csv")
CLUSTERS = Path(
    "results/mmseqs30_both/protein_clusters_cluster.tsv"
)
OUTPUT = Path("data/homology_split.csv")

# --------------------------------------------------
# Load peptide data
# --------------------------------------------------

df = pd.read_csv(DATA)

def normalize_id(x):
    x = str(x)

    if "uniprot.org/uniprot/" in x:
        return "UNIPROT|" + x.split("/")[-1]

    if "ncbi.nlm.nih.gov/protein/" in x:
        return "NCBI|" + x.split("/")[-1]

    return None

df["protein_key"] = df["protein_id"].apply(normalize_id)

# Keep proteins that we can map to retrieved sequences
df = df[df["protein_key"].notna()].copy()

# --------------------------------------------------
# Load MMseqs2 cluster assignments
# --------------------------------------------------

clusters = pd.read_csv(
    CLUSTERS,
    sep="\t",
    header=None,
    names=["cluster_id", "protein_key"]
)

protein_to_cluster = dict(
    zip(
        clusters["protein_key"],
        clusters["cluster_id"]
    )
)

df["cluster_id"] = df["protein_key"].map(
    protein_to_cluster
)

# Keep only proteins that were successfully clustered
df = df[df["cluster_id"].notna()].copy()

df = df.reset_index(drop=True)

print("=" * 60)
print("DATA AVAILABLE FOR HOMOLOGY SPLIT")
print("=" * 60)

print("Peptides:", len(df))
print("Proteins:", df["protein_key"].nunique())
print("Clusters:", df["cluster_id"].nunique())
print(
    "Positive rate:",
    round(100 * df["label"].mean(), 2),
    "%"
)

# --------------------------------------------------
# Make 10 group-disjoint stratified folds
# --------------------------------------------------

sgkf = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=SEED
)

df["fold"] = -1

X = np.zeros((len(df), 1))
y = df["label"].values
groups = df["cluster_id"].values

for fold_id, (_, fold_idx) in enumerate(
    sgkf.split(X, y, groups)
):
    df.loc[fold_idx, "fold"] = fold_id

# --------------------------------------------------
# Show fold statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("10-FOLD SUMMARY")
print("=" * 60)

for fold in range(10):

    part = df[df["fold"] == fold]

    print(
        f"Fold {fold}: "
        f"peptides={len(part)}, "
        f"positive={int(part['label'].sum())}, "
        f"positive_rate={100 * part['label'].mean():.2f}%, "
        f"clusters={part['cluster_id'].nunique()}"
    )

# --------------------------------------------------
# Choose the best validation/test pair
# --------------------------------------------------

total_n = len(df)
global_rate = df["label"].mean()

target_n = total_n * 0.10

best_pair = None
best_score = None

for val_fold in range(10):

    for test_fold in range(10):

        if val_fold == test_fold:
            continue

        val = df[df["fold"] == val_fold]
        test = df[df["fold"] == test_fold]

        # How far each fold is from 10% of peptides
        val_size_error = (
            abs(len(val) - target_n) / target_n
        )

        test_size_error = (
            abs(len(test) - target_n) / target_n
        )

        # How far each fold's class balance is
        # from the overall dataset
        val_rate_error = (
            abs(val["label"].mean() - global_rate)
            / global_rate
        )

        test_rate_error = (
            abs(test["label"].mean() - global_rate)
            / global_rate
        )

        score = (
            val_size_error
            + test_size_error
            + val_rate_error
            + test_rate_error
        )

        if best_score is None or score < best_score:
            best_score = score
            best_pair = (
                val_fold,
                test_fold
            )

val_fold, test_fold = best_pair

print("\nSelected validation fold:", val_fold)
print("Selected test fold:", test_fold)

# --------------------------------------------------
# Final train / val / test split
# --------------------------------------------------

df["split"] = "train"

df.loc[
    df["fold"] == val_fold,
    "split"
] = "val"

df.loc[
    df["fold"] == test_fold,
    "split"
] = "test"

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL HOMOLOGY-AWARE SPLIT")
print("=" * 60)

for split in ["train", "val", "test"]:

    part = df[df["split"] == split]

    print(f"\n{split.upper()}")

    print(
        "Peptides:",
        len(part)
    )

    print(
        "Percentage:",
        round(
            100 * len(part) / len(df),
            2
        ),
        "%"
    )

    print(
        "Proteins:",
        part["protein_key"].nunique()
    )

    print(
        "Clusters:",
        part["cluster_id"].nunique()
    )

    print(
        "Positive:",
        int(part["label"].sum())
    )

    print(
        "Positive rate:",
        round(
            100 * part["label"].mean(),
            2
        ),
        "%"
    )

# --------------------------------------------------
# Leakage checks
# --------------------------------------------------

print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

sets = {}

for split in ["train", "val", "test"]:

    part = df[df["split"] == split]

    sets[split] = {
        "clusters": set(
            part["cluster_id"]
        ),
        "proteins": set(
            part["protein_key"]
        ),
        "peptides": set(
            part["sequence"]
        )
    }

for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:

    print(
        f"{a}-{b} cluster overlap:",
        len(
            sets[a]["clusters"]
            & sets[b]["clusters"]
        )
    )

    print(
        f"{a}-{b} protein overlap:",
        len(
            sets[a]["proteins"]
            & sets[b]["proteins"]
        )
    )

    print(
        f"{a}-{b} peptide overlap:",
        len(
            sets[a]["peptides"]
            & sets[b]["peptides"]
        )
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)