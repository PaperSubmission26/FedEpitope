import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
N_SEARCH = 10000
VAL_FRACTION = 0.10

INPUT = Path("data/homology_split.csv")
OUTDIR = Path("data/federated")

OUTDIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# Load homology-aware dataset
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 70)
print("HOMOLOGY-AWARE DATASET")
print("=" * 70)

print("Total peptides:", len(df))
print(df["split"].value_counts())

train_pool = df[df["split"] == "train"].copy()
global_val = df[df["split"] == "val"].copy()
global_test = df[df["split"] == "test"].copy()

print("\nTraining pool:", len(train_pool))
print("Global validation:", len(global_val))
print("Global test:", len(global_test))


# ============================================================
# Biological client definitions
# ============================================================

client_organism_groups = {

    "client1": [
        "Severe acute respiratory syndrome coronavirus 2",
        "Severe acute respiratory syndrome coronavirus 2 Wuhan/Hu-1/2019",
        "SARS-CoV1",
        "SARS coronavirus Tor2",
        "Human coronavirus OC43",
        "Human coronavirus NL63",
        "Human coronavirus 229E",
        "Human coronavirus HKU1",
        "Middle East respiratory syndrome-related coronavirus",
        "Pangolin coronavirus",
    ],

    "client2": [
        "Trypanosoma cruzi strain CL Brener",
        "Schistosoma mansoni",
        "Onchocerca volvulus",
        "Plasmodium falciparum",
        "Leishmania donovani",
    ],

    "client3": [
        "Homo sapiens",
        "Mus musculus",
    ],

    "client4": [
        "Hepacivirus hominis",
        "Dengue virus",
        "Zika virus",
        "Hepatitis B virus",
        "West Nile virus",
    ],

    "client5": [
        "Streptococcus pyogenes",
        "Mycobacterium tuberculosis",
        "Staphylococcus aureus",
        "Bacillus anthracis",
        "Yersinia pestis",
        "Clostridium tetani",
        "Influenza A virus",
        "Human immunodeficiency virus 1",
    ],
}


# ============================================================
# Initial biological assignment
# ============================================================

assigned_organisms = set()
client_dfs = {}

print("\n" + "=" * 70)
print("INITIAL BIOLOGICAL ASSIGNMENT")
print("=" * 70)

for client_id, organisms in client_organism_groups.items():

    part = train_pool[
        train_pool["organism"].isin(organisms)
    ].copy()

    client_dfs[client_id] = part

    assigned_organisms.update(organisms)

    print(
        f"{client_id}: "
        f"{len(part)} peptides | "
        f"{part['organism'].nunique()} organisms | "
        f"{100 * part['label'].mean():.2f}% positive"
    )


# ============================================================
# Remaining organisms
# ============================================================

remaining = train_pool[
    ~train_pool["organism"].isin(assigned_organisms)
].copy()

print("\nRemaining peptides:", len(remaining))
print("Remaining organisms:", remaining["organism"].nunique())

remaining_orgs = (
    remaining["organism"]
    .value_counts()
    .sort_values(ascending=False)
)

client_ids = list(client_organism_groups.keys())

# Same round-robin rule as previous setup
for idx, organism in enumerate(remaining_orgs.index):

    target_client = client_ids[idx % len(client_ids)]

    org_df = remaining[
        remaining["organism"] == organism
    ]

    client_dfs[target_client] = pd.concat(
        [client_dfs[target_client], org_df],
        ignore_index=True
    )


# ============================================================
# Final client statistics
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLIENT ASSIGNMENT")
print("=" * 70)

for client_id, part in client_dfs.items():

    print(
        f"{client_id}: "
        f"{len(part)} peptides | "
        f"{part['organism'].nunique()} organisms | "
        f"{part['protein_key'].nunique()} proteins | "
        f"{part['cluster_id'].nunique()} clusters | "
        f"{100 * part['label'].mean():.2f}% positive"
    )


# ============================================================
# Make balanced cluster-disjoint validation set
# ============================================================

def find_balanced_cluster_split(part, client_seed):

    local_rng = np.random.default_rng(client_seed)

    cluster_stats = (
        part.groupby("cluster_id")
        .agg(
            n=("sequence", "size"),
            pos=("label", "sum")
        )
        .reset_index()
    )

    total_n = len(part)
    total_pos = int(part["label"].sum())

    target_n = total_n * VAL_FRACTION
    target_pos = total_pos * VAL_FRACTION

    global_rate = part["label"].mean()

    cluster_ids = cluster_stats["cluster_id"].values

    info = {
        row["cluster_id"]: (
            int(row["n"]),
            int(row["pos"])
        )
        for _, row in cluster_stats.iterrows()
    }

    best_clusters = None
    best_score = None

    # --------------------------------------------------------
    # Random search
    # --------------------------------------------------------

    for _ in range(N_SEARCH):

        shuffled = local_rng.permutation(cluster_ids)

        selected = []

        n = 0
        pos = 0

        for cid in shuffled:

            if n >= target_n:
                break

            cn, cp = info[cid]

            selected.append(cid)

            n += cn
            pos += cp

        if n == 0:
            continue

        rate = pos / n

        # Difference from desired validation size
        size_error = abs(n - target_n) / target_n

        # Difference from client's overall positive rate
        rate_error = abs(rate - global_rate) / max(
            global_rate,
            1e-8
        )

        # Positive-count error
        pos_error = abs(pos - target_pos) / max(
            target_pos,
            1
        )

        score = (
            size_error
            + rate_error
            + 0.5 * pos_error
        )

        if best_score is None or score < best_score:

            best_score = score
            best_clusters = set(selected)

    if best_clusters is None:
        raise RuntimeError(
            "Could not create validation split."
        )

    val_mask = part["cluster_id"].isin(best_clusters)

    c_val = part[val_mask].copy()
    c_train = part[~val_mask].copy()

    return c_train, c_val, best_score


# ============================================================
# Create client train / validation CSVs
# ============================================================

print("\n" + "=" * 70)
print("CLIENT TRAIN / VALIDATION FILES")
print("=" * 70)

columns = [
    "sequence",
    "label",
    "organism",
    "protein_key",
    "cluster_id"
]

for i, (client_id, part) in enumerate(client_dfs.items()):

    part = part.reset_index(drop=True)

    c_train, c_val, score = find_balanced_cluster_split(
        part,
        SEED + i
    )

    train_clusters = set(c_train["cluster_id"])
    val_clusters = set(c_val["cluster_id"])

    cluster_overlap = len(
        train_clusters & val_clusters
    )

    c_train[columns].to_csv(
        OUTDIR / f"{client_id}_train.csv",
        index=False
    )

    c_val[columns].to_csv(
        OUTDIR / f"{client_id}_val.csv",
        index=False
    )

    print(f"\n{client_id}")

    print(
        f"  Total client peptides: {len(part)}"
    )

    print(
        f"  Client positive rate: "
        f"{100 * part['label'].mean():.2f}%"
    )

    print(
        f"  Train: {len(c_train)} "
        f"({100 * len(c_train)/len(part):.2f}%)"
    )

    print(
        f"  Train positive rate: "
        f"{100 * c_train['label'].mean():.2f}%"
    )

    print(
        f"  Validation: {len(c_val)} "
        f"({100 * len(c_val)/len(part):.2f}%)"
    )

    print(
        f"  Validation positive rate: "
        f"{100 * c_val['label'].mean():.2f}%"
    )

    print(
        f"  Cluster overlap: {cluster_overlap}"
    )

    print(
        f"  Search score: {score:.4f}"
    )


# ============================================================
# Save global validation / test
# ============================================================

global_val[columns].to_csv(
    OUTDIR / "global_val.csv",
    index=False
)

global_test[columns].to_csv(
    OUTDIR / "global_test.csv",
    index=False
)


# ============================================================
# Dataset accounting
# ============================================================

client_total = sum(
    len(part)
    for part in client_dfs.values()
)

print("\n" + "=" * 70)
print("FINAL ACCOUNTING")
print("=" * 70)

print(
    "Training-pool peptides:",
    len(train_pool)
)

print(
    "Peptides assigned to clients:",
    client_total
)

print(
    "Unaccounted:",
    len(train_pool) - client_total
)


# ============================================================
# Global homology leakage checks
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL LEAKAGE SAFETY CHECK")
print("=" * 70)

train_clusters = set(
    train_pool["cluster_id"]
)

val_clusters = set(
    global_val["cluster_id"]
)

test_clusters = set(
    global_test["cluster_id"]
)

print(
    "Training vs global val:",
    len(train_clusters & val_clusters)
)

print(
    "Training vs global test:",
    len(train_clusters & test_clusters)
)

print(
    "Global val vs global test:",
    len(val_clusters & test_clusters)
)

print("\nDone.")