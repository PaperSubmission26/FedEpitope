import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
N_CLIENTS = 5

BASE = Path(".")
SOURCE_DIR = BASE / "data/federated"
OUT_DIR = BASE / "data/federated_iid"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load the exact samples used by the natural non-IID FedAvg
# ---------------------------------------------------------

source_files = [
    SOURCE_DIR / f"client{i}_train.csv"
    for i in range(1, N_CLIENTS + 1)
]

source_dfs = []
original_sizes = []

print("ORIGINAL NATURAL CLIENTS")
print("========================")

for i, f in enumerate(source_files, start=1):
    df = pd.read_csv(f)

    source_dfs.append(df)
    original_sizes.append(len(df))

    print(
        f"Client {i}: "
        f"N={len(df):,}, "
        f"positive={(df['label'] == 1).sum():,}, "
        f"rate={df['label'].mean():.4%}"
    )

all_train = pd.concat(
    source_dfs,
    ignore_index=True
)

print("\nPOOLED TRAINING DATA")
print("====================")
print(f"N:        {len(all_train):,}")
print(f"Positive: {(all_train['label'] == 1).sum():,}")
print(f"Negative: {(all_train['label'] == 0).sum():,}")
print(f"Rate:     {all_train['label'].mean():.4%}")

if all_train["sequence"].duplicated().any():
    raise ValueError(
        "Duplicate peptide sequences found in pooled training data."
    )

# ---------------------------------------------------------
# Stratified IID redistribution while preserving client sizes
#
# Strategy:
# - shuffle positives and negatives separately
# - determine approximately how many positives each client
#   should receive based on its original sample count
# - fill remaining slots with negatives
# ---------------------------------------------------------

rng = np.random.default_rng(SEED)

pos_df = all_train[
    all_train["label"] == 1
].copy()

neg_df = all_train[
    all_train["label"] == 0
].copy()

pos_df = pos_df.iloc[
    rng.permutation(len(pos_df))
].reset_index(drop=True)

neg_df = neg_df.iloc[
    rng.permutation(len(neg_df))
].reset_index(drop=True)

total_n = len(all_train)
total_pos = len(pos_df)

# Expected positives proportional to client size
raw_pos_targets = np.array(
    original_sizes,
    dtype=float
) * total_pos / total_n

pos_targets = np.floor(
    raw_pos_targets
).astype(int)

# Distribute any remaining positives according to
# largest fractional remainders
remaining = (
    total_pos
    - pos_targets.sum()
)

fractional = (
    raw_pos_targets
    - pos_targets
)

order = np.argsort(
    -fractional
)

for idx in order[:remaining]:
    pos_targets[idx] += 1

# Negative targets fill the remaining slots exactly
neg_targets = (
    np.array(original_sizes)
    - pos_targets
)

assert pos_targets.sum() == len(pos_df)
assert neg_targets.sum() == len(neg_df)

print("\nIID TARGET COUNTS")
print("=================")

for i in range(N_CLIENTS):
    print(
        f"Client {i+1}: "
        f"N={original_sizes[i]:,}, "
        f"target positive={pos_targets[i]:,}, "
        f"target negative={neg_targets[i]:,}, "
        f"target rate={pos_targets[i] / original_sizes[i]:.4%}"
    )

# ---------------------------------------------------------
# Assign data
# ---------------------------------------------------------

iid_clients = []

pos_start = 0
neg_start = 0

for i in range(N_CLIENTS):

    p = pos_targets[i]
    n = neg_targets[i]

    pos_part = pos_df.iloc[
        pos_start:pos_start + p
    ]

    neg_part = neg_df.iloc[
        neg_start:neg_start + n
    ]

    client_df = pd.concat(
        [pos_part, neg_part],
        ignore_index=True
    )

    # Shuffle rows within the client
    client_df = client_df.sample(
        frac=1,
        random_state=SEED + i
    ).reset_index(drop=True)

    iid_clients.append(client_df)

    pos_start += p
    neg_start += n

assert pos_start == len(pos_df)
assert neg_start == len(neg_df)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nFINAL IID CLIENTS")
print("=================")

for i, df in enumerate(
    iid_clients,
    start=1
):

    out = (
        OUT_DIR
        / f"client{i}_train.csv"
    )

    df.to_csv(
        out,
        index=False
    )

    print(
        f"Client {i}: "
        f"N={len(df):,}, "
        f"positive={(df['label'] == 1).sum():,}, "
        f"negative={(df['label'] == 0).sum():,}, "
        f"rate={df['label'].mean():.4%}, "
        f"organisms={df['organism'].nunique():,}, "
        f"proteins={df['protein_key'].nunique():,}, "
        f"clusters={df['cluster_id'].nunique():,}"
    )

    print("Saved:", out)

# ---------------------------------------------------------
# Strong sanity checks
# ---------------------------------------------------------

iid_all = pd.concat(
    iid_clients,
    ignore_index=True
)

assert len(iid_all) == len(all_train)

assert [
    len(df)
    for df in iid_clients
] == original_sizes

assert set(
    iid_all["sequence"]
) == set(
    all_train["sequence"]
)

assert not iid_all[
    "sequence"
].duplicated().any()

print("\n======================================")
print("IID PARTITION CREATED SUCCESSFULLY")
print("======================================")
print(
    f"Total samples preserved: "
    f"{len(iid_all):,}"
)
print(
    "Original client sample counts "
    "preserved exactly."
)
print(
    "All pooled training peptides "
    "used exactly once."
)
print(
    "IID partition is fixed with "
    f"partition seed {SEED}."
)
