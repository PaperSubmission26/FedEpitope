import pandas as pd

SPLIT_FILE = "data/homology_split.csv"
OVERLAP_FILE = "data/train_test_substring_overlaps.csv"
OUT_FILE = "data/global_test_strict.csv"

df = pd.read_csv(SPLIT_FILE)
overlap = pd.read_csv(OVERLAP_FILE)

test_df = df[df["split"].str.lower() == "test"].copy()

overlap_sequences = set(
    overlap["sequence"].astype(str).str.strip().str.upper()
)

test_df["sequence_norm"] = (
    test_df["sequence"].astype(str).str.strip().str.upper()
)

strict_df = test_df[
    ~test_df["sequence_norm"].isin(overlap_sequences)
].copy()

strict_df = strict_df.drop(columns=["sequence_norm"])

print(f"Original test size: {len(test_df):,}")
print(f"Removed substring-overlapping test peptides: {len(test_df) - len(strict_df):,}")
print(f"Strict test size: {len(strict_df):,}")

print("\nOriginal test positive rate:")
print(test_df["label"].mean())

print("\nStrict test positive rate:")
print(strict_df["label"].mean())

strict_df.to_csv(OUT_FILE, index=False)

print(f"\nSaved strict test set to: {OUT_FILE}")