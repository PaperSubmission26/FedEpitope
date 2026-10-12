import pandas as pd
from collections import defaultdict

INPUT = "data/homology_split.csv"

df = pd.read_csv(INPUT)

print("Columns:", df.columns.tolist())

# Adjust automatically to common column names
seq_col = None
for c in ["sequence", "Sequence", "peptide", "Peptide"]:
    if c in df.columns:
        seq_col = c
        break

split_col = None
for c in ["split", "Split", "partition", "Partition"]:
    if c in df.columns:
        split_col = c
        break

if seq_col is None or split_col is None:
    raise ValueError(
        f"Could not identify sequence/split columns. "
        f"Available columns: {df.columns.tolist()}"
    )

df[seq_col] = (
    df[seq_col]
    .astype(str)
    .str.strip()
    .str.upper()
)

df[split_col] = df[split_col].astype(str).str.strip().str.lower()

train = set(df.loc[df[split_col] == "train", seq_col])
val   = set(df.loc[df[split_col].isin(["val", "validation"]), seq_col])
test  = set(df.loc[df[split_col] == "test", seq_col])

print(f"\nTrain sequences: {len(train):,}")
print(f"Validation sequences: {len(val):,}")
print(f"Test sequences: {len(test):,}")


def audit_substring_overlap(reference_sequences, query_sequences, name):
    """
    Counts query peptides involved in substring containment with reference.

    A query is marked overlapping when either:
      1. query is contained in a reference peptide, OR
      2. a reference peptide is contained in query.

    Exact equality is excluded because exact overlap is audited separately.
    """

    reference_sequences = set(reference_sequences)
    query_sequences = set(query_sequences)

    ref_by_len = defaultdict(set)
    for s in reference_sequences:
        ref_by_len[len(s)].add(s)

    query_by_len = defaultdict(list)
    for q in query_sequences:
        query_by_len[len(q)].append(q)

    overlap_queries = set()
    query_inside_reference = set()
    reference_inside_query = set()

    # ---------------------------------------------------------
    # Direction 1:
    # Is the complete query sequence contained in a longer
    # reference sequence?
    #
    # Process one query length at a time to avoid storing every
    # possible substring of the full training set in memory.
    # ---------------------------------------------------------
    for L, queries_L in sorted(query_by_len.items()):

        query_set_L = set(queries_L)

        for ref in reference_sequences:
            if len(ref) <= L:
                continue

            for i in range(len(ref) - L + 1):
                sub = ref[i:i + L]

                if sub in query_set_L:
                    query_inside_reference.add(sub)

    # ---------------------------------------------------------
    # Direction 2:
    # Does the query contain a shorter reference peptide?
    # ---------------------------------------------------------
    ref_lengths = sorted(ref_by_len.keys())

    for q in query_sequences:
        qlen = len(q)

        found = False

        for L in ref_lengths:
            if L >= qlen:
                break

            ref_set_L = ref_by_len[L]

            for i in range(qlen - L + 1):
                if q[i:i + L] in ref_set_L:
                    reference_inside_query.add(q)
                    found = True
                    break

            if found:
                break

    overlap_queries = query_inside_reference | reference_inside_query

    n = len(query_sequences)
    n_overlap = len(overlap_queries)
    pct = 100.0 * n_overlap / n if n else 0.0

    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"Query contained in a longer reference peptide: "
        f"{len(query_inside_reference):,}"
    )
    print(
        f"Query contains a shorter reference peptide: "
        f"{len(reference_inside_query):,}"
    )
    print(
        f"Unique query peptides with any substring overlap: "
        f"{n_overlap:,} / {n:,} ({pct:.2f}%)"
    )

    return overlap_queries


train_test_overlap = audit_substring_overlap(
    train,
    test,
    "TRAIN <-> TEST SUBSTRING AUDIT"
)

train_val_overlap = audit_substring_overlap(
    train,
    val,
    "TRAIN <-> VALIDATION SUBSTRING AUDIT"
)

# Save affected peptides for inspection
pd.DataFrame(
    sorted(train_test_overlap),
    columns=["sequence"]
).to_csv(
    "data/train_test_substring_overlaps.csv",
    index=False
)

pd.DataFrame(
    sorted(train_val_overlap),
    columns=["sequence"]
).to_csv(
    "data/train_validation_substring_overlaps.csv",
    index=False
)

print("\nSaved:")
print("  data/train_test_substring_overlaps.csv")
print("  data/train_validation_substring_overlaps.csv")