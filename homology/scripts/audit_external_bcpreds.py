from pathlib import Path
import re
import pandas as pd

ROOT = Path(".")

BCPRED_DIR = ROOT / "external/epitope-prediction-paper/retraining/bcpreds"
INTERNAL_FILE = ROOT / "data/homology_split.csv"

OUT_DIR = ROOT / "data/external/BCPreds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN = OUT_DIR / "bcpreds_external_exactclean.csv"
OUT_OVERLAP = OUT_DIR / "bcpreds_exact_overlaps.csv"
OUT_CONFLICTS = OUT_DIR / "bcpreds_conflicting_labels.csv"
OUT_REPORT = OUT_DIR / "bcpreds_external_audit_summary.csv"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 8
MAX_LEN = 25


def normalize_sequence(seq):
    seq = str(seq).upper().strip()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.replace("-", "")
    return seq


def find_dataset_file(kind):
    candidates = []

    for path in BCPRED_DIR.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if kind == "pos":
            if name in {"pos.txt", "positive.txt", "positives.txt"}:
                candidates.append(path)

        elif kind == "neg":
            if name in {"neg.txt", "negative.txt", "negatives.txt"}:
                candidates.append(path)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {kind} file under {BCPRED_DIR}, "
            f"found: {candidates}"
        )

    return candidates[0]


def read_sequences(path, label):
    records = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()

            if not raw:
                continue

            if raw.startswith(">"):
                continue

            # Handles plain one-sequence-per-line files and
            # simple whitespace/comma separated formats.
            tokens = re.split(r"[\s,;\t]+", raw)

            seq = None

            for token in tokens:
                candidate = normalize_sequence(token)

                if (
                    len(candidate) >= 2
                    and set(candidate).issubset(STANDARD_AA)
                ):
                    seq = candidate
                    break

            if seq is None:
                continue

            records.append(
                {
                    "sequence": seq,
                    "label": label,
                    "source_file": path.name,
                    "source_line": line_no,
                }
            )

    return pd.DataFrame(records)


pos_file = find_dataset_file("pos")
neg_file = find_dataset_file("neg")

print("=" * 72)
print("BCPreds EXTERNAL BENCHMARK AUDIT")
print("=" * 72)

print("\nFiles:")
print("Positive:", pos_file)
print("Negative:", neg_file)

pos = read_sequences(pos_file, 1)
neg = read_sequences(neg_file, 0)

print("\nRAW PARSED")
print("Positive:", len(pos))
print("Negative:", len(neg))
print("Total:", len(pos) + len(neg))

df = pd.concat([pos, neg], ignore_index=True)

df["length"] = df["sequence"].str.len()

# ------------------------------------------------------------
# FedEpitope compatibility
# ------------------------------------------------------------

valid_mask = (
    df["length"].between(MIN_LEN, MAX_LEN)
    & df["sequence"].apply(
        lambda x: set(x).issubset(STANDARD_AA)
    )
)

invalid = df[~valid_mask].copy()
df = df[valid_mask].copy()

print("\nAFTER FEDEPITOPE COMPATIBILITY FILTER")
print("Kept:", len(df))
print("Removed:", len(invalid))

if len(df):
    print(
        "Length range:",
        int(df["length"].min()),
        "-",
        int(df["length"].max()),
    )

# ------------------------------------------------------------
# Label conflicts
# ------------------------------------------------------------

label_counts = (
    df.groupby("sequence")["label"]
    .nunique()
)

conflicting_sequences = set(
    label_counts[label_counts > 1].index
)

conflicts = df[
    df["sequence"].isin(conflicting_sequences)
].copy()

print("\nLABEL CONFLICT CHECK")
print(
    "Conflicting unique sequences:",
    len(conflicting_sequences),
)
print(
    "Rows belonging to conflicts:",
    len(conflicts),
)

if len(conflicts):
    conflicts.to_csv(
        OUT_CONFLICTS,
        index=False,
    )

df = df[
    ~df["sequence"].isin(conflicting_sequences)
].copy()

# ------------------------------------------------------------
# Same-label duplicates
# ------------------------------------------------------------

before_dedup = len(df)

df = (
    df.sort_values(
        ["sequence", "label", "source_line"]
    )
    .drop_duplicates(
        subset=["sequence"],
        keep="first",
    )
    .reset_index(drop=True)
)

duplicates_removed = before_dedup - len(df)

print("\nDEDUPLICATION")
print("Before:", before_dedup)
print("After:", len(df))
print(
    "Same-label duplicates removed:",
    duplicates_removed,
)

print("\nBEFORE FEDEPITOPE OVERLAP REMOVAL")
print("Unique sequences:", len(df))
print("Positive:", int(df["label"].sum()))
print(
    "Negative:",
    int(len(df) - df["label"].sum()),
)

# ------------------------------------------------------------
# Load complete FedEpitope strict benchmark
# ------------------------------------------------------------

internal = pd.read_csv(INTERNAL_FILE)

if "sequence" not in internal.columns:
    raise ValueError(
        "data/homology_split.csv does not contain 'sequence'."
    )

internal_sequences = set(
    internal["sequence"]
    .astype(str)
    .apply(normalize_sequence)
)

print("\nFEDEPITOPE REFERENCE SET")
print("Rows:", len(internal))
print(
    "Unique sequences:",
    len(internal_sequences),
)

# ------------------------------------------------------------
# Exact overlap
# ------------------------------------------------------------

overlap_mask = df["sequence"].isin(
    internal_sequences
)

overlap = df[overlap_mask].copy()
clean = df[~overlap_mask].copy()

overlap.to_csv(
    OUT_OVERLAP,
    index=False,
)

clean = clean[
    ["sequence", "label", "length"]
].reset_index(drop=True)

clean.to_csv(
    OUT_CLEAN,
    index=False,
)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

final_n = len(clean)
final_pos = int(clean["label"].sum())
final_neg = final_n - final_pos

print("\n" + "=" * 72)
print("FINAL EXACT-OVERLAP-CLEAN BCPreds SET")
print("=" * 72)

print("N:", final_n)
print("Positive:", final_pos)
print("Negative:", final_neg)

if final_n:
    print(
        "Positive rate:",
        f"{100 * clean['label'].mean():.2f}%"
    )

print("\nExact overlaps removed:", len(overlap))

if len(df):
    print(
        "Exact overlap fraction:",
        f"{100 * len(overlap) / len(df):.2f}%"
    )

overlap_pos = int(overlap["label"].sum())
overlap_neg = len(overlap) - overlap_pos

print("\nOVERLAP BY LABEL")
print("Positive overlaps:", overlap_pos)
print("Negative overlaps:", overlap_neg)

summary = pd.DataFrame(
    [
        {
            "raw_positive": len(pos),
            "raw_negative": len(neg),
            "raw_total": len(pos) + len(neg),
            "invalid_removed": len(invalid),
            "conflicting_unique_sequences":
                len(conflicting_sequences),
            "conflict_rows_removed":
                len(conflicts),
            "same_label_duplicates_removed":
                duplicates_removed,
            "unique_valid_before_overlap":
                len(df),
            "exact_overlap_removed":
                len(overlap),
            "exact_overlap_positive":
                overlap_pos,
            "exact_overlap_negative":
                overlap_neg,
            "final_external_total":
                final_n,
            "final_external_positive":
                final_pos,
            "final_external_negative":
                final_neg,
            "final_positive_rate":
                (
                    float(clean["label"].mean())
                    if final_n
                    else None
                ),
        }
    ]
)

summary.to_csv(
    OUT_REPORT,
    index=False,
)

print("\nSaved:")
print(OUT_CLEAN)
print(OUT_OVERLAP)
print(OUT_REPORT)

if len(conflicts):
    print(OUT_CONFLICTS)
