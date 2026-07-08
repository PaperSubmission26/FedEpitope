"""
check_leakage.py
================
Checks for sequence-level leakage between the FedEpitope training pool
and the held-out central test set.

Three levels of overlap are checked:
  Level 1 - Exact match         (identity = 100%)
  Level 2 - Substring match     (one sequence contained within another)
  Level 3 - Edit distance <= 1  (single substitution / insertion / deletion)

Usage:
    python check_leakage.py \
        --train_dir  data/clients/ \
        --test_file  data/central_test.csv \
        --seq_col    sequence \
        --out        leakage_report.csv

Outputs:
  - leakage_report.csv   : all flagged (test_seq, train_seq, level) pairs
  - Summary printed to stdout with percentages ready to paste into paper.

Runtime estimate on your dataset (~103K test x ~331K train):
  Level 1 : < 1 second   (set intersection)
  Level 2 : ~5-15 min    (nested loop with early-exit; Python)
  Level 3 : ~20-40 min   (edit distance on short peptides 8-25 aa)
  Total   : ~30-55 min on a single CPU core.
  On your A6000 (CPU side) expect the lower end of those ranges.

If you only have time for Level 1 + Level 2, pass --skip_edit_distance.
"""

import argparse
import glob
import os
import csv
from collections import defaultdict

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def edit_distance_leq1(a: str, b: str) -> bool:
    """Return True if Levenshtein distance between a and b is <= 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    # Allow length difference of 0 or 1
    if len(a) > len(b):
        a, b = b, a          # ensure len(a) <= len(b)
    # len(b) - len(a) is 0 or 1
    diffs = 0
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diffs += 1
            if diffs > 1:
                return False
            if len(a) == len(b):
                i += 1
            # if len(b) == len(a)+1, advance only j (deletion in a)
            j += 1
        else:
            i += 1
            j += 1
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",  default="data/clients/",
                        help="Glob pattern or directory containing client CSV files "
                             "(e.g. client1_train.csv … client5_train.csv)")
    parser.add_argument("--test_file",  default="data/central_test.csv",
                        help="Path to the central held-out test CSV")
    parser.add_argument("--seq_col",    default="sequence",
                        help="Column name containing peptide sequences")
    parser.add_argument("--out",        default="leakage_report.csv",
                        help="Output CSV with flagged pairs")
    parser.add_argument("--skip_edit_distance", action="store_true",
                        help="Skip the (slower) edit-distance check")
    args = parser.parse_args()

    # ── Load training sequences ───────────────────────────────────────────────
    train_files = glob.glob(os.path.join(args.train_dir, "*train*.csv"))
    if not train_files:
        # fallback: treat train_dir itself as a single file
        train_files = [args.train_dir]

    print(f"Loading training data from {len(train_files)} file(s)…")
    train_df = pd.concat(
        [pd.read_csv(f, usecols=[args.seq_col]) for f in train_files],
        ignore_index=True
    )
    train_seqs = set(train_df[args.seq_col].str.upper().dropna().unique())
    print(f"  Unique training sequences : {len(train_seqs):,}")

    # ── Load test sequences ───────────────────────────────────────────────────
    test_df = pd.read_csv(args.test_file, usecols=[args.seq_col])
    test_seqs_list = test_df[args.seq_col].str.upper().dropna().unique().tolist()
    test_seqs_set  = set(test_seqs_list)
    N_test = len(test_seqs_list)
    print(f"  Unique test  sequences    : {N_test:,}\n")

    flagged = []   # list of dicts

    # ── Level 1: Exact match ──────────────────────────────────────────────────
    exact = test_seqs_set & train_seqs
    print(f"Level 1 — Exact matches       : {len(exact):,}  "
          f"({100*len(exact)/N_test:.2f}% of test set)")
    for s in exact:
        flagged.append({"test_seq": s, "train_seq": s, "level": "exact"})

    # ── Level 2: Substring match ──────────────────────────────────────────────
    # Build a dict of train seqs by length for speed
    train_by_len = defaultdict(list)
    for s in train_seqs:
        train_by_len[len(s)].append(s)

    substring_count = 0
    already_flagged = {r["test_seq"] for r in flagged}

    print("Level 2 — Substring check …  (this may take a few minutes)")
    for t in test_seqs_list:
        if t in already_flagged:
            continue
        found = False
        lt = len(t)
        # t is substring of a train seq (train seq is longer)
        for l in range(lt + 1, 26):          # max peptide length 25
            for tr in train_by_len[l]:
                if t in tr:
                    flagged.append({"test_seq": t, "train_seq": tr, "level": "substring"})
                    substring_count += 1
                    found = True
                    break
            if found:
                break
        if found:
            continue
        # train seq is substring of t (t is longer)
        for l in range(8, lt):
            for tr in train_by_len[l]:
                if tr in t:
                    flagged.append({"test_seq": t, "train_seq": tr, "level": "substring"})
                    substring_count += 1
                    found = True
                    break
            if found:
                break

    already_flagged = {r["test_seq"] for r in flagged}
    print(f"Level 2 — Substring matches   : {substring_count:,}  "
          f"({100*substring_count/N_test:.2f}% of test set)")

    # ── Level 3: Edit distance <= 1 ───────────────────────────────────────────
    edit1_count = 0
    if not args.skip_edit_distance:
        print("Level 3 — Edit distance ≤ 1 … (this may take 20-40 min)")
        for t in test_seqs_list:
            if t in already_flagged:
                continue
            lt = len(t)
            found = False
            # only compare to train seqs of same or ±1 length
            for dl in [-1, 0, 1]:
                for tr in train_by_len[lt + dl]:
                    if edit_distance_leq1(t, tr):
                        flagged.append({"test_seq": t, "train_seq": tr, "level": "edit1"})
                        edit1_count += 1
                        found = True
                        break
                if found:
                    break
        print(f"Level 3 — Edit dist ≤ 1      : {edit1_count:,}  "
              f"({100*edit1_count/N_test:.2f}% of test set)")
    else:
        print("Level 3 — Skipped (--skip_edit_distance flag set)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_flagged = len(flagged)
    total_pct     = 100 * total_flagged / N_test

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total test sequences          : {N_test:,}")
    print(f"Total flagged (any level)     : {total_flagged:,}  ({total_pct:.2f}%)")
    print(f"  of which exact              : {len(exact):,}")
    print(f"  of which substring          : {substring_count:,}")
    if not args.skip_edit_distance:
        print(f"  of which edit dist ≤ 1     : {edit1_count:,}")
    print("="*60)

    if total_pct < 2.0:
        verdict = "LOW LEAKAGE — results are likely robust. Safe to report in paper."
    elif total_pct < 10.0:
        verdict = "MODERATE LEAKAGE — re-run evaluation on de-duplicated test set."
    else:
        verdict = "HIGH LEAKAGE — results need to be re-reported on cleaned test set."
    print(f"\nVerdict: {verdict}\n")

    # ── Save flagged pairs ────────────────────────────────────────────────────
    if flagged:
        pd.DataFrame(flagged).to_csv(args.out, index=False)
        print(f"Flagged pairs saved to: {args.out}")
    else:
        print("No leakage detected at any level.")


if __name__ == "__main__":
    main()