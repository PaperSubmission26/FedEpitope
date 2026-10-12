#!/usr/bin/env python3
"""
All-in-one updated independent NetBCE evaluation pipeline for FedEpitope.

Default workflow (--stage all):
  1) Parse/audit the official NetBCE train/test files.
  2) Build a leakage-clean independent NetBCE test set.
     By default, exact matches to BOTH the complete FedEpitope dataset and
     the NetBCE training set are removed, so the final comparison set is
     unseen by either training pipeline at the exact-peptide level.
  3) Create length-specific FASTA files for the official NetBCE predictor.
  4) Run the official NetBCE ensemble on the exact same cleaned peptides.
  5) Collect NetBCE probabilities.
  6) Evaluate updated FedAvg, FedProx, and Centralized LoRA checkpoints
     (seeds 42/43/44) on exactly the same cleaned set.
  7) Save individual predictions, all-run metrics, three-seed summaries,
     and a combined manuscript-ready summary including NetBCE.

Run from repository root:
    python run_netbce_external_updated.py

Useful partial stages:
    python run_netbce_external_updated.py --stage audit
    python run_netbce_external_updated.py --stage prepare
    python run_netbce_external_updated.py --stage netbce
    python run_netbce_external_updated.py --stage collect
    python run_netbce_external_updated.py --stage evaluate

If the official NetBCE code requires a separate legacy environment, run:
    python run_netbce_external_updated.py --stage audit
    python run_netbce_external_updated.py --stage prepare
    # activate NetBCE environment
    python run_netbce_external_updated.py --stage netbce
    python run_netbce_external_updated.py --stage collect
    # activate FedEpitope environment
    python run_netbce_external_updated.py --stage evaluate
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_sequence(seq: str) -> str:
    seq = str(seq).upper().strip()
    seq = re.sub(r"\s+", "", seq)
    return seq.replace("-", "")


def label_from_text(text: str):
    s = str(text).strip().lower()
    m = re.search(r"(?:label|lable)[_\-\s:=]*([01])(?:\D|$)", s)
    if m:
        return int(m.group(1))
    if re.search(r"\bnon[-_ ]?epitope\b", s) or re.search(r"\bnegative\b", s) or re.search(r"\bneg\b", s):
        return 0
    if re.search(r"\bepitope\b", s) or re.search(r"\bpositive\b", s) or re.search(r"\bpos\b", s):
        return 1
    return None


def is_sequence_token(token: str) -> bool:
    seq = normalize_sequence(token)
    return len(seq) >= 2 and set(seq).issubset(STANDARD_AA)


def parse_netbce_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    records = []
    pending_header = None

    for line_no, raw_line in enumerate(lines, start=1):
        raw = raw_line.strip()
        if not raw:
            continue
        if raw.startswith(">"):
            pending_header = raw[1:].strip()
            continue

        compact = normalize_sequence(raw)
        if pending_header is not None and is_sequence_token(compact):
            lab = label_from_text(pending_header)
            records.append({
                "sequence": compact,
                "label": lab,
                "source_line": line_no,
                "source_header": pending_header,
            })
            pending_header = None
            continue

        tokens = [t for t in re.split(r"[\t,;\s]+", raw) if t]
        seq_candidates = [normalize_sequence(t) for t in tokens if is_sequence_token(t)]
        label_candidates = []
        text_label = label_from_text(raw)
        if text_label is not None:
            label_candidates.append(text_label)
        if seq_candidates:
            for t in tokens:
                if t in {"0", "1"}:
                    label_candidates.append(int(t))

        if seq_candidates and label_candidates:
            unique_labels = sorted(set(label_candidates))
            if len(unique_labels) != 1:
                raise ValueError(f"Ambiguous label on line {line_no} of {path}: {raw!r}")
            records.append({
                "sequence": seq_candidates[0],
                "label": unique_labels[0],
                "source_line": line_no,
                "source_header": "",
            })
            pending_header = None
            continue

        if seq_candidates:
            raise ValueError(f"Found sequence without detectable label on line {line_no} of {path}: {raw!r}")

    if not records:
        raise RuntimeError(f"No labeled peptide records parsed from {path}")

    df = pd.DataFrame(records)
    if df["label"].isna().any():
        bad = df[df["label"].isna()].head(10)
        raise ValueError(f"Parsed {df['label'].isna().sum()} sequences without labels from {path}.\n{bad}")
    df["label"] = df["label"].astype(int)
    return df


def find_split_column(df: pd.DataFrame):
    for c in ["split", "set", "partition", "subset"]:
        if c in df.columns:
            return c
    return None


def audit_and_build(args):
    print("\n" + "=" * 80)
    print("STAGE 1: AUDIT + BUILD LEAKAGE-CLEAN NetBCE EXTERNAL BENCHMARK")
    print("=" * 80)

    out_dir = Path(args.audit_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_raw = parse_netbce_dataset(Path(args.netbce_test))
    train_raw = parse_netbce_dataset(Path(args.netbce_train))

    for d in [test_raw, train_raw]:
        d["sequence"] = d["sequence"].map(normalize_sequence)
        d["length"] = d["sequence"].str.len()

    raw_n = len(test_raw)
    raw_pos = int(test_raw["label"].sum())
    raw_neg = raw_n - raw_pos

    compatible = (
        test_raw["length"].between(args.min_len, args.max_len)
        & test_raw["sequence"].map(lambda s: set(s).issubset(STANDARD_AA))
    )
    invalid = test_raw[~compatible].copy()
    test = test_raw[compatible].copy()

    # Remove conflicting-label test sequences.
    test_label_counts = test.groupby("sequence")["label"].nunique()
    conflicting_sequences = set(test_label_counts[test_label_counts > 1].index)
    conflict_rows = test[test["sequence"].isin(conflicting_sequences)].copy()
    test = test[~test["sequence"].isin(conflicting_sequences)].copy()

    before_dedup = len(test)
    test = (
        test.sort_values(["sequence", "label", "source_line"])
        .drop_duplicates(subset=["sequence"], keep="first")
        .reset_index(drop=True)
    )
    same_label_dups_removed = before_dedup - len(test)

    # NetBCE train unique clean map.
    train_label_counts = train_raw.groupby("sequence")["label"].nunique()
    train_conflicts = set(train_label_counts[train_label_counts > 1].index)
    train_clean = train_raw[~train_raw["sequence"].isin(train_conflicts)].copy()
    train_unique = train_clean.drop_duplicates("sequence", keep="first")
    train_map = dict(zip(train_unique["sequence"], train_unique["label"]))

    test["overlap_netbce_train"] = test["sequence"].isin(train_map)
    test["netbce_train_label"] = test["sequence"].map(train_map)
    test["netbce_train_label_agrees"] = (
        test["overlap_netbce_train"] & (test["label"] == test["netbce_train_label"])
    )

    # Complete FedEpitope reference set.
    fed_path = Path(args.fedepitope_all)
    if not fed_path.exists():
        raise FileNotFoundError(fed_path)
    fed = pd.read_csv(fed_path)
    if "sequence" not in fed.columns:
        raise ValueError(f"{fed_path} has no 'sequence' column")
    fed["sequence"] = fed["sequence"].map(normalize_sequence)
    split_col = find_split_column(fed)
    fed_sequences = set(fed["sequence"])
    test["overlap_fedepitope_any"] = test["sequence"].isin(fed_sequences)

    if split_col:
        split_map = (
            fed.groupby("sequence")[split_col]
            .apply(lambda x: ";".join(sorted({str(v) for v in x.dropna()})))
            .to_dict()
        )
        test["fedepitope_split_overlap"] = test["sequence"].map(split_map).fillna("")
    else:
        test["fedepitope_split_overlap"] = ""

    overlap_fed = test[test["overlap_fedepitope_any"]].copy()
    overlap_net = test[test["overlap_netbce_train"]].copy()

    # Strongest fair benchmark: exact-disjoint from both training pools.
    remove_mask = test["overlap_fedepitope_any"].copy()
    if args.remove_netbce_train_overlap:
        remove_mask = remove_mask | test["overlap_netbce_train"]

    clean = test[~remove_mask].copy()
    clean_out = clean[["sequence", "label", "length"]].reset_index(drop=True)

    clean_file = out_dir / "netbce_external_exactclean.csv"
    clean_out.to_csv(clean_file, index=False)
    overlap_fed.to_csv(out_dir / "netbce_external_exact_overlaps.csv", index=False)
    overlap_net.to_csv(out_dir / "netbce_test_vs_netbce_train_exact_overlaps.csv", index=False)
    if len(conflict_rows):
        conflict_rows.to_csv(out_dir / "netbce_external_conflicting_labels.csv", index=False)
    if len(invalid):
        invalid.to_csv(out_dir / "netbce_external_incompatible_rows.csv", index=False)

    netbce_train_overlap = int(test["overlap_netbce_train"].sum())
    netbce_train_agree = (
        float(test.loc[test["overlap_netbce_train"], "netbce_train_label_agrees"].mean())
        if netbce_train_overlap else None
    )

    summary = pd.DataFrame([{
        "raw_test_total": raw_n,
        "raw_test_positive": raw_pos,
        "raw_test_negative": raw_neg,
        "length_or_aa_incompatible_removed": len(invalid),
        "conflicting_unique_sequences_removed": len(conflicting_sequences),
        "conflict_rows_removed": len(conflict_rows),
        "same_label_duplicates_removed": same_label_dups_removed,
        "unique_valid_before_overlap_removal": len(test),
        "fedepitope_exact_overlap_count": int(test["overlap_fedepitope_any"].sum()),
        "netbce_train_exact_overlap_count": netbce_train_overlap,
        "netbce_train_overlap_label_agreement": netbce_train_agree,
        "remove_netbce_train_overlap_from_final": bool(args.remove_netbce_train_overlap),
        "final_external_total": len(clean_out),
        "final_external_positive": int(clean_out["label"].sum()) if len(clean_out) else 0,
        "final_external_negative": int((clean_out["label"] == 0).sum()) if len(clean_out) else 0,
        "final_positive_rate": float(clean_out["label"].mean()) if len(clean_out) else None,
        "final_min_length": int(clean_out["length"].min()) if len(clean_out) else None,
        "final_max_length": int(clean_out["length"].max()) if len(clean_out) else None,
    }])
    summary_file = out_dir / "netbce_external_audit_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved clean benchmark: {clean_file}")
    print(f"Saved audit summary:   {summary_file}")
    if len(clean_out) == 0:
        raise RuntimeError("No examples remain after cleaning; stop evaluation.")


def prepare_fastas(args):
    print("\n" + "=" * 80)
    print("STAGE 2: PREPARE LENGTH-SPECIFIC FASTA FILES")
    print("=" * 80)
    df = pd.read_csv(args.clean_benchmark)
    out = Path(args.fasta_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if df["sequence"].duplicated().any():
        raise ValueError("Clean benchmark contains duplicate peptide sequences.")

    manifest_rows = []
    for length, g in df.groupby(df["sequence"].str.len()):
        fasta = out / f"len{int(length)}.fasta"
        with fasta.open("w") as f:
            for _, row in g.reset_index(drop=True).iterrows():
                f.write(f">{row['sequence']}\n{row['sequence']}\n")
        manifest_rows.append({"length": int(length), "n": len(g), "fasta": str(fasta)})

    manifest = pd.DataFrame(manifest_rows).sort_values("length")
    manifest.to_csv(out / "manifest.csv", index=False)
    print(manifest.to_string(index=False))


def run_official_netbce(args):
    print("\n" + "=" * 80)
    print("STAGE 3: RUN OFFICIAL NetBCE ENSEMBLE")
    print("=" * 80)

    manifest_path = Path(args.fasta_out_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)

    prediction_dir = Path(args.netbce_prediction_dir).resolve()
    script = prediction_dir / "NetBCE_prediction.py"
    if not script.exists():
        raise FileNotFoundError(script)

    # Paths passed to official script should work from its prediction directory.
    fasta_root = Path(args.fasta_out_dir).resolve()
    result_root = Path(args.netbce_result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        length = int(row["length"])
        fasta = fasta_root / f"len{length}.fasta"
        out_dir = result_root / f"len{length}"
        output_file = out_dir / "NetBCE_predictions.tsv"
        if output_file.exists() and not args.force_netbce:
            print(f"[skip] len{length}: {output_file} already exists")
            continue

        cmd = [
            sys.executable,
            "NetBCE_prediction.py",
            "-f", str(fasta),
            "-l", str(length),
            "-o", str(out_dir),
        ]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(prediction_dir), check=True)


def collect_netbce_predictions(args):
    print("\n" + "=" * 80)
    print("STAGE 4: COLLECT OFFICIAL NetBCE PREDICTIONS")
    print("=" * 80)

    bench = pd.read_csv(args.clean_benchmark)
    expected = set(bench["sequence"].astype(str).str.strip().str.upper())
    rows = []

    for path in sorted(Path(args.netbce_result_root).glob("len*/NetBCE_predictions.tsv")):
        d = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["protein_id", "sequence", "position", "probability"],
        )
        d["sequence"] = d["sequence"].astype(str).str.strip().str.upper()
        d["probability"] = pd.to_numeric(d["probability"], errors="raise")
        rows.append(d[["sequence", "probability"]])

    if not rows:
        raise FileNotFoundError(f"No NetBCE_predictions.tsv under {args.netbce_result_root}")

    pred = pd.concat(rows, ignore_index=True)
    if pred["sequence"].duplicated().any():
        dup = pred[pred["sequence"].duplicated(keep=False)].sort_values("sequence").head(20)
        raise ValueError(f"Duplicate NetBCE predictions detected:\n{dup}")

    got = set(pred["sequence"])
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(f"NetBCE prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    pred = bench[["sequence"]].merge(pred, on="sequence", how="left", validate="one_to_one")
    output = Path(args.netbce_prediction_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    print(f"Saved {len(pred):,} NetBCE predictions to {output}")


def metric_dict(y_true, y_score):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        matthews_corrcoef, precision_score, recall_score, confusion_matrix,
    )
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_test": len(y_true),
        "positive_rate": float(y_true.mean()),
        "auc_roc": float(auc),
        "auc_pr": float(ap),
        "f1_at_0.5": float(f1),
        "mcc_at_0.5": float(mcc),
        "precision_at_0.5": float(precision),
        "recall_at_0.5": float(recall),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_models(args):
    print("\n" + "=" * 80)
    print("STAGE 5: EVALUATE UPDATED FedEpitope CHECKPOINTS + NetBCE")
    print("=" * 80)

    # Heavy imports only when this stage is needed.
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_token_length

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    class PeptideDataset(Dataset):
        def __init__(self, dataframe, tokenizer):
            self.df = dataframe.reset_index(drop=True)
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            encoded = self.tokenizer(
                str(row["sequence"]), truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(row["label"]), dtype=torch.long),
            }

    def build_model(seed):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        lora_config = LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.1,
            target_modules=["query", "value"], bias="none", task_type="SEQ_CLS",
        )
        return get_peft_model(model, lora_config)

    def load_checkpoint(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(state)}")
        return state

    def set_saved_weights(model, state):
        model_params = dict(model.named_parameters())
        loaded = []
        missing_in_model = []
        for name, value in state.items():
            if name not in model_params:
                missing_in_model.append(name)
                continue
            target = model_params[name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(f"Shape mismatch for {name}: model={tuple(target.shape)}, checkpoint={tuple(value.shape)}")
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.append(name)
        if missing_in_model:
            raise RuntimeError(f"Checkpoint/model parameter names do not match. First missing: {missing_in_model[:20]}")
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        not_loaded = [name for name in trainable_names if name not in loaded]
        if not_loaded:
            raise RuntimeError(f"Not all trainable parameters restored. First missing: {not_loaded[:30]}")

    @torch.no_grad()
    def predict(model, loader, device):
        model.eval()
        labels, scores = [], []
        for batch in loader:
            y = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            labels.extend(y.numpy().tolist())
            scores.extend(probs.detach().cpu().numpy().tolist())
        return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)

    test_df = pd.read_csv(args.clean_benchmark)
    if not {"sequence", "label"}.issubset(test_df.columns):
        raise ValueError("Clean benchmark must contain sequence and label columns")

    output_dir = Path(args.result_out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loader = DataLoader(
        PeptideDataset(test_df, tokenizer),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; N={len(test_df):,}; positive rate={test_df['label'].mean():.4f}")

    jobs = []
    for seed in [42, 43, 44]:
        jobs.append(("FedAvg", seed, Path(f"results/fedavg_seed{seed}/best_global_weights.pt")))
    for seed in [42, 43, 44]:
        jobs.append(("FedProx", seed, Path(f"results/fedprox_seed{seed}/best_global_weights.pt")))
    for seed in [42, 43, 44]:
        jobs.append(("Centralized_LoRA", seed, Path(f"results/baseline_seed{seed}/centralized_weights.pt")))

    all_rows = []
    for i, (method, seed, checkpoint) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {method} seed {seed}: {checkpoint}")
        model = build_model(seed)
        state = load_checkpoint(checkpoint)
        set_saved_weights(model, state)
        model.to(device)
        y_true, y_score = predict(model, loader, device)
        metrics = metric_dict(y_true, y_score)
        all_rows.append({"method": method, "seed": seed, **metrics})
        pred_df = test_df.copy()
        pred_df["prediction_score"] = y_score
        pred_df.to_csv(output_dir / f"{method}_seed{seed}_predictions.csv", index=False)
        pd.DataFrame(all_rows).to_csv(output_dir / "all_runs_live.csv", index=False)
        print(f"AUROC={metrics['auc_roc']:.6f} AP={metrics['auc_pr']:.6f} F1={metrics['f1_at_0.5']:.6f} MCC={metrics['mcc_at_0.5']:.6f}")
        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "all_runs.csv", index=False)

    summary_rows = []
    for method, g in all_df.groupby("method"):
        row = {"method": method, "n_seeds": len(g), "N": int(g["n_test"].iloc[0]), "positive_rate": float(g["positive_rate"].iloc[0])}
        for source, short in [
            ("auc_roc", "auc_roc"), ("auc_pr", "auc_pr"),
            ("f1_at_0.5", "f1"), ("mcc_at_0.5", "mcc"),
            ("precision_at_0.5", "precision"), ("recall_at_0.5", "recall"),
        ]:
            row[f"{short}_mean"] = g[source].mean()
            row[f"{short}_std"] = g[source].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_3seeds.csv", index=False)

    # Add official NetBCE on the exact same benchmark.
    net_pred = pd.read_csv(args.netbce_prediction_csv)
    merged = test_df[["sequence", "label"]].merge(net_pred, on="sequence", how="left", validate="one_to_one")
    if merged["probability"].isna().any():
        raise ValueError(f"Missing NetBCE probabilities for {merged['probability'].isna().sum()} peptides")
    net_metrics = metric_dict(merged["label"], merged["probability"])
    net_row = {
        "method": "NetBCE", "n_seeds": 1, "N": net_metrics["n_test"],
        "positive_rate": net_metrics["positive_rate"],
        "auc_roc_mean": net_metrics["auc_roc"], "auc_roc_std": np.nan,
        "auc_pr_mean": net_metrics["auc_pr"], "auc_pr_std": np.nan,
        "f1_mean": net_metrics["f1_at_0.5"], "f1_std": np.nan,
        "mcc_mean": net_metrics["mcc_at_0.5"], "mcc_std": np.nan,
        "precision_mean": net_metrics["precision_at_0.5"], "precision_std": np.nan,
        "recall_mean": net_metrics["recall_at_0.5"], "recall_std": np.nan,
    }
    combined = pd.concat([summary, pd.DataFrame([net_row])], ignore_index=True)
    combined.to_csv(output_dir / "summary_with_netbce.csv", index=False)

    print("\nTHREE-SEED SUMMARY")
    print(summary.to_string(index=False))
    print("\nCOMBINED SUMMARY INCLUDING OFFICIAL NetBCE")
    print(combined.to_string(index=False))
    print(f"\nSaved outputs under: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["all", "audit", "prepare", "netbce", "collect", "evaluate"], default="all")

    p.add_argument("--netbce_test", default="external/NetBCE/data/testing dataset.txt")
    p.add_argument("--netbce_train", default="external/NetBCE/data/training dataset.txt")
    p.add_argument("--fedepitope_all", default="data/homology_split.csv")
    p.add_argument("--audit_out_dir", default="data/external/NetBCE")
    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--max_len", type=int, default=25)
    p.add_argument(
    "--remove_netbce_train_overlap",
    dest="remove_netbce_train_overlap",
    action="store_true",
    default=True,
    help="Remove NetBCE-test peptides found exactly in NetBCE training data from the final common external benchmark"
    )
    p.add_argument(
    "--no-remove_netbce_train_overlap",
    dest="remove_netbce_train_overlap",
    action="store_false"
    )




    p.add_argument("--clean_benchmark", default="data/external/NetBCE/netbce_external_exactclean.csv")
    p.add_argument("--fasta_out_dir", default="external/NetBCE/external_exactclean_fasta")
    p.add_argument("--netbce_prediction_dir", default="external/NetBCE/prediction")
    p.add_argument("--netbce_result_root", default="external/NetBCE/result/external_exactclean")
    p.add_argument("--netbce_prediction_csv", default="external/netbce_external_predictions.csv")
    p.add_argument("--force_netbce", action="store_true")

    p.add_argument("--result_out_dir", default="results/netbce_external_comparison")
    p.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--max_token_length", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    stages = ["audit", "prepare", "netbce", "collect", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "audit":
            audit_and_build(args)
        elif stage == "prepare":
            prepare_fastas(args)
        elif stage == "netbce":
            run_official_netbce(args)
        elif stage == "collect":
            collect_netbce_predictions(args)
        elif stage == "evaluate":
            evaluate_models(args)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("Key files:")
    print("  data/external/NetBCE/netbce_external_audit_summary.csv")
    print("  data/external/NetBCE/netbce_external_exactclean.csv")
    print("  results/netbce_external_comparison/all_runs.csv")
    print("  results/netbce_external_comparison/summary_3seeds.csv")
    print("  results/netbce_external_comparison/summary_with_netbce.csv")


if __name__ == "__main__":
    main()
