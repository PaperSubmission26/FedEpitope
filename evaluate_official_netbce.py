import os
import argparse
import subprocess
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef


def load_test_csv(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"sequence", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].astype(int)
    return df


def write_fasta(df, fasta_path):
    with open(fasta_path, "w") as f:
        for i, row in df.iterrows():
            seq = row["sequence"]
            label = int(row["label"])
            f.write(f">seq{i}|label={label}\n")
            f.write(f"{seq}\n")


def run_netbce(netbce_repo, fasta_path, output_dir, lengths):
    prediction_dir = os.path.join(netbce_repo, "prediction")
    prediction_script = os.path.join(prediction_dir, "NetBCE_prediction.py")

    if not os.path.exists(prediction_script):
        raise FileNotFoundError(f"Cannot find NetBCE_prediction.py at: {prediction_script}")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "NetBCE_prediction.py",
        "-f",
        os.path.abspath(fasta_path),
        "-o",
        os.path.abspath(output_dir),
        "-l",
    ] + [str(x) for x in lengths]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=prediction_dir, check=True)


def read_predictions(prediction_path):
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    pred = pd.read_csv(
        prediction_path,
        sep="\t",
        header=None,
        names=["protein_id", "candidate_epitope", "position", "score"],
    )

    pred["candidate_epitope"] = pred["candidate_epitope"].astype(str).str.upper().str.strip()
    pred["score"] = pred["score"].astype(float)

    return pred


def match_exact_full_sequence_scores(df, pred, results_dir):
    matched_scores = []

    for i, row in df.iterrows():
        seq = row["sequence"]
        label = int(row["label"])

        expected_id = f">seq{i}|label={label}"
        expected_position = f"1:{len(seq)}"

        hit = pred[
            (pred["protein_id"] == expected_id)
            & (pred["candidate_epitope"] == seq)
            & (pred["position"] == expected_position)
        ]

        if len(hit) != 1:
            debug_path = os.path.join(results_dir, "debug_unmatched_predictions.csv")
            pred[pred["protein_id"] == expected_id].to_csv(debug_path, index=False)

            raise ValueError(
                f"Expected exactly one full-sequence NetBCE score for row {i}, "
                f"but found {len(hit)}.\n"
                f"Sequence: {seq}\n"
                f"Expected protein_id: {expected_id}\n"
                f"Expected position: {expected_position}\n"
                f"Saved debug predictions to: {debug_path}"
            )

        matched_scores.append(float(hit.iloc[0]["score"]))

    return np.array(matched_scores)


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    df = load_test_csv(args.test_csv)

    print("=" * 80)
    print("Official NetBCE evaluation on de-duplicated NetBCE test set")
    print("=" * 80)
    print(f"Test CSV       : {args.test_csv}")
    print(f"NetBCE repo    : {args.netbce_repo}")
    print(f"Results dir    : {args.results_dir}")
    print(f"No. sequences  : {len(df)}")
    print(f"Positive labels: {int(df['label'].sum())}")
    print(f"Negative labels: {int((df['label'] == 0).sum())}")

    lengths = sorted(df["sequence"].str.len().unique().tolist())
    print(f"Exact sequence lengths passed to NetBCE: {lengths}")

    fasta_path = os.path.join(
        args.results_dir,
        "netbce_independent_test_for_official_netbce.fasta",
    )

    write_fasta(df, fasta_path)

    run_netbce(
        netbce_repo=args.netbce_repo,
        fasta_path=fasta_path,
        output_dir=args.results_dir,
        lengths=lengths,
    )

    prediction_path = os.path.join(args.results_dir, "NetBCE_predictions.tsv")
    pred = read_predictions(prediction_path)

    y_true = df["label"].values
    y_score = match_exact_full_sequence_scores(df, pred, args.results_dir)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "model": "NetBCE",
        "n_sequences": len(df),
        "netbce_auc_roc": roc_auc_score(y_true, y_score),
        "netbce_auc_pr": average_precision_score(y_true, y_score),
        "netbce_f1": f1_score(y_true, y_pred, zero_division=0),
        "netbce_mcc": matthews_corrcoef(y_true, y_pred),
    }

    metrics_path = os.path.join(args.results_dir, "official_netbce_metrics.csv")
    scores_path = os.path.join(args.results_dir, "official_netbce_scores.csv")

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    pd.DataFrame(
        {
            "sequence": df["sequence"],
            "label": y_true,
            "official_netbce_score": y_score,
        }
    ).to_csv(scores_path, index=False)

    print("\nOfficial NetBCE evaluation complete.")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved scores to : {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/netbce_independent_test.csv",
    )

    parser.add_argument(
        "--netbce_repo",
        type=str,
        required=True,
        help="Path to official NetBCE repo, e.g. ../NetBCE",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/netbce_official",
    )

    args = parser.parse_args()
    main(args)
