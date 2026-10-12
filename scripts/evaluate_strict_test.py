import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_LEN = 30


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class PeptideDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        seq = str(row["sequence"])
        label = int(row["label"])

        encoded = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long,
                ),
        }


# ============================================================
# Build same ESM-2 + LoRA architecture used in training
# ============================================================

def build_model(seed):

    set_seed(seed)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.1,
        target_modules=[
            "query",
            "value",
        ],
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        state = torch.load(
            path,
            map_location="cpu",
        )

    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(
            state["state_dict"],
            dict,
        )
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(
            f"Unexpected checkpoint type: "
            f"{type(state)}"
        )

    return state


def set_saved_weights(model, state):

    model_params = dict(
        model.named_parameters()
    )

    loaded = []
    missing_in_model = []

    for name, value in state.items():

        if name not in model_params:
            missing_in_model.append(name)
            continue

        target = model_params[name]

        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"model={tuple(target.shape)}, "
                f"checkpoint={tuple(value.shape)}"
            )

        target.data.copy_(
            value.to(
                device=target.device,
                dtype=target.dtype,
            )
        )

        loaded.append(name)

    if missing_in_model:

        print(
            "\nCheckpoint parameters "
            "not present in model:"
        )

        for name in missing_in_model[:20]:
            print(" ", name)

        raise RuntimeError(
            "Checkpoint/model parameter names "
            "do not match."
        )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    not_loaded = [
        name
        for name in trainable_names
        if name not in loaded
    ]

    if not_loaded:

        print(
            "\nTrainable model parameters "
            "missing from checkpoint:"
        )

        for name in not_loaded[:30]:
            print(" ", name)

        raise RuntimeError(
            "Not all trainable parameters "
            "were restored."
        )

    print(
        f"Loaded {len(loaded)} "
        f"checkpoint tensors successfully."
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    all_labels = []
    all_scores = []

    for batch in loader:

        labels = batch["labels"]

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probs = torch.softmax(
            outputs.logits,
            dim=-1,
        )[:, 1]

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_scores.extend(
            probs.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    y_true = np.asarray(
        all_labels,
        dtype=int,
    )

    y_score = np.asarray(
        all_scores,
        dtype=float,
    )

    # Fixed threshold.
    # No tuning is performed on BCPreds.
    y_pred = (
        y_score >= 0.5
    ).astype(int)

    auc_roc = roc_auc_score(
        y_true,
        y_score,
    )

    auc_pr = average_precision_score(
        y_true,
        y_score,
    )

    f1 = f1_score(
        y_true,
        y_pred,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    metrics = {
        "n_test":
            len(y_true),

        "positive_rate":
            float(y_true.mean()),

        "auc_roc":
            float(auc_roc),

        "auc_pr":
            float(auc_pr),

        "f1_at_0.5":
            float(f1),

        "mcc_at_0.5":
            float(mcc),

        "precision_at_0.5":
            float(precision),

        "recall_at_0.5":
            float(recall),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }

    return metrics, y_score


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    root = Path(".")

    test_file = (
    root
    / "data/global_test_strict.csv"
    )

    output_dir = (
    root
    / "results/"
      "strict_test_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Missing test set: {test_file}"
        )

    test_df = pd.read_csv(
        test_file
    )

    required = {
        "sequence",
        "label",
    }

    missing = (
        required
        - set(test_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("=" * 72)
    print("STRICT HOMOLOGY-AWARE TEST EVALUATION")
    print("=" * 72)

    print(
        "Test file:",
        test_file,
    )

    print(
        "N:",
        len(test_df),
    )

    print(
        "Positive:",
        int(test_df["label"].sum()),
    )

    print(
        "Negative:",
        int(
            len(test_df)
            - test_df["label"].sum()
        ),
    )

    print(
        "Positive rate:",
        f"{100 * test_df['label'].mean():.2f}%"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This dataset is evaluation-only."
    )

    print(
        "No training, checkpoint selection, "
        "threshold tuning, or hyperparameter "
        "selection uses BCPreds."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    test_dataset = PeptideDataset(
        test_df,
        tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    # --------------------------------------------------------
    # Frozen validation-selected checkpoints
    # --------------------------------------------------------

    jobs = []

    for seed in [42, 43, 44]:

        jobs.append(
            {
                "method":
                    "FedAvg",

                "seed":
                    seed,

                "checkpoint":
                    (
                        root
                        / f"results/fedavg_seed{seed}/"
                          "best_global_weights.pt"
                    ),
            }
        )

    for seed in [42, 43, 44]:

        jobs.append(
            {
                "method":
                    "FedProx",

                "seed":
                    seed,

                "checkpoint":
                    (
                        root
                        / f"results/fedprox_seed{seed}/"
                          "best_global_weights.pt"
                    ),
            }
        )

    for seed in [42, 43, 44]:

        jobs.append(
            {
                "method":
                    "Centralized_LoRA",

                "seed":
                    seed,

                "checkpoint":
                    (
                        root
                        / f"results/baseline_seed{seed}/"
                          "centralized_weights.pt"
                    ),
            }
        )

    all_results = []

    # --------------------------------------------------------
    # Sequential evaluation
    # --------------------------------------------------------

    for job_num, job in enumerate(
        jobs,
        start=1,
    ):

        method = job["method"]
        seed = job["seed"]
        checkpoint = job["checkpoint"]

        print("\n")
        print("=" * 72)

        print(
            f"[{job_num}/{len(jobs)}] "
            f"{method} -- seed {seed}"
        )

        print("=" * 72)

        print(
            "Checkpoint:",
            checkpoint,
        )

        model = build_model(
            seed
        )

        state = load_checkpoint(
            checkpoint
        )

        set_saved_weights(
            model,
            state,
        )

        model.to(device)

        metrics, scores = evaluate(
            model,
            test_loader,
            device,
        )

        row = {
            "method":
                method,

            "seed":
                seed,

            **metrics,
        }

        all_results.append(
            row
        )

        print("\nRESULT")

        print(
            f"AUROC : "
            f"{metrics['auc_roc']:.6f}"
        )

        print(
            f"AUPRC : "
            f"{metrics['auc_pr']:.6f}"
        )

        print(
            f"F1    : "
            f"{metrics['f1_at_0.5']:.6f}"
        )

        print(
            f"MCC   : "
            f"{metrics['mcc_at_0.5']:.6f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision_at_0.5']:.6f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall_at_0.5']:.6f}"
        )

        # Save model predictions
        pred_df = test_df.copy()

        pred_df[
            "prediction_score"
        ] = scores

        pred_file = (
            output_dir
            / (
                f"{method}_seed{seed}"
                "_predictions.csv"
            )
        )

        pred_df.to_csv(
            pred_file,
            index=False,
        )

        # Live backup
        pd.DataFrame(
            all_results
        ).to_csv(
            output_dir
            / "all_runs_live.csv",
            index=False,
        )

        del model
        del state

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # All runs
    # --------------------------------------------------------

    all_df = pd.DataFrame(
        all_results
    )

    all_file = (
        output_dir
        / "all_runs.csv"
    )

    all_df.to_csv(
        all_file,
        index=False,
    )

    # --------------------------------------------------------
    # Three-seed mean +/- SD
    # --------------------------------------------------------

    summary_rows = []

    for method, group in all_df.groupby(
        "method"
    ):

        summary_rows.append(
            {
                "method":
                    method,

                "n_seeds":
                    len(group),

                "auc_roc_mean":
                    group[
                        "auc_roc"
                    ].mean(),

                "auc_roc_std":
                    group[
                        "auc_roc"
                    ].std(ddof=1),

                "auc_pr_mean":
                    group[
                        "auc_pr"
                    ].mean(),

                "auc_pr_std":
                    group[
                        "auc_pr"
                    ].std(ddof=1),

                "f1_mean":
                    group[
                        "f1_at_0.5"
                    ].mean(),

                "f1_std":
                    group[
                        "f1_at_0.5"
                    ].std(ddof=1),

                "mcc_mean":
                    group[
                        "mcc_at_0.5"
                    ].mean(),

                "mcc_std":
                    group[
                        "mcc_at_0.5"
                    ].std(ddof=1),

                "precision_mean":
                    group[
                        "precision_at_0.5"
                    ].mean(),

                "precision_std":
                    group[
                        "precision_at_0.5"
                    ].std(ddof=1),

                "recall_mean":
                    group[
                        "recall_at_0.5"
                    ].mean(),

                "recall_std":
                    group[
                        "recall_at_0.5"
                    ].std(ddof=1),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        output_dir
        / "summary_3seeds.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\n")
    print("=" * 72)
    print("ALL INDIVIDUAL RUNS")
    print("=" * 72)

    print(
        all_df.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("THREE-SEED SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(all_file)
    print(summary_file)


if __name__ == "__main__":
    main()
