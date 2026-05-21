import torch
import pandas as pd
import numpy as np
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MAX_LEN      = 30
STANDARD_AA  = set('ACDEFGHIKLMNPQRSTVWY')
NETBCE_REPO  = '/tmp/NetBCE'   # cloned path
TEST_FILE    = os.path.join(NETBCE_REPO, 'data', 'testing dataset.txt')


# ── Step 1: Parse NetBCE FASTA ─────────────────────────────────────────────
def parse_netbce_fasta(filepath):
    """
    Parse NetBCE FASTA format.
    Header format: >seq_N_lable_0  (0=negative) or >seq_N_lable_1 (1=positive)
    Sequence may contain '-' gap characters — strip them.
    """
    records = []
    current_label = None
    current_seq   = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save previous record
                if current_seq and current_label is not None:
                    seq = ''.join(current_seq).replace('-', '').upper()
                    records.append({'sequence': seq, 'label': current_label})
                    current_seq = []
                # Parse label from header: >seq_N_lable_X
                parts = line.split('_')
                current_label = int(parts[-1])   # last token is 0 or 1
            else:
                current_seq.append(line)

    # Save last record
    if current_seq and current_label is not None:
        seq = ''.join(current_seq).replace('-', '').upper()
        records.append({'sequence': seq, 'label': current_label})

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} sequences — "
          f"{df['label'].sum()} positive, "
          f"{(df['label']==0).sum()} negative")
    return df


# ── Step 2: Filter ─────────────────────────────────────────────────────────
def filter_sequences(df):
    before = len(df)
    # Standard AA only (no gaps remain after replace above)
    df = df[df['sequence'].apply(
        lambda s: isinstance(s, str) and len(s) > 0 and
        set(s).issubset(STANDARD_AA)
    )].copy()
    # Length 8–25 to match your training pipeline
    df = df[df['sequence'].str.len().between(8, 25)]
    df = df.drop_duplicates(subset=['sequence'])
    print(f"  After filtering: {len(df)}/{before} retained")
    return df


# ── Step 3: Remove IEDB overlap ────────────────────────────────────────────
def remove_iedb_overlap(df):
    iedb_files = (
        ['data/central_test.csv'] +
        [f'data/client{i}_train.csv' for i in range(1, 6)] +
        [f'data/client{i}_val.csv'   for i in range(1, 6)]
    )
    iedb_seqs = set()
    for fpath in iedb_files:
        if os.path.exists(fpath):
            try:
                tmp = pd.read_csv(fpath)
                iedb_seqs.update(tmp['sequence'].str.upper().tolist())
            except Exception:
                pass
    before = len(df)
    df = df[~df['sequence'].isin(iedb_seqs)].copy()
    print(f"  Removed {before - len(df)} IEDB overlaps "
          f"— {len(df)} independent sequences remain")
    return df


# ── Dataset ────────────────────────────────────────────────────────────────
class EpitopeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LEN):
        self.sequences = df['sequence'].tolist()
        self.labels    = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx], return_tensors='pt',
            padding='max_length', truncation=True,
            max_length=self.max_length)
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Model ──────────────────────────────────────────────────────────────────
def build_model():
    model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=['query', 'value'])
    return get_peft_model(model, lora_config)


def load_model(weights_path, device):
    model = build_model().to(device)
    weights = torch.load(weights_path, map_location=device)
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    model.eval()
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device)
                      ).logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    return {
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pr':  average_precision_score(all_labels, all_probs),
        'f1':      f1_score(all_labels, preds, zero_division=0),
        'mcc':     matthews_corrcoef(all_labels, preds),
        'probs':   all_probs,
        'labels':  all_labels,
    }


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # ── Load & prepare NetBCE independent test set ─────────────────────────
    print(f"Loading NetBCE independent test set from:\n  {TEST_FILE}\n")
    netbce_df = parse_netbce_fasta(TEST_FILE)

    print("\nFiltering sequences (length 8-25, standard AA)...")
    netbce_df = filter_sequences(netbce_df)

    print("\nRemoving IEDB overlap...")
    netbce_df = remove_iedb_overlap(netbce_df)

    n_pos = int(netbce_df['label'].sum())
    n_neg = int((netbce_df['label'] == 0).sum())
    print(f"\nFinal NetBCE test set: {len(netbce_df)} sequences "
          f"| {n_pos} positive | {n_neg} negative "
          f"| {n_pos/len(netbce_df):.1%} positive rate")

    netbce_df.to_csv('data/netbce_independent_test.csv', index=False)
    print("Saved: data/netbce_independent_test.csv")

    # ── Also load IEDB test set for side-by-side comparison ───────────────
    iedb_test_df = pd.read_csv('data/central_test.csv')

    netbce_loader = DataLoader(EpitopeDataset(netbce_df, tokenizer),
                               batch_size=128, shuffle=False, num_workers=0)
    iedb_loader   = DataLoader(EpitopeDataset(iedb_test_df, tokenizer),
                               batch_size=128, shuffle=False, num_workers=0)

    # ── Evaluate ───────────────────────────────────────────────────────────
    models_to_eval = {
        'FedEpitope': 'results/best_global_weights.pt',
        'Centralised': 'results/centralised_weights.pt',
    }

    all_results = {}
    for model_name, weights_path in models_to_eval.items():
        if not os.path.exists(weights_path):
            print(f"\nSkipping {model_name} — not found: {weights_path}")
            continue
        print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")
        model = load_model(weights_path, device)

        nb = evaluate(model, netbce_loader, device)
        ie = evaluate(model, iedb_loader,   device)

        print(f"  NetBCE independent — AUC-ROC: {nb['auc_roc']:.4f} | "
              f"AUC-PR: {nb['auc_pr']:.4f} | "
              f"F1: {nb['f1']:.4f} | MCC: {nb['mcc']:.4f}")
        print(f"  IEDB held-out      — AUC-ROC: {ie['auc_roc']:.4f} | "
              f"AUC-PR: {ie['auc_pr']:.4f} | "
              f"F1: {ie['f1']:.4f} | MCC: {ie['mcc']:.4f}")
        print(f"  Gap (IEDB - NetBCE): "
              f"{ie['auc_roc'] - nb['auc_roc']:+.4f}")

        all_results[model_name] = {'netbce': nb, 'iedb': ie}
        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("No models evaluated. Exiting.")
        exit(1)

    # ── Save results CSV ───────────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        rows.append({
            'Model':              name,
            'NetBCE AUC-ROC':    round(res['netbce']['auc_roc'], 4),
            'NetBCE AUC-PR':     round(res['netbce']['auc_pr'],  4),
            'NetBCE F1':         round(res['netbce']['f1'],      4),
            'NetBCE MCC':        round(res['netbce']['mcc'],     4),
            'IEDB AUC-ROC':      round(res['iedb']['auc_roc'],   4),
            'IEDB AUC-PR':       round(res['iedb']['auc_pr'],    4),
            'IEDB F1':           round(res['iedb']['f1'],        4),
            'IEDB MCC':          round(res['iedb']['mcc'],       4),
            'Gap (IEDB-NetBCE)': round(
                res['iedb']['auc_roc'] - res['netbce']['auc_roc'], 4),
        })
    pd.DataFrame(rows).to_csv('results/netbce_validation.csv', index=False)
    print("\nSaved: results/netbce_validation.csv")

    # ── ROC figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['coral', 'steelblue']

    for ax, (dkey, dlabel, dsize) in zip(axes, [
        ('netbce', 'NetBCE Independent Benchmark\n(Xu & Zhao, GPB 2022)',
         f'n={len(netbce_df):,}'),
        ('iedb',   'IEDB Held-Out Test Set\n(same distribution as training)',
         f'n={len(iedb_test_df):,}'),
    ]):
        for (name, res), color in zip(all_results.items(), colors):
            m = res[dkey]
            fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC={m['auc_roc']:.4f})")
        ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5,label='Random (0.50)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC — {dlabel}\n{dsize}', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('FedEpitope: IEDB vs NetBCE Independent Validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure_netbce_roc.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Bar chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x, width = np.arange(len(all_results)), 0.30
    names = list(all_results.keys())

    bars1 = ax.bar(x - width/2,
                   [all_results[n]['netbce']['auc_roc'] for n in names],
                   width, label='NetBCE independent', color='coral', alpha=0.85)
    bars2 = ax.bar(x + width/2,
                   [all_results[n]['iedb']['auc_roc'] for n in names],
                   width, label='IEDB held-out', color='steelblue', alpha=0.85)

    ax.axhline(y=0.744, color='green', linestyle='--', linewidth=1.5,
               label='BepiPred-3.0 on IEDB (0.744)')
    ax.axhline(y=0.50,  color='grey',  linestyle=':',  linewidth=1.0,
               label='Random (0.50)')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_title('FedEpitope: IEDB vs NetBCE Independent Validation',
                 fontsize=13)
    ax.set_ylim(0.40, 0.90)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/figure_netbce_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INDEPENDENT VALIDATION SUMMARY")
    print(f"Benchmark: NetBCE testing dataset (Xu & Zhao, GPB 2022)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'NetBCE AUC':>12} {'IEDB AUC':>10} {'Gap':>8}")
    print("-"*48)
    for name, res in all_results.items():
        gap = res['iedb']['auc_roc'] - res['netbce']['auc_roc']
        print(f"{name:<15} {res['netbce']['auc_roc']:>12.4f} "
              f"{res['iedb']['auc_roc']:>10.4f} {gap:>+8.4f}")

    print(f"\nSaved: results/netbce_validation.csv")
    print(f"Saved: figures/figure_netbce_roc.png")
    print(f"Saved: figures/figure_netbce_bar.png")
    print(f"\n✅ Independent validation complete.")