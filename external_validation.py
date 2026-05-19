import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

MAX_LEN = 30

# Organisms that clients trained on — everything else is unseen
CLIENT_ORGANISMS = {
    # Client 1 - Coronavirus
    'Severe acute respiratory syndrome coronavirus 2',
    'Severe acute respiratory syndrome coronavirus 2 Wuhan/Hu-1/2019',
    'SARS-CoV1', 'SARS coronavirus Tor2',
    'Human coronavirus OC43', 'Human coronavirus NL63',
    'Human coronavirus 229E', 'Human coronavirus HKU1',
    'Middle East respiratory syndrome-related coronavirus',
    'Pangolin coronavirus',
    # Client 2 - Parasites
    'Trypanosoma cruzi strain CL Brener', 'Schistosoma mansoni',
    'Onchocerca volvulus', 'Plasmodium falciparum', 'Leishmania donovani',
    # Client 3 - Human
    'Homo sapiens', 'Mus musculus',
    # Client 4 - Flaviviruses
    'Hepacivirus hominis', 'Dengue virus', 'Zika virus',
    'Hepatitis B virus', 'West Nile virus',
    # Client 5 - Bacteria
    'Streptococcus pyogenes', 'Mycobacterium tuberculosis',
    'Staphylococcus aureus', 'Bacillus anthracis', 'Yersinia pestis',
    'Clostridium tetani', 'Influenza A virus',
    'Human immunodeficiency virus 1',
}

# ── Dataset ────────────────────────────────────────────────────────────────
class EpitopeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LEN):
        self.sequences  = df['sequence'].tolist()
        self.labels     = df['label'].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
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
    model   = build_model().to(device)
    weights = torch.load(weights_path, map_location=device)
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    model.eval()
    return model

# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device)).logits,
                dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)
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

    # ── Step 1: Prepare datasets ───────────────────────────────────────────
    full_test = pd.read_csv('data/central_test.csv')

    # External validation — sequences from organisms no client trained on
    external_df = full_test[
        ~full_test['organism'].isin(CLIENT_ORGANISMS)].copy()

    # Internal test — sequences from client-trained organisms
    internal_df = full_test[
        full_test['organism'].isin(CLIENT_ORGANISMS)].copy()

    print(f"Internal test set (client organisms): {len(internal_df)} sequences")
    print(f"  Positive rate: {internal_df['label'].mean():.1%}")
    print(f"  Unique organisms: {internal_df['organism'].nunique()}")
    print(f"\nExternal validation (unseen organisms): {len(external_df)} sequences")
    print(f"  Positive rate: {external_df['label'].mean():.1%}")
    print(f"  Unique organisms: {external_df['organism'].nunique()}")
    print(f"\nTop 10 external organisms:")
    print(external_df['organism'].value_counts().head(10).to_string())

    # Save external validation set
    external_df.to_csv('data/external_validation.csv', index=False)
    print(f"\nSaved data/external_validation.csv")

    # Build dataloaders
    external_loader = DataLoader(
        EpitopeDataset(external_df, tokenizer),
        batch_size=128, shuffle=False, num_workers=4)
    internal_loader = DataLoader(
        EpitopeDataset(internal_df, tokenizer),
        batch_size=128, shuffle=False, num_workers=4)

    # ── Step 2: Load models and evaluate ──────────────────────────────────
    models_to_eval = {
        'Federated': 'results/best_global_weights.pt',
        'Centralised': 'results/centralised_weights.pt',
    }

    all_results = {}
    for model_name, weights_path in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model = load_model(weights_path, device)

        ext_m = evaluate(model, external_loader, device)
        int_m = evaluate(model, internal_loader, device)

        print(f"  External (unseen organisms) — "
              f"AUC-ROC: {ext_m['auc_roc']:.4f} | "
              f"AUC-PR: {ext_m['auc_pr']:.4f} | "
              f"F1: {ext_m['f1']:.4f} | "
              f"MCC: {ext_m['mcc']:.4f}")
        print(f"  Internal (client organisms) — "
              f"AUC-ROC: {int_m['auc_roc']:.4f} | "
              f"AUC-PR: {int_m['auc_pr']:.4f} | "
              f"F1: {int_m['f1']:.4f} | "
              f"MCC: {int_m['mcc']:.4f}")
        print(f"  Generalisation gap (internal - external): "
              f"{int_m['auc_roc'] - ext_m['auc_roc']:+.4f}")

        all_results[model_name] = {
            'ext_auc_roc': ext_m['auc_roc'],
            'ext_auc_pr':  ext_m['auc_pr'],
            'ext_f1':      ext_m['f1'],
            'ext_mcc':     ext_m['mcc'],
            'ext_probs':   ext_m['probs'],
            'ext_labels':  ext_m['labels'],
            'int_auc_roc': int_m['auc_roc'],
            'int_auc_pr':  int_m['auc_pr'],
            'int_f1':      int_m['f1'],
            'int_mcc':     int_m['mcc'],
            'int_probs':   int_m['probs'],
            'int_labels':  int_m['labels'],
        }
        del model
        torch.cuda.empty_cache()

    # ── Step 3: Save results ───────────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        rows.append({
            'Model':              name,
            'External AUC-ROC':   round(res['ext_auc_roc'], 4),
            'External AUC-PR':    round(res['ext_auc_pr'],  4),
            'External F1':        round(res['ext_f1'],       4),
            'External MCC':       round(res['ext_mcc'],      4),
            'Internal AUC-ROC':   round(res['int_auc_roc'], 4),
            'Internal AUC-PR':    round(res['int_auc_pr'],  4),
            'Internal F1':        round(res['int_f1'],       4),
            'Internal MCC':       round(res['int_mcc'],      4),
            'Gap (Int-Ext)':      round(res['int_auc_roc'] -
                                        res['ext_auc_roc'], 4),
        })
    pd.DataFrame(rows).to_csv('results/external_validation.csv', index=False)

    # ── Step 4: Figure 1 — ROC curves ─────────────────────────────────────
    print("\nGenerating figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors    = ['coral', 'steelblue']

    for ax, (dataset_key, dataset_label) in zip(
            axes,
            [('ext', 'External Validation\n(944 Unseen Organisms)'),
             ('int', 'Internal Test\n(Client-Trained Organisms)')]):
        for (model_name, res), color in zip(all_results.items(), colors):
            probs  = res[f'{dataset_key}_probs']
            labels = res[f'{dataset_key}_labels']
            fpr, tpr, _ = roc_curve(labels, probs)
            auc          = res[f'{dataset_key}_auc_roc']
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model_name} (AUC={auc:.4f})")
        ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5,label='Random (0.50)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve — {dataset_label}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('External Validation: Generalisation to Unseen Organisms',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/figure_external_roc.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Step 5: Figure 2 — Bar chart comparison ────────────────────────────
    fig, ax    = plt.subplots(figsize=(10, 6))
    x          = np.arange(len(all_results))
    width      = 0.30
    names      = list(all_results.keys())
    ext_aucs   = [all_results[n]['ext_auc_roc'] for n in names]
    int_aucs   = [all_results[n]['int_auc_roc'] for n in names]

    bars1 = ax.bar(x - width/2, ext_aucs, width,
                   label='External (unseen organisms)',
                   color='coral', alpha=0.85)
    bars2 = ax.bar(x + width/2, int_aucs, width,
                   label='Internal (client organisms)',
                   color='steelblue', alpha=0.85)

    ax.axhline(y=0.74, color='green',  linestyle='--', linewidth=1.5,
               label='BepiPred-3.0 (0.74)')
    ax.axhline(y=0.62, color='orange', linestyle=':',  linewidth=1.5,
               label='BepiPred-2.0 independent (~0.62)')
    ax.axhline(y=0.50, color='grey',   linestyle=':',  linewidth=1.0,
               label='Random (0.50)')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_title('External Validation: Internal vs External Performance',
                 fontsize=13)
    ax.set_ylim(0.40, 0.90)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/figure_external_validation.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Step 6: Print final summary ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXTERNAL VALIDATION SUMMARY")
    print(f"External set: {len(external_df)} sequences from "
          f"{external_df['organism'].nunique()} unseen organisms")
    print(f"Internal set: {len(internal_df)} sequences from "
          f"{internal_df['organism'].nunique()} client-trained organisms")
    print(f"{'='*70}")
    print(f"\n{'Model':<15} {'External AUC':>14} {'Internal AUC':>14} "
          f"{'Gap':>8}")
    print("-"*55)
    print(f"{'Random':<15} {'0.5000':>14} {'0.5000':>14} {'—':>8}")
    print(f"{'BepiPred-2.0':<15} {'~0.6200':>14} {'~0.6000':>14} {'—':>8}")
    print(f"{'BepiPred-3.0':<15} {'N/A':>14} {'0.7400':>14} {'—':>8}")
    print("-"*55)
    for name, res in all_results.items():
        gap = res['int_auc_roc'] - res['ext_auc_roc']
        print(f"{name:<15} {res['ext_auc_roc']:>14.4f} "
              f"{res['int_auc_roc']:>14.4f} {gap:>+8.4f}")

    print(f"\nINTERPRETATION:")
    fed_ext = all_results['Federated']['ext_auc_roc']
    fed_int = all_results['Federated']['int_auc_roc']
    cen_ext = all_results['Centralised']['ext_auc_roc']

    if fed_ext > 0.62:
        print(f"✅ Federated model ({fed_ext:.4f}) outperforms BepiPred-2.0 "
              f"(~0.62) on unseen organisms")
    if fed_ext > 0.65:
        print(f"✅ Strong generalisation — model learned genuine "
              f"cross-pathogen epitope patterns")
    if abs(fed_ext - cen_ext) < 0.03:
        print(f"✅ Federated ≈ Centralised on external data "
              f"({fed_ext:.4f} vs {cen_ext:.4f}) — "
              f"privacy cost is minimal even for unseen pathogens")
    gap = fed_int - fed_ext
    print(f"\nFederated internal vs external gap: {gap:+.4f}")
    if gap < 0.05:
        print(f"✅ Small gap — model generalises well across organism types")
    elif gap < 0.10:
        print(f"⚠  Moderate gap — expected, model trained on specific "
              f"pathogen families")
    else:
        print(f"⚠  Large gap — model is more specialised than generalised")

    print(f"\n✅ External validation complete.")
    print(f"   Saved: results/external_validation.csv")
    print(f"   Saved: figures/figure_external_roc.png")
    print(f"   Saved: figures/figure_external_validation.png")
