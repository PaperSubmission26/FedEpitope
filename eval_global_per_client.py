"""
eval_global_per_client.py
--------------------------
Evaluates the best global federated checkpoint on each client's
pathogen-specific test set for seeds 42, 43, 44 in one run.

Single command (from project root):
    python eval_global_per_client.py

Outputs: results/r2_seed42/global_per_client_test.csv
         results/r2_seed43/global_per_client_test.csv
         results/r2_seed44/global_per_client_test.csv
"""

import os, random, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef)

# ── Hardcoded config — edit only if your paths differ ────────────────────────
SEEDS      = [42, 43, 44]
LORA_RANK  = 2
BATCH_SIZE = 64
DATA_DIR   = 'data'
CHECKPOINT_PATTERN = 'results/r2_seed{seed}/best_global_weights.pt'
RESULTS_PATTERN    = 'results/r2_seed{seed}'

CLIENT_NAMES = {1:'Coronavirus', 2:'Parasite', 3:'Human/Self',
                4:'Flavivirus',  5:'Bacteria'}

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── Dataset ───────────────────────────────────────────────────────────────────
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.encodings = tokenizer(sequences, padding='max_length',
                                   truncation=True, max_length=25,
                                   return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids':      self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels':         self.labels[idx]}

def load_client_test(client_id, tokenizer):
    path = os.path.join(DATA_DIR, f'client{client_id}_test.csv')
    seqs, labs = [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            seqs.append(row['sequence'])
            labs.append(int(row['label']))
    return PeptideDataset(seqs, labs, tokenizer)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(device):
    base = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    for name, p in base.named_parameters():
        if 'classifier' not in name:
            p.requires_grad = False
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_RANK,
                     lora_alpha=2*LORA_RANK, lora_dropout=0.1,
                     target_modules=['query','value'], bias='none')
    return get_peft_model(base, cfg).to(device)

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    for batch in loader:
        out = model(input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
        probs_all.extend(torch.softmax(out.logits, -1)[:,1].cpu().numpy())
        labels_all.extend(batch['labels'].numpy())
    p, l = np.array(probs_all), np.array(labels_all)
    preds = (p >= 0.5).astype(int)
    return {'auc_roc': roc_auc_score(l, p),
            'auc_pr':  average_precision_score(l, p),
            'f1':      f1_score(l, preds, zero_division=0),
            'mcc':     matthews_corrcoef(l, preds)}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")

    for seed in SEEDS:
        set_seed(seed)
        ckpt_path    = CHECKPOINT_PATTERN.format(seed=seed)
        results_dir  = RESULTS_PATTERN.format(seed=seed)
        out_path     = os.path.join(results_dir, 'global_per_client_test.csv')

        print(f"{'='*55}")
        print(f"Seed {seed} — loading {ckpt_path}")
        model = build_model(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device),
                              strict=False)
        model.eval()

        rows = []
        for cid, cname in CLIENT_NAMES.items():
            dataset = load_client_test(cid, tokenizer)
            g = torch.Generator(); g.manual_seed(seed + cid)
            loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0, generator=g)
            m = evaluate(model, loader, device)
            rows.append({'seed': seed, 'lora_rank': LORA_RANK,
                         'client_id': cid, 'client_name': cname,
                         'variant': 'Standard-FL', **m})
            print(f"  {cname:<15} AUC-ROC={m['auc_roc']:.4f}  "
                  f"AUC-PR={m['auc_pr']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")

        os.makedirs(results_dir, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  -> Saved: {out_path}\n")

    print("All done.")

if __name__ == '__main__':
    main()