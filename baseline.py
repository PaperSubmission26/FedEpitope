import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import warnings
import os
warnings.filterwarnings('ignore')

BATCH_SIZE = 64
LR         = 2e-4
MAX_LEN    = 30
EPOCHS     = 5

CLIENT_TRAIN_FILES = [f'data/client{i+1}_train.csv' for i in range(5)]
CLIENT_VAL_FILES   = [f'data/client{i+1}_val.csv'   for i in range(5)]
CLIENT_NAMES       = ['Coronavirus', 'Parasite', 'Human', 'Flavivirus', 'Bacteria']

class EpitopeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LEN):
        df = pd.read_csv(csv_path)
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
        'mcc':     matthews_corrcoef(all_labels, preds)
    }

def train_model(model, train_loader, val_loader, device, class_weights,
                epochs, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_auc  = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        m = evaluate(model, val_loader, device)
        print(f"  [{label} | Epoch {epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Val AUC: {m['auc_roc']:.4f}")
        if m['auc_roc'] > best_auc:
            best_auc   = m['auc_roc']
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model

def build_lora_model():
    model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=['query', 'value'])
    return get_peft_model(model, lora_config)

if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")

    test_ds     = EpitopeDataset('data/central_test.csv', tokenizer)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    print(f"Central test set: {len(test_ds)} samples\n")

    results = {}
    os.makedirs('results', exist_ok=True)

    # ── Baseline 1: Random ────────────────────────────────────────────────
    print("="*60)
    print("BASELINE 1: Random predictor")
    print("="*60)
    test_df   = pd.read_csv('data/central_test.csv')
    np.random.seed(42)
    rand_probs  = np.random.uniform(0, 1, len(test_df))
    rand_preds  = (rand_probs >= 0.5).astype(int)
    rand_labels = test_df['label'].values
    results['Random'] = {
        'auc_roc': roc_auc_score(rand_labels, rand_probs),
        'auc_pr':  average_precision_score(rand_labels, rand_probs),
        'f1':      f1_score(rand_labels, rand_preds, zero_division=0),
        'mcc':     matthews_corrcoef(rand_labels, rand_preds)
    }
    print(f"  AUC-ROC: {results['Random']['auc_roc']:.4f} | "
          f"AUC-PR: {results['Random']['auc_pr']:.4f} | "
          f"F1: {results['Random']['f1']:.4f} | "
          f"MCC: {results['Random']['mcc']:.4f}\n")

    # ── Baseline 2: Single client only (one per client) ───────────────────
    print("="*60)
    print("BASELINE 2: Single client training (each client independently)")
    print("="*60)
    for i in range(5):
        print(f"\n  Training Client {i+1} ({CLIENT_NAMES[i]}) independently...")
        train_df = pd.read_csv(CLIENT_TRAIN_FILES[i])
        pos      = train_df['label'].sum()
        neg      = len(train_df) - pos
        cw       = torch.tensor([1.0, neg/max(pos,1)],
                                dtype=torch.float).to(device)
        train_loader = DataLoader(
            EpitopeDataset(CLIENT_TRAIN_FILES[i], tokenizer),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            EpitopeDataset(CLIENT_VAL_FILES[i], tokenizer),
            batch_size=128, shuffle=False, num_workers=4)
        model = build_lora_model().to(device)
        model = train_model(model, train_loader, val_loader, device, cw,
                            EPOCHS, f"Client{i+1}-{CLIENT_NAMES[i]}")
        m = evaluate(model, test_loader, device)
        results[f'Single_Client{i+1}_{CLIENT_NAMES[i]}'] = m
        print(f"  → Test AUC-ROC: {m['auc_roc']:.4f} | "
              f"AUC-PR: {m['auc_pr']:.4f} | "
              f"F1: {m['f1']:.4f} | MCC: {m['mcc']:.4f}")
        del model
        torch.cuda.empty_cache()

    # ── Baseline 3: Frozen ESM-2 + linear classifier ──────────────────────
    print("\n" + "="*60)
    print("BASELINE 3: Frozen ESM-2 + linear head (no fine-tuning)")
    print("="*60)
    # Pool all client training data
    all_train = pd.concat([pd.read_csv(f) for f in CLIENT_TRAIN_FILES])
    all_val   = pd.concat([pd.read_csv(f) for f in CLIENT_VAL_FILES])
    all_train.to_csv('data/temp_all_train.csv', index=False)
    all_val.to_csv('data/temp_all_val.csv',     index=False)
    pos = all_train['label'].sum()
    neg = len(all_train) - pos
    cw  = torch.tensor([1.0, neg/max(pos,1)], dtype=torch.float).to(device)

    frozen_model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    # Freeze everything except classifier
    for name, param in frozen_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in frozen_model.parameters()
                    if p.requires_grad)
    print(f"  Trainable params (frozen ESM-2): {trainable:,}")
    frozen_model = frozen_model.to(device)
    train_loader = DataLoader(
        EpitopeDataset('data/temp_all_train.csv', tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        EpitopeDataset('data/temp_all_val.csv', tokenizer),
        batch_size=128, shuffle=False, num_workers=4)
    frozen_model = train_model(frozen_model, train_loader, val_loader,
                               device, cw, EPOCHS, "Frozen-ESM2")
    m = evaluate(frozen_model, test_loader, device)
    results['Frozen_ESM2'] = m
    print(f"  → Test AUC-ROC: {m['auc_roc']:.4f} | "
          f"AUC-PR: {m['auc_pr']:.4f} | "
          f"F1: {m['f1']:.4f} | MCC: {m['mcc']:.4f}")
    del frozen_model
    torch.cuda.empty_cache()

    # ── Baseline 4: Centralised ESM-2 + LoRA (upper bound) ────────────────
    print("\n" + "="*60)
    print("BASELINE 4: Centralised ESM-2 + LoRA (upper bound)")
    print("="*60)
    central_model = build_lora_model().to(device)
    central_model = train_model(central_model, train_loader, val_loader,
                                device, cw, EPOCHS, "Centralised-LoRA")
    m = evaluate(central_model, test_loader, device)
    results['Centralised_LoRA'] = m
    print(f"  → Test AUC-ROC: {m['auc_roc']:.4f} | "
          f"AUC-PR: {m['auc_pr']:.4f} | "
          f"F1: {m['f1']:.4f} | MCC: {m['mcc']:.4f}")
    torch.save({n: p.data.clone() for n, p in central_model.named_parameters()
                if p.requires_grad}, 'results/centralised_weights.pt')
    del central_model
    torch.cuda.empty_cache()

    # ── Also add federated result for comparison ───────────────────────────
    # Load best federated model and evaluate
    print("\n" + "="*60)
    print("FEDERATED MODEL (best checkpoint for comparison)")
    print("="*60)
    fed_model  = build_lora_model().to(device)
    fed_weights = torch.load('results/best_global_weights.pt',
                             map_location=device)
    for n, p in fed_model.named_parameters():
        if p.requires_grad and n in fed_weights:
            p.data.copy_(fed_weights[n])
    m = evaluate(fed_model, test_loader, device)
    results['Federated_LoRA'] = m
    print(f"  → Test AUC-ROC: {m['auc_roc']:.4f} | "
          f"AUC-PR: {m['auc_pr']:.4f} | "
          f"F1: {m['f1']:.4f} | MCC: {m['mcc']:.4f}")

    # ── Print final comparison table ───────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY — ALL MODELS ON CENTRAL TEST SET")
    print("="*60)
    print(f"{'Model':<35} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'MCC':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<35} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} "
              f"{m['f1']:>8.4f} {m['mcc']:>8.4f}")

    # Save results
    pd.DataFrame(results).T.to_csv('results/baseline_results.csv')
    print(f"\n✅ Baseline experiments complete. Saved to results/baseline_results.csv")

    # Cleanup temp files
    os.remove('data/temp_all_train.csv')
    os.remove('data/temp_all_val.csv')
