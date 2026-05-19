import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import warnings
import os
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
NUM_ROUNDS   = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS  = 5
BATCH_SIZE   = 64
LR           = 2e-4
MAX_LEN      = 30

CLIENT_TRAIN_FILES = [f'data/client{i+1}_train.csv' for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES   = [f'data/client{i+1}_val.csv'   for i in range(NUM_CLIENTS)]
CLIENT_NAMES       = ['Coronavirus', 'Parasite', 'Human', 'Flavivirus', 'Bacteria']

# ── Dataset ────────────────────────────────────────────────────────────────
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

# ── Model ──────────────────────────────────────────────────────────────────
def build_model():
    model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=['query', 'value']
    )
    return get_peft_model(model, lora_config)

# ── Trainable weight utilities — ALL trainable params, not just LoRA ───────
def get_trainable_weights(model):
    return {n: p.data.clone()
            for n, p in model.named_parameters() if p.requires_grad}

def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    return model

def print_weight_summary(weights, label):
    lora_keys       = [k for k in weights if 'lora' in k]
    classifier_keys = [k for k in weights if 'classifier' in k]
    print(f"  [{label}] total={len(weights)}, "
          f"lora={len(lora_keys)}, classifier={len(classifier_keys)}")

# ── Weighted FedAvg ────────────────────────────────────────────────────────
def federated_averaging(client_weights_list, client_sizes):
    total       = sum(client_sizes)
    proportions = [s / total for s in client_sizes]
    averaged    = {}
    for key in client_weights_list[0].keys():
        averaged[key] = sum(
            proportions[i] * client_weights_list[i][key].float()
            for i in range(len(client_weights_list))
        )
    return averaged

# ── Local training ─────────────────────────────────────────────────────────
def train_local(model, loader, optimizer, device, class_weights,
                round_num, client_id):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(LOCAL_EPOCHS):
        total_loss = 0
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            optimizer.zero_grad()
            loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits, labels
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [Round {round_num:02d} | Client {client_id+1} "
              f"{CLIENT_NAMES[client_id]:<12} | Epoch {epoch+1}/{LOCAL_EPOCHS}]"
              f" Loss: {total_loss/len(loader):.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            probs = torch.softmax(
                model(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device)).logits,
                dim=1
            )[:, 1].cpu().numpy()
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

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")

    train_loaders, val_loaders, class_weights_list, client_sizes = [], [], [], []
    for i in range(NUM_CLIENTS):
        tr  = EpitopeDataset(CLIENT_TRAIN_FILES[i], tokenizer)
        va  = EpitopeDataset(CLIENT_VAL_FILES[i],   tokenizer)
        df  = pd.read_csv(CLIENT_TRAIN_FILES[i])
        pos = df['label'].sum()
        neg = len(df) - pos
        w   = torch.tensor([1.0, neg / max(pos, 1)],
                           dtype=torch.float).to(device)
        train_loaders.append(DataLoader(tr, batch_size=BATCH_SIZE,
                                        shuffle=True,  num_workers=4))
        val_loaders.append(  DataLoader(va, batch_size=128,
                                        shuffle=False, num_workers=4))
        class_weights_list.append(w)
        client_sizes.append(len(tr))
        print(f"Client {i+1} ({CLIENT_NAMES[i]:<12}): "
              f"train={len(tr):>7}, val={len(va):>6}, "
              f"pos%={pos/len(df):.1%}, w_pos={neg/max(pos,1):.2f}")

    test_ds     = EpitopeDataset('data/central_test.csv', tokenizer)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    print(f"\nCentral test set: {len(test_ds)} samples")
    print(f"{'='*70}")
    print(f"Federated training: {NUM_ROUNDS} rounds | "
          f"{NUM_CLIENTS} clients | {LOCAL_EPOCHS} local epochs")
    print(f"{'='*70}\n")

    global_model = build_model().to(device)

    # Confirm what we are aggregating
    sample_weights = get_trainable_weights(global_model)
    print_weight_summary(sample_weights, "Weights to be aggregated")
    print()

    results    = {'round': [], 'auc_roc': [], 'auc_pr': [], 'f1': [], 'mcc': []}
    best_auc   = 0.0
    best_round = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─'*70}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─'*70}")

        client_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            optimizer   = torch.optim.AdamW(local_model.parameters(), lr=LR)
            train_local(local_model, train_loaders[client_id], optimizer,
                        device, class_weights_list[client_id],
                        round_num, client_id)
            m = evaluate(local_model, val_loaders[client_id], device)
            print(f"  [Round {round_num:02d} | Client {client_id+1} "
                  f"{CLIENT_NAMES[client_id]:<12} | Local val] "
                  f"AUC-ROC: {m['auc_roc']:.4f} | F1: {m['f1']:.4f}")
            client_weights.append(get_trainable_weights(local_model))
            del local_model
            torch.cuda.empty_cache()

        print(f"\n  → Weighted FedAvg over all trainable parameters...")
        averaged     = federated_averaging(client_weights, client_sizes)
        global_model = set_trainable_weights(global_model, averaged)

        metrics = evaluate(global_model, test_loader, device)
        results['round'].append(round_num)
        results['auc_roc'].append(metrics['auc_roc'])
        results['auc_pr'].append(metrics['auc_pr'])
        results['f1'].append(metrics['f1'])
        results['mcc'].append(metrics['mcc'])

        if metrics['auc_roc'] > best_auc:
            best_auc   = metrics['auc_roc']
            best_round = round_num
            os.makedirs('results', exist_ok=True)
            torch.save(get_trainable_weights(global_model),
                       'results/best_global_weights.pt')
            best_tag = " ← best so far"
        else:
            best_tag = ""

        print(f"\n  ★ GLOBAL MODEL | Round {round_num:02d} | "
              f"AUC-ROC: {metrics['auc_roc']:.4f} | "
              f"AUC-PR: {metrics['auc_pr']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"MCC: {metrics['mcc']:.4f}{best_tag}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/federated_rounds.csv', index=False)
    torch.save(get_trainable_weights(global_model),
               'results/global_weights_final.pt')

    print(f"\n{'='*70}")
    print(f"✅ Federated training complete")
    print(f"   Best AUC-ROC: {best_auc:.4f} at round {best_round}")
    print(f"   Saved to results/")
    print(f"{'='*70}")
