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

BATCH_SIZE   = 64
LR           = 2e-4
MAX_LEN      = 30
NUM_ROUNDS   = 20
LOCAL_EPOCHS = 3
NUM_CLIENTS  = 5

CLIENT_TRAIN_FILES = [f'data/client{i+1}_train.csv' for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES   = [f'data/client{i+1}_val.csv'   for i in range(NUM_CLIENTS)]
CLIENT_NAMES       = ['Coronavirus','Parasite','Human','Flavivirus','Bacteria']

class EpitopeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=MAX_LEN):
        df = pd.read_csv(csv_path)
        self.sequences  = df['sequence'].tolist()
        self.labels     = df['label'].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx], return_tensors='pt',
            padding='max_length', truncation=True, max_length=self.max_length)
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)}

def build_model(r=8):
    model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=r, lora_alpha=r*2,
        lora_dropout=0.1, target_modules=['query','value']))

def get_trainable_weights(model):
    return {n: p.data.clone() for n, p in model.named_parameters()
            if p.requires_grad}

def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    return model

def federated_averaging(client_weights_list, client_sizes):
    total       = sum(client_sizes)
    proportions = [s/total for s in client_sizes]
    averaged    = {}
    for key in client_weights_list[0].keys():
        averaged[key] = sum(proportions[i]*client_weights_list[i][key].float()
                            for i in range(len(client_weights_list)))
    return averaged

def train_local(model, loader, optimizer, device, class_weights):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(LOCAL_EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            loss = criterion(
                model(input_ids=batch['input_ids'].to(device),
                      attention_mask=batch['attention_mask'].to(device)).logits,
                batch['labels'].to(device))
            loss.backward()
            optimizer.step()

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
        'mcc':     matthews_corrcoef(all_labels, preds)}

def run_federated(train_loaders, val_loaders, test_loader, device,
                  class_weights_list, client_sizes, r=8,
                  num_rounds=NUM_ROUNDS, label=''):
    global_model = build_model(r=r).to(device)
    round_aucs   = []
    best_auc     = 0.0
    for round_num in range(1, num_rounds+1):
        client_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            optimizer   = torch.optim.AdamW(local_model.parameters(), lr=LR)
            train_local(local_model, train_loaders[client_id],
                        optimizer, device, class_weights_list[client_id])
            client_weights.append(get_trainable_weights(local_model))
            del local_model
            torch.cuda.empty_cache()
        averaged     = federated_averaging(client_weights, client_sizes)
        global_model = set_trainable_weights(global_model, averaged)
        m            = evaluate(global_model, test_loader, device)
        round_aucs.append(m['auc_roc'])
        if m['auc_roc'] > best_auc:
            best_auc = m['auc_roc']
        print(f"  [{label} | Round {round_num:02d}] AUC-ROC: {m['auc_roc']:.4f}")
    return best_auc, round_aucs

if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")
    os.makedirs('results', exist_ok=True)

    # Build dataloaders once — reused across all ablations
    train_loaders, val_loaders, class_weights_list, client_sizes = [], [], [], []
    for i in range(NUM_CLIENTS):
        tr  = EpitopeDataset(CLIENT_TRAIN_FILES[i], tokenizer)
        va  = EpitopeDataset(CLIENT_VAL_FILES[i],   tokenizer)
        df  = pd.read_csv(CLIENT_TRAIN_FILES[i])
        pos = df['label'].sum()
        neg = len(df) - pos
        train_loaders.append(DataLoader(tr, batch_size=BATCH_SIZE,
                                        shuffle=True,  num_workers=4))
        val_loaders.append(  DataLoader(va, batch_size=128,
                                        shuffle=False, num_workers=4))
        class_weights_list.append(
            torch.tensor([1.0, neg/max(pos,1)], dtype=torch.float).to(device))
        client_sizes.append(len(tr))

    test_ds     = EpitopeDataset('data/central_test.csv', tokenizer)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # ── Ablation 1: LoRA Rank ─────────────────────────────────────────────
    print("="*60)
    print("ABLATION 1: LoRA Rank (r = 2, 4, 8, 16)")
    print("="*60)
    rank_results = {}
    for r in [2, 4, 8, 16]:
        # Count params
        tmp = build_model(r=r)
        trainable_params = sum(p.numel() for p in tmp.parameters()
                               if p.requires_grad)
        lora_params = sum(p.numel() for n, p in tmp.named_parameters()
                          if p.requires_grad and 'lora' in n)
        del tmp
        print(f"\n  r={r} | trainable={trainable_params:,} | "
              f"lora_params={lora_params:,}")
        best_auc, round_aucs = run_federated(
            train_loaders, val_loaders, test_loader, device,
            class_weights_list, client_sizes, r=r, label=f'r={r}')
        rank_results[r] = {
            'best_auc':        best_auc,
            'trainable_params': trainable_params,
            'lora_params':      lora_params,
            'round_aucs':       round_aucs}
        print(f"  r={r} → Best AUC: {best_auc:.4f}")

    pd.DataFrame({
        'r':               list(rank_results.keys()),
        'best_auc':        [rank_results[r]['best_auc']        for r in rank_results],
        'trainable_params':[rank_results[r]['trainable_params'] for r in rank_results],
        'lora_params':     [rank_results[r]['lora_params']      for r in rank_results],
    }).to_csv('results/ablation_lora_rank.csv', index=False)
    print("\nLoRA rank ablation saved.")

    # ── Ablation 2: Convergence curve already from main run ───────────────
    # Already have results/federated_rounds.csv from Phase 4
    print("\n" + "="*60)
    print("ABLATION 2: Convergence curve — already saved from Phase 4")
    print("  See results/federated_rounds.csv")
    print("="*60)

    # ── Ablation 3: IID vs Non-IID ────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION 3: IID vs Non-IID")
    print("="*60)
    iid_train_loaders, iid_val_loaders, iid_cw_list, iid_sizes = [], [], [], []
    for i in range(NUM_CLIENTS):
        tr  = EpitopeDataset(f'data/iid_client{i+1}_train.csv', tokenizer)
        va  = EpitopeDataset(f'data/iid_client{i+1}_val.csv',   tokenizer)
        df  = pd.read_csv(f'data/iid_client{i+1}_train.csv')
        pos = df['label'].sum()
        neg = len(df) - pos
        iid_train_loaders.append(DataLoader(tr, batch_size=BATCH_SIZE,
                                            shuffle=True,  num_workers=4))
        iid_val_loaders.append(  DataLoader(va, batch_size=128,
                                            shuffle=False, num_workers=4))
        iid_cw_list.append(
            torch.tensor([1.0, neg/max(pos,1)], dtype=torch.float).to(device))
        iid_sizes.append(len(tr))
        print(f"  IID Client {i+1}: train={len(tr)}, "
              f"pos%={pos/len(df):.1%}")

    print("\n  Running IID federated training...")
    iid_best_auc, iid_round_aucs = run_federated(
        iid_train_loaders, iid_val_loaders, test_loader, device,
        iid_cw_list, iid_sizes, r=8, label='IID')

    # Load Non-IID round aucs from Phase 4
    noniid_df = pd.read_csv('results/federated_rounds.csv')
    pd.DataFrame({
        'round':           list(range(1, NUM_ROUNDS+1)),
        'iid_auc':         iid_round_aucs,
        'noniid_auc':      noniid_df['auc_roc'].tolist(),
    }).to_csv('results/ablation_iid_vs_noniid.csv', index=False)
    print(f"\n  IID best AUC:     {iid_best_auc:.4f}")
    print(f"  Non-IID best AUC: {noniid_df['auc_roc'].max():.4f}")
    print("  IID vs Non-IID comparison saved.")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print("\nLoRA Rank:")
    print(f"  {'r':>4} {'Best AUC':>10} {'LoRA Params':>12}")
    for r, res in rank_results.items():
        print(f"  {r:>4} {res['best_auc']:>10.4f} {res['lora_params']:>12,}")
    print(f"\nIID vs Non-IID:")
    print(f"  IID best AUC:     {iid_best_auc:.4f}")
    print(f"  Non-IID best AUC: {noniid_df['auc_roc'].max():.4f}")
    print("\nAblation experiments complete. Results saved to results/")
