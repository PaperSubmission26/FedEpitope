import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, matthews_corrcoef)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

PERSONAL_EPOCHS_V1 = 10
PERSONAL_LR_V1     = 1e-5

PERSONAL_EPOCHS_V2 = 10
PERSONAL_LR_V2     = 1e-4

PERSONAL_EPOCHS_V3 = 10
PERSONAL_LR_V3     = 1e-5
MU                 = 0.01

CLASSIFIER_PARAM_NAMES = {
    'base_model.model.classifier.modules_to_save.default.dense.weight',
    'base_model.model.classifier.modules_to_save.default.dense.bias',
    'base_model.model.classifier.modules_to_save.default.out_proj.weight',
    'base_model.model.classifier.modules_to_save.default.out_proj.bias',
}

CLIENT_TRAIN_FILES  = [f'data/client{i+1}_train.csv' for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES    = [f'data/client{i+1}_val.csv'   for i in range(NUM_CLIENTS)]
CLIENT_TEST_FILES   = [f'data/client{i+1}_test.csv'  for i in range(NUM_CLIENTS)]
CLIENT_NAMES        = ['Coronavirus','Parasite','Human','Flavivirus','Bacteria']

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
        'facebook/esm2_t12_35M_UR50D', num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=['query', 'value'])
    return get_peft_model(model, lora_config)

# ── Weight utilities ───────────────────────────────────────────────────────
def get_trainable_weights(model):
    return {n: p.data.clone()
            for n, p in model.named_parameters() if p.requires_grad}

def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    return model

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

# ── Standard local training ────────────────────────────────────────────────
def train_local_standard(model, loader, optimizer, device,
                          class_weights, epochs, label):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
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
        print(f"  [{label} | Epoch {epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(loader):.4f}")

# ── FedProx local training ─────────────────────────────────────────────────
def train_local_fedprox(model, loader, optimizer, device,
                         class_weights, epochs, global_weights_cpu,
                         mu, label):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Move global weights to GPU once before training loop
    global_weights_gpu = {n: w.to(device)
                          for n, w in global_weights_cpu.items()}
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            optimizer.zero_grad()
            ce_loss = criterion(
                model(input_ids=input_ids,
                      attention_mask=attention_mask).logits, labels)
            prox_loss = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and n in global_weights_gpu:
                    prox_loss += ((p - global_weights_gpu[n]) ** 2).sum()
            prox_loss = (mu / 2) * prox_loss
            loss      = ce_loss + prox_loss
            loss.backward()
            optimizer.step()
            total_loss += ce_loss.item()
        print(f"  [{label} | Epoch {epoch+1}/{epochs}] "
              f"CE Loss: {total_loss/len(loader):.4f}")

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
        'mcc':     matthews_corrcoef(all_labels, preds)
    }

# ── Phase 1: Standard federated training ──────────────────────────────────
def run_standard_federated(train_loaders, val_loaders,
                            central_test_loader, client_test_loaders,
                            device, class_weights_list, client_sizes):
    print(f"\n{'='*70}")
    print("PHASE 1: Standard Federated Training")
    print("  Global model evaluated on central test set each round")
    print(f"{'='*70}")

    global_model = build_model().to(device)
    round_aucs   = []
    best_auc     = 0.0
    best_weights = None

    for round_num in range(1, NUM_ROUNDS + 1):
        client_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            optimizer   = torch.optim.AdamW(local_model.parameters(), lr=LR)
            train_local_standard(
                local_model, train_loaders[client_id], optimizer,
                device, class_weights_list[client_id], LOCAL_EPOCHS,
                f"Round {round_num:02d} | "
                f"Client {client_id+1} {CLIENT_NAMES[client_id]}")
            m = evaluate(local_model, val_loaders[client_id], device)
            print(f"  [Round {round_num:02d} | Client {client_id+1} "
                  f"{CLIENT_NAMES[client_id]:<12} | Local val] "
                  f"AUC-ROC: {m['auc_roc']:.4f}")
            client_weights.append(get_trainable_weights(local_model))
            del local_model
            torch.cuda.empty_cache()

        averaged     = federated_averaging(client_weights, client_sizes)
        global_model = set_trainable_weights(global_model, averaged)

        # Evaluate on central test set
        metrics = evaluate(global_model, central_test_loader, device)
        round_aucs.append(metrics['auc_roc'])
        if metrics['auc_roc'] > best_auc:
            best_auc     = metrics['auc_roc']
            best_weights = get_trainable_weights(global_model)
        print(f"\n  ★ [Standard FL | Round {round_num:02d}] "
              f"Central test AUC: {metrics['auc_roc']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

    # Restore best global model
    global_model = set_trainable_weights(global_model, best_weights)

    # Evaluate global model on each client test set
    print(f"\n  Global model per-client test AUC (before personalisation):")
    global_per_client_aucs = []
    for client_id in range(NUM_CLIENTS):
        m = evaluate(global_model, client_test_loaders[client_id], device)
        global_per_client_aucs.append(m['auc_roc'])
        print(f"  Client {client_id+1} ({CLIENT_NAMES[client_id]:<12}): "
              f"{m['auc_roc']:.4f}")

    print(f"\n  Standard FL best central AUC: {best_auc:.4f}")
    print(f"  Standard FL avg per-client AUC: "
          f"{np.mean(global_per_client_aucs):.4f}")
    return (best_auc, round_aucs, global_model,
            best_weights, global_per_client_aucs)

# ── Personalisation — runs all 3 variants ─────────────────────────────────
def run_personalisation(variant_name, global_model, best_weights,
                         train_loaders, val_loaders, client_test_loaders,
                         device, class_weights_list,
                         epochs, lr, freeze_lora=False, use_fedprox=False):
    """
    Unified personalisation function for all 3 variants.

    variant_name : label for printing
    freeze_lora  : if True, only classifier head trains (Variant 2)
    use_fedprox  : if True, adds proximal term to loss (Variant 3)

    Each client:
      1. Starts from best global model weights
      2. Fine-tunes on LOCAL training data
      3. Evaluated on its OWN client test set (correct pFL evaluation)
    """
    print(f"\n{'='*70}")
    print(f"PERSONALISATION: {variant_name}")
    print(f"  epochs={epochs}, lr={lr}, "
          f"freeze_lora={freeze_lora}, use_fedprox={use_fedprox}")
    print(f"  Each client evaluated on its OWN pathogen test set")
    print(f"{'='*70}\n")

    # For FedProx: store global weights on CPU once
    if use_fedprox:
        global_weights_cpu = {n: p.cpu().clone()
                              for n, p in global_model.named_parameters()
                              if p.requires_grad}

    results = []
    for client_id in range(NUM_CLIENTS):
        personal_model = copy.deepcopy(global_model)
        personal_model = set_trainable_weights(personal_model, best_weights)

        if freeze_lora:
            # Variant 2: freeze LoRA, only classifier trains
            for n, p in personal_model.named_parameters():
                p.requires_grad = (n in CLASSIFIER_PARAM_NAMES)
            trainable_count = sum(p.numel()
                                  for p in personal_model.parameters()
                                  if p.requires_grad)
            assert trainable_count == (480*480 + 480 + 2*480 + 2), \
                f"Classifier freeze incorrect: {trainable_count} params"
            optimizer = torch.optim.AdamW(
                [p for p in personal_model.parameters()
                 if p.requires_grad], lr=lr)
        else:
            optimizer = torch.optim.AdamW(
                personal_model.parameters(), lr=lr)

        # Before personalisation — evaluate on client test set
        m_before = evaluate(personal_model,
                            client_test_loaders[client_id], device)
        print(f"  Client {client_id+1} ({CLIENT_NAMES[client_id]:<12}) "
              f"BEFORE: client test AUC={m_before['auc_roc']:.4f}")

        # Train
        if use_fedprox:
            train_local_fedprox(
                personal_model, train_loaders[client_id], optimizer,
                device, class_weights_list[client_id], epochs,
                global_weights_cpu, MU,
                f"{variant_name} Client {client_id+1} "
                f"{CLIENT_NAMES[client_id]}")
        else:
            train_local_standard(
                personal_model, train_loaders[client_id], optimizer,
                device, class_weights_list[client_id], epochs,
                f"{variant_name} Client {client_id+1} "
                f"{CLIENT_NAMES[client_id]}")

        # After personalisation — evaluate on client test set
        m_after = evaluate(personal_model,
                           client_test_loaders[client_id], device)
        m_val   = evaluate(personal_model,
                           val_loaders[client_id], device)
        print(f"  Client {client_id+1} ({CLIENT_NAMES[client_id]:<12}) "
              f"AFTER:  client test AUC={m_after['auc_roc']:.4f} | "
              f"val AUC={m_val['auc_roc']:.4f} | "
              f"improvement={m_after['auc_roc']-m_before['auc_roc']:+.4f}")

        results.append({
            'before': m_before['auc_roc'],
            'after':  m_after['auc_roc'],
            'f1':     m_after['f1'],
            'mcc':    m_after['mcc'],
            'auc_pr': m_after['auc_pr'],
        })
        del personal_model
        torch.cuda.empty_cache()

    avg_before = np.mean([r['before'] for r in results])
    avg_after  = np.mean([r['after']  for r in results])
    print(f"\n  {variant_name} avg client test AUC: "
          f"{avg_before:.4f} → {avg_after:.4f} "
          f"({avg_after - avg_before:+.4f})")
    return results, avg_after

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables',  exist_ok=True)

    # ── Pre-flight checks ──────────────────────────────────────────────────
    print("Running pre-flight checks...")
    test_model = build_model()
    actual_clf = {n for n, p in test_model.named_parameters()
                  if p.requires_grad and 'classifier' in n}
    assert actual_clf == CLASSIFIER_PARAM_NAMES, \
        f"Classifier mismatch: {actual_clf}"
    print("  ✅ Classifier param names verified")
    total = sum(p.numel() for p in test_model.parameters()
                if p.requires_grad)
    assert total == 416162, f"Expected 416162, got {total}"
    print(f"  ✅ Trainable params: {total:,}")
    m2  = copy.deepcopy(test_model)
    gw  = {n: p.data.clone() for n, p in test_model.named_parameters()
           if p.requires_grad}
    prx = sum(((p - gw[n]) ** 2).sum()
              for n, p in m2.named_parameters()
              if p.requires_grad and n in gw)
    assert prx.item() < 1e-6
    print("  ✅ FedProx proximal term verified")
    del test_model, m2, gw
    missing = []
    for i in range(1, 6):
        for s in ['train', 'val', 'test']:
            f = f'data/client{i}_{s}.csv'
            if not os.path.exists(f):
                missing.append(f)
    for f in ['data/central_test.csv', 'results/baseline_results.csv']:
        if not os.path.exists(f):
            missing.append(f)
    assert not missing, f"Missing: {missing}"
    print("  ✅ All data files present")
    print("  ✅ All checks passed\n")

    # ── Build dataloaders ──────────────────────────────────────────────────
    train_loaders, val_loaders       = [], []
    client_test_loaders              = []
    class_weights_list, client_sizes = [], []

    for i in range(NUM_CLIENTS):
        tr  = EpitopeDataset(CLIENT_TRAIN_FILES[i], tokenizer)
        va  = EpitopeDataset(CLIENT_VAL_FILES[i],   tokenizer)
        ct  = EpitopeDataset(CLIENT_TEST_FILES[i],  tokenizer)
        df  = pd.read_csv(CLIENT_TRAIN_FILES[i])
        pos = df['label'].sum()
        neg = len(df) - pos
        train_loaders.append(
            DataLoader(tr, batch_size=BATCH_SIZE,
                       shuffle=True, num_workers=4))
        val_loaders.append(
            DataLoader(va, batch_size=128,
                       shuffle=False, num_workers=4))
        client_test_loaders.append(
            DataLoader(ct, batch_size=128,
                       shuffle=False, num_workers=4))
        class_weights_list.append(
            torch.tensor([1.0, neg / max(pos, 1)],
                         dtype=torch.float).to(device))
        client_sizes.append(len(tr))
        print(f"Client {i+1} ({CLIENT_NAMES[i]:<12}): "
              f"train={len(tr):>7}, val={len(va):>6}, "
              f"client_test={len(ct):>6}, pos%={pos/len(df):.1%}")

    central_test_ds     = EpitopeDataset('data/central_test.csv', tokenizer)
    central_test_loader = DataLoader(central_test_ds, batch_size=128,
                                     shuffle=False, num_workers=4)
    print(f"Central test set: {len(central_test_ds)} samples\n")

    # Load single-client baselines — evaluated on central test (from Phase 5)
    baseline_df = pd.read_csv('results/baseline_results.csv', index_col=0)
    single_central_aucs = [
        baseline_df.loc['Single_Client1_Coronavirus', 'auc_roc'],
        baseline_df.loc['Single_Client2_Parasite',    'auc_roc'],
        baseline_df.loc['Single_Client3_Human',        'auc_roc'],
        baseline_df.loc['Single_Client4_Flavivirus',   'auc_roc'],
        baseline_df.loc['Single_Client5_Bacteria',     'auc_roc'],
    ]

    # ── Phase 1: Standard FL ──────────────────────────────────────────────
    (std_central_auc, std_round_aucs, global_model,
     best_weights, std_per_client_aucs) = run_standard_federated(
        train_loaders, val_loaders,
        central_test_loader, client_test_loaders,
        device, class_weights_list, client_sizes)

    torch.save(best_weights, 'results/pfl_v2_global_model.pt')

    # ── Phase 2: Three personalisation variants ────────────────────────────
    # Each variant evaluated on per-client test sets
    v1_results, v1_avg = run_personalisation(
        'pFL-LFT (lr=1e-5, 10 epochs)',
        global_model, best_weights,
        train_loaders, val_loaders, client_test_loaders,
        device, class_weights_list,
        epochs=PERSONAL_EPOCHS_V1, lr=PERSONAL_LR_V1,
        freeze_lora=False, use_fedprox=False)

    v2_results, v2_avg = run_personalisation(
        'pFL-CLF-Only (lr=1e-4, 10 epochs)',
        global_model, best_weights,
        train_loaders, val_loaders, client_test_loaders,
        device, class_weights_list,
        epochs=PERSONAL_EPOCHS_V2, lr=PERSONAL_LR_V2,
        freeze_lora=True, use_fedprox=False)

    v3_results, v3_avg = run_personalisation(
        f'pFL-FedProx (lr=1e-5, 10 epochs, mu={MU})',
        global_model, best_weights,
        train_loaders, val_loaders, client_test_loaders,
        device, class_weights_list,
        epochs=PERSONAL_EPOCHS_V3, lr=PERSONAL_LR_V3,
        freeze_lora=False, use_fedprox=True)

    # ── Collect results ────────────────────────────────────────────────────
    v1_aucs = [r['after'] for r in v1_results]
    v2_aucs = [r['after'] for r in v2_results]
    v3_aucs = [r['after'] for r in v3_results]

    # ── Save per-client results ────────────────────────────────────────────
    per_client_df = pd.DataFrame({
        'Client':            CLIENT_NAMES,
        'Local_Only_central':[round(a, 4) for a in single_central_aucs],
        'StdFL_client_test': [round(a, 4) for a in std_per_client_aucs],
        'pFL_LFT':           [round(a, 4) for a in v1_aucs],
        'pFL_CLF_Only':      [round(a, 4) for a in v2_aucs],
        'pFL_FedProx':       [round(a, 4) for a in v3_aucs],
    })
    per_client_df.to_csv('results/pfl_v2_per_client.csv', index=False)
    pd.DataFrame({'round': list(range(1, NUM_ROUNDS+1)),
                  'standard_fl': std_round_aucs
                  }).to_csv('results/pfl_v2_rounds.csv', index=False)

    # ── Figure 1: Per-client test AUC — all methods ────────────────────────
    print("\nGenerating figures...")
    fig, ax   = plt.subplots(figsize=(14, 6))
    x         = np.arange(NUM_CLIENTS)
    width     = 0.15
    data_list = [
        (std_per_client_aucs, 'Standard FL (client test)',  'coral'),
        (v1_aucs,             'pFL LFT',                    'mediumseagreen'),
        (v2_aucs,             'pFL Classifier-Only',        'gold'),
        (v3_aucs,             'pFL FedProx',                'mediumpurple'),
    ]
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for (data, label, color), offset in zip(data_list, offsets):
        bars = ax.bar(x + offset * width, data, width,
                      label=label, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Client', fontsize=13)
    ax.set_ylabel('AUC-ROC (Per-Client Test Set)', fontsize=13)
    ax.set_title('Personalised FL: Per-Client Test AUC\n'
                 '(Each model evaluated on its own pathogen test set)',
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(CLIENT_NAMES, fontsize=11)
    ax.set_ylim(0.50, 1.00)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/figure_pfl_v2_per_client.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Average AUC summary ─────────────────────────────────────
    std_avg    = np.mean(std_per_client_aucs)
    central_auc = baseline_df.loc['Centralised_LoRA', 'auc_roc']
    methods    = ['Std FL\n(client test)', 'pFL LFT',
                  'pFL CLF-Only', 'pFL FedProx']
    avg_aucs   = [round(std_avg, 4), round(v1_avg, 4),
                  round(v2_avg, 4),  round(v3_avg, 4)]
    bar_colors = ['coral', 'mediumseagreen', 'gold', 'mediumpurple']
    fig, ax    = plt.subplots(figsize=(9, 6))
    bars       = ax.bar(methods, avg_aucs, color=bar_colors, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Avg AUC-ROC (Per-Client Test Sets)', fontsize=13)
    ax.set_title('Personalised FL: Average Per-Client Performance',
                 fontsize=13)
    ax.set_ylim(0.50, 1.00)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/figure_pfl_v2_summary.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Standard FL convergence (central test) ──────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, NUM_ROUNDS+1), std_round_aucs,
            'b-o', markersize=4, linewidth=2, label='Standard FL')
    ax.axhline(y=central_auc, color='green', linestyle='--', linewidth=2,
               label=f'Centralised upper bound ({central_auc:.4f})')
    ax.axhline(y=std_avg, color='coral', linestyle=':',  linewidth=2,
               label=f'Std FL avg per-client ({std_avg:.4f})')
    ax.axhline(y=v1_avg,  color='mediumseagreen', linestyle=':', linewidth=2,
               label=f'pFL LFT avg ({v1_avg:.4f})')
    ax.axhline(y=v2_avg,  color='gold', linestyle='-.', linewidth=2,
               label=f'pFL CLF-Only avg ({v2_avg:.4f})')
    ax.axhline(y=v3_avg,  color='mediumpurple', linestyle=':', linewidth=2,
               label=f'pFL FedProx avg ({v3_avg:.4f})')
    ax.set_xlabel('Federated Round', fontsize=13)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_title('Standard FL Convergence + Personalisation Results',
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/figure_pfl_v2_convergence.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved 3 figures")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PERSONALISED FL v2 FINAL SUMMARY")
    print("Note: pFL variants evaluated on per-client test sets")
    print("      Standard FL also shown on per-client test sets for fair comparison")
    print("="*70)
    print(f"\n{'Method':<30} {'Avg per-client AUC':>20} {'vs StdFL':>10}")
    print("-"*62)
    for method, avg in [
        ('Standard FL (client test)', std_avg),
        ('pFL LFT',                   v1_avg),
        ('pFL Classifier-Only',       v2_avg),
        ('pFL FedProx',               v3_avg),
    ]:
        print(f"{method:<30} {avg:>20.4f} {avg-std_avg:>+10.4f}")

    print(f"\nPer-client breakdown (client test AUC):")
    print(f"{'Client':<13} {'StdFL':>8} {'LFT':>8} "
          f"{'CLF':>8} {'FedProx':>9} {'Best pFL':>10}")
    print("-"*60)
    for i in range(NUM_CLIENTS):
        best_pfl = max(v1_aucs[i], v2_aucs[i], v3_aucs[i])
        print(f"{CLIENT_NAMES[i]:<13} "
              f"{std_per_client_aucs[i]:>8.4f} "
              f"{v1_aucs[i]:>8.4f} "
              f"{v2_aucs[i]:>8.4f} "
              f"{v3_aucs[i]:>9.4f} "
              f"{best_pfl:>10.4f} "
              f"{'✅' if best_pfl > std_per_client_aucs[i] else '❌'}")

    wins = sum(1 for i in range(NUM_CLIENTS)
               if max(v1_aucs[i], v2_aucs[i], v3_aucs[i])
               > std_per_client_aucs[i])
    print(f"\nBest pFL variant outperforms Standard FL: "
          f"{wins}/{NUM_CLIENTS} clients")

    print(f"\n✅ Personalised FL v2 complete.")
    print(f"   Saved: results/pfl_v2_per_client.csv")
    print(f"   Saved: results/pfl_v2_rounds.csv")
    print(f"   Saved: figures/figure_pfl_v2_per_client.png")
    print(f"   Saved: figures/figure_pfl_v2_summary.png")
    print(f"   Saved: figures/figure_pfl_v2_convergence.png")
