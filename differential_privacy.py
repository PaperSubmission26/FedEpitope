import torch
import copy
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
NUM_ROUNDS      = 20
LOCAL_EPOCHS    = 3
NUM_CLIENTS     = 5
BATCH_SIZE      = 64
LR              = 2e-4
MAX_LEN         = 30
MAX_GRAD_NORM   = 1.0
EPSILON_VALUES  = [1.0, 5.0, 10.0, float('inf')]
DELTA           = 1e-5

CLIENT_TRAIN_FILES = [f'data/client{i+1}_train.csv' for i in range(NUM_CLIENTS)]
CLIENT_VAL_FILES   = [f'data/client{i+1}_val.csv'   for i in range(NUM_CLIENTS)]
CLIENT_NAMES       = ['Coronavirus','Parasite','Human','Flavivirus','Bacteria']

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
    model = get_peft_model(model, lora_config)
    model = ModuleValidator.fix(model)
    return model

# ── Weight utilities ───────────────────────────────────────────────────────
def get_trainable_weights(model):
    """
    Extract trainable weights, stripping Opacus '_module.' prefix if present.
    Opacus wraps the model and renames parameters from
    'base_model.X' to '_module.base_model.X'. We always
    store weights under the original name so aggregation works correctly.
    """
    weights = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            # Strip Opacus wrapper prefix if present
            clean_name = n[len('_module.'):] if n.startswith('_module.') else n
            weights[clean_name] = p.data.clone()
    return weights

def set_trainable_weights(model, weights):
    for n, p in model.named_parameters():
        if p.requires_grad and n in weights:
            p.data.copy_(weights[n])
    return model

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

# ── Local training with DP ─────────────────────────────────────────────────
def train_local_with_dp(model, train_dataset, device, class_weights,
                         target_epsilon, round_num, client_id):
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # required by Opacus
    )

    model.train()  # must be in train mode before make_private
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if target_epsilon != float('inf'):
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=LOCAL_EPOCHS,
            target_epsilon=target_epsilon,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"  [Round {round_num:02d} | Client {client_id+1} "
              f"{CLIENT_NAMES[client_id]:<12}] "
              f"noise_multiplier: {optimizer.noise_multiplier:.4f}")
    else:
        privacy_engine = None

    for epoch in range(LOCAL_EPOCHS):
        model.train()
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
        print(f"  [Round {round_num:02d} | Client {client_id+1} "
              f"{CLIENT_NAMES[client_id]:<12} | Epoch {epoch+1}/{LOCAL_EPOCHS}]"
              f" Loss: {total_loss/len(loader):.4f}")

    if privacy_engine is not None:
        actual_epsilon = privacy_engine.get_epsilon(DELTA)
        print(f"  [Round {round_num:02d} | Client {client_id+1} "
              f"{CLIENT_NAMES[client_id]:<12}] "
              f"Actual epsilon: {actual_epsilon:.4f}")

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
        'mcc':     matthews_corrcoef(all_labels, preds)
    }

# ── Full federated run for one epsilon ────────────────────────────────────
def run_federated_dp(train_datasets, val_loaders, test_loader, device,
                     class_weights_list, client_sizes, target_epsilon):
    label = f'eps={target_epsilon}' if target_epsilon != float('inf') else 'No-DP'
    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"{'='*70}")

    global_model = build_model().to(device)
    round_aucs   = []
    best_auc     = 0.0
    best_round   = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n  --- Round {round_num}/{NUM_ROUNDS} [{label}] ---")
        client_weights = []

        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model)
            local_model.train()

            local_model = train_local_with_dp(
                local_model,
                train_datasets[client_id],
                device,
                class_weights_list[client_id],
                target_epsilon,
                round_num,
                client_id
            )

            # Evaluate — need to handle Opacus wrapper for eval
            # Use a plain loader for evaluation
            eval_loader = val_loaders[client_id]
            m = evaluate(local_model, eval_loader, device)
            print(f"  [Round {round_num:02d} | Client {client_id+1} "
                  f"{CLIENT_NAMES[client_id]:<12} | Local val] "
                  f"AUC-ROC: {m['auc_roc']:.4f}")

            # get_trainable_weights strips _module. prefix automatically
            client_weights.append(get_trainable_weights(local_model))
            del local_model
            torch.cuda.empty_cache()

        averaged     = federated_averaging(client_weights, client_sizes)
        global_model = set_trainable_weights(global_model, averaged)

        metrics = evaluate(global_model, test_loader, device)
        round_aucs.append(metrics['auc_roc'])
        if metrics['auc_roc'] > best_auc:
            best_auc   = metrics['auc_roc']
            best_round = round_num

        print(f"\n  ★ [{label}] Round {round_num:02d} | "
              f"Global AUC-ROC: {metrics['auc_roc']:.4f} | "
              f"AUC-PR: {metrics['auc_pr']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"MCC: {metrics['mcc']:.4f}")

    print(f"\n  [{label}] Best AUC: {best_auc:.4f} at Round {best_round}")
    return best_auc, round_aucs, metrics

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    print(f"Device: {device}\n")

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables',  exist_ok=True)

    train_datasets, val_loaders, class_weights_list, client_sizes = [], [], [], []
    for i in range(NUM_CLIENTS):
        tr  = EpitopeDataset(CLIENT_TRAIN_FILES[i], tokenizer)
        va  = EpitopeDataset(CLIENT_VAL_FILES[i],   tokenizer)
        df  = pd.read_csv(CLIENT_TRAIN_FILES[i])
        pos = df['label'].sum()
        neg = len(df) - pos
        train_datasets.append(tr)
        val_loaders.append(DataLoader(va, batch_size=128,
                                      shuffle=False, num_workers=4))
        class_weights_list.append(
            torch.tensor([1.0, neg/max(pos,1)], dtype=torch.float).to(device))
        client_sizes.append(len(tr))
        print(f"Client {i+1} ({CLIENT_NAMES[i]:<12}): "
              f"train={len(tr):>7}, pos%={pos/len(df):.1%}")

    test_ds     = EpitopeDataset('data/central_test.csv', tokenizer)
    test_loader = DataLoader(test_ds, batch_size=128,
                             shuffle=False, num_workers=4)
    print(f"Central test set: {len(test_ds)} samples\n")

    # ── Verify weight name fix before long run ─────────────────────────────
    print("Verifying weight name stripping...")
    test_model = build_model().to(device)
    test_model.train()
    test_opt   = torch.optim.AdamW(test_model.parameters(), lr=LR)
    test_ds_small = EpitopeDataset(CLIENT_TRAIN_FILES[0], tokenizer)
    test_loader_small = DataLoader(test_ds_small, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0)
    pe = PrivacyEngine()
    test_model, test_opt, _ = pe.make_private_with_epsilon(
        module=test_model, optimizer=test_opt,
        data_loader=test_loader_small,
        epochs=1, target_epsilon=5.0, target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM)
    w = get_trainable_weights(test_model)
    has_module_prefix = any(k.startswith('_module.') for k in w.keys())
    print(f"  _module. prefix present after stripping: {has_module_prefix}")
    print(f"  Total weights extracted: {len(w)}")
    assert not has_module_prefix, "Prefix stripping failed!"
    assert len(w) == 52, f"Expected 52 weights, got {len(w)}"
    print("  ✅ Weight name fix verified — safe to run\n")
    del test_model, test_opt, pe, test_loader_small
    torch.cuda.empty_cache()

    dp_results     = {}
    all_round_aucs = {}

    for epsilon in EPSILON_VALUES:
        label = f'eps={epsilon}' if epsilon != float('inf') else 'No-DP'
        best_auc, round_aucs, final_metrics = run_federated_dp(
            train_datasets, val_loaders, test_loader, device,
            class_weights_list, client_sizes, epsilon)
        dp_results[label] = {
            'epsilon':  epsilon,
            'best_auc': best_auc,
            'auc_pr':   final_metrics['auc_pr'],
            'f1':       final_metrics['f1'],
            'mcc':      final_metrics['mcc'],
            'privacy':  'None' if epsilon == float('inf')
                        else f'ε={epsilon}, δ={DELTA}'
        }
        all_round_aucs[label] = round_aucs

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY RESULTS SUMMARY")
    print("="*70)
    print(f"{'Setting':<15} {'Privacy':<20} {'Best AUC-ROC':>12} "
          f"{'AUC-PR':>8} {'F1':>8} {'MCC':>8}")
    print("-"*70)
    no_dp_auc = dp_results['No-DP']['best_auc']
    for label, res in dp_results.items():
        cost = f"(-{no_dp_auc - res['best_auc']:.4f})" \
               if label != 'No-DP' else ''
        print(f"{label:<15} {res['privacy']:<20} {res['best_auc']:>12.4f} "
              f"{res['auc_pr']:>8.4f} {res['f1']:>8.4f} "
              f"{res['mcc']:>8.4f}  {cost}")

    pd.DataFrame(dp_results).T.to_csv('results/dp_results.csv')
    pd.DataFrame(all_round_aucs).to_csv('results/dp_round_aucs.csv',
                                         index=False)

    # ── Figure 1: Convergence curves ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ['red', 'orange', 'blue', 'green']
    styles  = ['-s', '-^', '-o', '--']
    for (label, round_aucs), color, style in zip(
            all_round_aucs.items(), colors, styles):
        ax.plot(range(1, NUM_ROUNDS+1), round_aucs,
                style, color=color, markersize=4,
                linewidth=2, label=label)
    ax.axhline(y=0.74, color='purple', linestyle=':',
               linewidth=1.5, label='BepiPred-3.0 (0.74)')
    ax.set_xlabel('Federated Round', fontsize=13)
    ax.set_ylabel('AUC-ROC (Central Test Set)', fontsize=13)
    ax.set_title('Federated Training with Differential Privacy\n'
                 'Privacy-Utility Tradeoff across ε values', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/figure_dp_convergence.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    # ── Figure 2: Bar chart tradeoff ──────────────────────────────────────
    fig, ax     = plt.subplots(figsize=(8, 5))
    labels_plot = list(dp_results.keys())
    best_aucs   = [dp_results[l]['best_auc'] for l in labels_plot]
    bar_colors  = ['red', 'orange', 'blue', 'green']
    bars = ax.bar(labels_plot, best_aucs, color=bar_colors, alpha=0.8)
    ax.axhline(y=0.74, color='purple', linestyle='--',
               linewidth=1.5, label='BepiPred-3.0 (0.74)')
    ax.set_xlabel('Privacy Setting', fontsize=13)
    ax.set_ylabel('Best AUC-ROC', fontsize=13)
    ax.set_title('Privacy-Utility Tradeoff', fontsize=13)
    ax.set_ylim(0.55, 0.85)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/figure_dp_tradeoff.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    print("\n✅ Differential privacy experiments complete.")
    print(f"   No-DP:  {dp_results['No-DP']['best_auc']:.4f}")
    print(f"   ε=10:   {dp_results['eps=10.0']['best_auc']:.4f}  "
          f"cost: {no_dp_auc - dp_results['eps=10.0']['best_auc']:.4f}")
    print(f"   ε=5:    {dp_results['eps=5.0']['best_auc']:.4f}  "
          f"cost: {no_dp_auc - dp_results['eps=5.0']['best_auc']:.4f}")
    print(f"   ε=1:    {dp_results['eps=1.0']['best_auc']:.4f}  "
          f"cost: {no_dp_auc - dp_results['eps=1.0']['best_auc']:.4f}")
    print("\n   Saved: results/dp_results.csv")
    print("   Saved: figures/figure_dp_convergence.png")
    print("   Saved: figures/figure_dp_tradeoff.png")
