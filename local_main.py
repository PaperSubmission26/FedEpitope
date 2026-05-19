import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# ── Dataset ────────────────────────────────────────────────────────────────
class EpitopeDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=30):
        df = pd.read_csv(csv_path)
        self.sequences = df['sequence'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
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
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Model ──────────────────────────────────────────────────────────────────
def build_model():
    model = EsmForSequenceClassification.from_pretrained(
        'facebook/esm2_t12_35M_UR50D',
        num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['query', 'value']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ── Training ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    return {
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pr':  average_precision_score(all_labels, all_probs),
        'f1':      f1_score(all_labels, preds, zero_division=0),
        'mcc':     matthews_corrcoef(all_labels, preds)
    }

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')

    # Use Client 2 (SARS-CoV1) — largest client dataset
    train_dataset = EpitopeDataset('data/client2_train.csv', tokenizer)
    val_dataset   = EpitopeDataset('data/client2_val.csv',   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

    # Class weights to handle imbalance
    train_df = pd.read_csv('data/client2_train.csv')
    pos = train_df['label'].sum()
    neg = len(train_df) - pos
    class_weights = torch.tensor([1.0, neg/pos], dtype=torch.float).to(device)
    print(f"Class weights — neg: 1.0, pos: {neg/pos:.2f}")

    model = build_model()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | "
              f"AUC-ROC: {metrics['auc_roc']:.4f} | AUC-PR: {metrics['auc_pr']:.4f} | "
              f"F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f}")

    print("Criteria: loss decreasing each epoch, AUC-ROC > 0.65")
