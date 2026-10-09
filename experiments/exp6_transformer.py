"""
EXPERIMENT 6 — TRANSFORMER BASELINE (multilingual DistilBERT)
Same train/test split as Exp 0-5. Fine-tunes distilbert-base-multilingual-cased
and compares against the tuned LinearSVC (Exp 4: test macro F1 0.7953).

CPU-only is fine — we subsample training for speed if no GPU.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)

DATA_PATH    = "data/news_balanced.csv"
RESULTS_DIR  = "results"
MODEL_NAME   = "distilbert-base-multilingual-cased"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
MAX_LEN      = 128
os.makedirs(RESULTS_DIR, exist_ok=True)

# CPU speed control — subsample train if no GPU
USE_GPU = torch.cuda.is_available()
TRAIN_SUBSET = None if USE_GPU else 8000   # 8k rows on CPU ≈ manageable
EPOCHS = 2
BATCH_SIZE = 16

print("=" * 60)
print("EXPERIMENT 6 — TRANSFORMER BASELINE (multilingual DistilBERT)")
print("=" * 60)
print(f"Device: {'cuda' if USE_GPU else 'cpu'}   "
      f"train subset: {TRAIN_SUBSET or 'full'}")

# [1/5] Load + split (identical to previous experiments)
df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str).tolist()
y = df["category"].tolist()

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_labels = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
if TRAIN_SUBSET and TRAIN_SUBSET < len(X_train):
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=TRAIN_SUBSET,
        random_state=RANDOM_STATE, stratify=y_train,
    )
print(f"[1/5] Train: {len(X_train):,}   Test: {len(X_test):,}   "
      f"Classes: {num_labels}")

# [2/5] Tokenize
print(f"[2/5] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids":      self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels":         self.labels[i],
        }

train_ds = NewsDataset(X_train, y_train)
test_ds  = NewsDataset(X_test,  y_test)

# [3/5] Model + Trainer
print(f"[3/5] Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

args = TrainingArguments(
    output_dir=f"{RESULTS_DIR}/exp6_checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    report_to="none",
    seed=RANDOM_STATE,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# [4/5] Train
print("[4/5] Fine-tuning ...")
t0 = time.time()
trainer.train()
train_time = time.time() - t0

# [5/5] Evaluate
print("[5/5] Evaluating on test set ...")
preds_out = trainer.predict(test_ds)
y_pred = np.argmax(preds_out.predictions, axis=-1)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f"\n      Test accuracy = {acc:.4f}")
print(f"      Test macro F1 = {f1m:.4f}")
print(f"      Train time    = {train_time:.1f}s")
print("\n" + classification_report(
    y_test, y_pred, target_names=le.classes_, digits=4
))

# Save results + summary
summary = pd.DataFrame([{
    "model":         MODEL_NAME,
    "train_rows":    len(X_train),
    "epochs":        EPOCHS,
    "test_accuracy": acc,
    "test_f1_macro": f1m,
    "train_time_s":  round(train_time, 1),
    "device":        "cuda" if USE_GPU else "cpu",
}])
summary.to_csv(f"{RESULTS_DIR}/exp6_transformer.csv", index=False)

with open(f"{RESULTS_DIR}/exp6_transformer.md", "w") as f:
    f.write("# Experiment 6 — Transformer Baseline (multilingual DistilBERT)\n\n")
    f.write(f"**Model:** `{MODEL_NAME}`\n\n")
    f.write(f"**Train rows:** {len(X_train):,}  "
            f"(device: {'cuda' if USE_GPU else 'cpu'}, epochs: {EPOCHS})\n\n")
    f.write(f"**Test accuracy:** {acc:.4f}\n\n")
    f.write(f"**Test macro F1:** {f1m:.4f}\n\n")
    f.write(f"**Train time:** {train_time:.1f}s\n\n")
    f.write("## Comparison vs LinearSVC (Exp 4)\n\n")
    f.write("| Model | Test acc | Test macro F1 |\n")
    f.write("|-------|----------|---------------|\n")
    f.write("| LinearSVC tuned (Exp 4) | 0.7958 | 0.7953 |\n")
    f.write(f"| {MODEL_NAME} | {acc:.4f} | {f1m:.4f} |\n\n")
    f.write("## Per-class metrics\n\n```\n")
    f.write(classification_report(y_test, y_pred,
                                  target_names=le.classes_, digits=4))
    f.write("\n```\n")

print(f"\nSaved: {RESULTS_DIR}/exp6_transformer.csv")
print(f"Saved: {RESULTS_DIR}/exp6_transformer.md")
print("Done.")