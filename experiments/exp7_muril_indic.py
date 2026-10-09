"""
EXPERIMENT 7 — MuRIL on native Indian-language news (L3Cube-IndicNews SHC)
Languages: Hindi, Bengali, Tamil
Columns in L3Cube CSVs: 'text' (headline) and 'labels' (category)
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
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)

RESULTS_DIR = "results"
MODEL_NAME = "google/muril-base-cased"
RANDOM_STATE = 42
MAX_LEN = 128
SUBSET = 2000
EPOCHS = 2
BATCH_SIZE = 16
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_GPU = torch.cuda.is_available()
print("=" * 60)
print("EXPERIMENT 7 — MuRIL on Indian-language news (L3Cube SHC)")
print("=" * 60)
print(f"Device: {'cuda' if USE_GPU else 'cpu'}")

LANG_FILE_MAP = {
    "hindi":   "Hindi/SHC/Hindi_SHC_Train.csv",
    "bengali": "Bengali/SHC/Bengal_SHC_Train.csv",
    "tamil":   "Tamil/SHC/Tamil_SHC_Train.csv",
}

print(f"Loading tokenizer: {MODEL_NAME}")
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


results = []

for lang_name, file_path in LANG_FILE_MAP.items():
    print(f"\n--- {lang_name.upper()} ---")

    # Download + load raw CSV
    try:
        csv_path = hf_hub_download(
            repo_id="ayushbagaria17/indic-nlp",
            filename=file_path,
            repo_type="dataset",
        )
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [SKIP] Could not load {lang_name}: {e}")
        continue

    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Raw rows: {len(df):,}")

    # L3Cube-IndicNews columns are literally 'text' and 'labels'
    text_col = "text"
    label_col = "labels"

    if text_col not in df.columns or label_col not in df.columns:
        print(f"  [SKIP] Expected columns 'text' and 'labels' not found in {lang_name}")
        continue

    # Subsample
    if SUBSET and SUBSET < len(df):
        df = df.sample(n=SUBSET, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Using rows: {len(df):,}")

    X = df[text_col].astype(str).tolist()
    y_raw = df[label_col].astype(str).tolist()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_labels = len(le.classes_)
    print(f"  Classes: {num_labels} -> {le.classes_.tolist()}")
    print(f"  Sample text: {X[0][:80]}")

    # Skip if too few classes / rows for stratified split
    if num_labels < 2 or len(df) < 20:
        print(f"  [SKIP] Not enough data or classes for {lang_name}")
        continue

    # Check every class has >=2 members (needed for stratified split)
    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        print(f"  [SKIP] Some classes have <2 members in {lang_name}; "
              f"min count = {counts.min()}")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    train_ds = NewsDataset(X_train, y_train)
    test_ds = NewsDataset(X_test, y_test)

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
        output_dir=f"{RESULTS_DIR}/exp7_{lang_name}_checkpoints",
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
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"  Training MuRIL on {lang_name}...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print(f"  {lang_name}: acc={acc:.4f}  macro F1={f1m:.4f}  ({train_time:.0f}s)")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, digits=4))

    results.append({
        "language":     lang_name,
        "rows":         len(df),
        "classes":      num_labels,
        "accuracy":     acc,
        "f1_macro":     f1m,
        "train_time_s": round(train_time, 1),
    })


if results:
    summary = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    summary.to_csv(f"{RESULTS_DIR}/exp7_muril_indic.csv", index=False)
    with open(f"{RESULTS_DIR}/exp7_muril_indic.md", "w") as f:
        f.write("# Experiment 7 — MuRIL on Indian Languages (L3Cube SHC)\n\n")
        f.write(summary.to_markdown(index=False))

    print(f"\nSaved: {RESULTS_DIR}/exp7_muril_indic.csv")
    print(f"Saved: {RESULTS_DIR}/exp7_muril_indic.md")
else:
    print("\n[WARN] No results — all languages skipped.")

print("Done.")