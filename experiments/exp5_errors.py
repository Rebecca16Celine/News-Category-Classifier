"""
EXPERIMENT 5 — ERROR ANALYSIS
Uses the tuned LinearSVC from Exp 4 (or defaults to LinearSVC(C=1.0) if not tuned).
Produces: confusion matrix, per-class metrics, top misclassified examples.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

DATA_PATH    = "data/news_balanced.csv"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 5 — ERROR ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[1/4] Train: {len(X_train):,}   Test: {len(X_test):,}")

# Best config from Exp 4 (fill in if you tuned)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                              sublinear_tf=True)),
    ("clf",   LinearSVC(C=1.0, random_state=RANDOM_STATE)),
])
print("[2/4] Fitting pipeline ...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
print(f"[3/4] Test accuracy = {acc:.4f}   macro F1 = {f1m:.4f}")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(f"{RESULTS_DIR}/exp5_confusion.csv")

report_str = classification_report(y_test, y_pred, digits=4)
print("\n" + report_str)

# Top misclassified examples
err_mask = (y_test.values != y_pred)
err_df = pd.DataFrame({
    "text":  X_test.values[err_mask],
    "true":  y_test.values[err_mask],
    "pred":  y_pred[err_mask],
})
# Short preview column
err_df["preview"] = err_df["text"].str.slice(0, 120)
err_df = err_df[["preview", "true", "pred", "text"]]
err_df.head(100).to_csv(f"{RESULTS_DIR}/exp5_misclassified.csv", index=False)

print(f"[4/4] Total errors: {err_mask.sum():,} of {len(y_test):,} "
      f"({100 * err_mask.mean():.2f}%)")
print("\nTop confusion pairs (true -> pred):")
pair_counts = (
    pd.DataFrame({"true": y_test, "pred": y_pred})[err_mask]
    .value_counts()
    .head(10)
)
print(pair_counts.to_string())

with open(f"{RESULTS_DIR}/exp5_errors.md", "w") as f:
    f.write("# Experiment 5 — Error Analysis\n\n")
    f.write(f"**Test accuracy:** {acc:.4f}\n\n")
    f.write(f"**Test macro F1:** {f1m:.4f}\n\n")
    f.write(f"**Total errors:** {err_mask.sum():,} / {len(y_test):,} "
            f"({100 * err_mask.mean():.2f}%)\n\n")
    f.write("## Per-class metrics\n\n```\n")
    f.write(report_str)
    f.write("\n```\n\n")
    f.write("## Top confusion pairs\n\n")
    f.write(pair_counts.to_frame("count").to_markdown())
    f.write("\n\n## Confusion matrix (rows=true, cols=pred)\n\n")
    f.write(cm_df.to_markdown())

print(f"\nSaved: {RESULTS_DIR}/exp5_errors.md")
print(f"Saved: {RESULTS_DIR}/exp5_confusion.csv")
print(f"Saved: {RESULTS_DIR}/exp5_misclassified.csv")
print("Done.")