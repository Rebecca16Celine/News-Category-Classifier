"""
EXPERIMENT 2 — MODEL COMPARISON
Fixed: preprocessing = raw text + TF-IDF
Same split as Exp 0 / Exp 1 (random_state=42, test_size=0.2, stratify=y)
"""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

DATA_PATH    = "data/news_balanced.csv"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 2 — MODEL COMPARISON")
print("=" * 60)

# [1/3] Load + split (identical to Exp 0 / Exp 1)
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df):,} rows")

# Adjust these two column names if Exp 1 uses different ones
X = df["text"].astype(str)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[1/3] Train: {len(X_train):,}   Test: {len(X_test):,}")

# [2/3] Define models
models = {
    "LogReg":        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LinearSVC":     LinearSVC(random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "SGD":           SGDClassifier(random_state=RANDOM_STATE),
}

# [3/3] Train + evaluate
print("[2/3] Training models...")
rows = []
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf",   clf),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = pipe.predict(X_test)
    row = {
        "model":               name,
        "accuracy":            accuracy_score(y_test, y_pred),
        "precision_weighted":  precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted":     recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted":         f1_score(y_test, y_pred, average="weighted"),
        "f1_macro":            f1_score(y_test, y_pred, average="macro"),
        "train_time_s":        round(train_time, 2),
    }
    rows.append(row)
    print(f"  --- {name:14s} acc={row['accuracy']:.4f}  "
          f"macroF1={row['f1_macro']:.4f}  ({train_time:.1f}s)")

results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

print("\n" + "=" * 60)
print("SUMMARY (sorted by accuracy)")
print("=" * 60)
print(results.to_string(index=False))

results.to_csv(f"{RESULTS_DIR}/exp2_models.csv", index=False)
with open(f"{RESULTS_DIR}/exp2_models.md", "w") as f:
    f.write("# Experiment 2 — Model Comparison\n\n")
    f.write(results.to_markdown(index=False))

print(f"\nSaved: {RESULTS_DIR}/exp2_models.csv")
print(f"Saved: {RESULTS_DIR}/exp2_models.md")
print("Done.")