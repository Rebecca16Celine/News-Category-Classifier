"""
EXPERIMENT 3 — 5-FOLD CROSS-VALIDATION
Fixed preprocessing = raw + TF-IDF. Same models as Exp 2.
Reports mean +/- std across folds so we know if differences are real.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

DATA_PATH    = "data/news_balanced.csv"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
N_SPLITS     = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 3 — 5-FOLD CROSS-VALIDATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = df["category"]
print(f"      Loaded {len(df):,} rows")

models = {
    "LogReg":        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LinearSVC":     LinearSVC(random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "SGD":           SGDClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rows = []
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf",   clf),
    ])
    t0 = time.time()
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    f1  = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    dt  = time.time() - t0
    rows.append({
        "model":         name,
        "acc_mean":      acc.mean(),
        "acc_std":       acc.std(),
        "f1_macro_mean": f1.mean(),
        "f1_macro_std":  f1.std(),
        "cv_time_s":     round(dt, 1),
    })
    print(f"  --- {name:14s} acc={acc.mean():.4f} +/- {acc.std():.4f}   "
          f"macroF1={f1.mean():.4f} +/- {f1.std():.4f}   ({dt:.1f}s)")

results = pd.DataFrame(rows).sort_values("acc_mean", ascending=False)
print("\n" + "=" * 60)
print("SUMMARY (sorted by mean accuracy)")
print("=" * 60)
print(results.to_string(index=False))

results.to_csv(f"{RESULTS_DIR}/exp3_cv.csv", index=False)
with open(f"{RESULTS_DIR}/exp3_cv.md", "w") as f:
    f.write("# Experiment 3 — Cross-Validation\n\n")
    f.write(results.to_markdown(index=False))

print(f"\nSaved: {RESULTS_DIR}/exp3_cv.csv")
print(f"Saved: {RESULTS_DIR}/exp3_cv.md")
print("Done.")