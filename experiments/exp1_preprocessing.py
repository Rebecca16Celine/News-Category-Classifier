"""
Experiment 1 — Preprocessing Comparison

Tests six preprocessing pipelines on the same data, features, and model.
Everything is fixed except the preprocessing step.

Options:
  A. Raw                       — no processing
  B. Basic                     — clean only
  C. Basic + Stopwords         — clean + remove stopwords
  D. Basic + Stemming          — clean + stem
  E. Basic + Lemmatization     — clean + lemmatize (baseline pipeline)
  F. Basic + Stopwords + Stem  — clean + stopwords + stem (classic NLP)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import TextPreprocessor


# ---------- CONFIG ----------
DATA_PATH      = "data/news_balanced.csv"
RESULTS_DIR    = "results"
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- THE SIX OPTIONS ----------
OPTIONS = [
    ("A_raw",                   "preprocess_raw"),
    ("B_basic",                 "preprocess_basic"),
    ("C_stopwords",             "preprocess_stopwords"),
    ("D_stemming",              "preprocess_stemming"),
    ("E_lemmatization",         "preprocess_lemmatization"),
    ("F_stopwords_stemming",    "preprocess_stopwords_stemming"),
]


def run_one_option(name, method_name, texts, labels, pre, indices):
    """Run one preprocessing option end to end. Returns a metrics dict."""
    print(f"\n  --- Option {name} ---")

    # Preprocess
    t0 = time.time()
    method = getattr(pre, method_name)
    processed = [method(t) for t in texts]
    pre_time = time.time() - t0
    print(f"      Preprocessing: {pre_time:.1f}s")

    # Train/test split using the SAME indices every time
    X_train = [processed[i] for i in indices["train"]]
    X_test  = [processed[i] for i in indices["test"]]
    y_train = [labels[i]   for i in indices["train"]]
    y_test  = [labels[i]   for i in indices["test"]]

    # Vectorize
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)

    # Train
    t0 = time.time()
    model = LinearSVC(C=0.5)
    model.fit(X_train_v, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred = model.predict(X_test_v)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    _, _, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"      Accuracy : {acc:.4f}")
    print(f"      Macro F1 : {f1_m:.4f}")

    return {
        "option": name,
        "accuracy": acc,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "preprocess_time_s": round(pre_time, 1),
        "train_time_s": round(train_time, 1),
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 1 — PREPROCESSING COMPARISON")
    print("=" * 60)

    # ---------- 1. LOAD DATA ----------
    print("\n[1/3] Loading data...")
    df = pd.read_csv(DATA_PATH)
    texts  = df["text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()
    print(f"      Loaded {len(texts):,} rows")

    # ---------- 2. FIXED SPLIT ----------
    # We compute the split indices ONCE and reuse for every option,
    # so all options see exactly the same train/test rows.
    print("\n[2/3] Building the fixed train/test split...")
    all_indices = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    indices = {"train": train_idx, "test": test_idx}
    print(f"      Train: {len(train_idx):,}   Test: {len(test_idx):,}")

    pre = TextPreprocessor()

    # ---------- 3. RUN EVERY OPTION ----------
    print("\n[3/3] Running all six preprocessing options...")
    results = []
    for name, method_name in OPTIONS:
        metrics = run_one_option(
            name, method_name, texts, labels, pre, indices
        )
        results.append(metrics)

    # ---------- SUMMARY TABLE ----------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False)

    print("\n" + "=" * 60)
    print("SUMMARY (sorted by accuracy)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # ---------- SAVE ----------
    out_csv = f"{RESULTS_DIR}/exp1_preprocessing.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ---------- MARKDOWN SUMMARY ----------
    out_md = f"{RESULTS_DIR}/exp1_preprocessing.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Experiment 1 — Preprocessing Comparison\n\n")
        f.write("## Setup\n")
        f.write("- Same data, same split, same TF-IDF, same SVM\n")
        f.write("- Only preprocessing changes\n\n")
        f.write("## Results\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
        except Exception:
            f.write(results_df.to_string(index=False))
        f.write("\n\n## Winner\n\n")
        top = results_df.iloc[0]
        f.write(
            f"**Option {top['option']}** "
            f"achieved the highest accuracy: {top['accuracy']:.4f}\n"
        )
    print(f"Saved: {out_md}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    