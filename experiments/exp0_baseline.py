"""
Experiment 0 — Baseline
Runs the CURRENT pipeline on the balanced dataset and saves results.

Pipeline:
    clean -> tokenize -> remove stopwords -> lemmatize
    + TF-IDF (max_features=50000, ngram 1-2, min_df=2, sublinear_tf)
    + Linear SVM (C=0.5)

Outputs:
    results/exp0_baseline.csv
    results/exp0_confusion.npy
    results/exp0_details.md
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Make imports work when running from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import TextPreprocessor


# ---------- CONFIG ----------
DATA_PATH       = "data/news_balanced.csv"
RESULTS_DIR     = "results"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("EXPERIMENT 0 — BASELINE")
    print("=" * 60)

    # ---------- 1. LOAD DATA ----------
    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    texts  = df["text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()
    print(f"      Loaded {len(texts):,} rows")
    print(f"      Categories: {len(set(labels))}")

    # ---------- 2. PREPROCESS ----------
    print("\n[2/5] Preprocessing (may take a few minutes)...")
    t0 = time.time()
    pre = TextPreprocessor()
    processed = [pre.preprocess(t, use_stemming=False) for t in texts]
    print(f"      Done in {time.time() - t0:.1f}s")

    # ---------- 3. SPLIT ----------
    print("\n[3/5] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"      Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ---------- 4. VECTORIZE + TRAIN ----------
    print("\n[4/5] TF-IDF + Linear SVM...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)
    print(f"      Feature matrix: {X_train_v.shape}")

    t0 = time.time()
    model = LinearSVC(C=0.5)
    model.fit(X_train_v, y_train)
    print(f"      Trained in {time.time() - t0:.1f}s")

    # ---------- 5. EVALUATE ----------
    print("\n[5/5] Evaluating...")
    y_pred = model.predict(X_test_v)

    acc = accuracy_score(y_test, y_pred)

    # weighted metrics (matches old train_model.py)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    # macro metrics (fairer across classes)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    # per-class metrics
    classes = sorted(set(labels))
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # ---------- PRINT ----------
    print("\n" + "-" * 60)
    print("OVERALL METRICS")
    print("-" * 60)
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Weighted Precision : {p_w:.4f}")
    print(f"  Weighted Recall    : {r_w:.4f}")
    print(f"  Weighted F1        : {f1_w:.4f}")
    print(f"  Macro F1           : {f1_m:.4f}")

    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    print(f"{'category':25} {'prec':>7} {'rec':>7} {'f1':>7} {'n':>6}")
    for i, c in enumerate(classes):
        print(
            f"{c:25} "
            f"{p_c[i]:>7.3f} {r_c[i]:>7.3f} {f1_c[i]:>7.3f} {sup_c[i]:>6}"
        )

    print("\n" + "-" * 60)
    print("TOP CONFUSIONS")
    print("-" * 60)
    conf_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                conf_pairs.append((cm[i, j], classes[i], classes[j]))
    conf_pairs.sort(reverse=True)
    for n, actual, predicted in conf_pairs[:10]:
        print(f"  {actual:25} -> {predicted:25} : {n}")

    # ---------- SAVE ----------
    metrics_df = pd.DataFrame({
        "category": classes,
        "precision": p_c,
        "recall": r_c,
        "f1": f1_c,
        "support": sup_c,
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/exp0_baseline.csv", index=False)
    np.save(f"{RESULTS_DIR}/exp0_confusion.npy", cm)

    # details.md
    with open(f"{RESULTS_DIR}/exp0_details.md", "w", encoding="utf-8") as f:
        f.write("# Experiment 0 — Baseline\n\n")
        f.write("## Pipeline\n")
        f.write("- Preprocessing: clean -> tokenize -> remove stopwords -> lemmatize\n")
        f.write("- Features: TF-IDF (max_features=50000, ngram 1-2, min_df=2, sublinear_tf)\n")
        f.write("- Model: LinearSVC(C=0.5)\n\n")
        f.write("## Overall Metrics\n")
        f.write(f"- Accuracy: {acc:.4f}\n")
        f.write(f"- Weighted F1: {f1_w:.4f}\n")
        f.write(f"- Macro F1: {f1_m:.4f}\n\n")
        f.write("## Per-Class Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n## Top Confusions\n\n")
        for n, actual, predicted in conf_pairs[:10]:
            f.write(f"- {actual} -> {predicted}: {n}\n")

    print(f"\nSaved:")
    print(f"  {RESULTS_DIR}/exp0_baseline.csv")
    print(f"  {RESULTS_DIR}/exp0_confusion.npy")
    print(f"  {RESULTS_DIR}/exp0_details.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
    