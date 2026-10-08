"""
EXPERIMENT 4 — HYPERPARAMETER TUNING (LinearSVC + TF-IDF)
Uses 5-fold CV on the training split; evaluated on the held-out test set.
"""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH    = "data/news_balanced.csv"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EXPERIMENT 4 — HYPERPARAMETER TUNING (LinearSVC)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X = df["text"].astype(str)
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[1/3] Train: {len(X_train):,}   Test: {len(X_test):,}")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf",   LinearSVC(random_state=RANDOM_STATE)),
])

param_grid = {
    "tfidf__max_features": [20000, 50000, None],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [True, False],
    "clf__C":              [0.1, 0.5, 1.0, 2.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    pipe, param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

print("[2/3] Running GridSearchCV ... (this may take several minutes)")
t0 = time.time()
grid.fit(X_train, y_train)
print(f"      Done in {time.time() - t0:.1f}s")

print("[3/3] Best params:")
for k, v in grid.best_params_.items():
    print(f"      {k} = {v}")
print(f"      best CV macro F1 = {grid.best_score_:.4f}")

y_pred = grid.best_estimator_.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred, average="macro")
print(f"\n      Test accuracy = {test_acc:.4f}")
print(f"      Test macro F1 = {test_f1:.4f}")

rows = []
for i, params in enumerate(grid.cv_results_["params"]):
    rows.append({
        **{k: str(v) for k, v in params.items()},
        "mean_test_f1_macro": grid.cv_results_["mean_test_score"][i],
        "std_test_f1_macro":  grid.cv_results_["std_test_score"][i],
        "rank":               grid.cv_results_["rank_test_score"][i],
    })
results = pd.DataFrame(rows).sort_values("rank")
results.to_csv(f"{RESULTS_DIR}/exp4_tuning.csv", index=False)

with open(f"{RESULTS_DIR}/exp4_tuning.md", "w") as f:
    f.write("# Experiment 4 — Hyperparameter Tuning (LinearSVC)\n\n")
    f.write(f"**Best params:** `{grid.best_params_}`\n\n")
    f.write(f"**Best CV macro F1:** {grid.best_score_:.4f}\n\n")
    f.write(f"**Held-out test accuracy:** {test_acc:.4f}\n\n")
    f.write(f"**Held-out test macro F1:** {test_f1:.4f}\n\n")
    f.write("## Top 10 configs\n\n")
    f.write(results.head(10).to_markdown(index=False))

print(f"\nSaved: {RESULTS_DIR}/exp4_tuning.csv")
print(f"Saved: {RESULTS_DIR}/exp4_tuning.md")
print("Done.")