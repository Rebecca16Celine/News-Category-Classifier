import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load trained models and stored results
with open("models.pkl", "rb") as f:
    data = pickle.load(f)

categories = data["categories"]

print("=" * 70)
print("11-CLASS ERROR ANALYSIS")
print("=" * 70)

for model_name, result in data["results"].items():

    print(f"\n{'=' * 70}")
    print(model_name.upper())
    print(f"{'=' * 70}")

    print(f"\nAccuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1-Score:  {result['f1']:.4f}")

    # We need y_test and y_pred, which were not saved.
    # Therefore, use the stored confusion matrix for now.
    cm = np.array(result["confusion_matrix"])

    print("\nConfusion Matrix:")
    print("Rows = Actual | Columns = Predicted")
    print()

    print("Categories:")
    for i, category in enumerate(categories):
        print(f"{i}: {category}")

    print("\nConfusion Matrix:")
    print(cm)

    # Calculate per-class metrics from confusion matrix
    true_positive = np.diag(cm)

    actual = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted != 0
    )

    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive, dtype=float),
        where=actual != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive, dtype=float),
        where=(precision + recall) != 0
    )

    print("\nPer-Class Results:")
    print("-" * 70)
    print(f"{'Category':25} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 70)

    for i, category in enumerate(categories):
        print(
            f"{category:25} "
            f"{precision[i]:>12.3f} "
            f"{recall[i]:>12.3f} "
            f"{f1[i]:>12.3f}"
        )

    # Find strongest confusion pairs
    print("\nTop Confusion Pairs:")
    print("-" * 70)

    confusion_pairs = []

    for i in range(len(categories)):
        for j in range(len(categories)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    (cm[i, j], categories[i], categories[j])
                )

    confusion_pairs.sort(reverse=True)

    for count, actual_category, predicted_category in confusion_pairs[:10]:
        print(
            f"{actual_category} -> {predicted_category}: "
            f"{count} articles"
        )