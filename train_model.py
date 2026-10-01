import pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from preprocessing import TextPreprocessor

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)


def load_dataset():
    """Load the balanced News Category Dataset."""

    csv_path = 'data/news_balanced.csv'

    print(f"Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path)

    # Make sure required columns exist
    if 'text' not in df.columns or 'category' not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'category' columns."
        )

    texts = df['text'].astype(str)
    labels = df['category'].astype(str)

    print(f"Loaded {len(df):,} articles.")

    return texts, labels


def main():
    print("=" * 60)
    print("News Category Classifier - Training")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading dataset...")
    texts, labels = load_dataset()

    print("\nDataset loaded:")
    print(f"  - Total samples: {len(texts):,}")
    print(f"  - Categories: {sorted(set(labels))}")

    print("\nCategory distribution:")
    print(pd.Series(labels).value_counts().sort_index())

    # Preprocess data
    print("\n2. Preprocessing text data...")
    preprocessor = TextPreprocessor()

    processed_texts = [
        preprocessor.preprocess(text, use_stemming=False)
        for text in texts
    ]

    print("  - Preprocessing complete")

    # Split data
    print("\n3. Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"  - Training set: {len(X_train):,} samples")
    print(f"  - Test set: {len(X_test):,} samples")

    # Create TF-IDF features
    print("\n4. Creating TF-IDF features...")

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"  - Feature matrix shape: {X_train_tfidf.shape}")

    # Train models
    print("\n5. Training models...")

    models = {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000),
        'svm': LinearSVC(C=0.5)
    }

    results = {}

    for name, model in models.items():

        print(f"\n  Training {name}...")

        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average='weighted'
        )

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred
        }

        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")

    # Save everything
    print("\n6. Saving models...")

    with open('models.pkl', 'wb') as f:

        pickle.dump({

            'models': {
                name: results[name]['model']
                for name in results
            },

            'tfidf': tfidf,

            'preprocessor': preprocessor,

            'results': {
                name: {
                    k: v
                    for k, v in results[name].items()
                    if k in [
                        'accuracy',
                        'precision',
                        'recall',
                        'f1',
                        'confusion_matrix'
                    ]
                }
                for name in results
            },

            'categories': sorted(set(labels)),

            'dataset_info': {
                'total_samples': len(texts),

                'categories': pd.Series(
                    labels
                ).value_counts().to_dict(),

                'avg_length': int(
                    np.mean([len(t) for t in texts])
                ),

                'processed_data': pd.DataFrame({
                    'text': texts,
                    'category': labels,
                    'processed_text': processed_texts
                }).to_dict('records')
            }

        }, f)

    print("\n" + "=" * 60)
    print("Training complete! Models saved to models.pkl")
    print("=" * 60)

    print("\nFinal Model Performance Summary:")
    print("-" * 40)

    for name, metrics in results.items():

        print(f"\n{name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1']:.3f}")


if __name__ == '__main__':
    main()