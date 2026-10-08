# Experiment 4 — Hyperparameter Tuning (LinearSVC)

**Best params:** `{'clf__C': 2.0, 'tfidf__max_features': None, 'tfidf__ngram_range': (1, 2), 'tfidf__sublinear_tf': True}`

**Best CV macro F1:** 0.7872

**Held-out test accuracy:** 0.7958

**Held-out test macro F1:** 0.7953

## Top 10 configs

|   clf__C | tfidf__max_features   | tfidf__ngram_range   | tfidf__sublinear_tf   |   mean_test_f1_macro |   std_test_f1_macro |   rank |
|---------:|:----------------------|:---------------------|:----------------------|---------------------:|--------------------:|-------:|
|      2   | None                  | (1, 2)               | True                  |             0.787197 |          0.00239452 |      1 |
|      1   | None                  | (1, 2)               | True                  |             0.78683  |          0.00215031 |      2 |
|      1   | None                  | (1, 2)               | False                 |             0.786375 |          0.00258191 |      3 |
|      2   | None                  | (1, 2)               | False                 |             0.786251 |          0.00230775 |      4 |
|      0.5 | None                  | (1, 2)               | True                  |             0.784396 |          0.00198644 |      5 |
|      0.5 | None                  | (1, 2)               | False                 |             0.784213 |          0.00240218 |      6 |
|      0.5 | 50000                 | (1, 2)               | True                  |             0.781997 |          0.00290045 |      7 |
|      0.5 | 50000                 | (1, 1)               | True                  |             0.781534 |          0.00414363 |      8 |
|      0.5 | None                  | (1, 1)               | True                  |             0.781534 |          0.00414363 |      8 |
|      0.5 | 50000                 | (1, 2)               | False                 |             0.781326 |          0.00323071 |     10 |