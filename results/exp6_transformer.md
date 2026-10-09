# Experiment 6 — Transformer Baseline (multilingual DistilBERT)

**Model:** `distilbert-base-multilingual-cased`

**Train rows:** 8,000  (device: cpu, epochs: 2)

**Test accuracy:** 0.7831

**Test macro F1:** 0.7825

**Train time:** 7683.8s

## Comparison vs LinearSVC (Exp 4)

| Model | Test acc | Test macro F1 |
|-------|----------|---------------|
| LinearSVC tuned (Exp 4) | 0.7958 | 0.7953 |
| distilbert-base-multilingual-cased | 0.7831 | 0.7825 |

## Per-class metrics

```
                    precision    recall  f1-score   support

          business     0.6828    0.6643    0.6734       852
     entertainment     0.8163    0.7981    0.8071       852
        food_drink     0.8141    0.8791    0.8454       852
   health_wellness     0.6776    0.6800    0.6788       853
         parenting     0.7875    0.8263    0.8064       852
          politics     0.8099    0.8052    0.8075       852
science_technology     0.7362    0.6870    0.7107       853
            sports     0.8471    0.8779    0.8622       852
      style_beauty     0.8554    0.8192    0.8369       852
            travel     0.7983    0.7946    0.7965       852

          accuracy                         0.7831      8522
         macro avg     0.7825    0.7832    0.7825      8522
      weighted avg     0.7825    0.7831    0.7825      8522

```
