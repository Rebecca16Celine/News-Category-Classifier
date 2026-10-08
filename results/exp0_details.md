# Experiment 0 — Baseline

## Pipeline
- Preprocessing: clean -> tokenize -> remove stopwords -> lemmatize
- Features: TF-IDF (max_features=50000, ngram 1-2, min_df=2, sublinear_tf)
- Model: LinearSVC(C=0.5)

## Overall Metrics
- Accuracy: 0.7951
- Weighted F1: 0.7945
- Macro F1: 0.7945

## Per-Class Metrics

| category           |   precision |   recall |       f1 |   support |
|:-------------------|------------:|---------:|---------:|----------:|
| business           |    0.735369 | 0.678404 | 0.705739 |       852 |
| entertainment      |    0.804816 | 0.745305 | 0.773918 |       852 |
| food_drink         |    0.837079 | 0.874413 | 0.855339 |       852 |
| health_wellness    |    0.69045  | 0.737397 | 0.713152 |       853 |
| parenting          |    0.800456 | 0.823944 | 0.81203  |       852 |
| politics           |    0.790304 | 0.82277  | 0.80621  |       852 |
| science_technology |    0.764557 | 0.708089 | 0.73524  |       853 |
| sports             |    0.849771 | 0.869718 | 0.859629 |       852 |
| style_beauty       |    0.879905 | 0.868545 | 0.874188 |       852 |
| travel             |    0.797497 | 0.82277  | 0.809936 |       852 |

## Top Confusions

- science_technology -> business: 57
- business -> science_technology: 57
- business -> health_wellness: 57
- science_technology -> health_wellness: 55
- business -> politics: 51
- politics -> business: 47
- parenting -> health_wellness: 47
- health_wellness -> parenting: 47
- entertainment -> politics: 47
- health_wellness -> food_drink: 38
