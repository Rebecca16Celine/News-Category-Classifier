# Experiment 1 — Preprocessing Comparison

## Setup
- Same data, same split, same TF-IDF, same SVM
- Only preprocessing changes

## Results

| option               |   accuracy |   precision_weighted |   recall_weighted |   f1_weighted |   f1_macro |   preprocess_time_s |   train_time_s |
|:---------------------|-----------:|---------------------:|------------------:|--------------:|-----------:|--------------------:|---------------:|
| F_stopwords_stemming |   0.792302 |             0.792178 |          0.792302 |      0.791771 |   0.79179  |                24.5 |            1.5 |
| A_raw                |   0.790307 |             0.79012  |          0.790307 |      0.78959  |   0.789608 |                 0   |            2.3 |
| C_stopwords          |   0.789251 |             0.788841 |          0.789251 |      0.78855  |   0.788568 |                 8.6 |            1.4 |
| D_stemming           |   0.788195 |             0.78817  |          0.788195 |      0.78769  |   0.787709 |                29.2 |            2.2 |
| E_lemmatization      |   0.787609 |             0.787706 |          0.787609 |      0.787101 |   0.787119 |                19.2 |            2.1 |
| B_basic              |   0.784558 |             0.784605 |          0.784558 |      0.784105 |   0.784123 |                 0.7 |            2.6 |

## Winner

**Option F_stopwords_stemming** achieved the highest accuracy: 0.7923
