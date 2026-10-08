# News Category Classifier

A machine learning-based NLP project that classifies news articles into 10 categories using TF-IDF feature extraction and machine learning algorithms.

## Categories

- Business
- Entertainment
- Food & Drink
- Health & Wellness
- Parenting
- Politics
- Science & Technology
- Sports
- Style & Beauty
- Travel

### Dataset Download

The dataset can be downloaded from Kaggle:

**News Category Dataset v3:**  
https://www.kaggle.com/datasets/rmisra/news-category-dataset

After downloading, place the `News_Category_Dataset_v3.json` file inside the `data/` folder:

```text
News-Category-Classifier/
└── data/
    └── News_Category_Dataset_v3.json
```
then run: python prepare_data.py

For the final experiment:

- 42,610 news articles
- 10 categories
- 4,261 articles per category
- Headline and short description combined
- Duplicate articles removed
- 80:20 stratified train-test split

The dataset itself is not included in this repository because of its size.

## Methodology

```text
News Dataset
     ↓
Data Cleaning
     ↓
Category Selection & Balancing
     ↓
Text Preprocessing
     ↓
80:20 Stratified Split
     ↓
TF-IDF Feature Extraction
     ↓
Machine Learning Models
     ↓
Performance Evaluation
     ↓
NiceGUI Web Application
```
Models
The project compares three machine learning models:
- Naive Bayes
- Logistic Regression
- Linear SVM
TF-IDF Configuration
- Maximum features: 50,000
- N-grams: Unigrams and Bigrams
- Minimum document frequency: 2
- Sublinear TF: Enabled

| Model | Accuracy |
|---|---:|
| Naive Bayes | 77.20% |
| Logistic Regression | 78.06% |
| Linear SVM | **79.36%** |

Linear SVM achieved the highest accuracy and was selected for the final application.
Application
The project includes a NiceGUI web application that provides:
- News category prediction
- Confidence distribution
- Model evaluation
- NLP pipeline visualization
- Dataset analytics

```
News-Category-Classifier/
├── app.py
├── config.py
├── model.py
├── preprocessing.py
├── prepare_data.py
├── train_model.py
├── analyze_results.py
├── models.pkl
├── requirements.txt
├── style.css
└── images/
```

Technologies Used
- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Naive Bayes
- Logistic Regression
- Linear SVM
- NiceGUI
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Experiments

| Exp | What | Best result | Commit |
|-----|------|-------------|--------|
| 0 | Baseline | 79.5% acc, macro F1 0.7945 | 5b593e6 |
| 1 | Preprocessing | F (stopwords+stemming) 79.2% acc | 2cf0ff2 |
| 2 | Models (single split) | LinearSVC 78.4% acc, macro F1 0.783 | <hash> |
| 3 | 5-fold CV | LinearSVC 0.7834 ± 0.0032, tied with SGD | <hash> |