import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    # ---------- BUILDING BLOCKS ----------

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ---------- ORIGINAL METHOD (kept intact) ----------

    def preprocess(self, text, use_stemming=True):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        else:
            tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)

    # ---------- NEW METHODS FOR EXPERIMENT 1 ----------

    def preprocess_raw(self, text):
        """Option A: no processing at all."""
        return text

    def preprocess_basic(self, text):
        """Option B: clean only."""
        return self.clean_text(text)

    def preprocess_stopwords(self, text):
        """Option C: clean + remove stopwords."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)

    def preprocess_stemming(self, text):
        """Option D: clean + stem."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.stem_tokens(tokens)
        return ' '.join(tokens)

    def preprocess_lemmatization(self, text):
        """Option E: clean + lemmatize."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)

    def preprocess_stopwords_stemming(self, text):
        """Option F: clean + stopwords + stem."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        return ' '.join(tokens)

    # ---------- PIPELINE VISUALIZATION (unchanged) ----------

    def get_pipeline_steps(self, text):
        steps = {}
        steps['raw'] = text
        steps['cleaned'] = self.clean_text(text)
        steps['tokenized'] = str(self.tokenize(steps['cleaned']))
        tokens = self.tokenize(steps['cleaned'])
        steps['no_stopwords'] = str(self.remove_stopwords(tokens))
        tokens_no_stop = self.remove_stopwords(tokens)
        steps['stemmed'] = str(self.stem_tokens(tokens_no_stop))
        steps['lemmatized'] = str(self.lemmatize_tokens(tokens_no_stop))
        return steps