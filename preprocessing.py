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
    
    def clean_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, use_stemming=True):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        else:
            tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)
    
    def get_pipeline_steps(self, text):
        """Returns all intermediate steps for visualization"""
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