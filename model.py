import pickle
import numpy as np
from preprocessing import TextPreprocessor

class NewsClassifierModel:
    def __init__(self, model_path='models.pkl'):
        try:
            with open(model_path, 'rb') as f:
                self.data = pickle.load(f)
            
            self.models = self.data['models']
            self.tfidf = self.data['tfidf']
            self.preprocessor = self.data['preprocessor']
            self.results = self.data['results']
            self.categories = self.data['categories']
            self.dataset_info = self.data['dataset_info']
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, text, model_name='svm'):
        processed = self.preprocessor.preprocess(text)
        vectorized = self.tfidf.transform([processed])
        
        model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized)[0]
        else:
            decision_values = model.decision_function(vectorized)[0]
            exp_values = np.exp(decision_values - np.max(decision_values))
            probabilities = exp_values / exp_values.sum()
        
        prediction = model.predict(vectorized)[0]
        
        prob_dict = {
            category: float(prob) 
            for category, prob in zip(self.categories, probabilities)
        }
        
        return prediction, prob_dict
    
    def get_pipeline_steps(self, text):
        return self.preprocessor.get_pipeline_steps(text)
    
    def get_pos_tags(self, text):
        from nltk import pos_tag, word_tokenize
        cleaned = self.preprocessor.clean_text(text)
        tokens = word_tokenize(cleaned)
        return pos_tag(tokens)
    
    def get_ner_tags(self, text):
        from nltk import ne_chunk
        pos_tags = self.get_pos_tags(text)
        ner_tree = ne_chunk(pos_tags)
        entities = []
        for subtree in ner_tree:
            if hasattr(subtree, 'label'):
                entity_text = ' '.join([word for word, tag in subtree.leaves()])
                entities.append((entity_text, subtree.label()))
        return entities
    
    def get_model_metrics(self, model_name):
        if model_name in self.results:
            return self.results[model_name]
        return None
    
    def get_all_metrics(self):
        return self.results
    
    def get_dataset_info(self):
        return self.dataset_info