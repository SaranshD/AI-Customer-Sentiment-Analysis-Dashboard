import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union


class SentimentClassifier:
    def __init__(self, model_name: str, device: int = None):
        self.model_name = model_name
        
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {'cuda' if self.device >= 0 else 'cpu'}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if self.device >= 0:
            self.model = self.model.to(self.device)
        
        self.labels = self.model.config.id2label
    
    def predict(self, texts: Union[str, List[str]], return_probs: bool = False) -> Union[Dict, List[Dict]]:
        if isinstance(texts, str):
            texts = [texts]
        
        all_probs = self._get_all_probabilities(texts)
        
        processed_results = []
        for probs in all_probs:
            max_label = max(probs, key=probs.get)
            max_score = probs[max_label]
            
            if 'positive' in max_label.lower():
                sentiment = 'Positive'
            elif 'negative' in max_label.lower():
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            processed = {
                'sentiment': sentiment,
                'confidence': max_score,
                'original_label': max_label
            }
            
            if return_probs:
                normalized_probs = {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 0.0}
                for label, score in probs.items():
                    if 'positive' in label.lower():
                        normalized_probs['Positive'] = score
                    elif 'negative' in label.lower():
                        normalized_probs['Negative'] = score
                    else:
                        normalized_probs['Neutral'] = score
                processed['probabilities'] = normalized_probs
            
            processed_results.append(processed)
        
        if len(processed_results) == 1:
            return processed_results[0]
        return processed_results
    
    def _get_all_probabilities(self, texts: List[str]) -> List[Dict[str, float]]:
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            if self.device >= 0:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            label_probs = {}
            for idx, prob in enumerate(probs):
                label = self.model.config.id2label[idx]
                label_probs[label] = prob.item()
            
            results.append(label_probs)
        
        return results
