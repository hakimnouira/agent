import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SourceScorerAgent:
    def __init__(self):
        # Path to your finetuned DeBERTa model
        model_path = "models/deberta_reputation_model_export"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Source scoring model loaded on: {self.device}")
    
    def score_source(self, source_type, source_name):
        """
        Score a source using your trained DeBERTa model.
        
        Args:
            source_type: "Web", "News Media", etc. (can be ignored if not used in training)
            source_name: "cnn.com", "bbc.com", etc.
        
        Returns:
            float: credibility score 1-5
        """
        # Format input based on your training data format
        # If you trained with just the domain:
        text = source_name
        
        # If you trained with "source_type | source_name" format:
        # text = f"{source_type} | {source_name}"
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy().flatten()
            
            # If your model outputs a single regression value:
            score = float(logits[0])
            
            # If your model outputs class probabilities (5 classes for scores 1-5):
            # predicted_class = logits.argmax()
            # score = float(predicted_class + 1)  # Convert 0-4 to 1-5
        
        # Clamp score between 1-5
        return max(1.0, min(5.0, score))
