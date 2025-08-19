from transformers import pipeline

class FakeNewsClassifier:
    def __init__(self):
        # Remove return_all_scores to simplify
        self.classifier = pipeline("text-classification", 
                                 model="Pulk17/Fake-News-Detection")
    
    def predict(self, text):
        result = self.classifier(text)
        print("DEBUG - Simplified result:", result)
        
        return {
            "prediction": result[0]['label'],
            "confidence": result[0]['score']
        }
