from transformers import pipeline
import numpy as np
from typing import Dict, List, Any
import re
from textblob import TextBlob

class FakeNewsClassifier:
    def __init__(self):
        """Initialize multiple classification approaches for robust detection"""
        
        # Primary transformer model
        try:
            self.primary_classifier = pipeline(
                "text-classification",
                model="Pulk17/Fake-News-Detection",
                return_all_scores=True
            )
            print("✅ Primary fake news model loaded successfully")
        except Exception as e:
            print(f"⚠️ Primary model failed to load: {e}")
            self.primary_classifier = None
        
        # Backup/alternative models for ensemble approach
        self.backup_models = []
        
        # Try to load additional models for ensemble
        backup_model_names = [
            "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis
        ]
        
        for model_name in backup_model_names:
            try:
                if "sst" in model_name:
                    model = pipeline("sentiment-analysis", model=model_name)
                else:
                    model = pipeline("text-classification", model=model_name)
                self.backup_models.append({"name": model_name, "pipeline": model})
                print(f"✅ Backup model {model_name} loaded")
            except Exception as e:
                print(f"⚠️ Backup model {model_name} failed: {e}")
        
        # Linguistic features for rule-based detection
        self.linguistic_patterns = {
            'sensational_words': [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'secret', 'exposed',
                'revealed', 'hidden', 'conspiracy', 'coverup', 'explosive', 'bombshell'
            ],
            'absolute_terms': [
                'always', 'never', 'all', 'none', 'every', 'completely', 'totally',
                'absolutely', 'definitely', '100%', 'proven', 'undeniable'
            ],
            'emotional_appeals': [
                'outrageous', 'disgusting', 'terrifying', 'horrifying', 'devastating',
                'miraculous', 'breakthrough', 'revolutionary', 'life-changing'
            ],
            'urgency_markers': [
                'urgent', 'breaking', 'alert', 'emergency', 'immediate', 'now',
                'quickly', 'before it\'s too late', 'act now', 'limited time'
            ]
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Comprehensive prediction using multiple approaches"""
        
        # Clean and preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Get predictions from all available models
        predictions = []
        
        # Primary model prediction
        if self.primary_classifier:
            try:
                primary_result = self.get_primary_prediction(cleaned_text)
                predictions.append({
                    "source": "primary_model",
                    "prediction": primary_result["label"],
                    "confidence": primary_result["confidence"],
                    "weight": 0.4  # Highest weight for primary model
                })
            except Exception as e:
                print(f"Primary model prediction failed: {e}")
        
        # Backup model predictions
        for model_info in self.backup_models:
            try:
                backup_result = self.get_backup_prediction(cleaned_text, model_info)
                if backup_result:
                    predictions.append({
                        "source": model_info["name"],
                        "prediction": backup_result["label"],
                        "confidence": backup_result["confidence"],
                        "weight": 0.2  # Lower weight for backup models
                    })
            except Exception as e:
                print(f"Backup model {model_info['name']} failed: {e}")
        
        # Rule-based linguistic analysis
        linguistic_result = self.analyze_linguistic_features(text)
        predictions.append({
            "source": "linguistic_analysis",
            "prediction": linguistic_result["prediction"],
            "confidence": linguistic_result["confidence"],
            "weight": 0.3  # Medium weight for rule-based approach
        })
        
        # Sentiment and readability analysis
        text_analysis = self.analyze_text_characteristics(text)
        predictions.append({
            "source": "text_analysis",
            "prediction": text_analysis["prediction"],
            "confidence": text_analysis["confidence"],
            "weight": 0.1  # Lower weight for text analysis
        })
        
        # Combine predictions using weighted ensemble
        final_prediction = self.ensemble_predictions(predictions)
        
        return {
            "prediction": final_prediction["label"],
            "confidence": final_prediction["confidence"],
            "individual_predictions": predictions,
            "text_characteristics": text_analysis,
            "linguistic_features": linguistic_result.get("features", {}),
            "ensemble_method": "weighted_voting",
            "models_used": len(predictions)
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better model performance"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Limit length to avoid model issues (most models have token limits)
        if len(text) > 1000:  # Keep first 1000 characters
            text = text[:1000] + "..."
        
        return text

    def get_primary_prediction(self, text: str) -> Dict[str, Any]:
        """Get prediction from primary fake news model"""
        result = self.primary_classifier(text)
        
        # Handle different output formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):  # Nested list format
                scores = result[0]
            else:  # Direct list format
                scores = result
            
            # Find the highest confidence prediction
            best_prediction = max(scores, key=lambda x: x['score'])
            
            # Normalize label names
            label = best_prediction['label'].upper()
            if label in ['FAKE', 'FALSE', '0']:
                normalized_label = 'FAKE'
            elif label in ['REAL', 'TRUE', '1']:
                normalized_label = 'REAL'
            else:
                normalized_label = label
            
            return {
                "label": normalized_label,
                "confidence": float(best_prediction['score']),
                "all_scores": scores
            }
        else:
            # Fallback for unexpected format
            return {
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "all_scores": result
            }

    def get_backup_prediction(self, text: str, model_info: Dict) -> Dict[str, Any]:
        """Get prediction from backup models"""
        pipeline_obj = model_info["pipeline"]
        model_name = model_info["name"]
        
        result = pipeline_obj(text)
        
        # Convert different model outputs to fake/real prediction
        if "sentiment" in model_name.lower():
            # Sentiment models: negative sentiment might correlate with fake news
            if result[0]['label'] == 'NEGATIVE' and result[0]['score'] > 0.8:
                return {"label": "FAKE", "confidence": result[0]['score'] * 0.6}  # Lower confidence
            else:
                return {"label": "REAL", "confidence": result[0]['score'] * 0.4}
        
        elif "detector" in model_name.lower():
            # AI-generated text detectors
            if result[0]['label'] in ['FAKE', 'AI-GENERATED'] and result[0]['score'] > 0.7:
                return {"label": "FAKE", "confidence": result[0]['score']}
            else:
                return {"label": "REAL", "confidence": 1 - result[0]['score']}
        
        else:
            # Generic text classification models
            if isinstance(result, list) and len(result) > 0:
                return {
                    "label": result[0]['label'].upper(),
                    "confidence": result[0]['score']
                }
        
        return None

    def analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features that may indicate misinformation"""
        text_lower = text.lower()
        features = {}
        total_risk_score = 0
        
        # Count pattern matches
        for pattern_type, words in self.linguistic_patterns.items():
            matches = [word for word in words if word in text_lower]
            count = len(matches)
            
            features[pattern_type] = {
                "count": count,
                "matches": matches,
                "risk_contribution": count * 0.1
            }
            total_risk_score += count * 0.1
        
        # Additional linguistic features
        
        # Exclamation marks (excessive use suggests sensationalism)
        exclamation_count = text.count('!')
        features["exclamation_usage"] = {
            "count": exclamation_count,
            "risk_contribution": min(exclamation_count * 0.05, 0.2)
        }
        total_risk_score += min(exclamation_count * 0.05, 0.2)
        
        # All caps usage (suggests shouting/sensationalism)
        caps_words = len([word for word in text.split() if word.isupper() and len(word) > 3])
        features["caps_usage"] = {
            "count": caps_words,
            "risk_contribution": min(caps_words * 0.03, 0.15)
        }
        total_risk_score += min(caps_words * 0.03, 0.15)
        
        # Question overuse (rhetorical questions common in misinformation)
        question_count = text.count('?')
        features["question_usage"] = {
            "count": question_count,
            "risk_contribution": min(question_count * 0.02, 0.1)
        }
        total_risk_score += min(question_count * 0.02, 0.1)
        
        # Normalize risk score
        risk_score = min(1.0, total_risk_score)
        
        # Determine prediction based on risk score
        if risk_score > 0.6:
            prediction = "FAKE"
            confidence = risk_score
        elif risk_score < 0.3:
            prediction = "REAL"
            confidence = 1 - risk_score
        else:
            prediction = "UNCERTAIN"
            confidence = 0.5
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_score": risk_score,
            "features": features
        }

    def analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze general text characteristics"""
        
        # Basic statistics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Sentiment analysis using TextBlob
        try:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity  # -1 to 1
            sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1
        except:
            sentiment_polarity = 0
            sentiment_subjectivity = 0.5
        
        # Reading difficulty (simplified)
        complex_words = len([word for word in text.split() if len(word) > 6])
        readability_score = complex_words / max(word_count, 1)
        
        # URL and link analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_count = len(re.findall(url_pattern, text))
        
        # Determine prediction based on characteristics
        risk_factors = 0
        
        # Extreme sentiment can indicate bias
        if abs(sentiment_polarity) > 0.7:
            risk_factors += 1
        
        # High subjectivity suggests opinion rather than fact
        if sentiment_subjectivity > 0.8:
            risk_factors += 1
        
        # Very short or very long articles can be suspicious
        if word_count < 50 or word_count > 2000:
            risk_factors += 0.5
        
        # Many URLs might indicate spam or link farming
        if urls_count > 3:
            risk_factors += 0.5
        
        # Very simple or very complex language can be suspicious
        if readability_score > 0.3 or readability_score < 0.1:
            risk_factors += 0.5
        
        risk_score = min(1.0, risk_factors / 4)  # Normalize to 0-1
        
        if risk_score > 0.6:
            prediction = "FAKE"
            confidence = risk_score * 0.7  # Lower confidence for text analysis
        else:
            prediction = "REAL"
            confidence = (1 - risk_score) * 0.7
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentiment_polarity": round(sentiment_polarity, 3),
            "sentiment_subjectivity": round(sentiment_subjectivity, 3),
            "readability_score": round(readability_score, 3),
            "urls_count": urls_count,
            "risk_factors": risk_factors
        }

    def ensemble_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Combine predictions using weighted ensemble voting"""
        
        if not predictions:
            return {"label": "UNCERTAIN", "confidence": 0.5}
        
        # Separate by prediction type
        fake_votes = []
        real_votes = []
        uncertain_votes = []
        
        total_weight = sum(pred["weight"] for pred in predictions)
        
        for pred in predictions:
            weighted_confidence = pred["confidence"] * pred["weight"]
            
            if pred["prediction"] in ["FAKE", "FALSE"]:
                fake_votes.append(weighted_confidence)
            elif pred["prediction"] in ["REAL", "TRUE"]:
                real_votes.append(weighted_confidence)
            else:
                uncertain_votes.append(weighted_confidence)
        
        # Calculate weighted scores
        fake_score = sum(fake_votes) / total_weight if fake_votes else 0
        real_score = sum(real_votes) / total_weight if real_votes else 0
        uncertain_score = sum(uncertain_votes) / total_weight if uncertain_votes else 0
        
        # Determine final prediction
        max_score = max(fake_score, real_score, uncertain_score)
        
        if max_score == fake_score and fake_score > 0.5:
            final_label = "FAKE"
            final_confidence = fake_score
        elif max_score == real_score and real_score > 0.5:
            final_label = "REAL"
            final_confidence = real_score
        else:
            final_label = "UNCERTAIN"
            final_confidence = max_score if max_score > 0 else 0.5
        
        return {
            "label": final_label,
            "confidence": round(final_confidence, 3),
            "vote_breakdown": {
                "fake_score": round(fake_score, 3),
                "real_score": round(real_score, 3),
                "uncertain_score": round(uncertain_score, 3)
            }
        }