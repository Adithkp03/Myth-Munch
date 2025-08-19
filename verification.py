import requests
import os

class FactChecker:
    def __init__(self):
        # Your real Google API key
        self.google_api_key = "AIzaSyA7VfLkk074kWuWauFrQKWPLG1x7s1O-EA"  # Replace with your actual key
        self.google_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def verify_claim(self, query):
        """Real-world claim verification using Google Fact Check API"""
        try:
            # Real Google Fact Check API
            google_result = self.check_google_factcheck(query)
            
            # Keep the simple keyword check as backup
            keyword_result = self.simple_keyword_check(query)
            
            return {
                "claim": query,
                "google_factcheck": google_result,
                "keyword_analysis": keyword_result,
                "overall_verdict": self.combine_verdicts(google_result, keyword_result)
            }
        except Exception as e:
            return {"error": str(e), "claim": query}
    
    def check_google_factcheck(self, query):
        """Real Google Fact Check API integration"""
        try:
            params = {
                'query': query,
                'key': self.google_api_key,
                'languageCode': 'en'
            }
            
            print(f"Querying Google Fact Check API for: {query}")
            response = requests.get(self.google_url, params=params, timeout=10)
            
            print(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                claims = data.get('claims', [])
                
                print(f"Found {len(claims)} fact-check results")
                
                if claims:
                    # Process real fact-check results
                    fact_checks = []
                    ratings = []
                    
                    for claim in claims[:3]:  # Top 3 results
                        for review in claim.get('claimReview', []):
                            fact_check = {
                                "publisher": review.get('publisher', {}).get('name', 'Unknown'),
                                "title": review.get('title', ''),
                                "rating": review.get('textualRating', ''),
                                "url": review.get('url', ''),
                                "date": review.get('reviewDate', '')
                            }
                            fact_checks.append(fact_check)
                            ratings.append(review.get('textualRating', '').lower())
                    
                    verdict = self.analyze_ratings(ratings)
                    
                    return {
                        "method": "real_google_api",
                        "found_fact_checks": len(fact_checks),
                        "fact_checks": fact_checks,
                        "verdict": verdict,
                        "confidence": 0.9 if len(fact_checks) > 0 else 0.3
                    }
                else:
                    return {
                        "method": "real_google_api",
                        "found_fact_checks": 0,
                        "verdict": "no_fact_checks_found",
                        "confidence": 0.1
                    }
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return {
                    "method": "real_google_api",
                    "error": f"API returned {response.status_code}",
                    "verdict": "api_error",
                    "confidence": 0.0
                }
                
        except Exception as e:
            print(f"Google API Exception: {str(e)}")
            return {
                "method": "real_google_api",
                "error": str(e),
                "verdict": "api_error",
                "confidence": 0.0
            }
    
    def analyze_ratings(self, ratings):
        """Analyze fact-check ratings to determine verdict"""
        false_keywords = ['false', 'fake', 'incorrect', 'misleading', 'pants on fire']
        true_keywords = ['true', 'correct', 'accurate', 'verified']
        mixed_keywords = ['mixed', 'mostly', 'partly', 'half']
        
        false_count = sum(1 for rating in ratings if any(kw in rating for kw in false_keywords))
        true_count = sum(1 for rating in ratings if any(kw in rating for kw in true_keywords))
        mixed_count = sum(1 for rating in ratings if any(kw in rating for kw in mixed_keywords))
        
        if false_count > true_count:
            return "disputed_false"
        elif true_count > false_count:
            return "verified_true"
        elif mixed_count > 0:
            return "mixed_evidence"
        else:
            return "disputed"
    
    def simple_keyword_check(self, query):
        """Keep the simple keyword check as backup"""
        false_patterns = [
            "vaccines cause autism",
            "climate change is a hoax", 
            "global warming is fake",
            "covid vaccine dangerous",
            "5g causes coronavirus"
        ]
        
        suspicious_words = ["hoax", "fake", "conspiracy", "secret", "hidden", "cover-up", "fabricated"]
        
        exact_match = any(pattern.lower() in query.lower() for pattern in false_patterns)
        suspicious_count = sum(1 for word in suspicious_words if word.lower() in query.lower())
        
        return {
            "method": "keyword_pattern_matching",
            "exact_false_pattern": exact_match,
            "suspicious_word_count": suspicious_count,
            "risk_score": min(1.0, (suspicious_count * 0.2) + (0.6 if exact_match else 0))
        }
    
    def combine_verdicts(self, google_result, keyword_result):
        """Combine Google API results with keyword analysis"""
        google_verdict = google_result.get("verdict", "unknown")
        google_confidence = google_result.get("confidence", 0.0)
        keyword_risk = keyword_result.get("risk_score", 0.0)
        
        # Google API takes priority if it found results
        if google_result.get("found_fact_checks", 0) > 0:
            if google_verdict == "disputed_false":
                return {"verdict": "fact_checked_false", "confidence": google_confidence}
            elif google_verdict == "verified_true":
                return {"verdict": "fact_checked_true", "confidence": google_confidence}
            elif google_verdict == "mixed_evidence":
                return {"verdict": "mixed_evidence", "confidence": 0.6}
            else:
                return {"verdict": "disputed", "confidence": 0.7}
        
        # Fall back to keyword analysis
        elif keyword_risk > 0.4:
            return {"verdict": "likely_false", "confidence": 0.6}
        else:
            return {"verdict": "no_red_flags", "confidence": 0.3}
