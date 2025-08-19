import requests
import re
from datetime import datetime
from typing import List, Dict, Any
import time

class FactChecker:
    def __init__(self):
        # Your real Google API key
        self.google_api_key = "AIzaSyA7VfLkk074kWuWauFrQKWPLG1x7s1O-EA" 
        self.google_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
        # General suspicious language patterns (not topic-specific)
        self.suspicious_patterns = {
            'absolute_claims': [
                r'\b(never|always|all|none|every|no one)\b.*\b(scientist|doctor|expert|government)\b',
                r'\b(100%|completely|totally|absolutely|definitely)\b.*\b(proven|false|true)\b'
            ],
            'emotional_language': [
                r'\b(shocking|exposed|revealed|hidden|secret|coverup|conspiracy)\b',
                r'\b(dangerous|deadly|toxic|poison|killer)\b',
                r'\b(miracle|amazing|incredible|unbelievable)\b'
            ],
            'authority_rejection': [
                r'\b(mainstream media|big pharma|government|they)\b.*\b(hiding|lying|covering)\b',
                r'\b(don\'t want you to know|they won\'t tell you|banned|censored)\b'
            ],
            'false_urgency': [
                r'\b(urgent|breaking|alert|warning|emergency)\b.*\b(share|spread|tell everyone)\b',
                r'\b(before it\'s too late|act now|time is running out)\b'
            ]
        }

    def verify_claim(self, query: str) -> Dict[str, Any]:
        """Comprehensive claim verification using multiple sources"""
        try:
            print(f"🔍 Starting comprehensive verification for: {query[:100]}...")
            
            # Step 1: Google Fact Check API (primary source)
            google_result = self.check_google_factcheck(query)
            
            # Step 2: Language pattern analysis (general, not topic-specific)
            language_analysis = self.analyze_language_patterns(query)
            
            # Step 3: Source credibility check (if URLs mentioned)
            source_analysis = self.analyze_mentioned_sources(query)
            
            # Step 4: Web evidence gathering (basic web search)
            web_evidence = self.gather_web_evidence(query)
            
            # Step 5: Combine all evidence
            overall_verdict = self.combine_all_evidence(
                google_result, language_analysis, source_analysis, web_evidence
            )
            
            return {
                "claim": query,
                "google_factcheck": google_result,
                "language_analysis": language_analysis,
                "source_analysis": source_analysis,
                "web_evidence": web_evidence,
                "overall_verdict": overall_verdict,
                "verification_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Verification failed: {str(e)}")
            return {"error": str(e), "claim": query}

    def check_google_factcheck(self, query: str) -> Dict[str, Any]:
        """Enhanced Google Fact Check API with better query processing"""
        try:
            # Extract key phrases for better API results
            key_phrases = self.extract_key_phrases(query)
            
            all_results = []
            
            # Search with main query
            main_result = self._search_factcheck_api(query)
            if main_result:
                all_results.extend(main_result.get('fact_checks', []))
            
            # Search with key phrases for better coverage
            for phrase in key_phrases[:2]:  # Limit to avoid rate limits
                phrase_result = self._search_factcheck_api(phrase)
                if phrase_result:
                    all_results.extend(phrase_result.get('fact_checks', []))
                time.sleep(0.5)  # Rate limiting
            
            # Remove duplicates and process results
            unique_results = self._remove_duplicate_factchecks(all_results)
            
            if unique_results:
                verdict = self.analyze_factcheck_ratings([fc.get('rating', '') for fc in unique_results])
                confidence = min(0.95, 0.7 + (len(unique_results) * 0.05))  # Higher confidence with more sources
                
                return {
                    "method": "google_factcheck_enhanced",
                    "found_fact_checks": len(unique_results),
                    "fact_checks": unique_results[:5],  # Top 5 results
                    "verdict": verdict,
                    "confidence": confidence,
                    "search_queries_used": [query] + key_phrases[:2]
                }
            else:
                return {
                    "method": "google_factcheck_enhanced",
                    "found_fact_checks": 0,
                    "verdict": "no_fact_checks_found",
                    "confidence": 0.1
                }
                
        except Exception as e:
            print(f"❌ Google API error: {str(e)}")
            return {
                "method": "google_factcheck_enhanced",
                "error": str(e),
                "verdict": "api_error",
                "confidence": 0.0
            }

    def _search_factcheck_api(self, query: str) -> Dict[str, Any]:
        """Search Google Fact Check API for a specific query"""
        params = {
            'query': query,
            'key': self.google_api_key,
            'languageCode': 'en'
        }
        
        response = requests.get(self.google_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get('claims', [])
            
            fact_checks = []
            for claim in claims:
                for review in claim.get('claimReview', []):
                    fact_check = {
                        "publisher": review.get('publisher', {}).get('name', 'Unknown'),
                        "title": review.get('title', ''),
                        "rating": review.get('textualRating', ''),
                        "url": review.get('url', ''),
                        "date": review.get('reviewDate', ''),
                        "claim_text": claim.get('text', '')
                    }
                    fact_checks.append(fact_check)
            
            return {"fact_checks": fact_checks}
        else:
            print(f"API returned {response.status_code}: {response.text}")
            return None

    def extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from claim for better search results"""
        # Remove common prefixes
        cleaned = re.sub(r'^(is it true that|did you know that|i heard that|apparently)\s+', '', query.lower())
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', cleaned)
        
        # Extract potential entity names (capitalized words/phrases)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract key noun phrases (simple approach)
        words = cleaned.split()
        key_phrases = []
        
        # Get longer phrases first
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if any(word in phrase for word in ['vaccine', 'climate', 'election', 'covid', 'virus', 'treatment', 'cure', 'study']):
                key_phrases.append(phrase)
        
        return quoted_phrases + entities + key_phrases[:3]

    def _remove_duplicate_factchecks(self, fact_checks: List[Dict]) -> List[Dict]:
        """Remove duplicate fact-checks based on URL and title similarity"""
        seen_urls = set()
        unique_checks = []
        
        for fc in fact_checks:
            url = fc.get('url', '')
            title = fc.get('title', '').lower()
            
            # Skip if we've seen this URL
            if url in seen_urls:
                continue
                
            # Skip if very similar title exists
            is_duplicate = False
            for existing in unique_checks:
                existing_title = existing.get('title', '').lower()
                if self._calculate_similarity(title, existing_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_checks.append(fc)
                seen_urls.add(url)
        
        return unique_checks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def analyze_language_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze language patterns that suggest misinformation"""
        query_lower = query.lower()
        
        pattern_scores = {}
        total_suspicious_score = 0
        
        for category, patterns in self.suspicious_patterns.items():
            category_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    category_matches.extend(matches)
            
            if category_matches:
                pattern_scores[category] = {
                    'matches': len(category_matches),
                    'examples': category_matches[:3]
                }
                total_suspicious_score += len(category_matches)
        
        # Additional checks
        exclamation_count = query.count('!')
        caps_ratio = sum(1 for c in query if c.isupper()) / max(len(query), 1)
        
        risk_score = min(1.0, (total_suspicious_score * 0.15) + (exclamation_count * 0.1) + (caps_ratio * 0.2))
        
        return {
            "method": "language_pattern_analysis",
            "pattern_matches": pattern_scores,
            "exclamation_marks": exclamation_count,
            "caps_ratio": round(caps_ratio, 3),
            "risk_score": round(risk_score, 3),
            "risk_level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
        }

    def analyze_mentioned_sources(self, query: str) -> Dict[str, Any]:
        """Analyze any sources or URLs mentioned in the claim"""
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, query)
        
        # Extract potential source mentions
        source_patterns = [
            r'\b(study|research|report|article|news|paper)\s+(?:from|by|in|at)\s+([A-Za-z\s]{2,30})',
            r'\b(according to|says|reports|claims)\s+([A-Za-z\s]{2,30})',
            r'\b([A-Za-z\s]{2,20})\s+(university|institute|organization|foundation|journal)'
        ]
        
        mentioned_sources = []
        for pattern in source_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    mentioned_sources.extend([m.strip() for m in match if m.strip()])
                else:
                    mentioned_sources.append(match.strip())
        
        # Simple credibility assessment
        credible_indicators = ['university', 'institute', 'journal', 'study', 'research', 'peer-reviewed']
        non_credible_indicators = ['blog', 'facebook', 'twitter', 'youtube', 'tiktok']
        
        credibility_score = 0
        for source in mentioned_sources:
            source_lower = source.lower()
            if any(indicator in source_lower for indicator in credible_indicators):
                credibility_score += 0.3
            elif any(indicator in source_lower for indicator in non_credible_indicators):
                credibility_score -= 0.2
        
        return {
            "urls_found": urls,
            "mentioned_sources": mentioned_sources[:5],
            "credibility_score": max(0, min(1, credibility_score)),
            "has_scientific_sources": any(word in query.lower() for word in ['study', 'research', 'journal', 'peer-reviewed'])
        }

    def gather_web_evidence(self, query: str) -> Dict[str, Any]:
        """Gather additional evidence from web (basic implementation)"""
        claim_indicators = {
            'statistical': bool(re.search(r'\b\d+%|\b\d+\s+(percent|times|fold|increase|decrease)\b', query)),
            'temporal': bool(re.search(r'\b(recent|new|latest|now|today|this year|2024|2025)\b', query, re.IGNORECASE)),
            'geographical': bool(re.search(r'\b(in|from|at)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)),
            'health_related': bool(re.search(r'\b(health|medical|disease|treatment|cure|vaccine|drug|medicine)\b', query, re.IGNORECASE)),
            'political': bool(re.search(r'\b(government|election|vote|political|policy|law)\b', query, re.IGNORECASE)),
            'scientific': bool(re.search(r'\b(science|scientific|research|study|experiment|data)\b', query, re.IGNORECASE))
        }
        
        return {
            "method": "basic_web_analysis",
            "claim_characteristics": claim_indicators,
            "requires_recent_data": claim_indicators['temporal'],
            "requires_expert_verification": claim_indicators['health_related'] or claim_indicators['scientific'],
            "complexity_level": sum(claim_indicators.values())
        }

    def analyze_factcheck_ratings(self, ratings: List[str]) -> str:
        """Analyze fact-check ratings to determine overall verdict"""
        if not ratings:
            return "unknown"
        
        ratings_lower = [r.lower() for r in ratings if r]
        
        false_indicators = ['false', 'fake', 'incorrect', 'misleading', 'pants on fire', 'mostly false', 'fabricated']
        true_indicators = ['true', 'correct', 'accurate', 'verified', 'mostly true', 'confirmed']
        mixed_indicators = ['mixed', 'partly', 'half true', 'half false', 'needs context', 'unproven']
        
        false_count = sum(1 for rating in ratings_lower if any(indicator in rating for indicator in false_indicators))
        true_count = sum(1 for rating in ratings_lower if any(indicator in rating for indicator in true_indicators))
        mixed_count = sum(1 for rating in ratings_lower if any(indicator in rating for indicator in mixed_indicators))
        
        total_ratings = len(ratings_lower)
        
        # Determine verdict based on majority
        if false_count / total_ratings > 0.5:
            return "disputed_false"
        elif true_count / total_ratings > 0.5:
            return "verified_true"
        elif mixed_count > 0 or (false_count > 0 and true_count > 0):
            return "mixed_evidence"
        else:
            return "disputed"

    def combine_all_evidence(self, google_result: Dict, language_analysis: Dict, 
                           source_analysis: Dict, web_evidence: Dict) -> Dict[str, Any]:
        """Combine all evidence sources to make final verdict"""
        
        # Start with Google Fact Check as primary source
        if google_result.get("found_fact_checks", 0) > 0:
            google_verdict = google_result.get("verdict", "unknown")
            google_confidence = google_result.get("confidence", 0.0)
            
            # Adjust confidence based on other factors
            confidence_adjustment = 0
            
            # Language analysis adjustment
            lang_risk = language_analysis.get("risk_score", 0)
            if google_verdict == "disputed_false" and lang_risk > 0.5:
                confidence_adjustment += 0.1  # High suspicious language supports false verdict
            elif google_verdict == "verified_true" and lang_risk > 0.5:
                confidence_adjustment -= 0.1  # Suspicious language contradicts true verdict
            
            # Source credibility adjustment  
            source_credibility = source_analysis.get("credibility_score", 0)
            if source_credibility > 0.5:
                confidence_adjustment += 0.05
            
            final_confidence = max(0.1, min(0.95, google_confidence + confidence_adjustment))
            
            # Map Google verdicts to final verdicts
            verdict_mapping = {
                "disputed_false": "fact_checked_false",
                "verified_true": "fact_checked_true", 
                "mixed_evidence": "mixed_evidence",
                "disputed": "disputed"
            }
            
            final_verdict = verdict_mapping.get(google_verdict, "disputed")
            
        else:
            # No fact-checks found - rely on language and source analysis
            lang_risk = language_analysis.get("risk_score", 0)
            source_credibility = source_analysis.get("credibility_score", 0)
            
            if lang_risk > 0.6:
                final_verdict = "likely_false"
                final_confidence = 0.6
            elif source_credibility > 0.5:
                final_verdict = "potentially_true"
                final_confidence = 0.4
            else:
                final_verdict = "insufficient_evidence"
                final_confidence = 0.2
        
        return {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 2),
            "evidence_summary": {
                "professional_fact_checks": google_result.get("found_fact_checks", 0),
                "language_risk_level": language_analysis.get("risk_level", "unknown"),
                "source_credibility": source_analysis.get("credibility_score", 0),
                "requires_expert_review": web_evidence.get("requires_expert_verification", False)
            }
        }