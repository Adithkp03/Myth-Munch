import json

class ExplanationGenerator:
    def __init__(self):
        # Enhanced templates for real fact-checking results
        self.templates = {
            "fact_checked_false": {
                "template": "This claim has been fact-checked by authoritative sources and found to be FALSE. {evidence} {recommendation}",
                "evidence_patterns": {
                    "vaccines cause autism": "Multiple large-scale scientific studies have found no link between vaccines and autism. This claim has been thoroughly debunked by medical experts.",
                    "climate change is a hoax": "Climate change is supported by overwhelming scientific consensus. This 'hoax' claim contradicts decades of peer-reviewed research.",
                    "global warming": "Global warming is well-documented through temperature records and scientific data from multiple independent sources.",
                    "covid": "COVID-19 health claims should be verified against WHO, CDC, and peer-reviewed medical research.",
                    "election": "Election fraud claims have been investigated and debunked by election officials and courts."
                }
            },
            "fact_checked_true": {
                "template": "This claim has been verified by fact-checking organizations as accurate. {evidence} {recommendation}"
            },
            "mixed_evidence": {
                "template": "Fact-checkers found mixed evidence for this claim. {evidence} {recommendation}"
            },
            "likely_false": {
                "template": "This claim appears to be misinformation. {evidence} {recommendation}",
                "evidence_patterns": {
                    "vaccines cause autism": "Multiple large-scale scientific studies have found no link between vaccines and autism. This claim has been thoroughly debunked by medical experts.",
                    "climate change is a hoax": "Climate change is supported by overwhelming scientific consensus. This 'hoax' claim contradicts decades of peer-reviewed research.",
                    "global warming": "Global warming is well-documented through temperature records and scientific data from multiple independent sources."
                }
            },
            "disputed": {
                "template": "This claim is disputed and should be verified. {evidence} {recommendation}"
            },
            "no_red_flags": {
                "template": "No immediate red flags detected, but we recommend verifying with multiple sources. {recommendation}"
            }
        }
    
    def generate_explanation(self, claim, classification, verification):
        """Generate human-readable explanation with real fact-check data"""
        try:
            # Get overall verdict
            verdict = verification.get("overall_verdict", {}).get("verdict", "unknown")
            confidence = verification.get("overall_verdict", {}).get("confidence", 0.0)
            
            # Get classification info
            ml_prediction = classification.get("prediction", "unknown")
            ml_confidence = classification.get("confidence", 0.0)
            
            # Generate explanation based on verdict
            explanation = self.create_explanation_text(claim, verdict, confidence, ml_prediction, ml_confidence, verification)
            
            # Add evidence summary
            evidence = self.summarize_evidence(verification)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(verdict, confidence)
            
            # Add fact-check citations
            citations = self.extract_citations(verification)
            
            return {
                "explanation": explanation,
                "evidence_summary": evidence,
                "recommendations": recommendations,
                "fact_check_citations": citations,
                "confidence_level": self.confidence_to_text(confidence),
                "audience_versions": {
                    "general_public": self.simplify_for_public(explanation, citations),
                    "technical": self.add_technical_details(explanation, classification, verification),
                    "social_media": self.create_social_media_summary(verdict, confidence, citations)
                }
            }
            
        except Exception as e:
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    def create_explanation_text(self, claim, verdict, confidence, ml_prediction, ml_confidence, verification):
        """Create main explanation text with real fact-check data"""
        if verdict == "fact_checked_false":
            base_text = f"This claim has been fact-checked by multiple authoritative sources and found to be FALSE (confidence: {confidence:.1%})."
            
            # Add specific evidence for known false patterns
            for pattern, evidence in self.templates["fact_checked_false"]["evidence_patterns"].items():
                if pattern.lower() in claim.lower():
                    base_text += f" {evidence}"
                    break
            
            # Add real fact-checker info
            google_check = verification.get("google_factcheck", {})
            fact_checks = google_check.get("fact_checks", [])
            if fact_checks:
                publishers = [fc.get("publisher", "Unknown") for fc in fact_checks[:2]]
                base_text += f" Professional fact-checkers including {', '.join(publishers)} have debunked this claim."
            
            return base_text
            
        elif verdict == "fact_checked_true":
            base_text = f"This claim has been verified by fact-checking organizations as TRUE (confidence: {confidence:.1%})."
            
            # Add fact-checker sources
            google_check = verification.get("google_factcheck", {})
            fact_checks = google_check.get("fact_checks", [])
            if fact_checks:
                publishers = [fc.get("publisher", "Unknown") for fc in fact_checks[:2]]
                base_text += f" Verified by {', '.join(publishers)}."
            
            return base_text
            
        elif verdict == "mixed_evidence":
            return f"Fact-checkers found mixed evidence for this claim (confidence: {confidence:.1%}). Some aspects may be accurate while others are disputed or lack sufficient evidence."
            
        elif verdict == "likely_false":
            base_text = f"Our analysis indicates this claim is likely false (confidence: {confidence:.1%})."
            
            # Add specific evidence for known false patterns
            for pattern, evidence in self.templates["likely_false"]["evidence_patterns"].items():
                if pattern.lower() in claim.lower():
                    base_text += f" {evidence}"
                    break
            
            return base_text
            
        elif verdict == "disputed":
            return f"This claim is disputed and requires careful verification (confidence: {confidence:.1%}). Multiple fact-checking sources show conflicting information."
            
        else:
            return f"No immediate red flags detected in this claim, but independent verification is recommended."
    
    def summarize_evidence(self, verification):
        """Summarize the evidence found including real fact-checks"""
        evidence = []
        
        # Google fact check evidence
        google_check = verification.get("google_factcheck", {})
        found_checks = google_check.get("found_fact_checks", 0)
        
        if found_checks > 0:
            verdict = google_check.get("verdict", "unknown")
            evidence.append(f"Found {found_checks} professional fact-checks with verdict: {verdict}")
            
            # Add specific publishers
            fact_checks = google_check.get("fact_checks", [])
            if fact_checks:
                publishers = list(set([fc.get("publisher", "Unknown") for fc in fact_checks]))
                evidence.append(f"Fact-checked by: {', '.join(publishers[:3])}")
                
                # Add recent dates
                dates = [fc.get("date", "") for fc in fact_checks if fc.get("date")]
                if dates:
                    recent_date = max(dates)[:10]  # Get YYYY-MM-DD format
                    evidence.append(f"Most recent fact-check: {recent_date}")
        
        # Keyword analysis evidence  
        keyword_analysis = verification.get("keyword_analysis", {})
        if keyword_analysis.get("exact_false_pattern"):
            evidence.append("Matches known misinformation patterns")
        
        suspicious_count = keyword_analysis.get("suspicious_word_count", 0)
        if suspicious_count > 0:
            evidence.append(f"Contains {suspicious_count} suspicious keywords")
        
        return evidence if evidence else ["No specific red flags in databases"]
    
    def extract_citations(self, verification):
        """Extract real citations from fact-checks"""
        citations = []
        
        google_check = verification.get("google_factcheck", {})
        fact_checks = google_check.get("fact_checks", [])
        
        for fc in fact_checks:
            citation = {
                "title": fc.get("title", "Untitled"),
                "publisher": fc.get("publisher", "Unknown"),
                "rating": fc.get("rating", ""),
                "url": fc.get("url", ""),
                "date": fc.get("date", "")[:10] if fc.get("date") else ""
            }
            citations.append(citation)
        
        return citations
    
    def generate_recommendations(self, verdict, confidence):
        """Generate actionable recommendations based on verdict"""
        if verdict in ["fact_checked_false", "likely_false"]:
            return [
                "❌ Do not share this claim - it has been debunked",
                "🔍 Check reputable fact-checkers like Snopes, Reuters, or FactCheck.org", 
                "📚 Look for peer-reviewed scientific studies if this involves health/science",
                "🚨 Be cautious of similar claims from the same source"
            ]
        elif verdict == "fact_checked_true":
            return [
                "✅ This claim appears to be accurate based on fact-checks",
                "📱 Safe to share, but include source citations",
                "🔄 Double-check if sharing older information"
            ]
        elif verdict in ["mixed_evidence", "disputed"]:
            return [
                "⚠️ Verify with multiple independent sources before sharing",
                "🔍 Check the original source and their methodology",
                "👥 Look for expert consensus on this topic",
                "📅 Verify information is current and not outdated"
            ]
        else:
            return [
                "🔍 Still verify with primary sources",
                "📅 Check publication date for currency", 
                "📰 Cross-reference with multiple news outlets",
                "🎓 Consult expert sources for specialized topics"
            ]
    
    def confidence_to_text(self, confidence):
        """Convert confidence score to human text"""
        if confidence >= 0.9:
            return "Very High Confidence"
        elif confidence >= 0.7:
            return "High Confidence"
        elif confidence >= 0.5:
            return "Moderate Confidence"
        else:
            return "Low Confidence"
    
    def simplify_for_public(self, explanation, citations):
        """Simplified version for general public"""
        simple = explanation.replace("analysis indicates", "our check shows")
        simple = simple.replace("confidence:", "certainty:")
        
        # Add simple citation summary
        if citations:
            simple += f"\n\n✅ Checked by trusted sources like {citations[0].get('publisher', 'fact-checkers')}."
        
        return simple
    
    def add_technical_details(self, explanation, classification, verification):
        """Add technical details for expert users"""
        tech_details = f"{explanation}\n\n🔧 Technical Details:\n"
        tech_details += f"- ML Classification: {classification.get('prediction')} ({classification.get('confidence'):.3f})\n"
        tech_details += f"- Risk Score: {verification.get('keyword_analysis', {}).get('risk_score', 0):.3f}\n"
        tech_details += f"- Pattern Matches: {verification.get('keyword_analysis', {}).get('exact_false_pattern')}\n"
        tech_details += f"- Google API Results: {verification.get('google_factcheck', {}).get('found_fact_checks', 0)} fact-checks found\n"
        
        # Add API response details
        google_check = verification.get("google_factcheck", {})
        if google_check.get("found_fact_checks", 0) > 0:
            ratings = [fc.get("rating", "") for fc in google_check.get("fact_checks", [])]
            tech_details += f"- Fact-Check Ratings: {', '.join(set(ratings))}"
        
        return tech_details
    
    def create_social_media_summary(self, verdict, confidence, citations):
        """Create shareable social media summary"""
        emoji_map = {
            "fact_checked_false": "❌",
            "fact_checked_true": "✅", 
            "mixed_evidence": "⚠️",
            "likely_false": "🚩",
            "disputed": "❓"
        }
        
        emoji = emoji_map.get(verdict, "🔍")
        
        if verdict == "fact_checked_false":
            summary = f"{emoji} FACT-CHECK: This claim is FALSE"
        elif verdict == "fact_checked_true":
            summary = f"{emoji} FACT-CHECK: This claim is TRUE"
        else:
            summary = f"{emoji} FACT-CHECK: This claim needs verification"
        
        summary += f" (Confidence: {confidence:.0%})"
        
        if citations:
            summary += f"\nSource: {citations[0].get('publisher', 'Fact-checkers')}"
        
        summary += "\n#FactCheck #Misinformation #VerifyBeforeSharing"
        
        return summary
