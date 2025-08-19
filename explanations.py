import json
from datetime import datetime
from typing import Dict, Any, List

class ExplanationGenerator:
    def __init__(self):
        # Dynamic templates that adapt to any topic
        self.base_templates = {
            "fact_checked_false": {
                "template": "This claim has been fact-checked by {num_sources} authoritative source{plural} and found to be {verdict_strength}. {evidence_summary} {recommendation}",
                "verdict_strength_map": {
                    "high": "FALSE",
                    "medium": "LARGELY FALSE", 
                    "low": "QUESTIONABLE"
                }
            },
            "fact_checked_true": {
                "template": "This claim has been verified by {num_sources} fact-checking organization{plural} as {verdict_strength}. {evidence_summary} {recommendation}",
                "verdict_strength_map": {
                    "high": "TRUE",
                    "medium": "LARGELY TRUE",
                    "low": "PARTIALLY TRUE"
                }
            },
            "mixed_evidence": {
                "template": "Fact-checkers found mixed evidence for this claim across {num_sources} source{plural}. {evidence_summary} {recommendation}"
            },
            "likely_false": {
                "template": "Our analysis indicates this claim is likely false based on {analysis_factors}. {evidence_summary} {recommendation}"
            },
            "insufficient_evidence": {
                "template": "There is insufficient evidence to verify this claim. {evidence_summary} {recommendation}"
            },
            "disputed": {
                "template": "This claim is disputed with conflicting information from {num_sources} source{plural}. {evidence_summary} {recommendation}"
            }
        }

    def generate_explanation(self, claim: str, classification: Dict, verification: Dict) -> Dict[str, Any]:
        """Generate comprehensive, dynamic explanation for any topic"""
        try:
            # Get overall verdict and confidence
            verdict = verification.get("overall_verdict", {}).get("verdict", "unknown")
            confidence = verification.get("overall_verdict", {}).get("confidence", 0.0)
            
            # Get classification info
            ml_prediction = classification.get("prediction", "unknown")
            ml_confidence = classification.get("confidence", 0.0)
            
            # Generate main explanation
            explanation = self.create_dynamic_explanation(claim, verdict, confidence, verification)
            
            # Summarize all evidence sources
            evidence_summary = self.create_comprehensive_evidence_summary(verification)
            
            # Generate topic-appropriate recommendations
            recommendations = self.generate_dynamic_recommendations(claim, verdict, confidence, verification)
            
            # Extract and format citations
            citations = self.extract_all_citations(verification)
            
            # Determine topic category for tailored advice
            topic_category = self.identify_topic_category(claim)
            
            return {
                "explanation": explanation,
                "evidence_summary": evidence_summary,
                "recommendations": recommendations,
                "fact_check_citations": citations,
                "confidence_level": self.confidence_to_text(confidence),
                "topic_category": topic_category,
                "verification_methods_used": self.list_verification_methods(verification),
                "audience_versions": {
                    "general_public": self.simplify_for_public(explanation, citations, topic_category),
                    "technical": self.add_technical_details(explanation, classification, verification),
                    "social_media": self.create_social_media_summary(verdict, confidence, citations, topic_category)
                },
                "reliability_indicators": self.assess_reliability_indicators(verification)
            }
            
        except Exception as e:
            return {"error": f"Explanation generation failed: {str(e)}"}

    def create_dynamic_explanation(self, claim: str, verdict: str, confidence: float, verification: Dict) -> str:
        """Create explanation that adapts to any topic and evidence available"""
        
        # Count evidence sources
        google_checks = verification.get("google_factcheck", {}).get("found_fact_checks", 0)
        language_risk = verification.get("language_analysis", {}).get("risk_level", "low")
        source_credibility = verification.get("source_analysis", {}).get("credibility_score", 0)
        
        num_sources = google_checks
        plural = "s" if num_sources != 1 else ""
        
        # Determine verdict strength based on confidence
        if confidence >= 0.8:
            strength = "high"
        elif confidence >= 0.6:
            strength = "medium" 
        else:
            strength = "low"
        
        # Get base template
        template_info = self.base_templates.get(verdict, {})
        template = template_info.get("template", "This claim requires further verification. {evidence_summary} {recommendation}")
        
        # Fill in template variables
        if verdict in ["fact_checked_false", "fact_checked_true"]:
            verdict_strength = template_info["verdict_strength_map"].get(strength, "UNCERTAIN")
            
            explanation = template.format(
                num_sources=num_sources,
                plural=plural,
                verdict_strength=verdict_strength,
                evidence_summary="",
                recommendation=""
            )
        elif verdict == "likely_false":
            analysis_factors = []
            if language_risk == "high":
                analysis_factors.append("suspicious language patterns")
            if source_credibility < 0.3:
                analysis_factors.append("lack of credible sources")
            if google_checks == 0:
                analysis_factors.append("absence of professional fact-checks")
            
            factors_text = ", ".join(analysis_factors) if analysis_factors else "multiple indicators"
            
            explanation = template.format(
                analysis_factors=factors_text,
                evidence_summary="",
                recommendation=""
            )
        else:
            explanation = template.format(
                num_sources=num_sources,
                plural=plural,
                evidence_summary="",
                recommendation=""
            )
        
        # Add specific context based on what we found
        if google_checks > 0:
            publishers = []
            fact_checks = verification.get("google_factcheck", {}).get("fact_checks", [])
            for fc in fact_checks[:3]:
                publisher = fc.get("publisher", "")
                if publisher and publisher not in publishers:
                    publishers.append(publisher)
            
            if publishers:
                explanation += f" Professional fact-checkers including {', '.join(publishers)} have examined this claim."
        
        # Add language analysis context if relevant
        if language_risk == "high":
            explanation += " The claim uses language patterns commonly associated with misinformation."
        
        # Add source analysis if relevant
        if source_credibility > 0.6:
            explanation += " The claim references credible sources."
        elif source_credibility < 0.3 and verification.get("source_analysis", {}).get("mentioned_sources"):
            explanation += " The sources mentioned in this claim may not be reliable."
        
        return explanation.strip()

    def create_comprehensive_evidence_summary(self, verification: Dict) -> List[str]:
        """Create comprehensive evidence summary from all sources"""
        evidence = []
        
        # Google Fact Check evidence
        google_check = verification.get("google_factcheck", {})
        found_checks = google_check.get("found_fact_checks", 0)
        if found_checks > 0:
            verdict = google_check.get("verdict", "unknown")
            evidence.append(f"✅ Found {found_checks} professional fact-check{('s' if found_checks != 1 else '')} with verdict: {verdict.replace('_', ' ')}")
            
            # Add publisher info
            fact_checks = google_check.get("fact_checks", [])
            if fact_checks:
                publishers = list(set([fc.get("publisher", "Unknown") for fc in fact_checks[:5]]))
                evidence.append(f"📰 Fact-checked by: {', '.join(publishers)}")
                
                # Add dates for recency
                dates = [fc.get("date", "") for fc in fact_checks if fc.get("date")]
                if dates:
                    recent_date = max(dates)[:10]
                    evidence.append(f"📅 Most recent fact-check: {recent_date}")
        
        # Language analysis evidence
        language_analysis = verification.get("language_analysis", {})
        risk_level = language_analysis.get("risk_level", "low")
        if risk_level != "low":
            pattern_matches = language_analysis.get("pattern_matches", {})
            if pattern_matches:
                evidence.append(f"⚠️ Language analysis: {risk_level} risk level detected")
                for category, details in pattern_matches.items():
                    evidence.append(f"   • {category.replace('_', ' ')}: {details['matches']} pattern(s) found")
        
        # Source credibility evidence  
        source_analysis = verification.get("source_analysis", {})
        mentioned_sources = source_analysis.get("mentioned_sources", [])
        if mentioned_sources:
            credibility = source_analysis.get("credibility_score", 0)
            evidence.append(f"📚 Sources mentioned: {', '.join(mentioned_sources[:3])}")
            if credibility > 0.5:
                evidence.append("✅ Sources appear credible")
            elif credibility < 0.3:
                evidence.append("❌ Source credibility concerns identified")
        
        # Web evidence characteristics
        web_evidence = verification.get("web_evidence", {})
        characteristics = web_evidence.get("claim_characteristics", {})
        if any(characteristics.values()):
            char_list = [k.replace('_', ' ') for k, v in characteristics.items() if v]
            evidence.append(f"🔍 Claim characteristics: {', '.join(char_list)}")
        
        return evidence if evidence else ["ℹ️ Limited verification data available"]

    def generate_dynamic_recommendations(self, claim: str, verdict: str, confidence: float, verification: Dict) -> List[str]:
        """Generate recommendations that adapt to the topic and evidence"""
        
        # Base recommendations by verdict
        base_recommendations = {
            "fact_checked_false": [
                "❌ Do not share this claim - it has been professionally debunked",
                "🔍 Always verify claims with multiple reputable fact-checking organizations"
            ],
            "fact_checked_true": [
                "✅ This claim appears accurate based on professional fact-checks",
                "📱 When sharing, include links to the original fact-check sources"
            ],
            "mixed_evidence": [
                "⚠️ Exercise caution - this claim has conflicting evidence",
                "🔍 Check multiple independent sources before drawing conclusions"
            ],
            "likely_false": [
                "🚩 Be skeptical of this claim - multiple red flags detected",
                "🔍 Look for verification from established fact-checking organizations"
            ],
            "insufficient_evidence": [
                "❓ More evidence needed to verify this claim",
                "🔍 Check back later as more information may become available"
            ]
        }
        
        recommendations = base_recommendations.get(verdict, [
            "🔍 Verify this claim with multiple independent sources",
            "📚 Look for peer-reviewed research if this involves scientific claims"
        ])
        
        # Add topic-specific recommendations
        topic_category = self.identify_topic_category(claim)
        
        if topic_category == "health":
            recommendations.extend([
                "🏥 For health claims, consult medical professionals and official health organizations",
                "📋 Be especially cautious with health misinformation as it can cause harm"
            ])
        elif topic_category == "science":
            recommendations.extend([
                "🧪 Look for peer-reviewed scientific studies and expert consensus",
                "📊 Be wary of claims that contradict established scientific understanding"
            ])
        elif topic_category == "politics":
            recommendations.extend([
                "🗳️ Verify political claims with multiple news sources across the political spectrum",
                "📰 Check with established news organizations and official government sources"
            ])
        elif topic_category == "breaking_news":
            recommendations.extend([
                "📺 Breaking news often contains inaccuracies - wait for confirmation",
                "🕐 Check multiple news sources and wait for updates"
            ])
        elif topic_category == "financial":
            recommendations.extend([
                "💰 Be extremely cautious with financial advice and investment claims",
                "🏦 Consult qualified financial professionals before making decisions"
            ])
        
        # Add confidence-based recommendations
        if confidence < 0.5:
            recommendations.append("⚡ Low confidence assessment - seek additional verification")
        
        # Add urgency warnings for suspicious language
        language_risk = verification.get("language_analysis", {}).get("risk_level", "low")
        if language_risk == "high":
            recommendations.insert(0, "🚨 This claim uses language typical of misinformation - be extra cautious")
        
        return recommendations[:6]  # Limit to 6 recommendations

    def identify_topic_category(self, claim: str) -> str:
        """Identify the general category of the claim for tailored advice"""
        claim_lower = claim.lower()
        
        category_keywords = {
            "health": ["health", "medical", "disease", "treatment", "cure", "vaccine", "drug", "medicine", "doctor", "hospital", "virus", "cancer", "covid"],
            "science": ["science", "scientific", "research", "study", "experiment", "climate", "global warming", "evolution", "space", "technology"],
            "politics": ["government", "election", "vote", "political", "policy", "law", "president", "politician", "congress", "senate"],
            "breaking_news": ["breaking", "urgent", "alert", "just in", "developing", "reports", "according to sources"],
            "financial": ["money", "investment", "stock", "market", "economy", "bitcoin", "crypto", "trading", "financial", "bank"],
            "celebrity": ["celebrity", "actor", "singer", "famous", "hollywood", "music", "movie", "entertainment"],
            "conspiracy": ["conspiracy", "cover-up", "secret", "hidden", "they don't want you to know", "illuminati", "deep state"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in claim_lower for keyword in keywords):
                return category
        
        return "general"

    def extract_all_citations(self, verification: Dict) -> List[Dict]:
        """Extract citations from all verification sources"""
        citations = []
        
        # Google fact-check citations
        google_check = verification.get("google_factcheck", {})
        fact_checks = google_check.get("fact_checks", [])
        
        for fc in fact_checks:
            citation = {
                "type": "fact_check",
                "title": fc.get("title", "Untitled Fact-Check"),
                "publisher": fc.get("publisher", "Unknown Publisher"),
                "rating": fc.get("rating", ""),
                "url": fc.get("url", ""),
                "date": fc.get("date", "")[:10] if fc.get("date") else "",
                "claim_text": fc.get("claim_text", "")[:200] + "..." if len(fc.get("claim_text", "")) > 200 else fc.get("claim_text", "")
            }
            citations.append(citation)
        
        # Add source analysis citations if any URLs were found
        source_analysis = verification.get("source_analysis", {})
        urls = source_analysis.get("urls_found", [])
        for url in urls[:3]:  # Limit to first 3 URLs
            citations.append({
                "type": "mentioned_source",
                "title": "Source mentioned in claim",
                "url": url,
                "note": "URL mentioned in the original claim"
            })
        
        return citations

    def assess_reliability_indicators(self, verification: Dict) -> Dict[str, Any]:
        """Assess various reliability indicators"""
        indicators = {
            "professional_fact_checks": verification.get("google_factcheck", {}).get("found_fact_checks", 0) > 0,
            "multiple_sources": verification.get("google_factcheck", {}).get("found_fact_checks", 0) > 1,
            "recent_verification": False,
            "credible_sources_mentioned": verification.get("source_analysis", {}).get("credibility_score", 0) > 0.5,
            "low_language_risk": verification.get("language_analysis", {}).get("risk_level", "high") == "low",
            "scientific_backing": verification.get("source_analysis", {}).get("has_scientific_sources", False)
        }
        
        # Check for recent verification
        fact_checks = verification.get("google_factcheck", {}).get("fact_checks", [])
        if fact_checks:
            from datetime import datetime, timedelta
            recent_threshold = datetime.now() - timedelta(days=365)
            for fc in fact_checks:
                date_str = fc.get("date", "")
                if date_str:
                    try:
                        check_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        if check_date > recent_threshold:
                            indicators["recent_verification"] = True
                            break
                    except:
                        pass
        
        # Calculate overall reliability score
        reliability_score = sum(indicators.values()) / len(indicators)
        
        return {
            "indicators": indicators,
            "reliability_score": round(reliability_score, 2),
            "reliability_level": "high" if reliability_score > 0.7 else "medium" if reliability_score > 0.4 else "low"
        }

    def list_verification_methods(self, verification: Dict) -> List[str]:
        """List all verification methods used"""
        methods = []
        
        if verification.get("google_factcheck", {}).get("found_fact_checks", 0) > 0:
            methods.append("Professional fact-checking databases")
        
        if verification.get("language_analysis", {}):
            methods.append("Language pattern analysis")
        
        if verification.get("source_analysis", {}).get("mentioned_sources"):
            methods.append("Source credibility assessment")
        
        if verification.get("web_evidence", {}):
            methods.append("Web evidence gathering")
        
        return methods

    def confidence_to_text(self, confidence: float) -> str:
        """Convert confidence score to human-readable text"""
        if confidence >= 0.9:
            return "Very High Confidence"
        elif confidence >= 0.75:
            return "High Confidence"
        elif confidence >= 0.6:
            return "Moderate Confidence"
        elif confidence >= 0.4:
            return "Low Confidence"
        else:
            return "Very Low Confidence"

    def simplify_for_public(self, explanation: str, citations: List[Dict], topic_category: str) -> str:
        """Create simplified explanation for general public"""
        # Remove technical jargon
        simplified = explanation.replace("analysis indicates", "our check shows")
        simplified = simplified.replace("confidence:", "certainty:")
        simplified = simplified.replace("verdict:", "result:")
        
        # Add category-specific simple advice
        category_advice = {
            "health": "For health information, always talk to your doctor or check with health authorities like the CDC or WHO.",
            "science": "For science claims, look for information from universities, scientific journals, or science organizations.",
            "politics": "For political claims, check multiple news sources and official government websites.",
            "financial": "For money advice, be very careful and talk to qualified financial advisors."
        }
        
        if topic_category in category_advice:
            simplified += f"\n\n💡 Quick tip: {category_advice[topic_category]}"
        
        # Add simple citation summary
        if citations:
            fact_check_citations = [c for c in citations if c.get("type") == "fact_check"]
            if fact_check_citations:
                publishers = [c.get("publisher", "fact-checkers") for c in fact_check_citations[:2]]
                simplified += f"\n\n✅ This has been checked by trusted sources like {', '.join(publishers)}."
        
        return simplified

    def add_technical_details(self, explanation: str, classification: Dict, verification: Dict) -> str:
        """Add technical details for expert/technical users"""
        tech_details = f"{explanation}\n\n🔧 Technical Analysis:\n"
        tech_details += f"• ML Classification: {classification.get('prediction', 'unknown')} ({classification.get('confidence', 0):.3f} confidence)\n"
        
        # Language analysis details
        lang_analysis = verification.get("language_analysis", {})
        tech_details += f"• Language Risk Score: {lang_analysis.get('risk_score', 0):.3f}\n"
        tech_details += f"• Suspicious Pattern Categories: {len(lang_analysis.get('pattern_matches', {}))}\n"
        
        # Source analysis details
        source_analysis = verification.get("source_analysis", {})
        tech_details += f"• Source Credibility Score: {source_analysis.get('credibility_score', 0):.3f}\n"
        tech_details += f"• URLs Found: {len(source_analysis.get('urls_found', []))}\n"
        
        # Fact-check details
        google_check = verification.get("google_factcheck", {})
        tech_details += f"• Professional Fact-Checks Found: {google_check.get('found_fact_checks', 0)}\n"
        if google_check.get("found_fact_checks", 0) > 0:
            ratings = [fc.get("rating", "") for fc in google_check.get("fact_checks", []) if fc.get("rating")]
            if ratings:
                tech_details += f"• Fact-Check Ratings: {', '.join(set(ratings))}\n"
        
        # Reliability assessment
        reliability = verification.get("overall_verdict", {}).get("evidence_summary", {})
        if reliability:
            tech_details += f"• Evidence Quality Indicators: {reliability}\n"
        
        return tech_details

    def create_social_media_summary(self, verdict: str, confidence: float, citations: List[Dict], topic_category: str) -> str:
        """Create shareable social media summary"""
        emoji_map = {
            "fact_checked_false": "❌",
            "fact_checked_true": "✅", 
            "mixed_evidence": "⚠️",
            "likely_false": "🚩",
            "insufficient_evidence": "❓",
            "disputed": "🤔"
        }
        
        emoji = emoji_map.get(verdict, "🔍")
        
        # Create verdict text
        verdict_text = {
            "fact_checked_false": "FACT-CHECK: This claim is FALSE",
            "fact_checked_true": "FACT-CHECK: This claim is TRUE", 
            "mixed_evidence": "FACT-CHECK: Mixed evidence - needs context",
            "likely_false": "FACT-CHECK: Likely false",
            "insufficient_evidence": "FACT-CHECK: Needs more verification",
            "disputed": "FACT-CHECK: Disputed claim"
        }
        
        summary = f"{emoji} {verdict_text.get(verdict, 'FACT-CHECK: Verification needed')}"
        summary += f" (Confidence: {confidence:.0%})"
        
        # Add source info
        fact_check_citations = [c for c in citations if c.get("type") == "fact_check"]
        if fact_check_citations:
            publisher = fact_check_citations[0].get("publisher", "Fact-checkers")
            summary += f"\nVerified by: {publisher}"
        
        # Add topic-specific hashtags
        topic_hashtags = {
            "health": "#HealthFacts #MedicalMisinformation",
            "science": "#ScienceFacts #ClimateScience", 
            "politics": "#PoliticalFacts #ElectionFacts",
            "breaking_news": "#BreakingNews #NewsVerification",
            "financial": "#FinancialFacts #InvestmentScam"
        }
        
        hashtags = topic_hashtags.get(topic_category, "#FactCheck #Misinformation")
        summary += f"\n{hashtags} #VerifyBeforeSharing"
        
        return summary