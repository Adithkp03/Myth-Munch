# Keep your existing ExplanationGenerator from before
# Or enhance it to work with the new verification results

import json
from datetime import datetime
from typing import Dict, Any, List

class ExplanationGenerator:
    def __init__(self):
        self.base_templates = {
            "SUPPORTS": {
                "template": "This claim is supported by evidence from {num_sources} source{plural}. {evidence_summary} {recommendation}",
                "verdict_map": {"high": "STRONGLY SUPPORTED", "medium": "SUPPORTED", "low": "WEAKLY SUPPORTED"}
            },
            "REFUTES": {
                "template": "This claim is refuted by evidence from {num_sources} source{plural}. {evidence_summary} {recommendation}",
                "verdict_map": {"high": "STRONGLY REFUTED", "medium": "REFUTED", "low": "WEAKLY REFUTED"}
            },
            "CONFLICTING_EVIDENCE": {
                "template": "There is conflicting evidence for this claim across {num_sources} source{plural}. {evidence_summary} {recommendation}"
            },
            "INSUFFICIENT_INFO": {
                "template": "There is insufficient information to verify this claim. {evidence_summary} {recommendation}"
            }
        }
    
    def generate(self, claim: str, verification: Dict) -> Dict:
        """Generate explanation from verification results"""
        try:
            verdict = verification.get("verdict", "INSUFFICIENT_INFO")
            confidence = verification.get("confidence", 0.0)
            evidence = verification.get("evidence", [])
            
            explanation = self.create_explanation(claim, verdict, confidence, evidence)
            evidence_summary = self.create_evidence_summary(evidence)
            recommendations = self.generate_recommendations(claim, verdict, confidence)
            
            return {
                "explanation": explanation,
                "evidence_summary": evidence_summary,
                "recommendations": recommendations,
                "confidence_level": self.confidence_text(confidence),
                "topic_category": self.identify_category(claim),
            }
        except Exception as e:
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    def create_explanation(self, claim: str, verdict: str, confidence: float, evidence: List[Dict]) -> str:
        num_sources = len(evidence)
        plural = "s" if num_sources != 1 else ""
        
        if confidence >= 0.8:
            strength = "high"
        elif confidence >= 0.6:
            strength = "medium"
        else:
            strength = "low"
        
        template_data = self.base_templates.get(verdict, {
            "template": "Analysis of this claim is inconclusive. {evidence_summary} {recommendation}"
        })
        
        explanation = template_data["template"].format(
            num_sources=num_sources,
            plural=plural,
            evidence_summary="",
            recommendation=""
        )
        
        # Add source details
        if evidence:
            sources = list(set([ev.get('source', 'unknown') for ev in evidence]))
            explanation += f" Sources include: {', '.join(sources)}."
        
        return explanation.strip()
    
    def create_evidence_summary(self, evidence: List[Dict]) -> List[str]:
        if not evidence:
            return ["No evidence found"]
        
        summary = []
        for ev in evidence[:5]:  # Top 5 pieces of evidence
            source = ev.get('source', 'Unknown')
            title = ev.get('title', 'Untitled')
            similarity = ev.get('similarity', 0)
            summary.append(f"• {source.title()}: {title} (relevance: {similarity:.2f})")
        
        return summary
    
    def generate_recommendations(self, claim: str, verdict: str, confidence: float) -> List[str]:
        base_recs = {
            "SUPPORTS": ["This claim appears to be supported by available evidence"],
            "REFUTES": ["This claim appears to be contradicted by available evidence"],
            "CONFLICTING_EVIDENCE": ["Exercise caution - evidence is mixed"],
            "INSUFFICIENT_INFO": ["More research needed to verify this claim"]
        }
        
        recs = base_recs.get(verdict, ["Verify with additional sources"])
        
        if confidence < 0.5:
            recs.append("Low confidence - seek additional verification")
        
        return recs
    
    def identify_category(self, claim: str) -> str:
        claim_lower = claim.lower()
        categories = {
            "health": ["health", "medical", "vaccine", "covid", "disease"],
            "science": ["science", "research", "climate", "study"],
            "politics": ["politics", "government", "election"],
            "technology": ["technology", "ai", "internet", "computer"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in claim_lower for keyword in keywords):
                return category
        return "general"
    
    def confidence_text(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
