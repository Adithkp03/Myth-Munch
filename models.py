from pydantic import BaseModel
from typing import List, Optional

class FactCheckRequest(BaseModel):
    claim: str
    include_evidence: Optional[bool] = True
    max_evidence_sources: Optional[int] = 8
    detailed_analysis: Optional[bool] = False

class EvidenceItem(BaseModel):
    source: str
    title: str
    url: str
    snippet: str
    credibility: float
    similarity: float
    timestamp: str

class FactCheckResponse(BaseModel):
    analysis_id: str
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence: List[EvidenceItem]
    gemini_verdict: str
    processing_time: float
