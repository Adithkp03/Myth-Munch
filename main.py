import hashlib
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import FactCheckRequest, FactCheckResponse
from verification import AdvancedFactChecker
from explanations import ExplanationGenerator
from config import SOURCE_CREDIBILITY

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Misinformation Detection Agent", version="v3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Initialize components
fact_checker = AdvancedFactChecker()
explainer = ExplanationGenerator()

@app.get("/")
async def root():
    return {"message": "AI Misinformation Detection Agent", "version": "v3.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "v3.0",
        "components": {
            "fact_checker": "active",
            "explainer": "active",
            "evidence_sources": ["wikipedia", "newsapi", "google_search"],
            "ai_models": ["gemini-1.5-flash", "sentence-transformers"]
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "cache_size": len(fact_checker.cache),
        "source_credibility": SOURCE_CREDIBILITY,
        "model_info": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "llm_model": "gemini-1.5-flash"
        }
    }

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_endpoint(request: FactCheckRequest):
    """Advanced fact-checking endpoint with evidence gathering"""
    analysis_id = hashlib.md5(request.claim.encode()).hexdigest()
    
    try:
        logger.info(f"Processing fact-check request: {request.claim[:100]}...")
        
        # Run fact-checking
        verification_result = await fact_checker.fact_check(
            request.claim, 
            request.max_evidence_sources
        )
        
        # Generate explanation
        explanation_result = explainer.generate(request.claim, verification_result)
        
        # Combine results
        response = FactCheckResponse(
            analysis_id=analysis_id,
            claim=verification_result["claim"],
            verdict=verification_result["verdict"],
            confidence=verification_result["confidence"],
            explanation=explanation_result.get("explanation", verification_result["explanation"]),
            evidence=[ev for ev in verification_result["evidence"]],
            gemini_verdict=verification_result["gemini_verdict"],
            processing_time=verification_result["processing_time"]
        )
        
        logger.info(f"Fact-check completed: {response.verdict} ({response.confidence})")
        return response
        
    except Exception as e:
        logger.exception(f"Fact-check error for claim '{request.claim}': {e}")
        raise HTTPException(status_code=500, detail=f"Fact-check failed: {str(e)}")

@app.post("/quick-check")
async def quick_check(request: FactCheckRequest):
    """Quick fact-check without extensive evidence gathering"""
    try:
        # Use cached results or faster processing
        result = await fact_checker.fact_check(request.claim, max_evidence=4)
        return {
            "claim": result["claim"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "explanation": result["explanation"][:200] + "...",
            "processing_time": result["processing_time"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
