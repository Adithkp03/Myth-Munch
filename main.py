from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from classifier import FakeNewsClassifier
from trends import TrendDetector
from verification import FactChecker
from explanations import ExplanationGenerator
import pandas as pd

# Create FastAPI app
app = FastAPI(
    title="AI Misinformation Detection Agent",
    description="Real-time fact-checking powered by professional sources & AI analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
classifier = FakeNewsClassifier()
trend_detector = TrendDetector()
fact_checker = FactChecker()
explainer = ExplanationGenerator()

# Routes
@app.get("/")
async def read_root():
    """Serve the main web interface"""
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Misinformation Detection Agent is running",
        "timestamp": pd.Timestamp.now().isoformat(),
        "components": {
            "classifier": "ready",
            "trend_detector": "ready", 
            "fact_checker": "ready",
            "explainer": "ready"
        }
    }

@app.post("/analyze-claim")
async def analyze_claim(request: Request):
    """Basic claim classification only"""
    try:
        data = await request.json()
        claim = data.get("claim", "")
        
        if not claim.strip():
            return {"error": "Please provide a claim to analyze"}
        
        result = classifier.predict(claim)
        return {
            "claim": claim,
            "analysis": result,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/verify-claim")
async def verify_claim(request: Request):
    """Fact-checking verification only"""
    try:
        data = await request.json()
        claim = data.get("claim", "")
        
        if not claim.strip():
            return {"error": "Please provide a claim to verify"}
        
        result = fact_checker.verify_claim(claim)
        return result
    except Exception as e:
        return {"error": f"Verification failed: {str(e)}"}

@app.post("/add-documents")
async def add_documents(request: Request):
    """Add documents for trend detection"""
    try:
        data = await request.json()
        documents = data.get("documents", [])
        
        if not documents:
            return {"error": "No documents provided"}
        
        # Validate documents
        valid_docs = [doc for doc in documents if isinstance(doc, str) and doc.strip()]
        
        if not valid_docs:
            return {"error": "No valid documents provided"}
        
        count = trend_detector.add_documents(valid_docs)
        return {
            "message": f"Added {len(valid_docs)} documents",
            "total_documents": count,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to add documents: {str(e)}"}

@app.post("/detect-trends")
async def detect_trends():
    """Detect trending topics from added documents"""
    try:
        result = trend_detector.detect_trends()
        result["timestamp"] = pd.Timestamp.now().isoformat()
        return result
    except Exception as e:
        return {"error": f"Trend detection failed: {str(e)}"}

@app.post("/full-analysis")
async def full_analysis(request: Request):
    """Complete misinformation analysis pipeline"""
    try:
        data = await request.json()
        claim = data.get("claim", "")
        
        if not claim.strip():
            return {"error": "Please provide a claim to analyze"}
        
        # Run all analysis components
        print(f"Starting full analysis for: {claim}")
        
        # Step 1: ML Classification
        classification = classifier.predict(claim)
        print(f"Classification complete: {classification.get('prediction')}")
        
        # Step 2: Fact-checking verification
        verification = fact_checker.verify_claim(claim)
        print(f"Verification complete: {verification.get('overall_verdict', {}).get('verdict')}")
        
        # Step 3: Human explanation generation
        explanation = explainer.generate_explanation(claim, classification, verification)
        print(f"Explanation generated with {len(explanation.get('fact_check_citations', []))} citations")
        
        return {
            "claim": claim,
            "fake_news_classification": classification,
            "fact_check_verification": verification,
            "human_explanation": explanation,
            "analysis_metadata": {
                "confidence_level": explanation.get("confidence_level", "Unknown"),
                "processing_timestamp": pd.Timestamp.now().isoformat(),
                "components_used": ["ML_classification", "fact_checking", "explanation_generation"],
                "api_sources": ["Google_Fact_Check_API", "keyword_patterns"]
            }
        }
    except Exception as e:
        print(f"Full analysis error: {str(e)}")
        return {"error": f"Full analysis failed: {str(e)}"}

@app.post("/batch-analysis")
async def batch_analysis(request: Request):
    """Analyze multiple claims at once"""
    try:
        data = await request.json()
        claims = data.get("claims", [])
        
        if not claims:
            return {"error": "No claims provided"}
        
        results = []
        for i, claim in enumerate(claims[:10]):  # Limit to 10 claims
            if isinstance(claim, str) and claim.strip():
                try:
                    classification = classifier.predict(claim)
                    verification = fact_checker.verify_claim(claim)
                    explanation = explainer.generate_explanation(claim, classification, verification)
                    
                    results.append({
                        "claim_index": i,
                        "claim": claim,
                        "verdict": verification.get("overall_verdict", {}).get("verdict", "unknown"),
                        "confidence": verification.get("overall_verdict", {}).get("confidence", 0.0),
                        "summary": explanation.get("explanation", ""),
                        "fact_checks_found": verification.get("google_factcheck", {}).get("found_fact_checks", 0)
                    })
                except Exception as e:
                    results.append({
                        "claim_index": i,
                        "claim": claim,
                        "error": str(e)
                    })
        
        return {
            "total_claims": len(claims),
            "processed_claims": len(results),
            "results": results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Batch analysis failed: {str(e)}"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        return {
            "system_status": "operational",
            "total_documents": len(trend_detector.documents),
            "trend_detection_ready": len(trend_detector.documents) >= 3,
            "api_integrations": {
                "google_fact_check": "active",
                "ml_classifier": "active",
                "trend_detector": "active",
                "explanation_generator": "active"
            },
            "capabilities": [
                "Real-time fact checking",
                "Professional source verification", 
                "Trend detection",
                "Multi-audience explanations",
                "Batch processing"
            ],
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Stats retrieval failed: {str(e)}"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {"error": "Endpoint not found", "available_endpoints": [
        "/", "/health", "/analyze-claim", "/verify-claim", 
        "/full-analysis", "/detect-trends", "/add-documents", 
        "/batch-analysis", "/stats"
    ]}

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return {"error": "Internal server error", "message": "Please try again or contact support"}

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 AI Misinformation Detection Agent starting up...")
    print("✅ Classifier initialized")
    print("✅ Trend detector initialized") 
    print("✅ Fact checker initialized")
    print("✅ Explanation generator initialized")
    print("🌐 Web interface available at http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 AI Misinformation Detection Agent shutting down...")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        reload_dirs=["./"],
        log_level="info"
    )
