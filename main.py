from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from classifier import FakeNewsClassifier
from verification import FactChecker
from explanations import ExplanationGenerator
from trends import TrendDetector

app = FastAPI(
    title="AI Misinformation Detection Agent",
    description="Real-time fact-checking powered by professional sources & AI analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

classifier = FakeNewsClassifier()
trend_detector = TrendDetector()
fact_checker = FactChecker()
explainer = ExplanationGenerator()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
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

@app.post("/verify-claim")
async def verify_claim(request: Request):
    data = await request.json()
    claim = data.get("claim", "")
    if not claim.strip():
        return {"error": "Please provide a claim to verify"}

    return fact_checker.verify_claim(claim)

@app.post("/full-analysis")
async def full_analysis(request: Request):
    data = await request.json()
    claim = data.get("claim", "")
    if not claim.strip():
        return {"error": "Please provide a claim to analyze"}

    classification = classifier.predict(claim)
    verification = fact_checker.verify_claim(claim)
    gfc = verification.get("google_factcheck", {})
    fact_verdict = verification.get("overall_verdict", {}).get("verdict", "")

    primary_predictions = classification.get("individual_predictions", [])
    if primary_predictions and isinstance(primary_predictions, list):
        primary = primary_predictions[0]
        pm_label = primary.get("label") or primary.get("prediction") or ""
        pm_conf = primary.get("confidence", 0.0)
    else:
        pm_label = classification.get("prediction", "")
        pm_conf = classification.get("confidence", 0.0)

    if gfc.get("found_fact_checks", 0) > 0 and fact_verdict in ("fact_checked_true", "verified_true"):
        final_verdict = fact_verdict
        final_confidence = verification["overall_verdict"]["confidence"]
    else:
        if pm_label in ("LABEL_0", "REAL", "TRUE", "POSITIVE"):
            final_verdict = "TRUE"
        elif pm_label in ("LABEL_1", "FAKE", "FALSE", "NEGATIVE"):
            final_verdict = "FALSE"
        else:
            final_verdict = "UNCERTAIN"
        final_confidence = pm_conf

    return {
        "claim": claim,
        "result": {
            "verdict": final_verdict,
            "confidence": round(final_confidence, 3)
        }
    }

@app.post("/add-documents")
async def add_documents(request: Request):
    data = await request.json()
    valid = [d for d in data.get("documents", []) if isinstance(d, str) and d.strip()]
    if not valid:
        return {"error": "No valid documents provided"}
    count = trend_detector.add_documents(valid)
    return {
        "message": f"Added {len(valid)} documents",
        "total_documents": count,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/detect-trends")
async def detect_trends():
    result = trend_detector.detect_trends()
    result["timestamp"] = pd.Timestamp.now().isoformat()
    return result

@app.post("/batch-analysis")
async def batch_analysis(request: Request):
    data = await request.json()
    claims = data.get("claims", [])[:10]
    results = []
    for i, c in enumerate(claims):
        if isinstance(c, str) and c.strip():
            cls = classifier.predict(c)
            ver = fact_checker.verify_claim(c)
            gfc = ver.get("google_factcheck", {})
            fact_verdict = ver.get("overall_verdict", {}).get("verdict", "")

            primary_predictions = cls.get("individual_predictions", [])
            if primary_predictions and isinstance(primary_predictions, list):
                primary = primary_predictions[0]
                pm_label = primary.get("label") or primary.get("prediction") or ""
                pm_conf = primary.get("confidence", 0.0)
            else:
                pm_label = cls.get("prediction", "")
                pm_conf = cls.get("confidence", 0.0)

            if gfc.get("found_fact_checks", 0) > 0 and fact_verdict in ("fact_checked_true", "verified_true"):
                verdict = fact_verdict
                conf = ver["overall_verdict"]["confidence"]
            else:
                if pm_label in ("LABEL_0", "REAL", "TRUE", "POSITIVE"):
                    verdict = "TRUE"
                elif pm_label in ("LABEL_1", "FAKE", "FALSE", "NEGATIVE"):
                    verdict = "FALSE"
                else:
                    verdict = "UNCERTAIN"
                conf = pm_conf
            results.append({"claim_index": i, "claim": c, "verdict": verdict, "confidence": round(conf, 3)})
    return {
        "total_claims": len(claims),
        "processed_claims": len(results),
        "results": results,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    return {
        "system_status": "operational",
        "total_documents": len(trend_detector.documents),
        "trend_detection_ready": len(trend_detector.documents) >= 3,
        "timestamp": pd.Timestamp.now().isoformat()
    }

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

@app.on_event("startup")
async def startup_event():
    print("🚀 AI Misinformation Detection Agent starting up...")
    print("✅ Classifier initialized")
    print("✅ Trend detector initialized")
    print("✅ Fact checker initialized")
    print("✅ Explanation generator initialized")
    print("🌐 Web interface available at http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 AI Misinformation Detection Agent shutting down...")

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
