# api.py — FastAPI REST API
# ── Endpoints ─────────────────────────────────────────────
# POST /inspect        → single image inspect karo
# POST /inspect/batch  → multiple images
# GET  /health         → server status check
# GET  /report/{id}    → batch report lo

import os, json, shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import agentops
# disable_instrumentations — LangGraph circular import fix
agentops.init(
    api_key=os.getenv("AGENTOPS_API_KEY"),
    tags=["quality-inspection", "api", "render"],
    instrument_llm_calls=False  # circular import avoid karo
)

from src.langgraph_pipeline.pipeline import run_inspection
from src.mcp_tools.quality_summarizer import QualitySummarizer
from src.mcp_tools.safety_detector import WorkerSafetyDetector

app = FastAPI(
    title="Quality Inspection Agent API",
    description="Agentic AI system for manufacturing quality inspection + worker safety",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer      = QualitySummarizer()
safety_detector = WorkerSafetyDetector()

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

# In-memory store for pending human reviews
pending_reviews: dict = {}


# ── Human review endpoint ─────────────────────────────────
@app.post("/review/{batch_id}")
def submit_human_review(batch_id: str, decision: str):
    """
    Human expert ka decision submit karo.
    decision: PASS / FAIL / REWORK
    """
    if decision not in ("PASS", "FAIL", "REWORK"):
        raise HTTPException(status_code=400, detail="decision must be PASS, FAIL, or REWORK")

    if batch_id not in pending_reviews:
        raise HTTPException(status_code=404, detail="No pending review for this batch")

    stored   = pending_reviews.pop(batch_id)
    prev     = stored["state"]
    app_g    = stored["app"]
    config   = stored["config"]

    # Human decision inject karo
    checkpoint = app_g.get_state(config)
    current    = dict(checkpoint.values) if checkpoint and checkpoint.values else {}
    current["human_decision"] = decision
    current["verdict"]        = decision
    current["awaiting_human"] = False
    app_g.update_state(config, {
        "human_decision": decision,
        "verdict":        decision,
        "awaiting_human": False
    })

    # Updated report save karo
    report_path = f"data/outputs/{batch_id}_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        report["final_verdict"]  = decision
        report["human_decision"] = decision
        report["awaiting_human"] = False
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    return {"batch_id": batch_id, "human_decision": decision, "verdict": decision}


# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ── Single image inspect ──────────────────────────────────
@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    # Save uploaded image
    batch_id   = f"BTH-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    image_path = f"data/uploads/{batch_id}_{file.filename}"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Run pipeline
        final_state, app_graph, config = run_inspection(image_path, batch_id)

        # Build report
        defects = final_state.get("defects_annotated") or []
        report  = {
            "batch_id":        batch_id,
            "timestamp":       datetime.now().isoformat(),
            "final_verdict":   final_state.get("verdict") or final_state.get("routing_decision"),
            "overall_severity": final_state.get("severity_level"),
            "confidence_score": final_state.get("confidence_score"),
            "defects": [
                {
                    "defect_id":  d.defect_id,
                    "class":      d.class_name,
                    "severity":   d.severity,
                    "confidence": d.confidence,
                    "bbox":       d.bbox,
                }
                for d in defects
            ],
            "awaiting_human": final_state.get("awaiting_human", False),
        }

        # Safety check
        try:
            safety = safety_detector.detect(image_path)
            report["safety"] = {
                "total_workers":    safety.total_workers,
                "avg_safety_score": safety.avg_safety_score,
                "safety_verdict":   safety.safety_verdict,
                "site_summary":     safety.site_summary,
            }
        except Exception as e:
            report["safety"] = {"error": str(e)}

        # AI Summary
        try:
            summary = summarizer.summarize(
                batch_id=batch_id,
                verdict=report["final_verdict"] or "UNKNOWN",
                severity=report["overall_severity"] or "NONE",
                confidence=report["confidence_score"] or 0.0,
                defects=report["defects"],
            )
            report["ai_summary"] = {
                "natural_language_summary": summary.natural_language_summary,
                "defect_details":           summary.defect_details,
                "recommended_action":       summary.recommended_action,
                "compliance_status":        summary.compliance_status,
                "risk_level":               summary.risk_level,
            }
        except Exception as e:
            report["ai_summary"] = {"error": str(e)}

        # Agar REVIEW state mein hai — partial report return karo
        # UI pe "awaiting_human: true" dikhega
        # User /review/{batch_id} endpoint se decision dega
        if report.get("awaiting_human"):
            # State save karo memory mein
            pending_reviews[batch_id] = {
                "state": final_state,
                "app":   app_graph,
                "config": config
            }

        # Save report
        report_path = f"data/outputs/{batch_id}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Get saved report ──────────────────────────────────────
@app.get("/report/{batch_id}")
def get_report(batch_id: str):
    path = f"data/outputs/{batch_id}_report.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(path) as f:
        return json.load(f)


# ── Run locally ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)