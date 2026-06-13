import sys
import os
import json
from datetime import datetime

# Windows Unicode fix — agentops emoji issue
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

import agentops
from src.langgraph_pipeline.pipeline import run_inspection
from src.mcp_tools.quality_summarizer import QualitySummarizer
from src.mcp_tools.safety_detector import WorkerSafetyDetector

summarizer      = QualitySummarizer()
safety_detector = WorkerSafetyDetector()

agentops.init(
    api_key=os.getenv("AGENTOPS_API_KEY"),
    tags=["quality-inspection", "yolo-v8", "langgraph"]
)


def inspect_product(image_path: str, batch_id: str = None) -> dict:
    if not batch_id:
        batch_id = f"BTH-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print(f"\n{'='*50}")
    print(f"  Quality Inspection System")
    print(f"  Batch: {batch_id}")
    print(f"  Image: {image_path}")
    print(f"{'='*50}")

    # Run pipeline — first pass
    final_state, app, config = run_inspection(image_path, batch_id)

    # Human-in-loop: REVIEW routing pe terminal se input lo
    if final_state and final_state.get("awaiting_human"):
        print(f"\n  [REVIEW] Low confidence — human input required")
        print(f"  Confidence : {final_state.get('confidence_score')}")
        print(f"  Severity   : {final_state.get('severity_level')}")
        defect_names = [d.class_name for d in (final_state.get('defects_annotated') or [])]
        print(f"  Defects    : {defect_names}")

        while True:
            human_dec = input("\n  Expert decision [PASS / FAIL / REWORK]: ").strip().upper()
            if human_dec in ("PASS", "FAIL", "REWORK"):
                break
            print("  Invalid — type PASS, FAIL, or REWORK")

        # Human decision — same app + config pass karo (same MemorySaver)
        prev_state = dict(final_state)
        # app aur config pehle se hain — nayi instance mat banao
        checkpoint = app.get_state(config)
        current = dict(checkpoint.values) if checkpoint and checkpoint.values else {}
        current["human_decision"] = human_dec
        current["awaiting_human"] = False
        current["verdict"] = human_dec
        app.update_state(config, {
            "human_decision": human_dec,
            "verdict": human_dec,
            "awaiting_human": False
        })
        final_state = {**prev_state, "human_decision": human_dec, "verdict": human_dec, "awaiting_human": False}

    # Build report
    report = {
        "batch_id":       batch_id,
        "image_path":     image_path,
        "timestamp":      datetime.now().isoformat(),
        "model_version":  "yolo-v8n",
        "defects": [
            {
                "defect_id":  d.defect_id,
                "class":      d.class_name,
                "severity":   d.severity,
                "confidence": d.confidence,
                "bbox":       d.bbox,
            }
            for d in (final_state.get("defects_annotated") or [])
        ],
        "overall_severity": final_state.get("severity_level"),
        "confidence_score": final_state.get("confidence_score"),
        "routing_decision": final_state.get("routing_decision"),
        "human_decision":   final_state.get("human_decision"),
        "final_verdict":    final_state.get("verdict"),
    }

    # ── Worker Safety Detection ───────────────────────────
    try:
        safety = safety_detector.detect(image_path)
        report["safety"] = {
            "total_workers":    safety.total_workers,
            "avg_safety_score": safety.avg_safety_score,
            "safety_verdict":   safety.safety_verdict,
            "site_summary":     safety.site_summary,
            "workers": [
                {
                    "worker_id":    w.worker_id,
                    "safety_score": w.safety_score,
                    "ppe":          w.ppe.to_dict(),
                    "missing_critical": w.ppe.missing_critical(),
                }
                for w in safety.workers
            ],
            "annotated_image": safety.annotated_image_path,
        }

        symbol_s = {
            "SAFE":       "[SAFE]",
            "UNSAFE":     "[UNSAFE]",
            "REVIEW":     "[SAFETY-REVIEW]",
            "NO_WORKERS": "[NO WORKERS]"
        }.get(safety.safety_verdict, "[?]")

        print(f"\n   --- Worker Safety (PPE) ---")
        print(f"   {symbol_s}  Site Score: {safety.avg_safety_score}%")
        print(f"   Workers detected : {safety.total_workers}")

        # PPE compliance table
        if safety.site_summary:
            print(f"   {'Gear':<10} {'Compliant':>10} {'Score':>8}")
            print(f"   {'-'*30}")
            for gear, data in safety.site_summary.items():
                critical_mark = " *" if data["critical"] else ""
                print(f"   {gear:<10} {data['compliant']:>4}/{safety.total_workers:<5} {data['compliance_pct']:>6.0f}%{critical_mark}")
            print(f"   (* = critical gear)")

        # Per-worker breakdown
        for w in safety.workers:
            missing = w.ppe.missing_critical()
            status  = "OK" if not missing else f"MISSING: {', '.join(missing)}"
            print(f"   {w.worker_id}: {w.safety_score:.0f}% — {status}")

        print(f"   Annotated: {safety.annotated_image_path}")

    except Exception as e:
        print(f"   [Safety] Skipped: {e}")
        report["safety"] = None

    # OpenAI Quality Summarizer — GPT-4o se natural language report
    try:
        summary = summarizer.summarize(
            batch_id=batch_id,
            verdict=report["final_verdict"] or "UNKNOWN",
            severity=report["overall_severity"] or "NONE",
            confidence=report["confidence_score"] or 0.0,
            defects=report["defects"],
            human_decision=report["human_decision"]
        )
        report["ai_summary"] = {
            "natural_language_summary": summary.natural_language_summary,
            "defect_details":           summary.defect_details,
            "recommended_action":       summary.recommended_action,
            "compliance_status":        summary.compliance_status,
            "risk_level":               summary.risk_level,
        }
    except Exception as e:
        print(f"   [Summarizer] Skipped: {e}")
        report["ai_summary"] = None

    os.makedirs("data/outputs", exist_ok=True)
    report_path = f"data/outputs/{batch_id}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    verdict = report["final_verdict"] or "UNKNOWN"
    symbol = {
        "PASS":   "[PASS]",
        "FAIL":   "[FAIL]",
        "REWORK": "[REWORK]",
        "REVIEW": "[REVIEW]"
    }.get(verdict, "[?]")

    print(f"\n{symbol}  VERDICT: {verdict}")
    print(f"   Defects:    {len(report['defects'])}")
    print(f"   Severity:   {report['overall_severity']}")
    print(f"   Confidence: {report['confidence_score']}")
    print(f"   Report:     {report_path}")

    # AI summary print karo
    if report.get("ai_summary"):
        s = report["ai_summary"]
        print(f"\n   --- AI Summary ---")
        print(f"   {s['natural_language_summary']}")
        print(f"\n   Defect Details : {s['defect_details']}")
        print(f"   Action         : {s['recommended_action']}")
        print(f"   Compliance     : {s['compliance_status']}")
        print(f"   Risk Level     : {s['risk_level']}")

    return report


def batch_inspect(image_dir: str):
    from pathlib import Path

    # Sirf original images — preprocessed/annotated skip
    all_imgs = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    images = [
        p for p in all_imgs
        if "_preprocessed" not in p.name
        and "_annotated" not in p.name
    ]

    if not images:
        print("No original images found in folder.")
        return []

    print(f"Found {len(images)} original image(s) to inspect")
    results = []

    for i, img_path in enumerate(sorted(images)):
        batch_id = f"BTH-{i+1:04d}"
        try:
            result = inspect_product(str(img_path), batch_id)
            results.append(result)
        except Exception as e:
            print(f"Error in {img_path.name}: {e}")

    passed  = sum(1 for r in results if r["final_verdict"] == "PASS")
    failed  = sum(1 for r in results if r["final_verdict"] == "FAIL")
    reviews = sum(1 for r in results if r["final_verdict"] in ("REVIEW", "REWORK"))
    unknown = sum(1 for r in results if r["final_verdict"] not in ("PASS","FAIL","REVIEW","REWORK"))

    print(f"\n{'='*40}")
    print(f"  Batch Summary")
    print(f"  Total:   {len(results)}")
    print(f"  Pass:    {passed}")
    print(f"  Fail:    {failed}")
    print(f"  Review:  {reviews}")
    if unknown:
        print(f"  Unknown: {unknown}")
    print(f"{'='*40}")

    try:
        agentops.end_trace(end_state="Success")
    except Exception:
        pass

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <image_path>")
        print("  python main.py --batch <folder>")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Folder path do: python main.py --batch data/sample_images/")
            sys.exit(1)
        batch_inspect(sys.argv[2])
    else:
        report = inspect_product(sys.argv[1])
        print(json.dumps(report, indent=2))