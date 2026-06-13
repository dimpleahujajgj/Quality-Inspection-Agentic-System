from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.mcp_tools.yolo_detector import (
    YOLODefectDetector, ImagePreprocessor,
    DefectAnnotator, DefectPrediction
)

CONFIDENCE_THRESHOLD = 0.005  # COCO model pe low — demo ke liye
MAX_RETRY = 3

preprocessor = ImagePreprocessor()
detector     = YOLODefectDetector(model_path="yolov8n.pt")
annotator    = DefectAnnotator()


class SeverityRulesEngine:
    DEFECT_SEVERITY_MAP = {
        "crack": "CRITICAL", "dent": "MODERATE",
        "scratch": "MINOR",  "stain": "MINOR",
    }
    def evaluate(self, defects):
        if not defects:
            return "NONE", []
        for d in defects:
            d.severity = self.DEFECT_SEVERITY_MAP.get(d.class_name, "MINOR")
        rank = {"CRITICAL":4,"MODERATE":3,"MINOR":2,"BORDERLINE":1,"NONE":0}
        worst = max(defects, key=lambda d: rank.get(d.severity, 0))
        if sum(1 for d in defects if d.severity=="CRITICAL") > 0:
            return "CRITICAL", defects
        if sum(1 for d in defects if d.severity=="MODERATE") > 1:
            return "BORDERLINE", defects
        if sum(1 for d in defects if d.severity=="MINOR") > 3:
            return "MODERATE", defects
        return worst.severity, defects

rules_engine = SeverityRulesEngine()


class InspectionState(TypedDict):
    image_path: str
    batch_id: str
    preprocessed_image: Optional[str]
    detection_result: Optional[Any]
    severity_level: Optional[str]
    defects_annotated: Optional[List[Any]]
    confidence_score: Optional[float]
    routing_decision: Optional[str]
    human_decision: Optional[str]
    awaiting_human: bool
    verdict: Optional[str]
    retry_count: int
    error: Optional[str]


def node_preprocess(state):
    try:
        clean = preprocessor.preprocess(state["image_path"])
        return {**state, "preprocessed_image": clean, "error": None}
    except Exception as e:
        return {**state, "error": str(e)}

def node_detect(state):
    try:
        result = detector.detect(state["preprocessed_image"])
        return {**state, "detection_result": result, "error": None}
    except Exception as e:
        return {**state, "error": str(e)}

def node_rules_engine(state):
    defects = state["detection_result"].defects
    severity, annotated = rules_engine.evaluate(defects)
    ann_path = annotator.annotate(state["preprocessed_image"], annotated)
    return {**state, "severity_level": severity, "defects_annotated": annotated, "preprocessed_image": ann_path}

def node_confidence_eval(state):
    defects = state.get("defects_annotated") or []
    if not defects:
        return {**state, "confidence_score": 1.0, "routing_decision": "PASS"}
    min_conf = min(d.confidence for d in defects)
    severity = state.get("severity_level", "NONE")
    if state.get("retry_count", 0) >= MAX_RETRY or min_conf < CONFIDENCE_THRESHOLD:
        decision = "REVIEW"
    elif severity in ("CRITICAL", "MODERATE", "BORDERLINE"):
        decision = "FAIL"
    else:
        decision = "PASS"
    return {**state, "confidence_score": round(min_conf, 4), "routing_decision": decision}

def node_human_review(state):
    return {**state, "awaiting_human": True}

def node_final_verdict(state):
    verdict = state.get("human_decision") or state.get("routing_decision") or "UNKNOWN"
    return {**state, "verdict": verdict, "awaiting_human": False}

def route_after_confidence(state):
    return "human_review" if state.get("routing_decision") == "REVIEW" else "final_verdict"


def build_inspection_graph():
    g = StateGraph(InspectionState)
    g.add_node("preprocess",      node_preprocess)
    g.add_node("detect",          node_detect)
    g.add_node("rules_engine",    node_rules_engine)
    g.add_node("confidence_eval", node_confidence_eval)
    g.add_node("human_review",    node_human_review)
    g.add_node("final_verdict",   node_final_verdict)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess",   "detect")
    g.add_edge("detect",       "rules_engine")
    g.add_edge("rules_engine", "confidence_eval")
    g.add_conditional_edges("confidence_eval", route_after_confidence,
        {"human_review": "human_review", "final_verdict": "final_verdict"})
    g.add_edge("human_review", "final_verdict")
    g.add_edge("final_verdict", END)
    # Har image ke liye fresh MemorySaver — cross-contamination nahi hoga
    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"]
    )


def run_inspection(image_path: str, batch_id: str, human_decision: str = None):
    """
    Har call pe fresh graph + fresh MemorySaver.
    Isliye har image independent hai — koi cross-contamination nahi.
    """
    app = build_inspection_graph()  # Fresh graph har baar
    config = {"configurable": {"thread_id": batch_id}}  # Unique thread_id per image

    if human_decision is None:
        initial: InspectionState = {
            "image_path": image_path, "batch_id": batch_id,
            "preprocessed_image": None, "detection_result": None,
            "severity_level": None, "defects_annotated": None,
            "confidence_score": None, "routing_decision": None,
            "human_decision": None, "awaiting_human": False,
            "verdict": None, "retry_count": 0, "error": None,
        }
        last = None
        for s in app.stream(initial, config=config, stream_mode="values"):
            last = s

        if last and last.get("routing_decision") == "REVIEW":
            last = {**last, "awaiting_human": True}

        return last, app, config

    else:
        # Human decision — same app instance use karo (same MemorySaver)
        checkpoint = app.get_state(config)
        current = dict(checkpoint.values) if checkpoint and checkpoint.values else {}
        current["human_decision"] = human_decision
        current["awaiting_human"] = False
        current["verdict"] = human_decision
        app.update_state(config, {
            "human_decision": human_decision,
            "verdict": human_decision,
            "awaiting_human": False
        })
        return current, app, config