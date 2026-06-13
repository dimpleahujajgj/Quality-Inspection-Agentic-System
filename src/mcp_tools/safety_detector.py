# src/mcp_tools/safety_detector.py
# ── Complete PPE Detection — 5 gear items ─────────────────
#
# PPE Items detected:
#   1. Helmet      — head region  (weight: 30%)
#   2. Safety Vest — torso region (weight: 25%)
#   3. Gloves      — hands region (weight: 20%)
#   4. Goggles     — eye region   (weight: 15%)
#   5. Safety Boots— feet region  (weight: 10%)
#
# Combined Safety Score = weighted average of all gear
#
# Verdict:
#   score >= 85  → SAFE
#   score >= 60  → REVIEW
#   score < 60   → UNSAFE

import cv2
import os
import base64
import json
from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI
from ultralytics import YOLO


# ── PPE Config ────────────────────────────────────────────
PPE_ITEMS = {
    "helmet":  {"weight": 0.30, "region": "head",   "critical": True},
    "vest":    {"weight": 0.25, "region": "torso",  "critical": True},
    "gloves":  {"weight": 0.20, "region": "hands",  "critical": False},
    "goggles": {"weight": 0.15, "region": "eyes",   "critical": False},
    "boots":   {"weight": 0.10, "region": "feet",   "critical": False},
}

SAFE_THRESHOLD   = 85.0
REVIEW_THRESHOLD = 60.0


@dataclass
class PPEStatus:
    helmet:  bool = False
    vest:    bool = False
    gloves:  bool = False
    goggles: bool = False
    boots:   bool = False
    helmet_conf:  float = 0.0
    vest_conf:    float = 0.0
    gloves_conf:  float = 0.0
    goggles_conf: float = 0.0
    boots_conf:   float = 0.0

    def combined_score(self) -> float:
        """Weighted PPE score — 0 to 100"""
        score = 0.0
        for gear, cfg in PPE_ITEMS.items():
            has_gear = getattr(self, gear)
            score += cfg["weight"] * (100 if has_gear else 0)
        return round(score, 1)

    def missing_critical(self) -> List[str]:
        """Critical gear jo nahi hai"""
        return [
            gear for gear, cfg in PPE_ITEMS.items()
            if cfg["critical"] and not getattr(self, gear)
        ]

    def to_dict(self) -> dict:
        return {
            "helmet":  {"detected": self.helmet,  "confidence": self.helmet_conf},
            "vest":    {"detected": self.vest,    "confidence": self.vest_conf},
            "gloves":  {"detected": self.gloves,  "confidence": self.gloves_conf},
            "goggles": {"detected": self.goggles, "confidence": self.goggles_conf},
            "boots":   {"detected": self.boots,   "confidence": self.boots_conf},
        }


@dataclass
class WorkerDetection:
    worker_id: str
    bbox: List[int]
    worker_confidence: float
    ppe: PPEStatus = field(default_factory=PPEStatus)

    @property
    def safety_score(self) -> float:
        return self.ppe.combined_score()


@dataclass
class SafetyResult:
    total_workers: int
    avg_safety_score: float
    safety_verdict: str
    workers: List[WorkerDetection]
    annotated_image_path: str
    site_summary: dict


# ── Main Detector ─────────────────────────────────────────
class WorkerSafetyDetector:

    PERSON_CLASS_ID = 0

    def __init__(self):
        self.person_model = YOLO("yolov8n.pt")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def detect(self, image_path: str) -> SafetyResult:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image load nahi hui: {image_path}")

        H, W = img.shape[:2]

        # Step 1: Person detect karo
        results = self.person_model(img, conf=0.25, verbose=False)[0]
        persons = []
        for box in results.boxes:
            if int(box.cls[0]) != self.PERSON_CLASS_ID:
                continue
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            persons.append({"bbox":[x1,y1,x2-x1,y2-y1], "conf":conf,
                            "x1":x1,"y1":y1,"x2":x2,"y2":y2})

        # Step 2: Har worker ka PPE check karo
        workers = []
        for i, p in enumerate(persons):
            ppe = self._check_full_ppe(img, p["x1"], p["y1"], p["x2"], p["y2"], H, W)
            workers.append(WorkerDetection(
                worker_id=f"W-{i+1:03d}",
                bbox=p["bbox"],
                worker_confidence=p["conf"],
                ppe=ppe
            ))

        # Step 3: Site-level safety score
        if not workers:
            avg_score = 100.0
            verdict   = "NO_WORKERS"
        else:
            avg_score = sum(w.safety_score for w in workers) / len(workers)
            avg_score = round(avg_score, 1)
            if avg_score >= SAFE_THRESHOLD:
                verdict = "SAFE"
            elif avg_score >= REVIEW_THRESHOLD:
                verdict = "REVIEW"
            else:
                verdict = "UNSAFE"

        # Step 4: Site summary
        site_summary = self._site_summary(workers)

        # Step 5: Annotate
        ann_path = self._annotate(img, workers, avg_score, verdict, image_path)

        return SafetyResult(
            total_workers=len(workers),
            avg_safety_score=avg_score,
            safety_verdict=verdict,
            workers=workers,
            annotated_image_path=ann_path,
            site_summary=site_summary
        )

    def _check_full_ppe(self, img, x1, y1, x2, y2, H, W) -> PPEStatus:
        """
        GPT-4o Vision se poore person ka PPE check karo.
        Ek hi API call mein sab 5 items check hote hain.
        """
        # Full person crop
        person_crop = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
        if person_crop.size == 0:
            return PPEStatus()

        try:
            _, buf = cv2.imencode(".jpg", person_crop)
            b64 = base64.b64encode(buf).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": """Analyze this worker image for PPE (Personal Protective Equipment).

Check for each item and respond ONLY with this JSON (no extra text):
{
  "helmet":  {"present": true/false, "confidence": 0.0-1.0},
  "vest":    {"present": true/false, "confidence": 0.0-1.0},
  "gloves":  {"present": true/false, "confidence": 0.0-1.0},
  "goggles": {"present": true/false, "confidence": 0.0-1.0},
  "boots":   {"present": true/false, "confidence": 0.0-1.0}
}

Rules:
- helmet: hard hat or safety helmet on head
- vest: high-visibility or safety vest on torso
- gloves: safety/work gloves on hands
- goggles: safety goggles or glasses on eyes
- boots: safety boots or steel-toe shoes on feet
- If body part not visible, set present: false, confidence: 0.5"""
                        }
                    ]
                }],
                max_tokens=200
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            parsed = json.loads(raw)

            return PPEStatus(
                helmet=parsed["helmet"]["present"],
                vest=parsed["vest"]["present"],
                gloves=parsed["gloves"]["present"],
                goggles=parsed["goggles"]["present"],
                boots=parsed["boots"]["present"],
                helmet_conf=parsed["helmet"]["confidence"],
                vest_conf=parsed["vest"]["confidence"],
                gloves_conf=parsed["gloves"]["confidence"],
                goggles_conf=parsed["goggles"]["confidence"],
                boots_conf=parsed["boots"]["confidence"],
            )

        except Exception as e:
            print(f"   [PPE Check] GPT-4o failed: {e} — using defaults")
            return PPEStatus()

    def _site_summary(self, workers: List[WorkerDetection]) -> dict:
        """Site-level PPE compliance summary"""
        if not workers:
            return {}

        total = len(workers)
        summary = {}
        for gear in PPE_ITEMS:
            count = sum(1 for w in workers if getattr(w.ppe, gear))
            summary[gear] = {
                "compliant":     count,
                "non_compliant": total - count,
                "compliance_pct": round(count/total*100, 1),
                "weight":         PPE_ITEMS[gear]["weight"],
                "critical":       PPE_ITEMS[gear]["critical"],
            }
        return summary

    def _annotate(self, img, workers, avg_score, verdict, image_path) -> str:
        """Color-coded annotation with PPE breakdown per worker"""
        annotated = img.copy()

        VERDICT_COLORS = {
            "SAFE":       (0, 200, 80),
            "UNSAFE":     (0, 0, 220),
            "REVIEW":     (0, 140, 255),
            "NO_WORKERS": (128, 128, 128),
        }
        v_color = VERDICT_COLORS.get(verdict, (128,128,128))

        for w in workers:
            x, y, ww, hh = w.bbox
            # Box color — green if safe, red if not
            box_color = (0,200,80) if w.safety_score >= SAFE_THRESHOLD else (0,0,220)
            cv2.rectangle(annotated, (x,y), (x+ww,y+hh), box_color, 2)

            # Worker ID + score
            main_label = f"{w.worker_id} Score:{w.safety_score:.0f}%"
            (lw,lh),_ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x,y-lh-6), (x+lw+4,y), box_color, -1)
            cv2.putText(annotated, main_label, (x+2,y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # PPE breakdown below box
            gear_icons = {
                "helmet":  "H", "vest":  "V", "gloves": "G",
                "goggles": "O", "boots": "B"
            }
            x_off = x
            for gear, icon in gear_icons.items():
                has = getattr(w.ppe, gear)
                c = (0,200,80) if has else (0,0,220)
                cv2.rectangle(annotated, (x_off, y+hh+2), (x_off+18, y+hh+18), c, -1)
                cv2.putText(annotated, icon, (x_off+3, y+hh+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                x_off += 20

        # Top overlay — site safety score
        overlay = f"Site Safety: {avg_score:.1f}% | {verdict}"
        cv2.rectangle(annotated, (8,8), (420,50), (0,0,0), -1)
        cv2.putText(annotated, overlay, (14,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, v_color, 2)

        out_path = image_path.replace(".", "_ppe_annotated.")
        cv2.imwrite(out_path, annotated)
        return out_path