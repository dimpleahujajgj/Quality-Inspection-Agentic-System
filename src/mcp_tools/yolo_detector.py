# src/mcp_tools/yolo_detector.py

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import agentops

# ── Data models ──────────────────────────────────────────
@dataclass
class DefectPrediction:
    defect_id: str
    class_name: str
    confidence: float
    bbox: List[int]
    severity: Optional[str] = None

@dataclass
class DetectionResult:
    image_path: str
    defects: List[DefectPrediction]
    model_version: str
    inference_ms: float
    raw_box_count: int
    final_box_count: int


# ── YOLO Detector ─────────────────────────────────────────
class YOLODefectDetector:
    # YOLOv8n pretrained COCO classes — manufacturing context mein map karo
    CLASS_NAMES = {
        0: "scratch",   1: "scratch",   2: "stain",    3: "stain",
        4: "dent",      5: "dent",      6: "crack",    7: "crack",
        8: "scratch",   9: "stain",    10: "dent",    11: "crack",
        12: "scratch", 13: "stain",    14: "dent",    15: "crack",
        16: "scratch", 17: "stain",    18: "dent",    19: "crack",
        20: "scratch", 21: "stain",    22: "dent",    23: "crack",
        24: "scratch", 25: "stain",    26: "dent",    27: "crack",
        28: "scratch", 29: "stain",    30: "dent",    31: "crack",
        32: "scratch", 33: "stain",    34: "dent",    35: "crack",
        36: "scratch", 37: "stain",    38: "dent",    39: "crack",
        40: "dent",    41: "crack",    42: "scratch",  43: "stain",
        44: "dent",    45: "crack",    46: "scratch",  47: "stain",
        48: "dent",    49: "crack",    50: "scratch",  51: "stain",
        52: "dent",    53: "crack",    54: "scratch",  55: "stain",
        56: "dent",    57: "crack",    58: "scratch",  59: "stain",
        60: "dent",    61: "crack",    62: "scratch",  63: "stain",
        64: "dent",    65: "crack",    66: "scratch",  67: "stain",
        68: "dent",    69: "crack",    70: "scratch",  71: "stain",
        72: "dent",    73: "crack",    74: "scratch",  75: "stain",
        76: "dent",    77: "crack",    78: "scratch",  79: "stain",
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold  # 0.01 = detect almost everything
        self.model_version = Path(model_path).stem

    def detect(self, image_path: str) -> DetectionResult:
        import time

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image load nahi hui: {image_path}")

        start = time.time()
        results = self.model(img, conf=self.conf_threshold, iou=0.5, verbose=False)
        inference_ms = (time.time() - start) * 1000

        result = results[0]
        final_boxes = result.boxes
        final_count = len(final_boxes)

        defects = []
        for i, box in enumerate(final_boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2 - x1, y2 - y1]
            cls_id = int(box.cls[0])
            cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            defects.append(DefectPrediction(
                defect_id=f"DEF-{i+1:03d}",
                class_name=cls_name,
                confidence=round(conf, 4),
                bbox=bbox
            ))

        # AgentOps — manually log as an event (new API)
        agentops.record(agentops.ActionEvent(
            action_type="yolo_inference",
            params={"image_path": image_path},
            returns={"defect_count": final_count, "inference_ms": round(inference_ms, 1)}
        ))

        return DetectionResult(
            image_path=image_path,
            defects=defects,
            model_version=self.model_version,
            inference_ms=round(inference_ms, 1),
            raw_box_count=final_count,
            final_box_count=final_count
        )


# ── Preprocessor ─────────────────────────────────────────
class ImagePreprocessor:
    TARGET_SIZE = (640, 640)

    def preprocess(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image nahi mili: {image_path}")

        h, w = img.shape[:2]
        scale = min(self.TARGET_SIZE[0]/w, self.TARGET_SIZE[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.full((*self.TARGET_SIZE, 3), 114, dtype=np.uint8)
        pad_x = (self.TARGET_SIZE[0] - new_w) // 2
        pad_y = (self.TARGET_SIZE[1] - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        out_path = image_path.replace(".", "_preprocessed.")
        cv2.imwrite(out_path, canvas)

        agentops.record(agentops.ActionEvent(
            action_type="image_preprocess",
            params={"input": image_path},
            returns={"output": out_path}
        ))
        return out_path


# ── Annotator ─────────────────────────────────────────────
class DefectAnnotator:
    SEVERITY_COLORS = {
        "CRITICAL":   (0, 0, 220),
        "MODERATE":   (0, 140, 255),
        "MINOR":      (0, 200, 80),
        "BORDERLINE": (0, 200, 220),
    }

    def annotate(self, image_path: str, defects: List[DefectPrediction]) -> str:
        img = cv2.imread(image_path)

        for d in defects:
            sev = d.severity or "MINOR"
            color = self.SEVERITY_COLORS.get(sev, (128, 128, 128))
            x, y, w, h = d.bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"{d.class_name} {sev} {d.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x, y-lh-6), (x+lw+4, y), color, -1)
            cv2.putText(img, label, (x+2, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        out_path = image_path.replace(".", "_annotated.")
        cv2.imwrite(out_path, img)

        agentops.record(agentops.ActionEvent(
            action_type="annotate_image",
            params={"defect_count": len(defects)},
            returns={"annotated_path": out_path}
        ))
        return out_path