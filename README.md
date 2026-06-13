# Quality Inspection Agent 🏭

An end-to-end **agentic AI system** for automated manufacturing quality inspection and worker safety compliance — built with YOLOv8, LangGraph, MCP, and GPT-4o.

---

## Architecture

```
Camera Input
     ↓
[MCP Preprocessor] → Image resize + normalize
     ↓
[MCP Anomaly Detector] → YOLOv8 defect detection
     ↓
[MCP Annotator] → Color-coded bounding boxes
     ↓
[Rules Engine] → Deterministic severity classification
     ↓
[Confidence Evaluator] → Score threshold check
     ↓
     ├── High confidence → Direct PASS / FAIL
     └── Low confidence → Human-in-Loop Review
                              ↓
                         Expert Decision
                              ↓
[MCP Quality Summarizer] → GPT-4o natural language report
     ↓
[Worker Safety Detector] → PPE compliance scoring
     ↓
[AgentOps] → Full session observability
```

---

## Key Design Decisions

### Why deterministic rules engine — not LLM — for decisioning?
Manufacturing requires **100% reproducible outputs** for regulatory compliance. LLMs are non-deterministic — same input can produce different outputs across runs. The rules engine guarantees: same defect → same severity → same verdict, always. LLM (GPT-4o) is used **only** for natural language summarization, not for any decision-making.

### Why LangGraph over CrewAI?
This pipeline requires **stateful orchestration** — conditional routing, human-in-loop pause/resume, retry logic with loop guards, and full state persistence across sessions. LangGraph's StateGraph is purpose-built for this. CrewAI is better suited for sequential agent workflows.

### Why MCP for tool layer?
Model Context Protocol provides a **standardized interface** for tool exposure — structured JSON in, structured JSON out. Each tool (preprocessor, detector, annotator, summarizer) is independently swappable without touching the orchestration layer.

---

## Features

- **Defect Detection** — YOLOv8 detects cracks, scratches, dents, stains with confidence scoring
- **Severity Classification** — Deterministic rules engine (CRITICAL / MODERATE / MINOR)
- **Confidence-based Routing** — High confidence → auto verdict | Low confidence → human review
- **Human-in-Loop** — LangGraph `interrupt_before` pauses workflow for expert decision
- **Worker Safety** — Full PPE detection (helmet, vest, gloves, goggles, boots) via GPT-4o Vision
- **AI Summary** — GPT-4o generates natural language defect reports with specific flaw details
- **Audit Trail** — Complete JSON reports with batch metadata, timestamps, model version
- **AgentOps Monitoring** — Full session replay, node timing, routing decisions tracked

---

## Tech Stack

| Component | Technology |
|---|---|
| Defect Detection | YOLOv8n (Ultralytics) |
| Orchestration | LangGraph (StateGraph) |
| Tool Interface | MCP (Model Context Protocol) |
| LLM | GPT-4o-mini (OpenAI) |
| Observability | AgentOps |
| Image Processing | OpenCV |

---

## Project Structure

```
quality-inspection-agent/
├── main.py                          # Entry point
├── .env                             # API keys (not committed)
├── src/
│   ├── mcp_tools/
│   │   ├── yolo_detector.py         # YOLO inference + preprocessor + annotator
│   │   ├── quality_summarizer.py    # GPT-4o defect summary
│   │   └── safety_detector.py      # Full PPE detection
│   └── langgraph_pipeline/
│       └── pipeline.py              # LangGraph StateGraph + rules engine
├── data/
│   ├── sample_images/               # Input images
│   └── outputs/                     # JSON reports + annotated images
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/quality-inspection-agent.git
cd quality-inspection-agent

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Add your API keys in .env
```

---

## Environment Variables

```env
AGENTOPS_API_KEY=your_agentops_key
OPENAI_API_KEY=your_openai_key
CONFIDENCE_THRESHOLD=0.80
MAX_RETRY_COUNT=3
```

---

## Usage

```bash
# Single image inspection
python main.py data/sample_images/product_001.jpg

# Batch inspection
python main.py --batch data/sample_images/
```

### Sample Output

```
==================================================
  Quality Inspection System
  Batch: BTH-0001
  Image: data/sample_images/product_001.jpg
==================================================

   --- Worker Safety (PPE) ---
   [UNSAFE]  Site Score: 0.0%
   Workers detected : 1
   Gear       Compliant    Score
   helmet        0/1          0% *
   vest          0/1          0% *

[PASS]  VERDICT: PASS
   Defects:    1
   Severity:   MINOR
   Confidence: 0.9223

   --- AI Summary ---
   Batch BTH-0001 inspection identified one minor surface scratch with 92%
   confidence. The scratch poses no structural risk and meets MINOR threshold.

   Defect Details : Single scratch at bbox [214,112,68,54], severity MINOR
   Action         : Log for trend analysis. Monitor future batches.
   Compliance     : COMPLIANT
   Risk Level     : LOW
```

---

## Confidence Score Formula

```
Final Confidence = Objectness Score × Class Probability
                 = "Is something here?" × "What is it?"

>= 0.80  →  High confidence → Direct verdict (PASS/FAIL)
 < 0.80  →  Low confidence  → Human review triggered
```

---

## PPE Scoring Weights

| Gear | Weight | Critical |
|---|---|---|
| Helmet | 30% | Yes |
| Safety Vest | 25% | Yes |
| Gloves | 20% | No |
| Goggles | 15% | No |
| Safety Boots | 10% | No |

Combined score ≥ 85% → SAFE | ≥ 60% → REVIEW | < 60% → UNSAFE

---

## AgentOps Dashboard

Every inspection session is tracked at [app.agentops.ai](https://app.agentops.ai):
- Node-level execution timing
- Routing decisions (PASS/FAIL/REVIEW)
- Human-in-loop events
- LLM call tracking

---

## Future Improvements

- Fine-tune YOLOv8 on NEU Metal Surface Defects dataset for higher confidence
- Add dedicated helmet detection YOLO model (replace GPT-4o Vision for speed)
- Active learning loop — human corrections feed back into model retraining
- Model drift detection — alert when confidence distribution shifts from baseline
- REST API wrapper for production deployment

---

## Author

**Dimple**  
Built for manufacturing quality automation with agentic AI.