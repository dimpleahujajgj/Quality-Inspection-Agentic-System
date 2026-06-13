# src/mcp_tools/quality_summarizer.py
# ── OpenAI GPT-4o Quality Summarizer — MCP tool ──────────
# 
# KEY DESIGN DECISION (interview mein yeh bolna):
# LLM sirf SUMMARIZATION ke liye use kiya — decisioning ke liye NAHI.
# Severity aur verdict rules engine se aata hai (deterministic).
# GPT-4o sirf human-readable report banata hai.

from openai import OpenAI
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QualitySummary:
    batch_id: str
    verdict: str
    natural_language_summary: str
    defect_details: str
    recommended_action: str
    compliance_status: str
    risk_level: str


class QualitySummarizer:
    """
    MCP Quality Summarizer tool.
    GPT-4o se defect data ko human-readable report mein convert karta hai.
    
    NOTE: LLM yahan sirf language generation ke liye hai.
    Verdict, severity, routing — sab pehle se rules engine ne decide kar diya.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # fast + cheap, interview ke liye perfect

    def summarize(
        self,
        batch_id: str,
        verdict: str,
        severity: str,
        confidence: float,
        defects: list,
        human_decision: Optional[str] = None
    ) -> QualitySummary:
        """
        Inspection results lo → GPT-4o se natural language summary banao.
        """

        # Defect details string banao
        if defects:
            defect_lines = "\n".join([
                f"- {d.get('class', 'unknown')} | severity: {d.get('severity', 'N/A')} | confidence: {d.get('confidence', 0):.2f} | bbox: {d.get('bbox', [])}"
                for d in defects
            ])
        else:
            defect_lines = "No defects detected."

        human_note = f"\nHuman expert reviewed and decided: {human_decision}" if human_decision else ""

        prompt = f"""You are a quality inspection AI assistant for a manufacturing plant.

Analyze the following inspection results and generate a detailed professional report.

INSPECTION DATA:
- Batch ID: {batch_id}
- Final Verdict: {verdict}
- Overall Severity: {severity}
- Model Confidence: {confidence:.2%}
- Defects Found:
{defect_lines}{human_note}

Generate a JSON response with exactly these fields:
{{
  "natural_language_summary": "3-4 sentence summary that MUST include: (1) what defects were found and where, (2) severity of each defect, (3) why verdict was given, (4) impact on product quality",
  "defect_details": "specific description of each flaw found — type, location, severity, and risk it poses",
  "recommended_action": "specific actionable next step — mention which defects need attention and how to fix them",
  "compliance_status": "one of: COMPLIANT / NON-COMPLIANT / REQUIRES-REVIEW",
  "risk_level": "one of: LOW / MEDIUM / HIGH / CRITICAL"
}}

Rules:
- ALWAYS mention specific defect types (crack, scratch, dent, stain) in the summary
- If no defects found, say product is clean and defect-free
- If human expert overrode the decision, mention that too
- Be specific — never say "a defect was found", say "a minor surface scratch was detected"
- compliance_status must match the verdict
- Respond ONLY with valid JSON, no extra text
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a manufacturing QA report generator. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # low temp = consistent outputs
            max_tokens=300
        )

        import json
        raw = response.choices[0].message.content.strip()
        # Clean JSON agar backticks aaye
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        return QualitySummary(
            batch_id=batch_id,
            verdict=verdict,
            natural_language_summary=parsed["natural_language_summary"],
            defect_details=parsed["defect_details"],
            recommended_action=parsed["recommended_action"],
            compliance_status=parsed["compliance_status"],
            risk_level=parsed["risk_level"]
        )