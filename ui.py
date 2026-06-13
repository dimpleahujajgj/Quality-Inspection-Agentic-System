# ui.py — Streamlit Web UI
# ── Run: streamlit run ui.py ──────────────────────────────

import streamlit as st
import requests
import json
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Quality Inspection Agent",
    page_icon="🏭",
    layout="wide"
)

# ── API URL — local ya Render ─────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Header ────────────────────────────────────────────────
st.title("🏭 Quality Inspection Agent")
st.markdown("**Agentic AI system for manufacturing quality inspection + worker safety**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("System Info")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: Online")
        st.caption(f"Last check: {health['timestamp'][:19]}")
    except:
        st.error("API: Offline — start api.py first")

    st.divider()
    st.markdown("**Pipeline:**")
    st.markdown("- YOLOv8 defect detection")
    st.markdown("- LangGraph orchestration")
    st.markdown("- GPT-4o quality summary")
    st.markdown("- PPE worker safety")
    st.markdown("- AgentOps monitoring")

# ── Main area ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Product or worker image upload karo",
        type=["jpg", "jpeg", "png"],
        help="Manufacturing product image ya worker image"
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

        if st.button("Run Inspection", type="primary", use_container_width=True):
            with st.spinner("Pipeline chal rahi hai..."):
                try:
                    response = requests.post(
                        f"{API_URL}/inspect",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=120
                    )
                    result = response.json()
                    st.session_state["result"] = result
                    st.success("Inspection complete!")
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("Results")

    if "result" in st.session_state:
        r = st.session_state["result"]

        # ── Verdict banner ────────────────────────────────
        verdict = r.get("final_verdict", "UNKNOWN")
        colors  = {"PASS":"green", "FAIL":"red", "REVIEW":"orange", "UNKNOWN":"gray"}
        color   = colors.get(verdict, "gray")
        st.markdown(f"### :{color}[{verdict}]")

        # ── Metrics ───────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Defects",    len(r.get("defects", [])))
        m2.metric("Severity",   r.get("overall_severity", "—"))
        m3.metric("Confidence", f"{(r.get('confidence_score') or 0)*100:.1f}%")

        # ── Defects table ─────────────────────────────────
        if r.get("defects"):
            st.markdown("**Defects detected:**")
            for d in r["defects"]:
                sev_color = {"CRITICAL":"red","MODERATE":"orange","MINOR":"green"}.get(d["severity"],"gray")
                st.markdown(f"- `{d['defect_id']}` — **{d['class']}** | :{sev_color}[{d['severity']}] | conf: {d['confidence']:.2f}")

        st.divider()

        # ── Safety section ────────────────────────────────
        if r.get("safety") and not r["safety"].get("error"):
            s = r["safety"]
            sv = s.get("safety_verdict", "—")
            sv_color = {"SAFE":"green","UNSAFE":"red","REVIEW":"orange"}.get(sv,"gray")
            st.markdown(f"**Worker Safety:** :{sv_color}[{sv}] — Score: {s.get('avg_safety_score', 0):.1f}%")

            if s.get("site_summary"):
                for gear, data in s["site_summary"].items():
                    pct = data["compliance_pct"]
                    icon = "✅" if pct == 100 else "❌"
                    critical = " ⚠" if data["critical"] else ""
                    st.markdown(f"  {icon} **{gear}**: {pct:.0f}%{critical}")

        st.divider()

        # ── AI Summary ────────────────────────────────────
        if r.get("ai_summary") and not r["ai_summary"].get("error"):
            ai = r["ai_summary"]
            st.markdown("**AI Summary:**")
            st.info(ai.get("natural_language_summary", ""))

            if ai.get("defect_details"):
                st.markdown(f"**Defect details:** {ai['defect_details']}")

            c1, c2 = st.columns(2)
            c1.markdown(f"**Compliance:** `{ai.get('compliance_status','—')}`")
            c2.markdown(f"**Risk:** `{ai.get('risk_level','—')}`")

            st.markdown(f"**Action:** {ai.get('recommended_action','—')}")

        # ── Human Review section ──────────────────────────
        if r.get("awaiting_human"):
            st.divider()
            st.warning("Low confidence — Human review required!")
            st.markdown(f"**Confidence:** {(r.get('confidence_score') or 0)*100:.1f}%")
            st.markdown(f"**Severity:** {r.get('overall_severity')}")

            col_p, col_f, col_r = st.columns(3)
            batch_id = r.get("batch_id")

            if col_p.button("PASS", use_container_width=True, type="primary"):
                res = requests.post(f"{API_URL}/review/{batch_id}?decision=PASS")
                st.success("Decision: PASS submitted!")
                st.session_state["result"]["final_verdict"] = "PASS"
                st.rerun()

            if col_f.button("FAIL", use_container_width=True):
                res = requests.post(f"{API_URL}/review/{batch_id}?decision=FAIL")
                st.error("Decision: FAIL submitted!")
                st.session_state["result"]["final_verdict"] = "FAIL"
                st.rerun()

            if col_r.button("REWORK", use_container_width=True):
                res = requests.post(f"{API_URL}/review/{batch_id}?decision=REWORK")
                st.warning("Decision: REWORK submitted!")
                st.session_state["result"]["final_verdict"] = "REWORK"
                st.rerun()

        # ── Raw JSON ──────────────────────────────────────
        with st.expander("View full JSON report"):
            st.json(r)

    else:
        st.info("Image upload karo aur 'Run Inspection' click karo")