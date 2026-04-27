"""
PRANAG-AI  —  Streamlit front-end for the Prompt Parser

Run:
    streamlit run app.py
"""
import json
import streamlit as st

from search_engine.vector_store import populate
from orchestrator.workflow import run_pipeline

# ── One-time initialisation ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising vector store …")
def init():
    populate()

init()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PRANAG-AI Prompt Parser", page_icon="🌾", layout="centered")
st.title("🌾 PRANAG-AI — Prompt Parser")
st.caption("Convert a natural language crop prompt into a structured `spec.json` for the simulation pipeline.")

with st.form("prompt_form"):
    prompt = st.text_area(
        "Enter your crop design prompt",
        placeholder='e.g. "I need wheat that can survive 48°C heat in Jodhpur with low rainfall"',
        height=100,
    )
    submitted = st.form_submit_button("🚀 Generate Spec")

if submitted and prompt.strip():
    with st.spinner("Running PRANAG pipeline …"):
        result = run_pipeline(prompt.strip())

    pipeline_id = result.get("pipeline_id", "—")
    spec        = result.get("spec")
    error       = result.get("error")

    st.divider()
    st.subheader("Pipeline Result")
    st.caption(f"Pipeline ID: `{pipeline_id}`")

    if spec:
        col1, col2, col3 = st.columns(3)
        col1.metric("Crop",        spec.get("crop", "—").title())
        col2.metric("Location",    spec.get("location", "—"))
        col3.metric("Temperature", f"{spec.get('temperature', '—')} °C")

        with st.expander("📋 Full spec.json", expanded=True):
            st.json(spec)

        if spec.get("retrieved_traits"):
            with st.expander("🔬 Retrieved traits"):
                for t in spec["retrieved_traits"]:
                    st.markdown(f"- {t}")

        if spec.get("scientific_basis"):
            with st.expander("📚 Scientific basis"):
                for f in spec["scientific_basis"]:
                    st.markdown(f"- {f}")

        confidence = spec.get("confidence", 0)
        st.progress(confidence, text=f"Confidence: {confidence:.0%}")

    else:
        st.error(f"Pipeline failed: {error or 'Unknown error'}")
        st.info("Make sure Ollama is running:  `ollama serve`  and the model is pulled.")

elif submitted:
    st.warning("Please enter a prompt before generating.")
