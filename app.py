import streamlit as st
from orchestrator.workflow import run_pipeline

st.title("PRANAG AI")

prompt = st.text_input("Enter prompt")

if st.button("Generate"):
    result = run_pipeline(prompt)
    st.json(result)
