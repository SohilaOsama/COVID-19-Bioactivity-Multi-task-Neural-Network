import streamlit as st

def show_about():
    st.sidebar.markdown("## About")
    st.sidebar.write("""
    This app predicts bioactivity class using two models:
    - **Multi-tasking Neural Network** (Predicts IC50 values)
    - **Decision Tree** (Predicts bioactivity class)
    
    It helps researchers analyze chemical compounds based on their SMILES representation.
    """)