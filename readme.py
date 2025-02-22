import streamlit as st

def show_readme():
    st.markdown("## 📖 ReadMe - Bioactivity Prediction")

    st.markdown("""
    ### ℹ️ **Overview**
    This application predicts the bioactivity of chemical compounds using Machine Learning models.

    ### 📝 **Instructions**
    1. Enter a **SMILES string** manually or **upload a TXT/CSV file** with SMILES in a single column.
    2. Choose between:
       - **Multi-Tasking Neural Network** (Predicts IC50 and pIC50)
       - **Decision Tree** (Classifies bioactivity)
    3. Click 'Predict' to get results.

    ### 📏 **SMILES String Requirements**
    - **Minimum Length:** 5 characters
    - **Maximum Length:** 500 characters
    - **Valid Examples:**
      - ✅ CCC(=O)OCC  
      - ✅ C1=CC=CC=C1  

    ### 📌 **Outputs**
    - **pIC50 Value**: Predicted binding affinity.
    - **IC50 (µM & ng/µL)**: Converts pIC50 to useful units.
    - **Bioactivity Classification**: Active or Inactive.
    - **Confidence Score**: Model certainty.
    - **Error Percentage**: Approximate prediction error.

    ### 🔗 **Helpful Resources**
    - Convert structures to SMILES: [decimer.ai](https://decimer.ai/)
    """)

    # ReadMe File Download
    readme_content = """
    Bioactivity Prediction App
    --------------------------
    This app helps predict the bioactivity class of compounds based on their SMILES notation.
    
    Input:
    - SMILES string (manual input or file upload)
    
    Outputs:
    - pIC50
    - IC50 (µM)
    - IC50 (ng/µL)
    - Bioactivity class
    - Confidence Score
    - Error percentage
    
    For more details, visit: [SRTA City](https://srtacity.org)
    """
    st.sidebar.download_button(
        label="📥 Download ReadMe",
        data=readme_content,
        file_name="README.txt",
        mime="text/plain"
    )