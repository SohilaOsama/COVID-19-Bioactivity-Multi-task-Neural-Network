import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer

# Load models and preprocessing steps
nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
stacking_clf = joblib.load('random_forest_model.pkl')
variance_threshold = joblib.load('variance_threshold.pkl')

# Custom CSS Styling
st.markdown("""
    <style>
        .stButton>button {border-radius: 8px; padding: 8px 16px; background-color: #4CAF50; color: white; font-size: 16px;}
        .stTextInput>div>div>input {border-radius: 8px; padding: 10px; font-size: 16px;}
        .stFileUploader>div>div>div>button {border-radius: 8px; padding: 10px; font-size: 16px;}
        .stMarkdown {font-size: 18px;}
        .reportview-container {background-color: #f5f5f5;}
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol)
    }

def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

def predict_with_nn(smiles):
    descriptors = calculate_descriptors(smiles)
    if descriptors is None:
        return None, None
    
    descriptors_df = pd.DataFrame([descriptors])
    fingerprints = smiles_to_morgan(smiles)
    if fingerprints is None:
        return None, None
    
    fingerprints_df = pd.DataFrame([fingerprints], columns=[str(i) for i in range(len(fingerprints))])
    combined_df = pd.concat([descriptors_df, fingerprints_df], axis=1)
    combined_scaled = scaler.transform(combined_df)
    combined_selected = pd.DataFrame(combined_scaled, columns=combined_df.columns)[selected_features]
    input_data = combined_selected.to_numpy()
    
    outputs = nn_model(input_data)
    pIC50 = outputs['output_0'].numpy()[0][0]
    bioactivity = 'Active' if outputs['output_1'].numpy()[0][0] > 0.5 else 'Inactive'
    return pIC50, bioactivity

def predict_with_stacking(smiles):
    fingerprints = smiles_to_morgan(smiles)
    if fingerprints is None:
        return None
    fingerprints_df = pd.DataFrame([fingerprints])
    X_filtered = variance_threshold.transform(fingerprints_df)
    prediction = stacking_clf.predict(X_filtered)
    class_mapping = {0: 'Inactive', 1: 'Intermediate', 2: 'Active'}
    return class_mapping[prediction[0]]

# Streamlit UI
st.set_page_config(page_title="Bioactivity Prediction", page_icon="🧪", layout="wide")
st.title("🔬 Bioactivity Prediction App")
st.markdown("Enter a SMILES notation below or upload a file to predict bioactivity.")
st.sidebar.header("📌 Instructions")
st.sidebar.write("1. Enter a SMILES string manually or upload a CSV file.")
st.sidebar.write("2. Select the prediction model.")
st.sidebar.write("3. Click 'Predict' to get results.")

model_choice = st.radio("Choose a Model:", ["Multi-Tasking Neural Network", "Decision Tree"])
smiles_input = st.text_input("Enter SMILES:", "")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if st.button("🔍 Predict"):
    if smiles_input:
        with st.spinner("Analyzing... Please wait."):
            if model_choice == "Multi-Tasking Neural Network":
                pIC50, bioactivity = predict_with_nn(smiles_input)
                if pIC50 is not None:
                    st.success(f"**pIC50 Prediction:** {pIC50:.2f}")
                    st.success(f"**Bioactivity:** {bioactivity}")
                else:
                    st.error("Invalid SMILES input.")
            else:
                bioactivity = predict_with_stacking(smiles_input)
                if bioactivity is not None:
                    st.success(f"**Bioactivity Class:** {bioactivity}")
                else:
                    st.error("Invalid SMILES input.")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "SMILES" not in df.columns:
            st.error("CSV file must have a 'SMILES' column.")
        else:
            results = []
            for smiles in df["SMILES"]:
                if model_choice == "Multi-Tasking Neural Network":
                    pIC50, bioactivity = predict_with_nn(smiles)
                    results.append([smiles, pIC50, bioactivity])
                else:
                    bioactivity = predict_with_stacking(smiles)
                    results.append([smiles, bioactivity])
            results_df = pd.DataFrame(results, columns=["SMILES", "pIC50", "Bioactivity"] if model_choice == "Multi-Tasking Neural Network" else ["SMILES", "Bioactivity"])
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download CSV", data=csv, file_name="bioactivity_predictions.csv", mime="text/csv")
    else:
        st.error("Please enter a SMILES string or upload a CSV file.")
