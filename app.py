import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np

# Load models and preprocessing steps
nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
stacking_clf = joblib.load('random_forest_model.pkl')
variance_threshold = joblib.load('variance_threshold.pkl')

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol)
        }
    return None

def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

def predict_with_nn(smiles):
    descriptors = calculate_descriptors(smiles)
    if descriptors:
        descriptors_df = pd.DataFrame([descriptors])
        fingerprints = smiles_to_morgan(smiles)
        fingerprints_df = pd.DataFrame([fingerprints], columns=[str(i) for i in range(len(fingerprints))])
        combined_df = pd.concat([descriptors_df, fingerprints_df], axis=1)
        combined_scaled = scaler.transform(combined_df)
        combined_selected = pd.DataFrame(combined_scaled, columns=combined_df.columns)[selected_features]
        input_data = combined_selected.to_numpy()
        outputs = nn_model(input_data)
        pIC50 = outputs['output_0'].numpy()[0][0]
        bioactivity = 'active' if outputs['output_1'].numpy()[0][0] > 0.5 else 'inactive'
        return pIC50, bioactivity
    return None, None

def predict_with_stacking(smiles):
    fingerprints = smiles_to_morgan(smiles)
    if fingerprints:
        fingerprints_df = pd.DataFrame([fingerprints])
        X_filtered = variance_threshold.transform(fingerprints_df)
        prediction = stacking_clf.predict(X_filtered)
        class_mapping = {0: 'inactive', 1: 'intermediate', 2: 'active'}
        return class_mapping[prediction[0]]
    return None

def convert_pIC50_to_uM(pIC50):
    return 10 ** (-pIC50) * 1e6

def convert_pIC50_to_ng_per_uL(pIC50, mol_weight):
    return convert_pIC50_to_uM(pIC50) * mol_weight / 1000

st.set_page_config(page_title="Bioactivity Prediction", page_icon="🧪", layout="wide")
st.markdown("""
    <style>
    .stButton>button { border-radius: 8px; padding: 10px 20px; font-size: 16px; }
    .stTextInput>div>div>input { border-radius: 8px; padding: 10px; }
    .stFileUploader>div>button { border-radius: 8px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Bioactivity Prediction from SMILES")
st.image("images/Drug.png", use_column_width=True)
st.markdown("Enter a SMILES string or upload a file to predict the bioactivity class of compounds.")
st.sidebar.markdown("""## About
This app predicts bioactivity class using two models:
- **Neural Network** (pIC50 & IC50 values)
- **Decision Tree** (Bioactivity class)
""")

model_choice = st.radio("Choose a model:", ["Multi-Tasking Neural Network", "Decision Tree"], horizontal=True)
smiles_input = st.text_input("Enter SMILES:")
uploaded_file = st.file_uploader("Upload a CSV file with SMILES", type=["csv"])

if st.button("Predict"):
    if smiles_input:
        with st.spinner("Predicting..."):
            if model_choice == "Multi-Tasking Neural Network":
                pIC50, bioactivity = predict_with_nn(smiles_input)
                if pIC50 is not None:
                    mol_weight = calculate_descriptors(smiles_input)['MolWt']
                    st.success(f"Predicted pIC50: {pIC50:.2f}")
                    st.success(f"Predicted IC50: {convert_pIC50_to_uM(pIC50):.2f} µM")
                    st.success(f"Predicted IC50: {convert_pIC50_to_ng_per_uL(pIC50, mol_weight):.2f} ng/µL")
                    st.success(f"Predicted Bioactivity: {bioactivity}")
                else:
                    st.error("Invalid SMILES string.")
            else:
                bioactivity = predict_with_stacking(smiles_input)
                if bioactivity:
                    st.success(f"Predicted Bioactivity Class: {bioactivity}")
                else:
                    st.error("Invalid SMILES string.")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "SMILES" in df.columns:
            results = []
            for smiles in df["SMILES"]:
                if model_choice == "Multi-Tasking Neural Network":
                    pIC50, bioactivity = predict_with_nn(smiles)
                    if pIC50 is not None:
                        mol_weight = calculate_descriptors(smiles)['MolWt']
                        results.append([smiles, pIC50, convert_pIC50_to_uM(pIC50), convert_pIC50_to_ng_per_uL(pIC50, mol_weight), bioactivity])
                    else:
                        results.append([smiles, "Error", "Error", "Error", "Error"])
                else:
                    bioactivity = predict_with_stacking(smiles)
                    results.append([smiles, bioactivity if bioactivity else "Error"])
            if model_choice == "Multi-Tasking Neural Network":
                results_df = pd.DataFrame(results, columns=["SMILES", "pIC50", "IC50 (µM)", "IC50 (ng/µL)", "Bioactivity"])
            else:
                results_df = pd.DataFrame(results, columns=["SMILES", "Bioactivity"])
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "bioactivity_predictions.csv", "text/csv")
            st.success("Predictions completed.")
        else:
            st.error("CSV must contain a column named 'SMILES'.")
    else:
        st.error("Enter a SMILES string or upload a file.")
