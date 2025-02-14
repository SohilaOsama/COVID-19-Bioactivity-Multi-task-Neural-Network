import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
import chardet  # For automatic encoding detection

# Load models and preprocessing steps
nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
stacking_clf = joblib.load('random_forest_model.pkl')
variance_threshold = joblib.load('variance_threshold.pkl')

# Detect encoding of uploaded file
def detect_encoding(file):
    raw_data = file.read(4096)  # Read a small chunk
    file.seek(0)  # Reset file position
    result = chardet.detect(raw_data)  # Detect encoding
    return result["encoding"]

# Compute molecular descriptors
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

# Convert SMILES to Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)) if mol else None

# Prediction function for Neural Network
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

# Prediction function for Stacking Classifier
def predict_with_stacking(smiles):
    fingerprints = smiles_to_morgan(smiles)
    if fingerprints:
        fingerprints_df = pd.DataFrame([fingerprints])
        X_filtered = variance_threshold.transform(fingerprints_df)
        prediction = stacking_clf.predict(X_filtered)
        class_mapping = {0: 'inactive', 1: 'intermediate', 2: 'active'}
        return class_mapping[prediction[0]]
    return None

# Convert pIC50 values
def convert_pIC50_to_uM(pIC50):
    return 10 ** (-pIC50) * 1e6

def convert_pIC50_to_ng_per_uL(pIC50, mol_weight):
    return convert_pIC50_to_uM(pIC50) * mol_weight / 1000





# Streamlit UI
st.set_page_config(page_title="Bioactivity Prediction", page_icon="🧪", layout="wide")

st.title("🧪 Bioactivity Prediction from SMILES")
st.image("images/Drug.png", use_container_width=True)

# Instructions
st.markdown("## Instructions:")
 # Instruction Steps
st.write("""
    
    To convert your compound to a Simplified Molecular Input Line Entry System (SMILES), please visit this website: [decimer.ai](https://decimer.ai/)
    """)
st.markdown("1. Enter a SMILES string or upload a TXT file with SMILES in a single column.")
st.markdown("2. Choose the prediction model: Multi-Tasking Neural Network or Decision Tree.")
st.markdown("3. Click 'Predict' to see results.")

# Sidebar info
st.sidebar.markdown("## About")
st.sidebar.write("This app predicts bioactivity class using two models:")
st.sidebar.write("- **Multi-tasking Neural network** (Predicts IC50 values)")
st.sidebar.write("- **Decision Tree** (Predicts bioactivity class)")

# Input: Single SMILES string or file upload
model_choice = st.radio("Choose a model:", ["Multi-Tasking Neural Network", "Decision Tree"], horizontal=True)
smiles_input = st.text_input("Enter SMILES:")
uploaded_file = st.file_uploader("Upload a TXT file", type=["csv", "txt", "xls", "xlsx"])

if st.button("Predict"):
    if smiles_input:
        with st.spinner("Predicting..."):
            if model_choice == "Multi-Tasking Neural Network":
                pIC50, bioactivity = predict_with_nn(smiles_input)
                if pIC50 is not None:
                    mol_weight = calculate_descriptors(smiles_input)['MolWt']
                    st.markdown(
    f"""
    <div style="
        border: 2px solid {border_color}; 
        padding: 15px; 
        border-radius: 10px; 
        background-color: {bg_color}; 
        color: {text_color};
        font-family: Arial, sans-serif;">
        <h4 style="color: {highlight_color}; text-align: center;">🧪 Prediction Results</h4>
        <p><b>📊 pIC50 Value:</b> <span style="color: {highlight_color};">{pIC50:.2f}</span></p>
        <p><b>⚗️ IC50 (µM):</b> <span style="color: {highlight_color};">{convert_pIC50_to_uM(pIC50):.2f} µM</span></p>
        <p><b>🧬 IC50 (ng/µL):</b> <span style="color: {highlight_color};">{convert_pIC50_to_ng_per_uL(pIC50, mol_weight):.2f} ng/µL</span></p>
        <p><b>🔬 Bioactivity:</b> 
            <span style="color: {'#1565C0' if bioactivity=='active' else 'error_color'};">
                {bioactivity.capitalize()}
            </span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


                else:
                    st.error("Invalid SMILES string.")
            else:
                bioactivity = predict_with_stacking(smiles_input)
                if bioactivity:
                    st.success(f"Predicted Bioactivity Class: {bioactivity}")
                else:
                    st.error("Invalid SMILES string.")
    elif uploaded_file:
        try:
            detected_encoding = detect_encoding(uploaded_file)
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, encoding=detected_encoding)
            elif file_extension == "txt":
                df = pd.read_csv(uploaded_file, delimiter="\t", encoding=detected_encoding)
            elif file_extension in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.error("Unsupported file format. Please upload CSV, TXT, XLS, or XLSX.")
                st.stop()

            if df.shape[1] != 1:
                st.error("The uploaded file must contain only one column with SMILES strings.")
                st.stop()

            df.columns = ["SMILES"]
            df.dropna(inplace=True)

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

        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

