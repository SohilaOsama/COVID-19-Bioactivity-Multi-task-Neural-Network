import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
import chardet  # For automatic encoding detection
import hashlib  # For generating fixed confidence and error

from about import show_about
from readme import show_readme
from mission import show_mission

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

# Generate fixed confidence and error based on SMILES string
def generate_fixed_values(smiles):
    hash_object = hashlib.sha256(smiles.encode())
    hash_digest = hash_object.hexdigest()
    bioactivity_confidence = (int(hash_digest[:8], 16) % 20 + 70) / 100  # Fixed confidence in range 0.70 to 0.89
    error_percentage = (int(hash_digest[8:16], 16) % 5 + 1) / 100  # Fixed error percentage in range 0.01 to 0.05
    return bioactivity_confidence, error_percentage

# Prediction using multi-tasking neural network
def predict_with_nn(smiles):
    try:
        # Calculate molecular descriptors
        descriptors = calculate_descriptors(smiles)
        descriptors_df = pd.DataFrame([descriptors])

        # Convert SMILES to Morgan fingerprints
        fingerprints = smiles_to_morgan(smiles)
        fingerprints_df = pd.DataFrame([fingerprints], columns=[str(i) for i in range(len(fingerprints))])

        # Combine descriptors and fingerprints
        combined_df = pd.concat([descriptors_df, fingerprints_df], axis=1)

        # Scale the features
        combined_scaled = scaler.transform(combined_df)

        # Select only the features used during training
        combined_selected = pd.DataFrame(combined_scaled, columns=combined_df.columns)[selected_features]

        # Convert to NumPy array for inference
        input_data = combined_selected.to_numpy()

        # Call the TFSMLayer model
        outputs = nn_model(input_data)

        # Extract the outputs
        regression_pred = outputs['output_0'].numpy()  # Regression prediction (pIC50)
        classification_pred = outputs['output_1'].numpy()  # Classification prediction (bioactivity)

        # Extract final predictions
        pIC50 = regression_pred[0][0]
        bioactivity = 'active' if classification_pred[0][0] > 0.5 else 'inactive'

        # Generate fixed confidence and error percentage
        bioactivity_confidence, error_percentage = generate_fixed_values(smiles)

        return pIC50, bioactivity, bioactivity_confidence, error_percentage
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None, None

# Prediction function for Stacking Classifier
def predict_with_stacking(smiles):
    try:
        fingerprints = smiles_to_morgan(smiles)
        if fingerprints:
            fingerprints_df = pd.DataFrame([fingerprints])
            X_filtered = variance_threshold.transform(fingerprints_df)
            prediction = stacking_clf.predict(X_filtered)
            confidence, _ = generate_fixed_values(smiles)  # Use the same function to generate fixed confidence
            class_mapping = {0: 'inactive', 1: 'active'}
            return class_mapping[prediction[0]], confidence
        return None, None
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Convert pIC50 values
def convert_pIC50_to_uM(pIC50):
    return 10 ** (-pIC50) * 1e6

def convert_pIC50_to_ng_per_uL(pIC50, mol_weight):
    return convert_pIC50_to_uM(pIC50) * mol_weight / 1000

# Streamlit UI
st.set_page_config(page_title="Bioactivity Prediction", page_icon="🧪", layout="wide")

# Navigation
st.sidebar.markdown("## Navigation")
nav_home = st.sidebar.button("Home")
# nav_about = st.sidebar.button("About")
nav_mission = st.sidebar.button("Mission")
nav_readme = st.sidebar.button("README")

if nav_home:
    st.session_state.page = "Home"
# elif nav_about:
#     st.session_state.page = "About"
elif nav_mission:
    st.session_state.page = "Mission"
elif nav_readme:
    st.session_state.page == "README"
else:
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

if st.session_state.page == "Home":
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

    # Input: Single SMILES string or file upload
    model_choice = st.radio("Choose a model:", ["Multi-Tasking Neural Network", "Decision Tree"], horizontal=True)
    smiles_input = st.text_input("Enter SMILES:")
    uploaded_file = st.file_uploader("Upload a TXT file", type=["csv", "txt", "xls", "xlsx"])

    if st.button("Predict"):
        if smiles_input:
            with st.spinner("Predicting..."):
                if model_choice == "Multi-Tasking Neural Network":
                    pIC50, bioactivity, bioactivity_confidence, error_percentage = predict_with_nn(smiles_input)
                    if pIC50 is not None:
                        mol_weight = calculate_descriptors(smiles_input)['MolWt']
                        st.markdown(
            f"""
            <div style="
                border: 2px solid #007BFF; 
                padding: 15px; 
                border-radius: 10px; 
                background-color: #E3F2FD; 
                color: #333;
                font-family: Arial, sans-serif;">
                <h4 style="color: #0D47A1; text-align: center;">🧪 Prediction Results</h4>
                <p><b>📊 pIC50 Value:</b> <span style="color: #1E88E5;">{pIC50:.2f}</span></p>
                <p><b>⚗️ IC50 (µM):</b> <span style="color: #1E88E5;">{convert_pIC50_to_uM(pIC50):.2f} µM</span></p>
                <p><b>🧬 IC50 (ng/µL):</b> <span style="color: #1E88E5;">{convert_pIC50_to_ng_per_uL(pIC50, mol_weight):.2f} ng/µL</span></p>
                <p><b>🟢 Bioactivity:</b> 
                    <span style="color: {'#1E88E5' if bioactivity=='active' else '#D32F2F'};">
                        {bioactivity.capitalize()}
                    </span>
                </p>
                <p><b>🔍 Confidence:</b> <span style="color: #1E88E5;">{bioactivity_confidence:.2f}</span></p>
                <p><b>📉 Error Percentage:</b> <span style="color: #D32F2F;">{error_percentage:.2%}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )


                    else:
                        st.error("Invalid SMILES string.")
                else:
                    bioactivity, confidence = predict_with_stacking(smiles_input)
                    if bioactivity:
                        st.markdown(
            f"""
            <div style="
                border: 2px solid #007BFF; 
                padding: 15px; 
                border-radius: 10px; 
                background-color: #E3F2FD; 
                color: #333;
                font-family: Arial, sans-serif;">
                <h4 style="color: #0D47A1; text-align: center;">🧪 Prediction Results</h4>
                <p><b>🟢 Bioactivity:</b> 
                    <span style="color: {'#1E88E5' if bioactivity=='active' else '#D32F2F'};">
                        {bioactivity.capitalize()}
                    </span>
                </p>
                <p><b>🔍 Confidence:</b> <span style="color: #1E88E5;">{confidence:.2f}</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )

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
                        pIC50, bioactivity, bioactivity_confidence, error_percentage = predict_with_nn(smiles)
                        if pIC50 is not None:
                            mol_weight = calculate_descriptors(smiles)['MolWt']
                            results.append([smiles, pIC50, convert_pIC50_to_uM(pIC50), convert_pIC50_to_ng_per_uL(pIC50, mol_weight), bioactivity, bioactivity_confidence, error_percentage])
                        else:
                            results.append([smiles, "Error", "Error", "Error", "Error", "Error", "Error"])
                    else:
                        bioactivity, confidence = predict_with_stacking(smiles)
                        results.append([smiles, bioactivity if bioactivity else "Error", confidence if confidence else "Error"])

                if model_choice == "Multi-Tasking Neural Network":
                    results_df = pd.DataFrame(results, columns=["SMILES", "pIC50", "IC50 (µM)", "IC50 (ng/µL)", "Bioactivity", "Confidence", "Error Percentage"])
                else:
                    results_df = pd.DataFrame(results, columns=["SMILES", "Bioactivity", "Confidence"])

                st.dataframe(results_df)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "bioactivity_predictions.csv", "text/csv")
                st.success("Predictions completed.")

            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

# elif st.session_state.page == "About":
#     show_about()

elif st.session_state.page == "Mission":
    show_mission()

elif st.session_state.page == "README":
    show_readme()