import streamlit as st
import pandas as pd
import os
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np

# Load models and preprocessing steps
# Neural network model (multi-tasking)
nn_model = TFSMLayer('multi_tasking_model_converted', call_endpoint='serving_default')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Stacking classifier model
stacking_clf = joblib.load('random_forest_model.pkl')
variance_threshold = joblib.load('variance_threshold.pkl')

# Function to calculate molecular descriptors
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol)
        }
    except Exception as e:
        raise ValueError(f"Error calculating descriptors for SMILES: {smiles}. Details: {e}")

# Function to convert SMILES to Morgan fingerprints
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except Exception as e:
        raise ValueError(f"Error generating fingerprints for SMILES: {smiles}. Details: {e}")

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

        return pIC50, bioactivity

    except Exception as e:
        raise ValueError(f"Error during NN prediction: {e}")

# Prediction using stacking classifier
def predict_with_stacking(smiles):
    try:
        # Convert SMILES to Morgan fingerprints
        fingerprints = smiles_to_morgan(smiles)

        # Convert to DataFrame for compatibility with preprocessing
        fingerprints_df = pd.DataFrame([fingerprints])

        # Apply VarianceThreshold to remove redundant features
        X_filtered = variance_threshold.transform(fingerprints_df)

        # Predict using the stacking model
        prediction = stacking_clf.predict(X_filtered)

        # Map prediction back to class names
        class_mapping_reverse = {0: 'inactive', 1: 'intermediate', 2: 'active'}
        predicted_class = class_mapping_reverse[prediction[0]]
        return predicted_class

    except Exception as e:
        raise ValueError(f"Error during stacking model prediction: {e}")

# Function to convert pIC50 to micromolar (µM)
def convert_pIC50_to_uM(pIC50):
    ic50_uM = 10 ** (-pIC50) * 1e6
    return ic50_uM

# Function to convert pIC50 to nanograms per microliter (ng/µL)
def convert_pIC50_to_ng_per_uL(pIC50, mol_weight):
    ic50_uM = convert_pIC50_to_uM(pIC50)
    ic50_ng_per_uL = ic50_uM * mol_weight / 1000
    return ic50_ng_per_uL

# Streamlit UI
st.set_page_config(
    page_title="Bioactivity Prediction",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Bioactivity Prediction from SMILES")
st.image("https://github.com/SohilaOsama/COVID-19-Bioactivity-Multi-task-Neural-Network/blob/688f014639a49039043f9efe7b4f945af0213520/images/bioactivity_image.png", use_container_width=True)
st.write("Welcome to the Bioactivity Prediction App! Enter a SMILES string or upload a file to predict the bioactivity class of compounds.")

# Model selection
model_choice = st.radio("Choose a prediction model:", ["Multi-Tasking Neural Network", "Decision Tree"])

# User input for SMILES
smiles_input = st.text_input("SMILES", help="Enter the SMILES notation of the compound here.")

# File upload for multiple SMILES
uploaded_file = st.file_uploader("Upload a TXT or CSV file with SMILES strings", type=["txt", "csv"])

if st.button("Predict"):
    if smiles_input:
        with st.spinner("Predicting..."):
            try:
                if model_choice == "Multi-Tasking Neural Network":
                    pIC50, bioactivity = predict_with_nn(smiles_input)
                    ic50_uM = convert_pIC50_to_uM(pIC50)
                    mol_weight = calculate_descriptors(smiles_input)['MolWt']
                    ic50_ng_per_uL = convert_pIC50_to_ng_per_uL(pIC50, mol_weight)
                    st.success(f"Predicted pIC50: {pIC50:.2f}")
                    st.success(f"Predicted IC50: {ic50_uM:.2f} µM")
                    st.success(f"Predicted IC50: {ic50_ng_per_uL:.2f} ng/µL")
                    st.success(f"Predicted Bioactivity: {bioactivity}")
                elif model_choice == "decision tree":
                    bioactivity = predict_with_stacking(smiles_input)
                    st.success(f"Predicted Bioactivity Class: {bioactivity}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif uploaded_file:
        try:
            if uploaded_file.name.endswith('.txt'):
                smiles_list = uploaded_file.read().decode("utf-8").splitlines()
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                smiles_list = df.iloc[:, 0].tolist()  # Assuming SMILES are in the first column

            predictions = []
            for smiles in smiles_list:
                try:
                    if model_choice == "Multi-Tasking Neural Network":
                        pIC50, bioactivity = predict_with_nn(smiles)
                        ic50_uM = convert_pIC50_to_uM(pIC50)
                        mol_weight = calculate_descriptors(smiles)['MolWt']
                        ic50_ng_per_uL = convert_pIC50_to_ng_per_uL(pIC50, mol_weight)
                        predictions.append((smiles, pIC50, ic50_uM, ic50_ng_per_uL, bioactivity))
                    elif model_choice == "decision tree":
                        bioactivity = predict_with_stacking(smiles)
                        predictions.append((smiles, bioactivity))
                except Exception as e:
                    predictions.append((smiles, 'Error', 'Error', 'Error', 'Error'))

            # Create a DataFrame for the predictions
            if model_choice == "Multi-Tasking Neural Network":
                predictions_df = pd.DataFrame(predictions, columns=["SMILES", "pIC50", "IC50 (µM)", "IC50 (ng/µL)", "Bioactivity"])
            else:
                predictions_df = pd.DataFrame(predictions, columns=["SMILES", "Bioactivity"])

            # Convert DataFrame to CSV
            csv = predictions_df.to_csv(index=False).encode('utf-8')

            # Provide download link
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='bioactivity_predictions.csv',
                mime='text/csv',
            )
            st.success("Predictions completed and file is ready for download.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.error("Please enter a valid SMILES string or upload a file.")

# Add an about section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts the bioactivity class of a compound based on its SMILES notation.
    - **Models Available**:
      - Multi-Tasking Neural Network
      - decision tree
    """
)