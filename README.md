

```markdown
# COVID-19 Drug Discovery: Multi-task Neural Network & Stacking Classifier

Welcome to the **COVID-19 Drug Discovery** project! This app is designed to predict the **bioactivity** of compounds based on their SMILES (Simplified Molecular Input Line Entry System) representation. It combines **machine learning** and **deep learning** models to provide accurate and flexible bioactivity predictions for drug discovery and cheminformatics applications.

---

## **Key Features**
- **Multi-Task Neural Network**: Predicts both:
  - **pIC50**: Regression output representing the inhibitory concentration of a compound.
  - **Bioactivity**: Classification output (active/inactive).
- **Desion Tree Classifier**:
  - Predicts bioactivity class using an ensemble model (`inactive`, `intermediate`, `active`).
- Supports both **single compound prediction** and **batch predictions** from `.txt` or `.csv` files.
- Clean and intuitive **Streamlit-based UI**.

---

## **Technologies Used**
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Machine Learning Models**: Stacking Classifier (Random Forest, Logistic Regression)
- **Data Processing**: RDKit, pandas, scikit-learn
- **Web Interface**: Streamlit

---

## **Installation**

### **Prerequisites**
Make sure you have the following installed on your system:
- Python 3.8 or higher
- Git
- pip (Python package manager)

### **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/SohilaOsama/COVID-19-Drug-Discovery--Multi-task-Neural-Network.git
   cd COVID-19-Drug-Discovery--Multi-task-Neural-Network
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present in the project directory:
   - `multi_tasking_model_converted/`: SavedModel directory for the neural network.
   - `random_forest_model.pkl`: Trained stacking classifier model.
   - `scaler.pkl`: Preprocessing scaler for the neural network.
   - `variance_threshold.pkl`: Feature selector for the stacking classifier.
   - `images/`: Folder containing the app's image assets.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## **How to Use**

### **Single SMILES Prediction**
1. Enter a valid SMILES string in the **SMILES** text box.
2. Select the prediction model:
   - **Multi-Task Neural Network**: For pIC50 and bioactivity prediction.
   - **Stacking Classifier**: For bioactivity class prediction (`inactive`, `intermediate`, `active`).
3. Click the **Predict** button to view the results.

### **Batch Prediction**
1. Upload a `.txt` or `.csv` file containing SMILES strings (one SMILES per line or column).
2. Select the prediction model.
3. Download the prediction results as a `.csv` file.

---

## **Project Workflow**
1. **Input Data**: SMILES strings.
2. **Preprocessing**:
   - **Neural Network**: Molecular descriptors (LogP, MolWt, NumHDonors, NumHAcceptors) and Morgan fingerprints.
   - **Stacking Classifier**: VarianceThreshold and Morgan fingerprints.
3. **Model Prediction**:
   - Outputs:
     - **pIC50** (regression, Neural Network).
     - **Bioactivity** (classification, Neural Network & Stacking Classifier).
4. **Output Results**: Displayed in the app or downloadable as `.csv`.

---

## **Example Predictions**
| SMILES                  | pIC50  | Bioactivity (NN) | Bioactivity Class (Stacking) |
|-------------------------|--------|------------------|------------------------------|
| `CCO`                  | 5.64   | Active           | Active                       |
| `CC(=O)Nc1ccc(O)cc1`   | 4.12   | Inactive         | Intermediate                 |
| `CC1=CC=CC=C1O`        | 6.20   | Active           | Active                       |

---

## **Screenshots**
### **Main Interface**
![Main UI](images/bioactivity_image.png)

### **Batch Prediction Results**
![Batch Results](images/batch_results_example.png)

---

## **Folder Structure**
```
COVID-19-Drug-Discovery--Multi-task-Neural-Network/
│
├── app.py                     # Main Streamlit app
├── requirements.txt           # Required Python libraries
├── multi_tasking_model_converted/  
│   ├── saved_model.pb         # Neural network SavedModel
│   └── variables/             # Model variables
├── DecisonTree.pkl    # Stacking classifier model
├── scaler.pkl                 # Scaler for NN preprocessing
├── variance_threshold.pkl     # Feature selector for stacking classifier
├── images/                    # Image assets for UI
└── README.md                  # Project documentation
```

---

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes and push to your fork:
   ```bash
   git push origin feature-branch
   ```
4. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
For questions or feedback, please reach out to:
- **Email**: [sohilaosama1661@gmail.com](mailto:your.email@example.com)
- **GitHub**: [SohilaOsama](https://github.com/SohilaOsama)

---

Happy Predicting! 🎉
```

---

### **How to Use**
1. Save this as `README.md` in the root of your project directory.
2. Commit and push it to your GitHub repository:
   ```bash
   git add README.md
   git commit -m "Added professional README.md"
   git push origin main
   ```

Let me know if you need further adjustments! 😊
