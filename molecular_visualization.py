import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem

def show_molecular_visualization():
    st.header("3D Molecular Visualization 🌐")
    
    # Input: SMILES strings
    smiles_input = st.text_area("Enter SMILES strings 📝 (separated by newline or comma):")
    style = st.selectbox("Select style:", ["stick", "sphere", "line"])
    color_scheme = st.selectbox("Select color scheme:", ["Jmol", "white Carbon"])

    if st.button("Visualize"):
        if smiles_input:
            smiles_list = [smiles.strip() for smiles in smiles_input.replace(',', '\n').split('\n') if smiles.strip()]
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    AllChem.Compute2DCoords(mol)
                    if style == "stick":
                        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                        drawer.FinishDrawing()
                        svg = drawer.GetDrawingText().replace('\n', '')
                        st.image(svg, use_column_width=True)
                    elif style == "sphere":
                        img = Draw.MolToImage(mol, size=(300, 300), kekulize=True, wedgeBonds=True, fitImage=True, options=Draw.DrawingOptions())
                        st.image(img)
                    elif style == "line":
                        img = Draw.MolToImage(mol, size=(300, 300), kekulize=False, options=Draw.DrawingOptions())
                        st.image(img)
                else:
                    st.error(f"Invalid SMILES string: {smiles}")
        else:
            st.error("Please enter at least one SMILES string.")