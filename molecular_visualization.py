import streamlit as st
import py3Dmol
import base64
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit_ketcher as sk

def generate_3d_view(smiles, style="stick"):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)  # Add hydrogen atoms
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # Generate 3D coordinates using ETKDGv3
        AllChem.UFFOptimizeMolecule(mol)  # Optimize structure

        mol_block = Chem.MolToMolBlock(mol)

        viewer = py3Dmol.view(width=500, height=500)
        viewer.addModel(mol_block, "mol")

        if style == "stick":
            viewer.setStyle({"stick": {}})
        elif style == "sphere":
            viewer.setStyle({"sphere": {"scale": 0.3}})
        elif style == "line":
            viewer.setStyle({"line": {}})

        viewer.zoomTo()
        return viewer
    return None

def generate_2d_view(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    svg = drawer.GetDrawingText()
    svg_base64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{svg_base64}" width="300"/>'

def show_molecular_visualization():
    st.header("Molecular Visualization 🌐")

    # Use streamlit_ketcher to get the SMILES string
    ketcher_smiles = sk.st_ketcher(key="ketcher")

    # Standard text input for SMILES string
    smiles_input = st.text_input("Or enter SMILES string:")

    # Determine which SMILES input to use
    if not smiles_input and ketcher_smiles:
        smiles_input = ketcher_smiles

    vis_mode = st.radio("Choose visualization mode:", ["2D", "3D"])
    style = st.selectbox("Select style:", ["stick", "sphere", "line"])

    if st.button("Visualize"):
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                AllChem.Compute2DCoords(mol)
                
                if vis_mode == "3D":
                    viewer = generate_3d_view(smiles_input, style)
                    if viewer is not None:
                        viewer_html = viewer._make_html()
                        st.components.v1.html(viewer_html, height=500)
                    else:
                        st.error(f"Invalid SMILES string: {smiles_input}")
                else:
                    st.markdown(generate_2d_view(mol), unsafe_allow_html=True)
            else:
                st.error(f"Invalid SMILES string: {smiles_input}")
        else:
            st.error("Please enter at least one SMILES string.")

if __name__ == "__main__":
    show_molecular_visualization()