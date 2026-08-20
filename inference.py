import os
from dataclasses import dataclass

import biopandas
import docktgrid
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from docktgrid.transforms import RandomRotation
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.docktdeep.dataset import VoxelDataset
from src.docktdeep.models import Baseline


def convert_mol2_to_pdb(input_mol2, output_pdb=None):
    """
    Convert a molecule from MOL2 format to PDB format using RDKit.

    Parameters:
        input_mol2 (str): Path to the input MOL2 file.
        output_pdb (str): Path to the output PDB file.
    """
    if output_pdb is None:
        output_pdb = input_mol2.replace(".mol2", ".pdb")
    # Load the molecule from the MOL2 file. Set removeHs=False if you want to keep explicit hydrogens.
    mol = Chem.MolFromMol2File(input_mol2, removeHs=False)
    if mol is None:
        raise ValueError(
            f"Failed to read molecule from {input_mol2}. Check if the file is valid."
        )

    # Write the molecule to a PDB file
    # You can either write directly to a file:
    Chem.MolToPDBFile(mol, output_pdb)

def convert_sdf_to_pdb(input_sdf, output_pdb=None):
    """
    Convert a molecule from SDF format to PDB format using RDKit.

    Parameters:
        input_sdf (str): Path to the input SDF file.
        output_pdb (str): Path to the output PDB file.
    """
    if output_pdb is None:
        output_pdb = input_sdf.replace(".sdf", ".pdb")
    # Load the molecule from the SDF file. Set removeHs=False if you want to keep explicit hydrogens.
    mol = Chem.MolFromMolFile(input_sdf, removeHs=False)
    if mol is None:
        raise ValueError(
            f"Failed to read molecule from {input_sdf}. Check if the file is valid."
        )

    # Write the molecule to a PDB file
    Chem.MolToPDBFile(mol, output_pdb)

def read_multimol2(multimol2_path: str) -> list[biopandas.mol2.PandasMol2]:
    parser = docktgrid.molparser.MolecularParser()
    mols = list(biopandas.mol2.split_multimol2(multimol2_path))
    mols_return = []
    for mol in mols:
        with open("tmp.mol2", "w") as f:
            for line in mol[1]:
                f.write(line)
        mols_return.append(parser.parse_file("tmp.mol2", "mol2"))
    os.remove("tmp.mol2")
    return mols_return


def add_cofacs_to_protein(protein_file_name, input_rootdir, cofacs_dir):
    # cofacs are expected to be all listed inside 'cofacs_dir'

    with open(os.path.join(input_rootdir, protein_file_name), "r") as f:
        pdb_lines = f.readlines()

    # convert all cofacs in .mol2 to pdb format first
    for cofac_file in os.listdir(os.path.join(input_rootdir, cofacs_dir)):
        if cofac_file.endswith(".mol2"):
            mol2_fpath = os.path.join(input_rootdir, cofacs_dir, cofac_file)
            convert_mol2_to_pdb(mol2_fpath)
        elif cofac_file.endswith(".sdf"):
            sdf_fpath = os.path.join(input_rootdir, cofacs_dir, cofac_file)
            convert_sdf_to_pdb(sdf_fpath)

    cofacs_file_contents = []
    for cofac_file in os.listdir(os.path.join(input_rootdir, cofacs_dir)):
        if cofac_file.endswith(".pdb"):
            with open(os.path.join(input_rootdir, cofacs_dir, cofac_file), "r") as f:
                atom_lines = []
                for line in f.readlines():
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_lines.append(line)
                cofacs_file_contents.append(atom_lines)

    last_ter_idx = 0
    for idx, line in enumerate(pdb_lines):
        if line.startswith("TER"):
            last_ter_idx = idx

    for cofac_file in cofacs_file_contents:
        for idx, line in enumerate(cofac_file):
            pdb_lines.insert(last_ter_idx + idx + 1, line)

    output_fname = os.path.join(
        input_rootdir, os.path.splitext(protein_file_name)[0] + "-cofacs.pdb"
    )
    with open(output_fname, "w") as f:
        for line in pdb_lines:
            f.write(line)

    return output_fname


# function to get the model from a checkpoint
def get_model(module: pl.LightningModule, ckpt_path: str):
    model = module.load_from_checkpoint(ckpt_path)
    model.eval().cuda()
    return model


@dataclass
class Results:
    preds: np.ndarray
    labels: np.ndarray


def make_inference(dataset, model, nrots=25):
    preds = np.zeros(len(dataset))
    stds = np.zeros(len(dataset))

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), total=len(dataset), desc="Inference", position=1, leave=False):
            preds_rots = np.zeros(nrots)
            for j in tqdm(range(nrots), total=nrots, desc="Rotations", position=2, leave=False):
                x, _ = dataset[i]
                preds_rots[j] = model(x.unsqueeze(0)).cpu().numpy().squeeze()
            preds[i] = preds_rots.mean()
            stds[i] = preds_rots.std()

    return Results([(p, s) for p, s in zip(preds, stds)], dataset.labels.numpy())


def get_dataset(voxel_grid, protein=str, ligands=str, read_ligands_func=read_multimol2):
    # protein should be a string with the path to the protein file in PDB format
    # ligands should be a string with the path to the ligands file in multi-MOL2 format

    ligand_mols = read_ligands_func(ligands)
    proteins_mols = [protein] * len(ligand_mols)

    data = VoxelDataset(
        protein_files=proteins_mols,
        ligand_files=ligand_mols,
        labels=[0] * len(ligand_mols),
        voxel=voxel_grid,
        transform=[RandomRotation()],
        molecular_dropout=0.0,
    )

    return data


def predict_and_save_to_file(
    voxel,
    model,
    protein_path,
    cofacs_dir,
    ligands_dir,
    output_dir,
    model_hash,
    nrots=5,
    group: str = "",
    read_ligands_func=read_multimol2,
    ligand_type = "mol2"
):
    data = {}
    max_length = 0

    protein_file = protein_path
    # Se houver cofatores, adiciona ao arquivo da proteína
    if cofacs_dir and os.path.exists(cofacs_dir):
        protein_file = add_cofacs_to_protein(
            os.path.basename(protein_path), os.path.dirname(protein_path), os.path.basename(cofacs_dir)
        )

    ligand_files = [
        os.path.join(ligands_dir, file)
        for file in os.listdir(ligands_dir)
        if file.endswith(f".{ligand_type}")
    ]
    ligand_files = sorted(ligand_files)

    print(f"Processing {os.path.basename(protein_file)}...")
    for ligand in tqdm(ligand_files, total=len(ligand_files), desc="Ligands", position=0):
        dataset = get_dataset(voxel, protein_file, ligand, read_ligands_func)
        results = make_inference(dataset, model, nrots)  # make predictions

        if ligand.endswith("_docked.mol2"):
            ligand_name = os.path.basename(ligand).replace("_docked.mol2", "")
        elif ligand.endswith(f".{ligand_type}"):
            ligand_name = os.path.basename(ligand).replace(f".{ligand_type}", "")
        else:
            ligand_name = os.path.basename(ligand)
        data[f"{ligand_name}"] = results.preds
        max_length = max(max_length, len(results.preds))

    # save results
    for key in data:
        data[key] = data[key] + [None] * (max_length - len(data[key]))
    df = pd.DataFrame(data).T
    df.columns = [f"top{i+1}" for i in range(max_length)]

    # split columns into mean and std
    expanded_columns = {}
    for col in df.columns:
        expanded_columns[f"{col}_mean"] = df[col].apply(
            lambda x: x[0] if pd.notnull(x) else None
        )
        expanded_columns[f"{col}_std"] = df[col].apply(
            lambda x: x[1] if pd.notnull(x) else None
        )
    df_expanded = pd.DataFrame(expanded_columns, index=df.index)
    df_rounded = df_expanded.round(3)

    # remove protein file with cofactors if it was created
    if cofacs_dir and protein_file != protein_path:
        os.remove(protein_file)

    # save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(protein_path))[0]}_{group}_model={model_hash}.csv",
    )
    df_rounded.to_csv(output_file, index=True)


def predict(
    voxel,
    protein_path: str,
    cofacs_dir: str,
    ligands_dir: str,
    output_dir: str,
    model_hash: str,
    model: pl.LightningModule,
    group: str = "results",
    read_ligands_func=read_multimol2,
    ligand_type = "mol2"
):
    predict_and_save_to_file(
        voxel=voxel,
        model=model,
        protein_path=protein_path,
        cofacs_dir=cofacs_dir,
        ligands_dir=ligands_dir,
        output_dir=output_dir,
        model_hash=model_hash,
        nrots=5,
        group=group,
        read_ligands_func=read_ligands_func,
        ligand_type=ligand_type
    )


def main():
    # define the voxel parameters
    model_hash = "788f06d9409e4e4e87377564"  # unique identifier for the model
    model_path = "epoch=1499-step=448500.ckpt"

    voxel = docktgrid.VoxelGrid(
        views=[docktgrid.view.VolumeView(), docktgrid.view.BasicView()],
        vox_size=1.0,
        box_dims=[24.0, 24.0, 24.0],
    )

    model = get_model(Baseline, model_path)

    root_docking_path = "/storage/applied-projects/dockthor-results/HSA"
    root_receptor_path = "/storage/applied-projects/receptors/HSA"
    root_cofactors_path = "cofactors"
    root_docktdeep_output_dir = "/storage/applied-projects/docktdeep-results/HSA"
    for domain in os.listdir(root_docking_path):
        domain_path = os.path.join(root_docking_path, domain)
        cofactors_dir = os.path.join(root_cofactors_path, domain)
        docktdeep_output_dir = os.path.join(root_docktdeep_output_dir, domain)
        os.makedirs(docktdeep_output_dir, exist_ok=True)
        if os.path.isdir(domain_path):
            for receptor in os.listdir(domain_path):
                ligands_dir = os.path.join(domain_path, receptor)
                if os.path.isdir(ligands_dir):
                    protein_path = os.path.join(root_receptor_path, receptor, receptor + ".pdb")
                    try:
                        print(f"Processing protein: {protein_path}, cofactors: {cofactors_dir}, ligands: {ligands_dir}, results: {docktdeep_output_dir}")
                        predict(
                            voxel=voxel,
                            protein_path=protein_path,
                            cofacs_dir=cofactors_dir,
                            ligands_dir=ligands_dir,
                            output_dir=docktdeep_output_dir,
                            model_hash=model_hash,
                            model=model,
                            group=domain,
                        )
                        print()
                    except Exception as e:
                        print(f"Processing protein: {protein_path}, cofactors: {cofactors_dir}, ligands: {ligands_dir}, results: {docktdeep_output_dir}",
                            f"\nError: {e}")
                        print()
                        continue

def read_pdbligand(pdbligand_path: str):
    mols_return = []
    parser = docktgrid.molparser.MolecularParser()
    if os.path.isdir(pdbligand_path):
        for file in os.listdir(pdbligand_path):
            if file.endswith(".pdb"):
                pdbligand_path_full = os.path.join(pdbligand_path, file)                
                mols_return.append(parser.parse_file(pdbligand_path_full, "pdb"))
    elif os.path.isfile(pdbligand_path) and pdbligand_path.endswith(".pdb"):
        mols_return.append(parser.parse_file(pdbligand_path, "pdb"))
    return mols_return

def main_shirley():
    model_hash = "788f06d9409e4e4e87377564"  # unique identifier for the model
    model_path = "ckpts/epoch=1499-step=448500.ckpt"

    voxel = docktgrid.VoxelGrid(
        views=[docktgrid.view.VolumeView(), docktgrid.view.BasicView()],
        vox_size=1.0,
        box_dims=[24.0, 24.0, 24.0],
    )

    model = get_model(Baseline, model_path)

    root_receptors = "/storage/joserdf/projects/shirley/receptors"
    root_docking = "/storage/joserdf/projects/shirley/dockthor-results"
    root_compounds = "/storage/joserdf/projects/shirley/compounds"
    cofactors_dir = None
    docktdeep_output_dir = "/storage/joserdf/projects/shirley/docktdeep-results"

    for receptor in os.listdir(root_receptors):
        receptor_dir = os.path.join(root_receptors, receptor)
        if os.path.isdir(receptor_dir):
            protein_path = os.path.join(receptor_dir, receptor + "_protein.pdb")
            if os.path.exists(protein_path):
                for group, ligands_dir, read_func, file_ext in [
                    ("FEP_ref", os.path.join(root_compounds, receptor), read_pdbligand, "pdb"),
                    ("DockThor", os.path.join(root_docking, receptor + "_protein"), read_multimol2, "mol2"),
                ]:
                    if os.path.isdir(ligands_dir):
                        try:
                            print(f"Processing protein: {protein_path}, cofactors: {cofactors_dir}, ligands: {ligands_dir}, results: {docktdeep_output_dir}")
                            predict(
                                voxel=voxel,
                                protein_path=protein_path,
                                cofacs_dir=cofactors_dir,
                                ligands_dir=ligands_dir,
                                output_dir=docktdeep_output_dir,
                                model_hash=model_hash,
                                model=model,
                                group=group,
                                read_ligands_func=read_func,
                                ligand_type=file_ext
                            )
                            print()
                        except Exception as e:
                            print(f"Processing protein: {protein_path}, cofactors: {cofactors_dir}, ligands: {ligands_dir}, results: {docktdeep_output_dir}",
                                f"\nError: {e}")
                            print()
                            continue


if __name__ == "__main__":
    main_shirley()