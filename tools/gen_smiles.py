#!/usr/bin/env python3
"""Generate canonical ligand SMILES for PDBbind2020 complexes.

The processed ligand pickles only store 3D coordinates + element symbols (no
bonds, no SMILES). This script reconstructs each ligand's molecular graph from
the 3D coordinates using RDKit distance-based bond perception (covalent radius
sum * ``--bond-tol``), removes explicit hydrogens, sanitizes and canonicalizes.

Outputs (all under ``--out-dir``):
  ``ligand_smiles.csv``      per-complex ``complex_id, smiles``
  ``ligands_unique.txt``     one canonical SMILES per unique ligand (for ChemBERTa)
  ``complex_to_smiles.json`` ``{complex_id: smiles}`` (convenience)

Note: bond orders are perceived from geometry only; stereo may be less reliable
than an authoritative source. Good enough for frozen ChemBERTa features.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Geometry import Point3D

rdBase.DisableLog("rdApp.*")
PERIODIC = Chem.GetPeriodicTable()

# Canonical 1-letter amino acids (unused here, kept for symmetry with ESM-2).
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("data/pdbbind2020/index-pfam.csv"))
    p.add_argument("--root-dir", type=Path, default=Path("data/pdbbind2020/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--ligand-pattern", type=str, default="{c}_ligand_rnum.pdb.pkl")
    p.add_argument("--bond-tol", type=float, default=1.35,
                   help="Bond threshold as multiple of covalent radius sum.")
    return p.parse_args()


def coords_to_smiles(coords, elements, bond_tol):
    coords = np.asarray(coords, dtype=float)
    els = np.asarray(elements)
    n = len(els)
    m = Chem.RWMol()
    for e in els:
        m.AddAtom(Chem.Atom(str(e)))
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(float(coords[0, i]), float(coords[1, i]), float(coords[2, i])))
    m.AddConformer(conf)
    for i in range(n):
        ri = PERIODIC.GetRcovalent(m.GetAtomWithIdx(i).GetAtomicNum())
        for j in range(i + 1, n):
            rj = PERIODIC.GetRcovalent(m.GetAtomWithIdx(j).GetAtomicNum())
            d = float(np.linalg.norm(coords[:, i] - coords[:, j]))
            if 0.1 < d < bond_tol * (ri + rj):
                m.AddBond(i, j, Chem.BondType.SINGLE)
    mol = Chem.Mol(m)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def main():
    args = parse_args()
    import pandas as pd

    df = pd.read_csv(args.index, low_memory=False)
    rows, smiles_to_id = [], {}
    n_ok = n_fail = 0
    for cid in df["id"]:
        path = args.root_dir / args.ligand_pattern.format(c=cid)
        if not path.exists():
            continue
        import pickle
        try:
            lig = pickle.load(open(path, "rb"))
            coords = lig.coords.numpy() if hasattr(lig.coords, "numpy") else np.array(lig.coords)
            elements = np.array(lig.element_symbols)
            smi = coords_to_smiles(coords, elements, args.bond_tol)
            if not smi:
                n_fail += 1
                continue
            rows.append((cid, smi))
            smiles_to_id.setdefault(smi, cid)
            n_ok += 1
        except Exception:
            n_fail += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "ligand_smiles.csv"
    pd.DataFrame(rows, columns=["complex_id", "smiles"]).to_csv(out_csv, index=False)

    unique = sorted(smiles_to_id)
    (args.out_dir / "ligands_unique.txt").write_text("\n".join(unique) + "\n")

    (args.out_dir / "complex_to_smiles.json").write_text(json.dumps(dict(rows)))

    print(f"complexos com SMILES: {n_ok}, falhas: {n_fail}, ligantes únicos: {len(unique)}")
    print(f"escrito: {out_csv}, ligands_unique.txt, complex_to_smiles.json")


if __name__ == "__main__":
    main()
