#!/usr/bin/env python3
"""Derive protein amino-acid sequences from processed PDBbind2020 pickles.

Each protein pickle holds a biopandas ``PandasPdb`` whose ``ATOM`` records carry
``residue_name``/``chain_id``/``residue_number``. We reconstruct the 1-letter
sequence (ordered by chain then residue number, using only CA atoms), deduplicate
by sequence, and write:
  ``proteins_unique.fasta``     unique sequences (id = seq-hash)
  ``complex_to_seq.json``       ``{complex_id: seq_hash}``
  ``seqs.json``                 ``{seq_hash: {"sequence", "chains", "n_residues"}}``
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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
    p.add_argument("--protein-pattern", type=str, default="{c}_protein_prep.pdb.pkl")
    return p.parse_args()


def seq_from_pickle(path):
    import pickle
    p = pickle.load(open(path, "rb"))
    df = p.molecule_object.df["ATOM"]
    ca = df[df["atom_name"] == "CA"].sort_values(["chain_id", "residue_number"])
    chains = []
    for chain, g in ca.groupby("chain_id", sort=False):
        seq = "".join(AA3TO1.get(r, "X") for r in g["residue_name"])
        chains.append((str(chain), seq))
    return chains


def main():
    args = parse_args()
    import pandas as pd

    df = pd.read_csv(args.index, low_memory=False)
    seqs = {}          # seq_hash -> {"sequence", "chains"}
    complex_seq = {}   # complex_id -> seq_hash
    n_ok = n_fail = 0
    for cid in df["id"]:
        path = args.root_dir / args.protein_pattern.format(c=cid)
        if not path.exists():
            continue
        try:
            chains = seq_from_pickle(path)
            full = "".join(s for _, s in chains)
            if not full:
                n_fail += 1
                continue
            h = hashlib.sha1(full.encode()).hexdigest()[:16]
            seqs.setdefault(h, {"sequence": full, "chains": chains})
            complex_seq[str(cid)] = h
            n_ok += 1
        except Exception:
            n_fail += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.out_dir / "proteins_unique.fasta"
    with open(fasta, "w") as f:
        for h, info in seqs.items():
            f.write(f">{h}\n{info['sequence']}\n")
    (args.out_dir / "seqs.json").write_text(json.dumps(seqs))
    (args.out_dir / "complex_to_seq.json").write_text(json.dumps(complex_seq))

    lens = [len(i["sequence"]) for i in seqs.values()]
    print(f"complexos com sequência: {n_ok}, falhas: {n_fail}, sequências únicas: {len(seqs)}")
    print(f"comprimento sequência: min={min(lens) if lens else '-'}, "
          f"max={max(lens) if lens else '-'}, médio={np.mean(lens):.1f}" if lens else "")
    print(f"escrito: {fasta}, seqs.json, complex_to_seq.json")


if __name__ == "__main__":
    main()
