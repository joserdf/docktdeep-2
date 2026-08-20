#!/usr/bin/env python3
"""Pre-compute frozen ChemBERTa embeddings for unique ligand SMILES.

Reads ``data/embeddings/ligands_unique.txt`` (one canonical SMILES per line),
runs ChemBERTa (HuggingFace ``seyonec/ChemBERTa-zinc-base-v1``) and caches the
``<cls>``-pooled embedding (768-d) as one ``.npy`` per SMILES hash under
``data/embeddings/chemberta/<smiles_hash>.npy``.

Also writes:
  ``chemberta/smiles_to_hash.json``  ``{smiles: hash}``
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    smiles_path = args.embeddings_dir / "ligands_unique.txt"
    smiles = [s.strip() for s in smiles_path.read_text().splitlines() if s.strip()]
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"{len(smiles)} SMILES únicos")

    model_id = "seyonec/ChemBERTa-zinc-base-v1"
    out_dir = args.embeddings_dir / "chemberta"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    smiles_to_hash = {}
    with torch.no_grad():
        for i in range(0, len(smiles), args.batch_size):
            batch = smiles[i : i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            last = model(**enc).last_hidden_state
            # <cls> token embedding
            cls_vec = last[:, 0, :]
            for smi, vec in zip(batch, cls_vec):
                h = hashlib.sha1(smi.encode()).hexdigest()[:16]
                smiles_to_hash[smi] = h
                np.save(out_dir / f"{h}.npy", vec.cpu().numpy())
    (out_dir / "smiles_to_hash.json").write_text(json.dumps(smiles_to_hash))
    print(f"ok: {out_dir} ({len(smiles_to_hash)} embeddings)")
    print("done")


if __name__ == "__main__":
    main()
