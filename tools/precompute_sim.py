#!/usr/bin/env python3
"""Precompute pairwise similarity matrices for the similarity-term ablation.

Sources (all under ``data/embeddings/``):
  ``complex_to_seq.json``      {complex_id: protein_hash}   (19,118 complexes)
  ``seqs.json``                {protein_hash: {"sequence": str}} (12,717 uniques)
  ``complex_to_smiles.json``   {complex_id: canonical_smiles} (18,746 complexes)

Outputs (under ``--out-dir``, default ``data/embeddings/sim/``):
  ``S_prot.npz``    uint8 (0-100) pairwise sequence identity (PSI), rows = unique
                    proteins (sorted by hash) + 1 shared zero sentinel row
  ``prot_map.json`` {complex_id: matrix row index}
  ``S_lig.npz``     uint8 (0-100) Tanimoto of Morgan fingerprints, rows = unique
                    SMILES (sorted) + 1 shared zero sentinel row
  ``lig_map.json``  {complex_id: matrix row index} (missing SMILES -> sentinel)

Conventions (see docs/proposta-ablacao-similaridade.md, sections 3-5):
  - PSI: position-wise identity on UNALIGNED full-length sequences; denominator
    = min(len_i, len_j); PAD positions never match (unknown 'X' residues DO
    count as matches, standard for position-wise identity). Interpretation:
    off-diagonal mass concentrates on same-deposited-protein pairs.
  - Tanimoto: Morgan radius 2, 2048 bits, one fingerprint per unique SMILES.
  - Sentinel row: complexes absent from the source maps point to it; its row
    and column are all zero, so they contribute zero gradient downstream
    (same convention as IFP rows without a positive partner).
  - Diagonal = 100 for real rows, 0 for the sentinel.

Deterministic: unique rows are sorted; chunked int32 GEMMs (no float noise).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

AA2I = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8,
    "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16,
    "W": 17, "Y": 18, "V": 19,
}
X_CODE = 20
PAD_CODE = 21  # excluded from matches


def parse_args():
    emb = Path("data/embeddings")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-map", type=Path, default=emb / "complex_to_seq.json")
    p.add_argument("--seqs", type=Path, default=emb / "seqs.json")
    p.add_argument("--smiles-map", type=Path, default=emb / "complex_to_smiles.json")
    p.add_argument("--out-dir", type=Path, default=emb / "sim")
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-bits", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=256,
                   help="Row chunk size for the GEMMs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Only the first N unique rows (0 = all); for quick tests.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Protein: PSI                                                                 #
# --------------------------------------------------------------------------- #

def encode_sequences(hashes, seqs_by_hash):
    """Return (codes, lens, n_sentinel_rows).

    codes: uint8 (N, Lmax) with AA2I codes, X_CODE, PAD_CODE (right-padded).
    lens:  int32 (N,) true lengths.
    """
    missing = [h for h in hashes if h not in seqs_by_hash]
    if missing:
        raise SystemExit(f"{len(missing)} hashes referenced by the map but "
                         f"absent from seqs.json (ex.: {missing[:3]})")
    seqs = [seqs_by_hash[h]["sequence"] for h in hashes]
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
    lmax = int(lens.max())
    codes = np.full((len(seqs), lmax), PAD_CODE, dtype=np.uint8)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            codes[i, j] = X_CODE if ch == "X" else AA2I[ch]
    return codes, lens


def psi_matrix(codes, lens, chunk):
    """Position-wise identity, denominator min-length, uint8 percent (0-100).

    M[i, j] = number of positions (among 1..min(L_i, L_j)) where the residue
    matches (PAD never matches). Computed as a one-hot Gram: each residue
    channel contributes (codes==c) @ (codes==c).T.

    Exactness: entries are 0/1 and row sums are < 2^24, so the float32 GEMM
    (BLAS) returns exact integers -- numpy's integer matmul does not dispatch
    to BLAS and is ~50x slower.
    """
    n = len(lens)
    lmax = codes.shape[1]
    nch = X_CODE + 1  # channels 0..20 (PAD excluded)
    onehot = np.zeros((n, nch * lmax), dtype=np.float32)
    for c in range(nch):
        onehot[:, c * lmax:(c + 1) * lmax] = codes == c
    print(f"  one-hot f32: {onehot.nbytes / 2**30:.1f} GiB")

    m = np.zeros((n, n), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, chunk):
        a = onehot[i:i + chunk]
        for j in range(i, n, chunk):
            b = onehot[j:j + chunk]
            block = a @ b.T
            m[i:i + chunk, j:j + chunk] = block
            m[j:j + chunk, i:i + chunk] = block.T
        if (i // chunk) % 5 == 0:
            el = time.time() - t0
            done = (i + chunk) * n
            print(f"  prot GEMM: {i + chunk}/{n} rows "
                  f"({100 * done / (n * n):.0f}%), {el:.0f}s elapsed")
    # divide by min-length (per outer chunk to avoid a second (n, n) matrix)
    out = np.zeros((n, n), dtype=np.uint8)
    for i in range(0, n, chunk):
        denom = np.minimum(lens[i:i + chunk, None], lens[None, :]).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.rint(100.0 * m[i:i + chunk] /
                          np.where(denom > 0, denom, 1.0))
        out[i:i + chunk] = np.clip(pct, 0, 100).astype(np.uint8)
    return out


def build_prot(args):
    print(f"[S_prot] seq map: {args.seq_map}")
    c2h = json.loads(args.seq_map.read_text())
    seqs = json.loads(args.seqs.read_text())
    hashes = sorted(set(c2h.values()))
    if args.limit:
        hashes = hashes[: args.limit]
    print(f"[S_prot] {len(hashes)} unique proteins "
          f"({len(c2h)} complexes referenced)")
    codes, lens = encode_sequences(hashes, seqs)
    print(f"[S_prot] lengths: min {lens.min()}, max {lens.max()}, "
          f"mean {lens.mean():.0f}")
    s = psi_matrix(codes, lens, args.chunk)
    # append shared sentinel row (all zero, diagonal 0)
    n = len(hashes)
    full = np.zeros((n + 1, n + 1), dtype=np.uint8)
    full[:n, :n] = s
    np.savez(args.out_dir / "S_prot.npz", S=full, hashes=np.array(hashes),
             lens=lens, sentinel=n)
    hash_to_row = {h: i for i, h in enumerate(hashes)}
    prot_map = {c: hash_to_row.get(c2h[c], n) for c in c2h}
    args.out_dir.joinpath("prot_map.json").write_text(json.dumps(prot_map))
    n_sent = sum(1 for v in prot_map.values() if v == n)
    return full, prot_map, n_sent, len(hashes)


# --------------------------------------------------------------------------- #
# Ligand: Morgan Tanimoto                                                      #
# --------------------------------------------------------------------------- #

def morgan_bits(smiles_list, radius, nbits):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    rdBase.DisableLog("rdApp.*")
    # RDKit >= 2025: ExplicitBitVect.ToBitString() is a '0'/'1' string,
    # one char per bit, bit i at position i. Cross-checked once below
    # against the (deprecated) reference implementation.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    f = np.zeros((len(smiles_list), nbits), dtype=np.int8)
    bad, checked = [], False
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(smi)
            continue
        s = fpgen.GetFingerprint(mol).ToBitString()
        f[i] = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 0x30
        if not checked:
            ref = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
            if not all(int(s[k]) == ref[k] for k in range(0, nbits, 97)):
                raise SystemExit("MorganGenerator disagrees with the reference "
                                 "GetMorganFingerprintAsBitVect")
            checked = True
    if bad:
        raise SystemExit(f"{len(bad)} SMILES failed to parse (ex.: {bad[:3]})")
    return f, bad


def tanimoto_matrix(f, chunk):
    """Same float32-BLAS exactness argument as psi_matrix (bits, sums < 2^24)."""
    n = len(f)
    a32 = f.astype(np.float32)
    inter = np.zeros((n, n), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, chunk):
        a = a32[i:i + chunk]
        for j in range(i, n, chunk):
            b = a32[j:j + chunk]
            block = a @ b.T
            inter[i:i + chunk, j:j + chunk] = block
            inter[j:j + chunk, i:i + chunk] = block.T
        if (i // chunk) % 5 == 0:
            print(f"  lig GEMM: {i + chunk}/{n} rows, "
                  f"{time.time() - t0:.0f}s elapsed")
    pop = a32.sum(axis=1)
    out = np.zeros((n, n), dtype=np.uint8)
    for i in range(0, n, chunk):
        denom = pop[i:i + chunk, None] + pop[None, :] - inter[i:i + chunk]
        # denom > 0 always here (real rows have >=1 bit)
        pct = np.rint(100.0 * inter[i:i + chunk] / np.where(denom > 0, denom, 1.0))
        out[i:i + chunk] = np.clip(pct, 0, 100).astype(np.uint8)
    return out


def build_lig(args):
    c2s = json.loads(args.smiles_map.read_text())
    smiles = sorted(set(c2s.values()))
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"[S_lig] {len(smiles)} unique SMILES ({len(c2s)} complexes referenced)")
    f, _ = morgan_bits(smiles, args.morgan_radius, args.morgan_bits)
    print(f"  fingerprints: {f.nbytes / 2**20:.0f} MiB, "
          f"mean popcount {f.sum(axis=1).mean():.0f}")
    s = tanimoto_matrix(f, max(args.chunk, 512))
    n = len(smiles)
    full = np.zeros((n + 1, n + 1), dtype=np.uint8)
    full[:n, :n] = s
    np.savez(args.out_dir / "S_lig.npz", S=full, smiles=np.array(smiles),
             sentinel=n)
    smile_to_row = {smi: i for i, smi in enumerate(smiles)}
    lig_map = {c: smile_to_row.get(smi, n) for c, smi in c2s.items()}
    args.out_dir.joinpath("lig_map.json").write_text(json.dumps(lig_map))
    n_sent = sum(1 for v in lig_map.values() if v == n)
    return full, lig_map, n_sent, len(smiles)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #

def validate(name, s, n_real, sentinel):
    diag = np.diag(s[:n_real])
    ok_diag = bool((diag == 100).all())
    sent_ok = bool((s[sentinel] == 0).all() and (s[:, sentinel] == 0).all())
    off = s[np.triu_indices(n_real, k=1)]
    print(f"[validate {name}] shape={s.shape} dtype={s.dtype} "
          f"({s.nbytes / 2**20:.0f} MiB)")
    print(f"  diagonal real rows == 100: {ok_diag}")
    print(f"  sentinel row/col all zero: {sent_ok}")
    print(f"  off-diagonal: mean {off.mean():.2f}, p50 {np.percentile(off, 50):.0f}, "
          f"p99 {np.percentile(off, 99):.0f}, "
          f"==100: {(off == 100).mean() * 100:.3f}%, "
          f">0: {(off > 0).mean() * 100:.2f}%")
    if not (ok_diag and sent_ok):
        raise SystemExit(f"validation FAILED for {name}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    s_prot, prot_map, n_prot_sent, n_prot = build_prot(args)
    validate("S_prot", s_prot, n_prot, n_prot)
    print(f"  complexes -> sentinel: {n_prot_sent}/{len(prot_map)}")

    s_lig, lig_map, n_lig_sent, n_lig = build_lig(args)
    validate("S_lig", s_lig, n_lig, n_lig)
    print(f"  complexes -> sentinel: {n_lig_sent}/{len(lig_map)}")

    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}/")


if __name__ == "__main__":
    main()
