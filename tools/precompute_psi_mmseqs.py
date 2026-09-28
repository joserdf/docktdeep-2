#!/usr/bin/env python3
"""Aligned pairwise sequence identity (PSI) with MMseqs2, in the ``S_prot.npz`` format.

``precompute_sim.py:psi_matrix`` compares residues by *position*, without an
alignment, so any N-terminal shift or indel drops the identity to chance
(PLAN-SPLIT-EIXOS §3.2).  This tool runs an MMseqs2 all-vs-all search over the
same unique sequences and writes a drop-in replacement matrix.

Identity of a pair = identical aligned residues / min(len_i, len_j), the same
denominator as the unaligned PSI, so a fragment of a longer chain still scores
high.  Identical residues come from ``fident * alnlen`` (this MMseqs2 build
writes ``nident`` as 0).  The pair keeps the larger of its two directions
(query/target), which makes the matrix symmetric.  Pairs MMseqs2 does not report
score 0.

Rows are the unique protein hashes sorted, plus one zero sentinel row, exactly as
``precompute_sim.py`` orders them, so ``prot_map.json`` is shared.

Outputs (under ``--out-dir``): ``S_prot_aln.npz`` (S, hashes, lens, sentinel).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen"]


def parse_args():
    emb = Path("data/embeddings")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-map", type=Path, default=emb / "complex_to_seq.json")
    p.add_argument("--seqs", type=Path, default=emb / "seqs.json")
    p.add_argument("--out-dir", type=Path, default=emb / "sim")
    p.add_argument("--mmseqs", type=str, default=str(Path.home() / ".local/bin/mmseqs"))
    p.add_argument("--sensitivity", type=float, default=7.5)
    p.add_argument("--max-seqs", type=int, default=20000,
                   help="prefilter hits kept per query; above the largest family")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--m8", type=Path, default=None,
                   help="reuse an existing search result (columns: %s)" % ",".join(COLS))
    return p.parse_args()


def search(args, hashes, seqs, work: Path) -> Path:
    fasta = work / "prot.fasta"
    with fasta.open("w") as f:
        for h in hashes:
            f.write(f">{h}\n{seqs[h]['sequence']}\n")
    m8 = work / "hits.m8"
    subprocess.run([args.mmseqs, "easy-search", str(fasta), str(fasta), str(m8), str(work / "tmp"),
                    "-s", str(args.sensitivity), "--max-seqs", str(args.max_seqs), "-e", "10",
                    "--threads", str(args.threads), "--format-output", ",".join(COLS)],
                   check=True, stdout=subprocess.DEVNULL)
    return m8


def main():
    args = parse_args()
    c2h = json.loads(args.seq_map.read_text())
    seqs = json.loads(args.seqs.read_text())
    hashes = sorted(set(c2h.values()))
    n = len(hashes)
    lens = np.array([len(seqs[h]["sequence"]) for h in hashes], dtype=np.int32)
    row = {h: i for i, h in enumerate(hashes)}

    with tempfile.TemporaryDirectory() as tmp:
        m8 = args.m8 or search(args, hashes, seqs, Path(tmp))
        hits = pd.read_csv(m8, sep="\t", header=None, usecols=range(len(COLS)), names=COLS,
                           dtype={"query": str, "target": str})
    if args.m8:
        print(f"[S_prot_aln] reused {m8}")
    i = hits["query"].map(row).to_numpy()
    j = hits["target"].map(row).to_numpy()
    ident = hits.fident.to_numpy() * hits.alnlen.to_numpy()
    psi = np.minimum(np.rint(100 * ident / np.minimum(lens[i], lens[j])), 100).astype(np.uint8)

    s = np.zeros((n + 1, n + 1), dtype=np.uint8)
    np.maximum.at(s, (i, j), psi)
    s[:n, :n] = np.maximum(s[:n, :n], s[:n, :n].T)
    s[np.arange(n), np.arange(n)] = 100
    per_query = hits.groupby("query").size()
    print(f"[S_prot_aln] {len(hits)} hits over {n} sequences; "
          f"max hits per query {per_query.max()} (cap {args.max_seqs})")
    off = s[:n, :n][np.triu_indices(n, k=1)]
    print(f"[S_prot_aln] off-diagonal: >0 {(off > 0).mean() * 100:.2f}%, "
          f">=30 {(off >= 30).mean() * 100:.3f}%, >=90 {(off >= 90).mean() * 100:.3f}%")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "S_prot_aln.npz", S=s, hashes=np.array(hashes), lens=lens, sentinel=n)
    print(f"[S_prot_aln] wrote {args.out_dir / 'S_prot_aln.npz'}")


if __name__ == "__main__":
    main()
