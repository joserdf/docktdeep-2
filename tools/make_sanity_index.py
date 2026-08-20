#!/usr/bin/env python3
"""Build a small PDBbind2020 index for M1 / Fase 0 sanity runs.

Samples a handful of complexes per split from the full index, keeping only
complexes whose protein and ligand voxelization pickles exist in ``processed/``.
The output CSV has the columns the datamodule actually uses: ``id``, ``delta_g``
and the split column (``random_split``). This keeps M1 sanity runs fast and
reproducible without touching the 22 GB dataset.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        type=Path,
        default=Path("data/pdbbind2020/index-pfam.csv"),
        help="Full PDBbind2020 index CSV.",
    )
    p.add_argument(
        "--root-dir",
        type=Path,
        default=Path("data/pdbbind2020/processed"),
        help="Directory with the processed pickles.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/pdbbind2020/index-sanity.csv"),
        help="Output sanity index CSV.",
    )
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--validation", type=int, default=8)
    p.add_argument("--test", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--split-column", type=str, default="random_split")
    p.add_argument(
        "--protein-pattern", type=str, default="{c}_protein_prep.pdb.pkl"
    )
    p.add_argument(
        "--ligand-pattern", type=str, default="{c}_ligand_rnum.pdb.pkl"
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.index, low_memory=False)

    def has_pkl(complex_id: str) -> bool:
        return os.path.exists(
            args.root_dir / args.protein_pattern.format(c=complex_id)
        ) and os.path.exists(args.root_dir / args.ligand_pattern.format(c=complex_id))

    df = df[df["id"].apply(has_pkl)]

    parts = []
    for split, n in [
        ("train", args.train),
        ("validation", args.validation),
        ("test", args.test),
    ]:
        sub = df[df[args.split_column] == split]
        if len(sub) < n:
            print(
                f"warning: split {split!r} has only {len(sub)} rows (< {n}); taking all"
            )
            n = len(sub)
        parts.append(sub.sample(n, random_state=args.seed))

    sanity = pd.concat(parts)
    cols = [
        c
        for c in ["id", "delta_g", args.split_column]
        if c in sanity.columns
    ]
    sanity = sanity[cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sanity.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sanity)} rows)")
    print(sanity[args.split_column].value_counts().to_string())


if __name__ == "__main__":
    main()
