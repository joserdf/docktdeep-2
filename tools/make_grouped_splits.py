#!/usr/bin/env python3
"""Build grouped (Pfam-cluster) split columns for the 2^3 ablation grid.

Motivation
----------
The ``pfam_split_r*_f*`` columns that ship with ``index-pfam.csv`` reproduce the
Pfam-CV protocol of Zhu, Yang & Huang (JCIM 2022, doi 10.1021/acs.jcim.2c01149).
That protocol holds out whole Pfam clusters as the *test* fold, but then carves
the *validation* set at random out of the remaining folds. Measured on this very
index, 97-98% of the validation complexes share a Pfam cluster with training
while 0% of the test complexes do. Since ``train.py`` selects checkpoints with
``ModelCheckpoint(monitor="val_pearsonr")``, model selection ends up optimizing a
distribution that the test set does not follow.

This tool fixes that: the validation slice is itself carved at cluster
granularity, so the selection signal and the evaluation signal agree.

Output columns
--------------
``grp_cluster``   Pfam cluster label (from the paper's ``pdbbind_2020_cluster_result.csv``).
``grp_holdout``   ``dev`` / ``test`` -- the frozen hold-out, decided once.
``grp_stratum``   ``dev`` / ``ood`` / ``casf`` -- how each hold-out row may be reported.
``grp_cv_o{o}``   ``train`` / ``validation`` / ``test`` -- grouped CV over ``dev``.
``grp_final``     ``train`` / ``validation`` / ``test`` -- final refit, test = hold-out.

Rows outside the filtered population get NaN everywhere, which ``PDBbind._split_dataset``
already ignores.

The CASF core set
-----------------
The 261 CASF-2016 core-set complexes sit in 31 clusters whose combined footprint is
~56% of the population, so pinning whole clusters to the hold-out is not possible.
Instead the core-set *complexes* are pinned (you never train on one) and flagged as
the ``casf`` stratum: literature-comparable, but still family-overlapping with train.
The ``ood`` stratum is the cluster-disjoint one. Never average the two together.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MEGA_DEFAULT = "pkinase"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("data/pdbbind2020/index-pfam.csv"))
    p.add_argument("--clusters", type=Path,
                   default=Path("data/pdbbind2020/pdbbind_2020_cluster_result.csv"),
                   help="Pfam cluster table from hnlab/generalization_benchmark.")
    p.add_argument("--out", type=Path, default=Path("data/pdbbind2020/index-grouped.csv"))
    p.add_argument("--cl1", dest="cl1", action="store_true", default=True,
                   help="Keep only LP-PDBBind cleanliness level CL1 (default).")
    p.add_argument("--no-cl1", dest="cl1", action="store_false",
                   help="Skip the CL1 filter, keeping ~4.4k more complexes.")
    p.add_argument("--folds", type=int, default=4,
                   help="Outer grouped folds over dev. Default 4, which is what makes "
                        "the mega-cluster fit as a fold of its own.")
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.18,
                   help="Fraction of each outer training pool reserved as a grouped "
                        "validation slice.")
    p.add_argument("--mega-cluster", type=str, default=MEGA_DEFAULT,
                   help="Cluster too large to balance; kept out of the hold-out and "
                        "out of every validation slice, and given its own outer fold. "
                        "Pass '' to disable the special case.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def greedy_partition(sizes, k, rng):
    """Assign whole clusters to k parts, largest first, always into the smallest part.

    Largest-first (LPT) balances far better than the shuffle-then-fill loop used in
    the paper's ``fold_generate.ipynb``; the seeded shuffle only breaks ties among
    equal-sized clusters, so different seeds still give different partitions.
    """
    order = list(sizes)
    rng.shuffle(order)
    order.sort(key=lambda c: -sizes[c])
    parts = [[] for _ in range(k)]
    totals = [0] * k
    for c in order:
        i = int(np.argmin(totals))
        parts[i].append(c)
        totals[i] += sizes[c]
    return parts, totals


def take_clusters_until(sizes, target, rng, exclude=()):
    """Pick whole clusters at random until their combined size reaches ``target``."""
    pool = [c for c in sizes if c not in exclude]
    rng.shuffle(pool)
    chosen, total = [], 0
    for c in pool:
        if total >= target:
            break
        chosen.append(c)
        total += sizes[c]
    return set(chosen), total


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    idx = pd.read_csv(args.index, low_memory=False)
    clu = pd.read_csv(args.clusters, low_memory=False)
    df = idx.merge(clu[["pdb", "PCV_cluster"]], left_on="id", right_on="pdb", how="left")
    df = df.drop(columns=["pdb"]).rename(columns={"PCV_cluster": "grp_cluster"})

    # --- population: filter first, so leakage control acts on the final set ---
    keep = df.grp_cluster.notna() & (df.random_split != "ERR") & (~df.covalent)
    if args.cl1:
        keep &= df.CL1
    pop = df[keep]
    sizes = pop.groupby("grp_cluster").size().to_dict()
    print(f"population: {len(pop)} complexes in {len(sizes)} clusters "
          f"(CL1={'on' if args.cl1 else 'off'})")

    mega = args.mega_cluster if args.mega_cluster in sizes else None
    if mega:
        print(f"mega-cluster {mega!r}: {sizes[mega]} complexes "
              f"({100 * sizes[mega] / len(pop):.1f}% of the population)")

    # --- 1. frozen hold-out, carved at cluster granularity ---
    target = args.holdout_frac * len(pop)
    test_clusters, n_test = take_clusters_until(
        sizes, target, rng, exclude={mega} if mega else set()
    )
    df["grp_holdout"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[keep, "grp_holdout"] = "dev"
    is_test = keep & df.grp_cluster.isin(test_clusters)
    df.loc[is_test, "grp_holdout"] = "test"

    # CASF core-set complexes are pinned individually: their clusters are far too
    # large to move wholesale, so they stay family-overlapping and get their own
    # stratum rather than being silently mixed into the OOD number.
    is_casf = keep & (df.coreset_v2016 == True) & ~is_test  # noqa: E712
    df.loc[is_casf, "grp_holdout"] = "test"
    df["grp_stratum"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[keep, "grp_stratum"] = "dev"
    df.loc[is_test, "grp_stratum"] = "ood"
    df.loc[is_casf, "grp_stratum"] = "casf"

    dev = df[keep & (df.grp_holdout == "dev")]
    dev_sizes = dev.groupby("grp_cluster").size().to_dict()
    print(f"hold-out: {int(is_test.sum())} ood ({len(test_clusters)} clusters) "
          f"+ {int(is_casf.sum())} casf = {int(is_test.sum() + is_casf.sum())} "
          f"({100 * (is_test.sum() + is_casf.sum()) / len(pop):.1f}%)")
    print(f"dev: {len(dev)} complexes in {len(dev_sizes)} clusters")

    # --- 2. grouped outer CV over dev, mega-cluster as a fold of its own ---
    rest = {c: n for c, n in dev_sizes.items() if c != mega}
    if mega and mega in dev_sizes:
        parts, totals = greedy_partition(rest, args.folds - 1, rng)
        parts = [[mega]] + parts
        totals = [dev_sizes[mega]] + totals
    else:
        parts, totals = greedy_partition(rest, args.folds, rng)

    for o, (fold_clusters, total) in enumerate(zip(parts, totals), start=1):
        col = f"grp_cv_o{o}"
        df[col] = pd.Series(np.nan, index=df.index, dtype=object)
        in_dev = keep & (df.grp_holdout == "dev")
        in_fold = in_dev & df.grp_cluster.isin(fold_clusters)

        # The validation slice is carved from the *clusters* of the training pool,
        # never at random -- this is the whole point of the tool.
        pool = {c: n for c, n in dev_sizes.items()
                if c not in fold_clusters and c != mega}
        val_clusters, _ = take_clusters_until(
            pool, args.val_frac * sum(pool.values()), rng
        )
        in_val = in_dev & df.grp_cluster.isin(val_clusters)

        df.loc[in_dev, col] = "train"
        df.loc[in_val, col] = "validation"
        df.loc[in_fold, col] = "test"
        print(f"  {col}: train={int((df[col] == 'train').sum())} "
              f"val={int((df[col] == 'validation').sum())} "
              f"test={int((df[col] == 'test').sum())} "
              f"({len(fold_clusters)} test clusters"
              f"{', mega' if mega in fold_clusters else ''})")

    # --- 3. final refit column: train on dev, evaluate on the frozen hold-out ---
    pool = {c: n for c, n in dev_sizes.items() if c != mega}
    val_clusters, _ = take_clusters_until(pool, args.val_frac * sum(pool.values()), rng)
    df["grp_final"] = pd.Series(np.nan, index=df.index, dtype=object)
    in_dev = keep & (df.grp_holdout == "dev")
    df.loc[in_dev, "grp_final"] = "train"
    df.loc[in_dev & df.grp_cluster.isin(val_clusters), "grp_final"] = "validation"
    df.loc[keep & (df.grp_holdout == "test"), "grp_final"] = "test"
    print(f"  grp_final: train={int((df.grp_final == 'train').sum())} "
          f"val={int((df.grp_final == 'validation').sum())} "
          f"test={int((df.grp_final == 'test').sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows, "
          f"{sum(c.startswith('grp_') for c in df.columns)} grp_* columns)")


if __name__ == "__main__":
    main()
