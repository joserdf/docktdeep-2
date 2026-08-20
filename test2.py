import torch
from itertools import combinations
from time import time
from train import run, parse_args
import sys

def subconjuntos_uniao(A, B):
    U = A.union(B)
    elementos = list(U)
    resultado = []
    
    for r in range(len(elementos) + 1):
        for c in combinations(elementos, r):
            if c:
                resultado.append(tuple(c))
    
    return set(resultado)

if __name__ == "__main__":

    original_views = [
        #["BasicView"],
        #["VolumeView"],
        ["BasicView", "VolumeView"],
    ]

    basic_views = [
        ["BasicViewLigProt"],
        #["VolumeViewLigProt"],
        #["VolumeViewLigProt", "VolumeViewComplex"],
        #["BasicViewLigProt", "VolumeViewLigProt"],
        #["BasicViewLigProt", "VolumeViewComplex"],
        #["BasicViewLigProt", "VolumeViewLigProt", "VolumeViewComplex"],
        #["BasicViewComplex", "VolumeViewLigProt"],
        #["BasicViewComplex", "VolumeViewLigProt", "VolumeViewComplex"],
        #["BasicViewLigProt", "BasicViewComplex"],
        #["BasicViewLigProt", "BasicViewComplex", "VolumeViewLigProt"],
        #["BasicViewLigProt", "BasicViewComplex", "VolumeViewComplex"],
        #["BasicViewLigProt", "BasicViewComplex", "VolumeViewLigProt", "VolumeViewComplex"],
    ]

    new_views = [
        #["NewViewLigProt"],
        #["NewViewLigProt", "VolumeViewLigProt"],
        #["NewViewLigProt", "VolumeViewComplex"],
        #["NewViewLigProt", "VolumeViewLigProt", "VolumeViewComplex"],
        #["NewViewComplex", "VolumeViewLigProt"],
        #["NewViewComplex", "VolumeViewLigProt", "VolumeViewComplex"],
        #["NewViewLigProt", "NewViewComplex"],
        #["NewViewLigProt", "NewViewComplex", "VolumeViewLigProt"],
        #["NewViewLigProt", "NewViewComplex", "VolumeViewComplex"],
        #["NewViewLigProt", "NewViewComplex", "VolumeViewLigProt", "VolumeViewComplex"],
    ]

    occs = ["vdw", "vdw2_matmul", "gaussian", "vdw2", "gaussian_amax"]

    split_column = "coreset_v2016"

    experiments = []

    for occ in occs:
        if occ == "vdw":
            views = basic_views
        else:
            views = basic_views + new_views
        for view in views:
            experiments.append({
                "split_column": split_column,
                "view": view,
                "occupancy": occ,
            })
    
    # order experiments by view and occupancy
    experiments.sort(key=lambda x: (len(x['view']), '_'.join(sorted(x['view'])), x['occupancy']))

    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")

    for i, experiment in enumerate(experiments):
        split_column = experiment.get("split_column", "random_split")
        view = experiment.get("view", ["VolumeView", "BasicView"])
        occupancy = experiment.get("occupancy", "vdw")
        experiment_name = f"docktdeep_split_{split_column}_view_{'_'.join(view)}_occ_{occupancy}"

        namespace = {
            "model": "Baseline",
            "experiment": experiment_name,
            "depthwise-convs": True,
            "adaptive-pooling": True,
            "optim": "AdamW",
            "max-epochs": 1500,
            "batch-size": 64,
            "lr": 0.00087469,
            "beta1": 0.25693012,
            "eps": 0.00032933,
            "dropout": 0.25348994,
            "wdecay": 0.0000169,
            "molecular-dropout": 0.06,
            "molecular-dropout-unit": "complex",
            "random-rotation": True,
            "dataframe-path": "/data/mpds/pdbbind2020/index-pfam.csv",
            "root-dir": "/data/mpds/pdbbind2020/processed/",
            "ligand-path-pattern": "{c}_ligand_rnum.pdb.pkl",
            "protein-path-pattern": "{c}_protein_prep.pdb.pkl",
            "split-column": split_column,
            "occupancy": occupancy,
            "view": view,
            "merge-val-test": True,
            "num-workers": 0,
        }
        _old_argv = sys.argv
        for key, value in namespace.items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f"--{key.replace('_', '-')}")
            elif isinstance(value, list):
                sys.argv.append(f"--{key.replace('_', '-')}")
                for v in value:
                    sys.argv.append(str(v))
            else:
                sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        try:
            args = parse_args()
        finally:
            sys.argv = _old_argv

        print(f"Starting experiment {i+1}/{total_experiments}: {experiment_name}")
        start_time = time()
        run(args)
        end_time = time()
        print(f"Finished experiment {i+1}/{total_experiments} in {(end_time - start_time)/3600:.2f} hours: {experiment_name}")

        torch.cuda.empty_cache()
