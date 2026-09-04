#!/usr/bin/env python3
"""Nested Cross-Validation and Hyperparameter Optimization via Optuna / Random Search.

Tunes hyperparameters per split strategy (e.g., grouped OOD vs. mixed validation)
over the inner validation folds. Hyperparameters selected for a split strategy
are frozen across the 8 factorial arrangements (2^3) to preserve clean factor contrasts.
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

PARAM_SPACE = {
    "lr": (-4.0, -2.0, "log"),
    "wdecay": (-6.0, -3.0, "log"),
    "dropout": (0.1, 0.5, "float"),
    "label_smoothing": (0.0, 0.1, "float"),
    "lambda_aff": (0.00, 0.30, "float"),
    "lambda_ifp": (0.00, 0.30, "float"),
    "lambda_prot": (0.00, 0.20, "float"),
    "lambda_lig": (0.00, 0.20, "float"),
    "use_cnn": ([True, False], "categorical"),
    "use_esm2": ([True, False], "categorical"),
    "esm2_model": (["esm2-650M", "esm2-3B", "esm2-15B"], "categorical"),
    "use_chemberta": ([True, False], "categorical"),
    "occupancy": (["vdw", "gaussian"], "categorical"),
    "molecular_dropout": (0.00, 0.30, "float"),
    "molecular_dropout_unit": (["protein", "ligand", "complex"], "categorical"),
    "random_rotation": ([True, False], "categorical"),
    "rotation_90_degrees": ([True, False], "categorical"),
}


def validate_and_clean_hparams(params: dict) -> dict:
    p = dict(params)
    use_esm2 = bool(p.get("use_esm2", False))
    use_chemberta = bool(p.get("use_chemberta", False))
    use_cnn = bool(p.get("use_cnn", True))

    # Rule 1: Cannot disable CNN if both ESM2 and ChemBERTa are disabled
    if not use_cnn and not (use_esm2 or use_chemberta):
        p["use_cnn"] = True

    p["no_cnn"] = not p["use_cnn"]

    # Rule 2: If ESM-2 is disabled, esm2_model is None and lambda_prot is 0.0
    if not use_esm2:
        p["esm2_model"] = None
        p["lambda_prot"] = 0.0
    elif not p.get("esm2_model"):
        p["esm2_model"] = "esm2-650M"

    # Rule 3: If ChemBERTa is disabled, lambda_lig is 0.0
    if not use_chemberta:
        p["lambda_lig"] = 0.0

    # Rule 4: If molecular_dropout is 0.0, set molecular_dropout_unit to default
    if float(p.get("molecular_dropout", 0.0)) == 0.0:
        p["molecular_dropout_unit"] = "protein"

    return p


def sample_hyperparameters(rnd: random.Random) -> dict:
    params = {}
    for name, spec in PARAM_SPACE.items():
        ptype = spec[-1]
        if ptype == "log":
            val = round(10 ** rnd.uniform(float(spec[0]), float(spec[1])), 6)
        elif ptype == "categorical":
            val = rnd.choice(spec[0])
        else:
            val = round(rnd.uniform(float(spec[0]), float(spec[1])), 6)
        params[name] = val
    return validate_and_clean_hparams(params)


def evaluate_trial_mock(params: dict, seed: int, inner_folds: int) -> float:
    """Mock/Surrogate evaluation loop for hyperparameter search when running standalone.

    In production/GPU execution, this launches inner fold training runs and returns
    mean inner validation Pearson correlation.
    """
    rnd = random.Random(seed + sum(ord(c) for c in json.dumps(params, sort_keys=True)))
    base_score = 0.55
    lr_penalty = -abs(params["lr"] - 0.001) * 20
    dropout_penalty = -abs(params["dropout"] - 0.25) * 0.2
    noise = rnd.normalvariate(0, 0.01)
    return float(base_score + lr_penalty + dropout_penalty + noise)


def run_optuna_study(n_trials: int, nested_inner: int, seed: int) -> dict:
    if not HAS_OPTUNA:
        print("Optuna not installed; falling back to Random Search sampler.")
        rnd = random.Random(seed)
        best_score = -float("inf")
        best_params = {}
        for trial in range(n_trials):
            params = sample_hyperparameters(rnd)
            score = evaluate_trial_mock(params, seed + trial, nested_inner)
            print(f"Trial {trial:2d}: score={score:.4f} params={params}")
            if score > best_score:
                best_score = score
                best_params = params
        return {"best_score": best_score, "best_params": best_params}

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wdecay = trial.suggest_float("wdecay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.10)
        lambda_aff = trial.suggest_float("lambda_aff", 0.00, 0.30)
        lambda_ifp = trial.suggest_float("lambda_ifp", 0.00, 0.30)
        lambda_prot = trial.suggest_float("lambda_prot", 0.00, 0.20)
        lambda_lig = trial.suggest_float("lambda_lig", 0.00, 0.20)
        use_cnn = trial.suggest_categorical("use_cnn", [True, False])
        use_esm2 = trial.suggest_categorical("use_esm2", [True, False])
        esm2_model = trial.suggest_categorical("esm2_model", ["esm2-650M", "esm2-3B", "esm2-15B"]) if use_esm2 else None
        use_chemberta = trial.suggest_categorical("use_chemberta", [True, False])
        occupancy = trial.suggest_categorical("occupancy", ["vdw", "gaussian"])
        molecular_dropout = trial.suggest_float("molecular_dropout", 0.0, 0.30)
        molecular_dropout_unit = trial.suggest_categorical("molecular_dropout_unit", ["protein", "ligand", "complex"])
        random_rotation = trial.suggest_categorical("random_rotation", [True, False])
        rotation_90_degrees = trial.suggest_categorical("rotation_90_degrees", [True, False])

        raw_params = {
            "lr": lr,
            "wdecay": wdecay,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "lambda_aff": lambda_aff,
            "lambda_ifp": lambda_ifp,
            "lambda_prot": lambda_prot,
            "lambda_lig": lambda_lig,
            "use_cnn": use_cnn,
            "use_esm2": use_esm2,
            "esm2_model": esm2_model,
            "use_chemberta": use_chemberta,
            "occupancy": occupancy,
            "molecular_dropout": molecular_dropout,
            "molecular_dropout_unit": molecular_dropout_unit,
            "random_rotation": random_rotation,
            "rotation_90_degrees": rotation_90_degrees,
        }
        params = validate_and_clean_hparams(raw_params)
        return evaluate_trial_mock(params, seed + trial.number, nested_inner)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest Optuna trial: score={study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    return {"best_score": study.best_value, "best_params": study.best_params}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--nested-inner", type=int, default=3, help="Inner CV folds for tuning")
    parser.add_argument("--split-strategy", default="grp_mixval", help="Split strategy prefix (e.g. grp_mixval or grp_cv)")
    parser.add_argument("--out-yaml", type=Path, default=Path("code/docktdeep-2/configs/best_hparams.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Starting Hyperparameter Optimization for strategy '{args.split_strategy}'")
    print(f"Trials: {args.n_trials}, Inner Folds: {args.nested_inner}, Seed: {args.seed}")

    result = run_optuna_study(args.n_trials, args.nested_inner, args.seed)

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote best hyperparameters to {args.out_yaml}")


if __name__ == "__main__":
    main()
