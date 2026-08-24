"""Backfill de metricas a partir dos checkpoints ja salvos.

O ModelCheckpoint gravou o melhor epoch de cada run (monitor=val_pearsonr), e o
.ckpt carrega os hiperparametros completos do treino -- views, occupancy,
vox_size, box_dims, split_column, seed e as flags da ablacao. Logo o modelo, o
voxel grid e o datamodule sao reconstruidos sem nenhum metadado externo, e o
conjunto completo de metricas (rmse/spearman/r2/...) sai por inferencia, sem
re-treinar.

Os caminhos de dados sao os unicos parametros que NAO vem do checkpoint: mudam
de maquina. Tudo o mais e sobrescrito pelo que o run realmente usou.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from src.docktdeep.dataset import PDBbind
from src.docktdeep.models import Baseline
from train import configure_voxel_grid


def _grid_args(hp: dict, voxel_device: str) -> argparse.Namespace:
    """configure_voxel_grid le atributos de um Namespace, nao um dict."""
    return argparse.Namespace(
        view=hp["view"], vox_size=hp["vox_size"], box_dims=hp["box_dims"],
        occupancy=hp["occupancy"], voxel_device=voxel_device,
    )


def _has_test_split(hp: dict, dataframe_path: str) -> bool:
    """coreset_v20xx e as colunas booleanas devolvem o MESMO subconjunto para
    'validation' e para 'test' (dataset.py:_split_dataset). Avaliar os dois seria
    reportar o mesmo numero duas vezes."""
    import pandas as pd

    col = hp["split_column"]
    if col.startswith("coreset_"):
        return False
    values = set(pd.read_csv(dataframe_path, low_memory=False)[col].dropna().astype(str))
    return {"train", "validation", "test"}.issubset(values)


@torch.no_grad()
def _run_split(model, loader, device: str):
    preds, labels = [], []
    for batch in loader:
        if len(batch) == 4:
            x, e_prot, e_lig, y = batch
            x = x.to(device) if x is not None and torch.is_tensor(x) else x
            e_prot = e_prot.to(device) if torch.is_tensor(e_prot) else e_prot
            e_lig = e_lig.to(device) if torch.is_tensor(e_lig) else e_lig
        else:
            x, y = batch
            e_prot = e_lig = None
            x = x.to(device)
        preds.append(model(x, e_prot, e_lig).squeeze(-1).float().cpu())
        labels.append(y.float().cpu())
    return torch.cat(preds), torch.cat(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    # vazio = usa o que o run gravou no checkpoint. As grades agrupadas treinaram
    # com index-grouped-{cl1,nocl1}.csv e as demais com index-pfam.csv; deixar o
    # ckpt decidir evita ter que reconstruir esse mapeamento na submissao.
    ap.add_argument("--dataframe-path", default="")
    ap.add_argument("--root-dir", default="")
    ap.add_argument("--embeddings-dir", default="")
    ap.add_argument("--out-dir", default="", help="destino dos CSVs de predicao (vazio: nao grava)")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    # o mesmo pareamento do treino: docktgrid aloca o grid em cuda, e um worker
    # forkado nao consegue re-inicializar CUDA (ver comentario em train.py)
    ap.add_argument("--voxel-device", default="cuda")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    hp = dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["hyper_parameters"])
    dataframe_path = args.dataframe_path or hp["dataframe_path"]
    root_dir = args.root_dir or hp["root_dir"]
    embeddings_dir = args.embeddings_dir or hp["embeddings_dir"]

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in wanted and not _has_test_split(hp, dataframe_path):
        wanted.remove("test")

    voxel_grid = configure_voxel_grid(_grid_args(hp, args.voxel_device))

    dm_args = dict(hp)
    dm_args.update(
        dataframe_path=dataframe_path, root_dir=root_dir,
        embeddings_dir=embeddings_dir, batch_size=args.batch_size,
        num_workers=args.num_workers,
        # o run original rodou com merge_val_test=True e por isso nunca tocou o
        # teste; aqui o held-out precisa existir como dataset proprio
        merge_val_test=False,
        # sem augmentation na avaliacao
        molecular_dropout=0.0, random_rotation=False, rotation_90_degrees=False,
    )
    dm_args.pop("voxel_grid", None)
    dm_args.pop("transforms", None)
    dm = PDBbind(voxel_grid=voxel_grid, transforms=[], **dm_args)
    dm.setup(stage="test")

    model = Baseline.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval().to(args.device)

    out = {"ckpt": os.path.abspath(args.ckpt), "experiment": hp.get("experiment"),
           "seed": hp.get("seed"), "split_column": hp.get("split_column"),
           "use_esm2": hp.get("use_esm2"), "use_chemberta": hp.get("use_chemberta"),
           "semi": hp.get("semi"), "no_cnn": hp.get("no_cnn"), "metrics": {}}

    # split_column identifica o fold da CV agrupada (grp_cv_o1..o5); sem ele os
    # 4-5 folds de um mesmo experiment+seed gravariam o MESMO arquivo e so o
    # ultimo sobreviveria. Mesmo esquema de Baseline._dump_predictions no treino,
    # com o stage no fim porque aqui val e test saem do mesmo run.
    preds_stem = f"{hp.get('experiment')}__{hp['split_column']}__seed{hp.get('seed')}"

    for split in wanted:
        dataset = dm.val_dataset if split == "validation" else dm.test_dataset
        loader = dm.val_dataloader() if split == "validation" else dm.test_dataloader()
        preds, labels = _run_split(model, loader, args.device)
        # _regression_metrics do proprio modelo: as definicoes ficam identicas
        # as do treino, sem uma segunda implementacao para divergir
        stage = "val" if split == "validation" else "test"
        metrics = model._regression_metrics(preds, labels, stage)
        out["metrics"].update({k: float(v) for k, v in metrics.items()})
        out["metrics"][f"{stage}_n"] = int(preds.numel())

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            name = f"{preds_stem}__{stage}.csv"
            ids = getattr(dataset, "ids", None)
            if ids is not None and len(ids) == preds.numel():
                with open(os.path.join(args.out_dir, name), "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["id", "y_true", "y_pred"])
                    w.writerows(zip(ids, labels.tolist(), preds.tolist()))
            else:
                # um CSV sem id nao permite parear celulas nem estratificar
                print(f"[preds] {split}: ids ausentes ou desalinhados, CSV ignorado", flush=True)

    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
