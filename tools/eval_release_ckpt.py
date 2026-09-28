"""Predicoes do checkpoint do release upstream (docktdeep v0.2.0) no hold-out.

O `docktdeep-model.ckpt` do release nao carrega no `Baseline` atual: ele tem
`fc1.*`/`linear.*`, e o commit 78b19cf (M3) trocou esse par por `f_proj`/`head`.
Em vez de remapear chaves para uma arquitetura que evoluiu (latente, cabecas,
embeddings), este script reconstroi a arquitetura do proprio commit do release
(`git_hash` gravado no ckpt, 52da955: `models/baseline.py`) e carrega o
state_dict com `strict=True` -- qualquer divergencia aborta.

Duas diferencas de protocolo em relacao aos nossos runs, ambas tratadas aqui:

* Alvo. O upstream treina em `delta_g` (kcal/mol); nos, em `pki`. A predicao
  e convertida pela razao delta_g/pki do proprio dataframe (-1.3635, desvio
  7e-4), de modo que r nao muda e RMSE/MAE saem em unidades pK.
* Contaminacao. O `cmd` gravado no ckpt sugere o split default
  `random_split`, mas o proprio ckpt desmente: 448.500 passos em 1.500 epocas
  = 299 passos/epoca x batch 64 ~ 19,1 mil complexos -- o PDBbind inteiro, nao
  os 13.579 do `random_split == train` (seriam 213 passos). `monitor=None`, sem
  early stop. Medido em 27/09/2026 no diamante-03: r ~0,985 em TODAS as
  fatias, inclusive CASF e `random_split == test`. O release e um modelo de
  producao ajustado em 100% dos dados; nenhuma fatia do nosso hold-out e limpa
  para ele. A coluna `random_split` vai no CSV para documentar isso, nao para
  recortar um subconjunto "limpo".

Uso (da raiz do docktdeep-2):
    .venv/bin/python tools/eval_release_ckpt.py \\
        --ckpt ckpts/docktdeep-model.ckpt \\
        --dataframe-path ../../data/pdbbind2020/index-grouped-nocl1.csv \\
        --root-dir ../../data/pdbbind2020/processed \\
        --out ../ga-036-broker/results/refit/release-upstream__grp_final__seed42.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.docktdeep.dataset import PDBbind  # noqa: E402
from train import configure_voxel_grid  # noqa: E402

from src.docktdeep.models.blocks import ConvGroup, ConvGroupDepthwise  # noqa: E402


class FCGroup(torch.nn.Sequential):
    """A do commit 52da955, com o BatchNorm1d(1000) literal de la."""

    def __init__(self, in_c, out_c, dropout_rate):
        super().__init__(
            torch.nn.Linear(in_c, out_c, bias=False),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
        )


class ReleaseBaseline(torch.nn.Module):
    """`Baseline` do commit 52da955, so a parte que tem peso."""

    def __init__(self, hp: dict):
        super().__init__()
        conv = ConvGroupDepthwise if hp["depthwise_convs"] else ConvGroup
        c_in = hp["input_size"][0]
        self.conv_layers = torch.nn.Sequential(
            conv(c_in, 64, 5), conv(64, 128, 5), conv(128, 256, 5))
        flatten = torch.nn.Flatten()
        self.flatten = (torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d((2, 2, 2)), flatten)
                        if hp["adaptive_pooling"] else flatten)
        self.fc1 = FCGroup(256 * (2**3 if hp["adaptive_pooling"] else 3**3),
                           hp["num_fc_units"][0], hp["dropout"])
        self.linear = torch.nn.Linear(hp["num_fc_units"][-1], 1)

    def forward(self, x):
        return self.linear(self.fc1(self.flatten(self.conv_layers(x))))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataframe-path", required=True)
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--split-column", default="grp_final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    model = ReleaseBaseline(hp)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval().to(a.device)

    # O release nao grava occupancy: o VoxelGrid de 52da955 usava o default do
    # docktgrid, que e o `vdw` daqui (CustomVoxelGrid delega a base).
    grid = configure_voxel_grid(argparse.Namespace(
        view=hp["view"], vox_size=hp["vox_size"], box_dims=hp["box_dims"],
        occupancy="vdw", voxel_device=a.device))
    dm = PDBbind(voxel_grid=grid, transforms=[], batch_size=a.batch_size,
                 dataframe_path=a.dataframe_path, root_dir=a.root_dir,
                 split_column=a.split_column, target_column="pki",
                 num_workers=0, molecular_dropout=0.0)
    dm.setup(stage="test")

    preds, labels = [], []
    for x, y in dm.test_dataloader():
        preds.append(model(x.to(a.device)).squeeze(-1).float().cpu())
        labels.append(y.float().cpu())
    dg = torch.cat(preds)
    y = torch.cat(labels)

    df = pd.read_csv(a.dataframe_path, low_memory=False).set_index("id")
    razao = float((df["delta_g"] / df["pki"]).mean())
    pk = dg / razao
    ids = dm.test_dataset.ids
    if len(ids) != pk.numel():
        sys.exit(f"{len(ids)} ids para {pk.numel()} predicoes: CSV nao gravado")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "y_true", "y_pred", "random_split"])
        for i, t, p in zip(ids, y.tolist(), pk.tolist()):
            w.writerow([i, t, p, df.at[i, "random_split"]])
    r = torch.corrcoef(torch.stack((pk, y)))[0, 1].item()
    print(f"[release] n={len(ids)} razao_dg/pk={razao:.4f} r_pooled={r:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
