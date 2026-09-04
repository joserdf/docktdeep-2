#!/usr/bin/env python3
"""Gera configs/simab/*.yaml (16 células) a partir da grade de ablação 2^4 + ∅.

Documentação das 16 células de ablação de termos de similaridade (fator C). As
runs reais usam as flags do tools/submit_simab.py (broker); os yaml são o mapa
legível (mesmo padrão de configs/grid/NN_*.yaml).
"""
import itertools
import os
from pathlib import Path

TERMS = ["ifp", "aff", "prot", "lig"]
OUT = Path(__file__).resolve().parent.parent / "configs" / "simab"

CELLS = {"s00_empty": []}
for r in range(1, len(TERMS) + 1):
    for combo in itertools.combinations(TERMS, r):
        CELLS[f"s{len(CELLS)}_" + "_".join(combo)] = list(combo)

TEMPLATE = """# {title}
# Célula {name} — ablação de termos de similaridade (fator C), grade 2^4 + ∅.
# K = {terms}
# Base: arr05 (CNN base, random_split, merge-val-test) + Huber + label-smoothing
# 0.05 + RDrop@0.025. yaware OFF (default). λ_k = λ0 = 0.025; orçamento 0.125.
factors:
  esm2: off
  chemberta: off
  semi: on
  sim_terms: {terms_list}      # subconjunto de {{ifp, aff, prot, lig}}
  yaware: false                # obrigatório (R0 senão herdaria o InfoNCE antigo)

model: Baseline

data:
  dataframe_path: data/pdbbind2020/index-pfam.csv
  root_dir: data/pdbbind2020/processed
  split_column: random_split
  target_column: pki
  merge_val_test: true
  batch_size: 64
  num_workers: 8
  vox_size: 1.0
  box_dims: [24.0, 24.0, 24.0]
  view: ["VolumeView", "BasicView"]
  occupancy: gaussian
  random_rotation: true

train:
  max_epochs: 200
  optimizer: AdamW
  lr: 0.00087469
  beta1: 0.25693012
  eps: 0.00032933
  dropout: 0.25348994
  wdecay: 0.0000169
  molecular_dropout: 0.06
  molecular_dropout_unit: complex
  depthwise_convs: true
  adaptive_pooling: true
  loss: huber
  label_smoothing: 0.05
  lambda_semi: 0.025     # RDrop
  sim_lambda: 0.025      # cada termo de similaridade
  sim_lambda_max: 0.125  # orçamento do bloco
  sim_mat_dir: data/embeddings/sim
  sim_prot_map: data/embeddings/sim/prot_map.json
  sim_lig_map: data/embeddings/sim/lig_map.json
{ifp_line}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, terms in CELLS.items():
        is_r0 = not terms
        title = "R0 (baseline, sem termos)" if is_r0 else f"K={terms}"
        ifp_line = ("  ifp_path: data/pdbbind2020/ifp.parquet"
                    if "ifp" in terms else "# (sem termo ifp: sem --ifp-path)")
        content = TEMPLATE.format(
            title=title, name=name,
            terms="∅" if is_r0 else terms,
            terms_list="[]" if is_r0 else "[" + ", ".join(terms) + "]",
            ifp_line=ifp_line,
        )
        (OUT / f"{name}.yaml").write_text(content)
    print(f"gerados {len(CELLS)} yaml em {OUT}")


if __name__ == "__main__":
    main()
