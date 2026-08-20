# Grade de Ablação 2³ — mapeamento de configs

Cada arquivo `NN_*.yaml` define um arranjo da grade fatorial 2³ (ver `.claude/PLAN.md` §4/§7):

| # | A: ESM-2 | B: ChemBERTa | C: Reg/Semi | config |
|---|:--------:|:------------:|:-----------:|--------|
| 1 | off | off | off | `01_base.yaml` |
| 2 | on  | off | off | `02_esm2.yaml` |
| 3 | off | on  | off | `03_chemberta.yaml` |
| 4 | on  | on  | off | `04_esm2_chemberta.yaml` |
| 5 | off | off | on  | `05_semi.yaml` |
| 6 | on  | off | on  | `06_esm2_semi.yaml` |
| 7 | off | on  | on  | `07_chemberta_semi.yaml` |
| 8 | on  | on  | on  | `08_all.yaml` |

Os campos `command_flags` mapeiam 1:1 para flags do `train.py`
(`--use-esm2`, `--use-chemberta`, `--semi`). Essas flags **ainda não existem** no
`train.py`; serão implementadas em M2 (embeddings ESM-2/ChemBERTa), M3
(condicionamento do modelo) e M4 (regularizadores + `L_semi`). O arranjo **#1 é
executável hoje** (Fase 0 / sanity).

## Comando base (arranjo 01)

```bash
python train.py \
  --model Baseline \
  --experiment grid/01 \
  --remote \
  --seed 42 \
  --depthwise-convs \
  --adaptive-pooling \
  --optim AdamW \
  --max-epochs 1500 \
  --batch-size 64 \
  --lr 0.00087469 \
  --beta1 0.25693012 \
  --eps 0.00032933 \
  --dropout 0.25348994 \
  --wdecay 0.0000169 \
  --molecular-dropout 0.06 \
  --molecular-dropout-unit complex \
  --random-rotation \
  --occupancy gaussian \
  --view VolumeView BasicView \
  --dataframe-path data/pdbbind2020/index-pfam.csv \
  --root-dir data/pdbbind2020/processed \
  --protein-path-pattern "{c}_protein_prep.pdb.pkl" \
  --ligand-path-pattern "{c}_ligand_rnum.pdb.pkl" \
  --split-column random_split \
  --merge-val-test
```

- Label da regressão: `delta_g` (coluna do índice; média ≈ −8.67, range −20.8…−0.5).
- Aim remoto: `--remote` + env `AIM_REPO=aim://<host>:43800` (ver broker).
- Sanity (M1): reduzir `--max-epochs 2`, `--batch-size 16`, e usar um `index-pfam.csv`
  reduzido (ver `tools/make_sanity_index.py`).

## Status de implementação por arranjo

| # | Flags necessárias | Implementado? |
|---|-------------------|---------------|
| 1 | —                | ✅ executável hoje |
| 2 | `--use-esm2`     | 🔜 M2 + M3 |
| 3 | `--use-chemberta`| 🔜 M2 + M3 |
| 4 | `--use-esm2 --use-chemberta` | 🔜 M2 + M3 |
| 5 | `--semi`         | 🔜 M4 |
| 6 | `--use-esm2 --semi` | 🔜 M2+M3+M4 |
| 7 | `--use-chemberta --semi` | 🔜 M2+M3+M4 |
| 8 | `--use-esm2 --use-chemberta --semi` | 🔜 M2+M3+M4 |

> Hipóteses: mesmo corpus, splits e seeds em todos os arranjos (spec 05); ≥3 seeds por
> config; média ± IC no relatório (PLAN.md §7, §8, §9).
