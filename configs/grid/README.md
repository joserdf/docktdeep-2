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
  --target-column pki \
  --merge-val-test
```

- Aim remoto: `--remote` + env `AIM_REPO=aim://<host>:43800` (ver broker).
- Sanity (M1): reduzir `--max-epochs 2`, `--batch-size 16`, e usar um `index-pfam.csv`
  reduzido (ver `tools/make_sanity_index.py`).

### Label da regressão: `pki` (unidades de pK)

O índice traz o mesmo alvo em duas colunas, ligadas pela identidade termodinâmica
ΔG = −RT·ln(10)·pK:

| coluna | unidade | média | dp | range |
|--------|---------|-------|-----|-------|
| `pki` | pK (pKd/pKi/pIC50) | 6.36 | 1.86 | 0.40 … 15.22 |
| `delta_g` | kcal/mol | −8.67 | 2.53 | −20.76 … −0.54 |

Medido no `index-pfam.csv`: `delta_g/pki = −1.36355` (dp 7.3e−04), contra
−RT·ln(10) = −1.36425 a 298.15 K, com Pearson r = −0.999999. São **a mesma
variável reescalada**, então a escolha não muda o que o modelo consegue aprender
(Pearson e Spearman são idênticos); muda o que os números significam:

- **Comparabilidade:** KDEEP, Pafnucy, OnionNet e IGN/GraphBAR reportam RMSE em
  unidades de pK. Reportar RMSE em ΔG ao lado deles infla o número em 36%.
- **Hiperparâmetros com escala:** `--huber-beta` e `--yaware-sigma` são distâncias
  nas unidades do rótulo. Com `pki`, `1.0` ≈ 0.54 dp — uma década de afinidade.

Por isso o default é `--target-column pki`. Para reproduzir experimentos antigos,
passe `--target-column delta_g` (as métricas de correlação batem; o RMSE sai
multiplicado por 1.364 e o sinal da predição inverte).

## Status de implementação por arranjo

| # | Flags necessárias | Implementado? |
|---|-------------------|---------------|
| 1 | —                | ✅ executável |
| 2 | `--use-esm2`     | ✅ (M2/M3) |
| 3 | `--use-chemberta`| ✅ (M2/M3) |
| 4 | `--use-esm2 --use-chemberta` | ✅ (M2/M3) |
| 5 | `--semi`         | ✅ (M4) |
| 6 | `--use-esm2 --semi` | ✅ (M2/M3/M4) |
| 7 | `--use-chemberta --semi` | ✅ (M2/M3/M4) |
| 8 | `--use-esm2 --use-chemberta --semi` | ✅ (M2/M3/M4) |

> Todos os arranjos executáveis desde M4 (março de M2-M4). Embeddings pré-computados em
> `data/embeddings/` (ESM-2 650M/150M/35M/8M + ChemBERTa). Flags de fator C: `--semi`,
> `--yaware` (contrastivo ancorado na afinidade), `--loss huber`, `--label-smoothing`,
> `--lambda-semi`. Arranjos 5-8 usam `--yaware` (InfoNCE inspirado na própria afinidade,
> independe de embeddings). Ver `tools/*.py`.

> Hipóteses: mesmo corpus, splits e seeds em todos os arranjos (spec 05); ≥3 seeds por
> config; média ± IC no relatório (PLAN.md §7, §8, §9).
