"""Cabecas do modelo: a de regressao (afinidade) e a de projecao (fator C).

As duas leem o MESMO latente `z`, ja depois da concatenacao dos ramos (ver
`latent.py`). E o que faz o gradiente dos termos contrastivos atravessar o
concat e alcancar `proj_prot`/`proj_lig`: com o gargalo aplicado antes do
concat, como era ate aqui, a cabeca de projecao via so o ramo convolucional e
as projecoes dos embeddings eram treinadas apenas pela perda de regressao.
"""

import torch

__all__ = ["build_regression_head", "build_projection_head"]


def build_regression_head(
    in_dim: int, fc_units: list[int], dropout: float
) -> torch.nn.Sequential:
    """MLP que termina num escalar (pKi).

    Um bloco `Linear -> BatchNorm -> ReLU -> Dropout` por entrada de `fc_units`,
    e um `Linear(.., 1)` no fim. Os indices resultantes sao as chaves `head.N.*`
    do `state_dict`, entao a ordem dos quatro modulos e parte do contrato.
    """
    layers: list[torch.nn.Module] = []
    prev = in_dim
    for units in fc_units:
        layers += [
            torch.nn.Linear(prev, units, bias=False),
            torch.nn.BatchNorm1d(units),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
        ]
        prev = units
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


def build_projection_head(
    in_dim: int, proj_dim: int, dropout: float = 0.1
) -> torch.nn.Sequential:
    """p(z) para o objetivo semi-supervisionado.

    O `Dropout` no meio nao e regularizacao: e a UNICA fonte de estocasticidade
    que a consistencia R-Drop compara entre as duas projecoes do mesmo exemplo
    (nao ha dropout algum no caminho ate `z`). Por isso ele governa a magnitude
    de `L_rdrop`, e por isso virou hiperparametro: com o valor fixo em 0.1,
    `--lambda-rdrop` e este dropout eram conjuntamente nao-identificaveis —
    dobrar um e reduzir o outro pela metade dava aproximadamente a mesma perda.
    O default 0.1 reproduz todas as campanhas anteriores.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, proj_dim, bias=False),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(proj_dim, proj_dim, bias=False),
    )
