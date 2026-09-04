"""Cabecas do modelo: a de regressao (afinidade) e a de projecao (fator C).

As duas leem o MESMO latente `z`, ja depois da concatenacao dos ramos (ver
`latent.py`). E o que faz o gradiente dos termos contrastivos atravessar o
concat e alcancar `proj_prot`/`proj_lig`: com o gargalo aplicado antes do
concat, como era ate aqui, a cabeca de projecao via so o ramo convolucional e
as projecoes dos embeddings eram treinadas apenas pela perda de regressao.
"""

import torch

__all__ = ["build_regression_head", "build_projection_head", "FCGroup"]


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


def build_projection_head(in_dim: int, proj_dim: int) -> torch.nn.Sequential:
    """p(f) para o objetivo semi-supervisionado.

    O `Dropout` no meio nao e regularizacao: e a fonte de estocasticidade que a
    consistencia R-Drop compara entre duas passadas do mesmo exemplo.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, proj_dim, bias=False),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(proj_dim, proj_dim, bias=False),
    )


class FCGroup(torch.nn.Sequential):
    """Cabeca de regressao do upstream (docktdeep v0.2.0). NAO USADA.

    Era o `fc1` do `Baseline` original (`conv -> flatten -> fc1 -> linear`), ate
    o condicionamento por embeddings trocar o par `fc1`/`linear` pelo laco de
    `build_regression_head`. Mantida como registro da linhagem; note o
    `BatchNorm1d(1000)` fixo, que so nao quebrava porque o default de
    `--num-fc-units` e `[1000]`. `stn.py` tem a sua propria versao, corrigida e
    em uso.
    """

    def __init__(self, in_c, out_c, dropout_rate, **kwargs):
        super().__init__(
            torch.nn.Linear(in_c, out_c, bias=False),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
        )
