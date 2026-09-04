"""O espaco latente `z`, onde o aprendizado contrastivo age.

Fica **depois** da concatenacao dos ramos, e nao antes: essa e a diferenca em
relacao ao antigo `f_proj`. Com o gargalo antes do concat, o gradiente dos
termos contrastivos parava em `conv`/`f_proj` e nunca alcancava `proj_prot` e
`proj_lig`; depois do concat, ele atravessa a concatenacao e chega nas duas.

Desligar o gargalo (`--latent-dim 0`) faz a cabeca de regressao ler direto a
concatenacao — que, sem embeddings, e exatamente o `flatten -> fc1 -> linear`
do docktdeep upstream, e serve de braco de comparacao.
"""

import torch

from .blocks import linear_bn_relu

__all__ = ["build_latent_projection"]


def build_latent_projection(in_dim: int, latent_dim: int) -> torch.nn.Sequential:
    """Projeta a concatenacao dos ramos no espaco latente compartilhado."""
    return linear_bn_relu(in_dim, latent_dim)
