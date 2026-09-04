"""Projecoes dos embeddings congelados (fatores A e B da grade de ablacao).

ESM-2 (proteina) e ChemBERTa (ligante) entram pre-calculados e nunca sao
treinados; o que se aprende aqui e so a projecao que leva cada um para
`emb_proj_dim` antes da concatenacao com o latente `f`.
"""

import torch

from .blocks import linear_bn_relu

__all__ = ["ESM2_DIMS", "E_LIG_DIM", "esm2_dim", "build_embedding_projection"]

# ESM-2 model key -> embedding dim (matches tools/precompute_esm2.py).
ESM2_DIMS = {
    "esm2-650M": 1280,
    "esm2-3B": 2560,
    "esm2-15B": 5120,
    "esm2-150M": 640,
    "esm2-35M": 480,
    "esm2-8M": 320,
}
E_LIG_DIM = 768  # ChemBERTa (zinc-base)

_DEFAULT_ESM2_DIM = ESM2_DIMS["esm2-650M"]


def esm2_dim(model_key: str) -> int:
    """Dimensao do embedding ESM-2; cai no 650M se a chave for desconhecida."""
    return ESM2_DIMS.get(model_key, _DEFAULT_ESM2_DIM)


def build_embedding_projection(in_dim: int, out_dim: int) -> torch.nn.Sequential:
    """Condicionamento de um embedding congelado antes da concatenacao."""
    return linear_bn_relu(in_dim, out_dim)
