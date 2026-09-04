"""Ramo convolucional: grade de voxels -> vetor achatado de features.

O ramo para no `flatten`. O gargalo que antes vinha logo em seguida (`f_proj`)
mudou de lugar: agora vive em `latent.py` e age **depois** da concatenacao com
os embeddings, para que o contrastivo alcance as duas projecoes.

As pecas voltam separadas (`conv_layers`, `flatten`) em vez de um unico
`nn.Module` porque viram atributos de primeiro nivel do `Baseline`, e sao os
nomes desses atributos que formam as chaves do `state_dict`.
"""

import torch

from .blocks import ConvGroup, ConvGroupDepthwise

__all__ = ["CONV_CHANNELS", "KERNEL_SIZE", "flat_dim", "build_voxel_encoder"]

CONV_CHANNELS = (64, 128, 256)
KERNEL_SIZE = 5


def flat_dim(adaptive_pooling: bool) -> int:
    """Numero de features na saida do ramo.

    Sem pooling adaptativo o valor pressupoe uma grade que caia em 3x3x3 depois
    dos tres MaxPool (ou seja, 24 voxels por eixo na entrada).
    """
    return CONV_CHANNELS[-1] * (2**3 if adaptive_pooling else 3**3)


def build_voxel_encoder(
    in_channels: int,
    depthwise_convs: bool,
    adaptive_pooling: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, int]:
    """Monta o ramo 3D.

    Devolve `(conv_layers, flatten, out_dim)`, aplicados nessa ordem; `out_dim`
    e a largura que o ramo entrega para a concatenacao.
    """
    conv = ConvGroupDepthwise if depthwise_convs else ConvGroup
    widths = zip((in_channels, *CONV_CHANNELS[:-1]), CONV_CHANNELS)
    conv_layers = torch.nn.Sequential(
        *(conv(in_c, out_c, KERNEL_SIZE) for in_c, out_c in widths)
    )

    flatten = torch.nn.Flatten()
    if adaptive_pooling:
        flatten = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d((2, 2, 2)), flatten
        )

    return conv_layers, flatten, flat_dim(adaptive_pooling)
