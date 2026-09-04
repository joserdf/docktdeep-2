"""Blocos convolucionais reutilizaveis.

Definidos aqui porque `baseline.py` e `stn.py` usavam copias identicas de
`ConvGroupDepthwise`. Nenhum deles guarda estado alem dos proprios pesos: sao
`nn.Sequential` prontos, e a ordem interna das camadas define as chaves do
`state_dict` — mudar essa ordem invalida os checkpoints ja treinados.
"""

import torch

__all__ = ["ConvGroup", "ConvGroupDepthwise", "linear_bn_relu"]


class ConvGroup(torch.nn.Sequential):
    """Conv3d densa -> BatchNorm -> ReLU -> MaxPool (reduz cada eixo pela metade)."""

    def __init__(self, in_c, out_c, kernel_size, **kwargs):
        super().__init__(
            torch.nn.Conv3d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
                **kwargs,
            ),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((2, 2, 2)),
        )


class ConvGroupDepthwise(torch.nn.Sequential):
    """Idem `ConvGroup`, com a convolucao fatorada em depthwise + pointwise."""

    def __init__(self, in_c, out_c, kernel_size, **kwargs):
        super().__init__(
            self.depthwise_separable_conv(in_c, out_c, kernel_size, **kwargs),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((2, 2, 2)),
        )

    def depthwise_separable_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        kernels_per_layer: int = 1,
    ):
        """3D depthwise conv layer."""
        conv3d = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = torch.nn.Conv3d(
            in_channels=in_channels * kernels_per_layer,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        return torch.nn.Sequential(conv3d, pointwise_conv)


def linear_bn_relu(in_c: int, out_c: int) -> torch.nn.Sequential:
    """Linear sem bias -> BatchNorm -> ReLU.

    A projecao base do modelo: o `bias` sai porque a BatchNorm logo em seguida
    ja tem o seu proprio deslocamento. Usada pelas projecoes dos embeddings e
    pelo espaco latente.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_c, out_c, bias=False),
        torch.nn.BatchNorm1d(out_c),
        torch.nn.ReLU(inplace=True),
    )
