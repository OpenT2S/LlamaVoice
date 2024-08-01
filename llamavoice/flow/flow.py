# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Basic Flow modules used in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Optional, Tuple, Union

import torch


class FlipFlow(torch.nn.Module):
    """Flip flow module."""

    def forward(
        self, x: torch.Tensor, *args, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Flipped tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        x = torch.flip(x, [1])
        if not inverse:
            logdet = x.new_zeros(x.size(0))
            return x, logdet
        else:
            return x


class LogFlow(torch.nn.Module):
    """Log flow module."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        inverse: bool = False,
        eps: float = 1e-5,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            inverse (bool): Whether to inverse the flow.
            eps (float): Epsilon for log.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        if not inverse:
            y = torch.log(torch.clamp_min(x, eps)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffineFlow(torch.nn.Module):
    """Elementwise affine flow module."""

    def __init__(self, channels: int):
        """Initialize ElementwiseAffineFlow module.

        Args:
            channels (int): Number of channels.

        """
        super().__init__()
        self.channels = channels
        self.register_parameter("m", torch.nn.Parameter(torch.zeros(channels, 1)))
        self.register_parameter("logs", torch.nn.Parameter(torch.zeros(channels, 1)))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_lengths (Tensor): Length tensor (B,).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        if not inverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class Transpose(torch.nn.Module):
    """Transpose module for torch.nn.Sequential()."""

    def __init__(self, dim1: int, dim2: int):
        """Initialize Transpose module."""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose."""
        return x.transpose(self.dim1, self.dim2)


class DilatedDepthSeparableConv(torch.nn.Module):
    """Dilated depth-separable conv module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        layers: int,
        dropout_rate: float = 0.0,
        eps: float = 1e-5,
    ):
        """Initialize DilatedDepthSeparableConv module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            dropout_rate (float): Dropout rate.
            eps (float): Epsilon for layer norm.

        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        groups=channels,
                        dilation=dilation,
                        padding=padding,
                    ),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(
                        channels,
                        eps=eps,
                        elementwise_affine=True,
                    ),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        1,
                    ),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(
                        channels,
                        eps=eps,
                        elementwise_affine=True,
                    ),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        if g is not None:
            x = x + g
        for f in self.convs:
            y = f(x * x_mask)
            x = x + y
        return x * x_mask
