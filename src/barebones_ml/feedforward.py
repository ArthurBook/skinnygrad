""" Attention module. """

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,  # (I)
        hidden_dims: int,  # (H)
        output_dims: int,  # (O)
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self._hidden = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(
        self,
        x: torch.Tensor,  # (..., I)
    ) -> torch.Tensor:  # (..., O)
        return self._hidden(x)


class LinearProjection(nn.Module):
    def __init__(
        self,
        input_dims: int,  # (I)
        output_dims: int,  # (O)
    ) -> None:  #
        super().__init__()
        self._weights = nn.Parameter(torch.randn(input_dims, output_dims))  # (I, O)

    def forward(
        self,
        x: torch.Tensor,  # (..., I)
    ) -> torch.Tensor:  # (..., O)
        return torch.einsum("...i,io->...o", x, self._weights)
