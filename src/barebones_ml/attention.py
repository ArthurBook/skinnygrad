""" Attention module. """

import abc
from typing import Protocol

import torch
from torch import nn

NINF = -1e9


class Attention(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        query: torch.Tensor,  # (..., Q)
        key: torch.Tensor,  # (..., K)
        value: torch.Tensor,  # (..., V)
        mask: torch.Tensor | None,  # (..., Q, K)
    ) -> torch.Tensor: ...  # (..., V)


class ScaledDotProductAttention(nn.Module, Attention):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.query_proj = LinearProjection(embed_size, embed_size)
        self.key_proj = LinearProjection(embed_size, embed_size)
        self.value_proj = LinearProjection(embed_size, embed_size)

    def forward(
        self,
        query: torch.Tensor,  # (..., Q)
        key: torch.Tensor,  # (..., K)
        value: torch.Tensor,  # (..., V)
        mask: torch.Tensor | None = None,  # (..., Q, K)
    ) -> torch.Tensor:  # (..., V)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        return calc_scaled_dot_product_attention_values(query, key, value, mask)


###
# Helper functions
###
def calc_scaled_dot_product_attention_values(
    query: torch.Tensor,  # (..., Q)
    key: torch.Tensor,  # (..., K)
    value: torch.Tensor,  # (..., V)
    mask: torch.Tensor | None,  # (..., Q, K)
) -> torch.Tensor:  # (..., Q, V)
    """
    return attention values for each query

    Time complexity: (Batch * Query Seq * Key Seq * Embed)
    Space complexity: (Batch * Query Seq * Embed)
    """
    scores = calc_scaled_dot_product_attention_scores(query, key, mask)  # (..., Q, K)
    return torch.einsum("...qk,...kv->...qv", scores, value)  # (..., Q, E)


def calc_scaled_dot_product_attention_scores(
    query: torch.Tensor,  # (..., Q, E)
    key: torch.Tensor,  # (..., K, E)
    mask: torch.Tensor | None = None,  # Bool (..., Q, K)
) -> torch.Tensor:  # (..., Q, K)
    """
    mask: Bool tensor with True for _ignored_ values
    return attention scores for each query-key pair

    Time complexity: (Batch * Q * K * E)
    Space complexity: (Batch * Q * K)
    """
    scores = torch.einsum("...qe,...ke->...qk", query, key)  # (..., Q, K)
    scores /= query.size(-1) ** 0.5  # Scale by sqrt(d_k)
    if mask is not None:  # inplace fill with -inf
        torch.where(mask, torch.tensor(NINF, device=scores.device), scores, out=scores)
    torch.softmax(scores, dim=-1, out=scores)
    return scores


if __name__ == "__main__":
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 2, 4)
    V = torch.randn(2, 2, 4)
    attn = ScaledDotProductAttention(4)
    attn(Q, K, V, mask=None).shape

    nn.MultiheadAttention(4, 1, batch_first=True)(Q, K, V)[1].shape
