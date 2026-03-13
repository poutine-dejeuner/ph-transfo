"""Graph Transformer layers and encoder.

Implements multi-head attention adapted for graph-structured data, with an
optional learned structural bias derived from the adjacency matrix.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGraphAttention(nn.Module):
    """Multi-head self-attention for graphs with optional structural bias.

    Attention scores are computed as::

        A = softmax( (Q K^T) / sqrt(d_head) + B )

    where *B* is an optional structural bias (e.g. derived from the adjacency
    matrix) that encourages the model to respect graph topology.

    Args:
        d_model: Total model dimension (must be divisible by *num_heads*).
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability applied to the attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            x: Node features of shape ``(N, d_model)`` or ``(B, N, d_model)``.
            attention_bias: Optional additive bias of shape broadcastable to
                ``(..., N, N)``.

        Returns:
            Output tensor with the same shape as *x*.
        """
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)  # → (1, N, d_model)

        B, N, _ = x.shape
        H, D = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)

        if attention_bias is not None:
            # Support (N,N), (B,N,N) or (B,H,N,N) bias tensors
            while attention_bias.dim() < 4:
                attention_bias = attention_bias.unsqueeze(0)
            scores = scores + attention_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)

        if not is_batched:
            out = out.squeeze(0)
        return out


class GraphTransformerLayer(nn.Module):
    """Single pre-norm graph transformer layer.

    Applies multi-head attention followed by a position-wise feed-forward
    network, both with residual connections and layer normalisation::

        x = x + Dropout(Attention(LayerNorm(x), adj))
        x = x + FFN(LayerNorm(x))

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (default: ``4 * d_model``).
        dropout: Dropout probability.
        use_edge_bias: If ``True``, add a learned scalar bias to attention
            scores based on whether an edge exists between each node pair.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = MultiHeadGraphAttention(d_model, num_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learned edge-presence bias: index 0 → no edge, index 1 → edge
        self.edge_bias: Optional[nn.Parameter]
        if use_edge_bias:
            self.edge_bias = nn.Parameter(torch.zeros(2))
        else:
            self.edge_bias = None

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape ``(N, d_model)`` or ``(B, N, d_model)``.
            adj: Adjacency matrix of shape ``(N, N)`` or ``(B, N, N)``.
                Binary or weighted; non-zero entries are treated as edges.

        Returns:
            Output tensor with the same shape as *x*.
        """
        attention_bias: Optional[torch.Tensor] = None
        if adj is not None and self.edge_bias is not None:
            adj_bin = (adj > 0).float()
            attention_bias = (
                adj_bin * self.edge_bias[1] + (1.0 - adj_bin) * self.edge_bias[0]
            )

        x = x + self.dropout(self.attention(self.norm1(x), attention_bias))
        x = x + self.ff(self.norm2(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """Stack of :class:`GraphTransformerLayer` modules.

    Args:
        d_model: Model dimension.
        num_layers: Number of stacked layers.
        num_heads: Number of attention heads per layer.
        d_ff: Feed-forward hidden dimension (default: ``4 * d_model``).
        dropout: Dropout probability.
        use_edge_bias: Whether to use learned adjacency bias in each layer.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(d_model, num_heads, d_ff, dropout, use_edge_bias)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply all transformer layers.

        Args:
            x: Node features of shape ``(N, d_model)`` or ``(B, N, d_model)``.
            adj: Optional adjacency matrix.

        Returns:
            Encoded node features with the same shape as *x*.
        """
        for layer in self.layers:
            x = layer(x, adj)
        return self.norm(x)
