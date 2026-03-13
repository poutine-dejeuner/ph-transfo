"""PHGraphTransformer: graph transformer with persistence homology featurization.

Combines:

1. **PHFeaturizer** — computes Betti number curves (β0, β1) from the graph
   topology and uses them as additional topological features.
2. **GraphTransformerEncoder** — multi-head self-attention transformer operating
   over the set of graph nodes.

The PH features are injected in two ways:

* **Global PH features**: a single Betti-curve vector for the whole graph,
  projected to ``d_model`` and *added* to every node embedding (providing a
  global topological context).
* **Node PH features**: per-node [β0, β1] from the sublevel-set filtration,
  *concatenated* to the raw node features before embedding.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .graph_transformer import GraphTransformerEncoder
from .ph_featurizer import PHFeaturizer


class PHGraphTransformer(nn.Module):
    """Graph Transformer with Persistence Homology Betti number featurization.

    Args:
        node_in_dim: Dimension of the raw input node feature vectors.
        d_model: Internal transformer dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (default: ``4 * d_model``).
        dropout: Dropout probability.
        num_filtration_steps: Number of filtration levels for the PH featurizer.
        out_dim: Output dimension (e.g. number of classes).
        readout: Graph-level readout pooling — one of ``'mean'``, ``'sum'``,
            or ``'max'``.
        use_node_ph: Append node-level Betti features [β0, β1] to the raw
            node features before embedding.
        use_graph_ph: Add a projected global Betti-curve feature to all node
            embeddings after the initial linear embedding.
    """

    def __init__(
        self,
        node_in_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        num_filtration_steps: int = 50,
        out_dim: int = 1,
        readout: str = "mean",
        use_node_ph: bool = True,
        use_graph_ph: bool = True,
    ) -> None:
        super().__init__()

        if readout not in ("mean", "sum", "max"):
            raise ValueError("readout must be 'mean', 'sum', or 'max'")

        self.readout = readout
        self.use_node_ph = use_node_ph
        self.use_graph_ph = use_graph_ph

        # PH featurizer (runs on CPU with numpy)
        self.ph_featurizer = PHFeaturizer(
            num_filtration_steps=num_filtration_steps,
            max_homology_dim=1,
        )

        # Node embedding: raw features + optional node-level PH features
        total_in_dim = node_in_dim + (2 if use_node_ph else 0)
        self.node_embedding = nn.Linear(total_in_dim, d_model)

        # Projection for global graph-level PH features
        self.ph_graph_proj: Optional[nn.Linear]
        if use_graph_ph:
            ph_graph_dim = self.ph_featurizer.feature_dim
            self.ph_graph_proj = nn.Linear(ph_graph_dim, d_model)
        else:
            self.ph_graph_proj = None

        # Transformer encoder
        self.encoder = GraphTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Graph-level output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Union[torch.Tensor, np.ndarray],
        adj: Optional[torch.Tensor] = None,
        edge_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """Compute graph-level predictions.

        Args:
            node_features: Shape ``(N, node_in_dim)`` or ``(B, N, node_in_dim)``.
            edge_index: Shape ``(2, E)`` edge indices (integer).
            adj: Optional pre-computed adjacency matrix ``(N, N)`` or
                ``(B, N, N)``.  Computed from *edge_index* when not provided.
            edge_weights: Optional edge weights ``(E,)`` used for the PH
                filtration and for building *adj* when not provided.

        Returns:
            Predictions of shape ``(out_dim,)`` for unbatched input, or
            ``(B, out_dim)`` for batched input.
        """
        if node_features.dim() == 3:
            return self._forward_batched(node_features, edge_index, adj, edge_weights)

        N = node_features.shape[0]
        device = node_features.device

        # Convert edge_index / edge_weights to numpy for the featurizer
        ei_np, ew_np = _to_numpy(edge_index, edge_weights)

        x = node_features  # (N, node_in_dim)

        # 1. Node-level PH features
        if self.use_node_ph:
            node_ph = self.ph_featurizer.compute_node_features(ei_np, N)
            node_ph_t = torch.tensor(node_ph, dtype=torch.float32, device=device)
            x = torch.cat([x, node_ph_t], dim=-1)  # (N, node_in_dim + 2)

        # 2. Linear node embedding
        x = self.node_embedding(x)  # (N, d_model)

        # 3. Global PH features broadcast to all nodes
        if self.use_graph_ph and self.ph_graph_proj is not None:
            graph_ph = self.ph_featurizer.compute_graph_features(ei_np, N, ew_np)
            graph_ph_t = torch.tensor(graph_ph, dtype=torch.float32, device=device)
            ph_emb = self.ph_graph_proj(graph_ph_t)  # (d_model,)
            x = x + ph_emb.unsqueeze(0)  # broadcast: (N, d_model)

        # 4. Build adjacency matrix if not supplied
        if adj is None:
            adj = _build_adj(edge_index, N, edge_weights, device)

        # 5. Graph transformer encoder
        x = self.encoder(x, adj)  # (N, d_model)

        # 6. Graph readout
        graph_emb = _readout(x, self.readout)  # (d_model,)

        # 7. Output head
        return self.output_head(graph_emb)  # (out_dim,)

    def _forward_batched(
        self,
        node_features: torch.Tensor,
        edge_index: Union[torch.Tensor, np.ndarray],
        adj: Optional[torch.Tensor],
        edge_weights: Optional[Union[torch.Tensor, np.ndarray]],
    ) -> torch.Tensor:
        """Process a batch of graphs that share the same topology."""
        B = node_features.shape[0]
        results = [
            self.forward(
                node_features[b],
                edge_index,
                adj[b] if adj is not None else None,
                edge_weights,
            )
            for b in range(B)
        ]
        return torch.stack(results, dim=0)  # (B, out_dim)

    # ------------------------------------------------------------------
    # Extra utility: node-level embeddings (for downstream tasks)
    # ------------------------------------------------------------------

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: Union[torch.Tensor, np.ndarray],
        adj: Optional[torch.Tensor] = None,
        edge_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """Return transformer node embeddings without the output head.

        Useful for node-level downstream tasks.

        Args:
            node_features: Shape ``(N, node_in_dim)``.
            edge_index: Shape ``(2, E)``.
            adj: Optional adjacency matrix ``(N, N)``.
            edge_weights: Optional edge weights ``(E,)``.

        Returns:
            Node embeddings of shape ``(N, d_model)``.
        """
        N = node_features.shape[0]
        device = node_features.device
        ei_np, ew_np = _to_numpy(edge_index, edge_weights)

        x = node_features
        if self.use_node_ph:
            node_ph = self.ph_featurizer.compute_node_features(ei_np, N)
            node_ph_t = torch.tensor(node_ph, dtype=torch.float32, device=device)
            x = torch.cat([x, node_ph_t], dim=-1)

        x = self.node_embedding(x)

        if self.use_graph_ph and self.ph_graph_proj is not None:
            graph_ph = self.ph_featurizer.compute_graph_features(ei_np, N, ew_np)
            graph_ph_t = torch.tensor(graph_ph, dtype=torch.float32, device=device)
            ph_emb = self.ph_graph_proj(graph_ph_t)
            x = x + ph_emb.unsqueeze(0)

        if adj is None:
            adj = _build_adj(edge_index, N, edge_weights, device)

        return self.encoder(x, adj)  # (N, d_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(
    edge_index: Union[torch.Tensor, np.ndarray],
    edge_weights: Optional[Union[torch.Tensor, np.ndarray]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if isinstance(edge_weights, torch.Tensor):
        edge_weights = edge_weights.detach().cpu().numpy()
    return edge_index, edge_weights


def _build_adj(
    edge_index: Union[torch.Tensor, np.ndarray],
    num_nodes: int,
    edge_weights: Optional[Union[torch.Tensor, np.ndarray]],
    device: torch.device,
) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index).long()
    edge_index = edge_index.to(device)
    if edge_weights is not None:
        if isinstance(edge_weights, np.ndarray):
            edge_weights = torch.from_numpy(edge_weights).float()
        adj[edge_index[0], edge_index[1]] = edge_weights.to(device).float()
    else:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def _readout(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=0)
    elif mode == "sum":
        return x.sum(dim=0)
    else:  # max
        return x.max(dim=0).values
