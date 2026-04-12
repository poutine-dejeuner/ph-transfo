"""GPS (General Powerful Scalable) Graph Transformer for molecular regression.

Uses PyG's GPSConv: local MPNN message passing + global multi-head attention.
Supports optional topological positional encoding (Betti curves + persistence stats)
injected additively after projection, and edge-level filtration features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool


@dataclass
class GPSTransformerHParams:
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    # Topology PE
    topo_dim: int = 0  # input dim of topo features (0 = disabled)
    topo_pe_layers: int = 2  # MLP depth for topo PE projection
    # Edge features
    edge_topo: bool = False  # use edge filtration values


class GPSTransformer(nn.Module):
    """GPS Graph Transformer with optional topological positional encoding.

    Architecture:
      1. Linear node embedding → hidden_dim
      2. (Optional) Topo PE: MLP(topo_x) added to node embeddings
      3. Edge embedding: bond distances + (optional) Rips filtration values
      4. N × GPSConv (GIN local + Transformer global)
      5. Graph-level mean pooling
      6. MLP prediction head
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hparams: GPSTransformerHParams | None = None,
    ) -> None:
        super().__init__()
        if hparams is None:
            hparams = GPSTransformerHParams()
        self.hp = hparams
        h = hparams.hidden_dim

        self.node_embed = nn.Linear(input_dim, h)

        # Topological positional encoding: project topo features → hidden_dim
        if hparams.topo_dim > 0:
            layers = []
            in_d = hparams.topo_dim
            for i in range(hparams.topo_pe_layers - 1):
                layers.extend([nn.Linear(in_d, h), nn.ReLU(), nn.LayerNorm(h)])
                in_d = h
            layers.append(nn.Linear(in_d, h))
            self.topo_pe = nn.Sequential(*layers)
        else:
            self.topo_pe = None

        # Edge embedding: distance (1) + optional filtration value (1)
        edge_in_dim = 2 if hparams.edge_topo else 1
        self.edge_embed = nn.Linear(edge_in_dim, h)

        self.gps_layers = nn.ModuleList()
        for _ in range(hparams.num_layers):
            gin_nn = nn.Sequential(
                nn.Linear(h, h * 2),
                nn.ReLU(),
                nn.Linear(h * 2, h),
            )
            gin_conv = GINEConv(gin_nn, edge_dim=h)

            gps = GPSConv(
                channels=h,
                conv=gin_conv,
                heads=hparams.num_heads,
                dropout=hparams.dropout,
                attn_type="multihead",
                attn_kwargs={"dropout": hparams.attn_dropout},
            )
            self.gps_layers.append(gps)

        self.final_norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, output_dim),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x.float()
        edge_index = batch.edge_index
        batch_index = batch.batch

        # Node embedding
        x = self.node_embed(x)

        # Additive topological positional encoding
        if self.topo_pe is not None and hasattr(batch, "topo_x") and batch.topo_x is not None:
            topo_pe = self.topo_pe(batch.topo_x.float())
            x = x + topo_pe

        # Edge features
        if hasattr(batch, "pos") and batch.pos is not None:
            edge_dist = torch.norm(
                batch.pos[edge_index[0]] - batch.pos[edge_index[1]], dim=1, keepdim=True
            )
        else:
            edge_dist = torch.ones(edge_index.size(1), 1, device=x.device)

        if self.hp.edge_topo and hasattr(batch, "edge_filt") and batch.edge_filt is not None:
            edge_raw = torch.cat([edge_dist, batch.edge_filt.float()], dim=1)
        else:
            edge_raw = edge_dist

        edge_attr = self.edge_embed(edge_raw)

        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, batch_index, edge_attr=edge_attr)

        x = self.final_norm(x)
        x = global_mean_pool(x, batch_index)
        out = self.head(x)
        return out.squeeze(-1)
