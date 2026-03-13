"""Graph Transformer with TOGL-style local Betti number computation.

Combines multi-head self-attention on graph nodes with a topological
aggregation layer that learns filtrations, computes persistent homology
via GUDHI (through ``torch_topological``), and maps persistence pairs to
per-node features via learnable coordinate functions — following the TOGL
approach (Horn et al., "Topological Graph Neural Networks", ICLR 2022).

The persistent homology backend uses GUDHI's ``SimplexTree`` with
lower-star filtrations, as implemented in ``torch_topological.nn.graphs``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import gudhi as gd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_topological.utils import pairwise


# ---------------------------------------------------------------------------
# Coordinate transforms  (learnable maps  (birth, death) → ℝ^d)
# ---------------------------------------------------------------------------


class TriangleTransform(nn.Module):
    """Triangle coordinate function: ReLU(death − |t − birth|)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.t = nn.Parameter(torch.randn(output_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2]  (birth, death)
        return F.relu(x[:, 1:2] - torch.abs(self.t - x[:, 0:1]))


class GaussianTransform(nn.Module):
    """Gaussian coordinate function centred on learnable (t₁, t₂)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.t = nn.Parameter(torch.randn(2, output_dim) * 0.1)
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2]
        diff = x[:, :, None] - self.t  # [N, 2, output_dim]
        return torch.exp(-diff.pow(2).sum(dim=1) / (2 * self.sigma.pow(2)))


class LineTransform(nn.Module):
    """Linear projection of (birth, death)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RationalHatTransform(nn.Module):
    """Rational-hat coordinate function (Hofer et al., 2019)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.randn(2, output_dim) * 0.1)
        self.r = nn.Parameter(torch.randn(1, output_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2]
        norm = torch.norm(x[:, :, None] - self.c, p=1, dim=1)  # [N, out]
        return 1.0 / (1.0 + norm) - 1.0 / (1.0 + torch.abs(self.r.abs() - norm))


COORD_TRANSFORMS: dict[str, type[nn.Module]] = {
    "triangle": TriangleTransform,
    "gaussian": GaussianTransform,
    "line": LineTransform,
    "rational_hat": RationalHatTransform,
}


# ---------------------------------------------------------------------------
# GUDHI-based persistent homology (following torch_topological's approach)
# ---------------------------------------------------------------------------


def _compute_ph_single_graph(
    vertices: torch.Tensor,
    f_vertices: torch.Tensor,
    edges: torch.Tensor,
    f_edges: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Compute H0 persistence diagram for a single graph via GUDHI.

    Uses a lower-star filtration: each simplex receives the max filtration
    value of its vertices.  Returns a per-node persistence diagram of shape
    ``[n_nodes, 2]`` where each row is ``(birth, death)`` in filtration-value
    space, preserving the gradient through the filtration values.

    Follows the ``torch_topological.nn.graphs.TOGL._compute_persistent_homology``
    implementation.
    """
    n_nodes = len(vertices)

    st = gd.SimplexTree()

    for v, f in zip(vertices.tolist(), f_vertices.tolist()):
        st.insert([v], filtration=f)

    for (u, v), f in zip(edges.tolist(), f_edges.tolist()):
        st.insert([u, v], filtration=f)

    st.make_filtration_non_decreasing()
    st.expansion(2)
    st.persistence()

    generators = st.lower_star_persistence_generators()
    generators_regular = generators[0]

    # Default: every node paired with itself (trivial persistence)
    persistence_diagram = torch.stack((f_vertices, f_vertices), dim=1)

    if len(generators_regular) > 0 and len(generators_regular[0]) > 0:
        gen0 = torch.as_tensor(generators_regular[0]) - offset
        gen0 = gen0.sort(dim=0, stable=True)[0]
        # Map generators back to filtration values (preserves gradients)
        persistence_diagram[gen0[:, 0], 1] = f_vertices[gen0[:, 1]]

    return persistence_diagram


# ---------------------------------------------------------------------------
# Topology layer (TOGL-style, using GUDHI via torch_topological)
# ---------------------------------------------------------------------------


class TopologyLayer(nn.Module):
    """Compute per-node topological features via learned filtrations and PH.

    1. An MLP maps node features → ``num_filtrations`` scalar filtration values.
    2. GUDHI computes H0 persistence diagrams (lower-star filtration).
    3. Learnable coordinate functions map each (birth, death) pair to features.
    4. Results are combined with input via residual + BatchNorm or concat + linear.
    """

    def __init__(
        self,
        features_in: int,
        features_out: int,
        num_filtrations: int = 4,
        filtration_hidden: int = 32,
        coord_funs: dict[str, int] | None = None,
        residual_and_bn: bool = True,
        share_filtration_parameters: bool = True,
        apply_tanh: bool = False,
    ) -> None:
        super().__init__()
        if coord_funs is None:
            coord_funs = {"triangle": 4, "line": 4}

        self.num_filtrations = num_filtrations
        self.residual_and_bn = residual_and_bn

        total_coord = sum(coord_funs.values())
        self.total_coord = total_coord

        # Coordinate function modules
        self.coord_modules = nn.ModuleList(
            [COORD_TRANSFORMS[name](dim) for name, dim in coord_funs.items()]
        )

        # Filtration MLP
        act = nn.Tanh() if apply_tanh else nn.Identity()
        if share_filtration_parameters:
            self.filtration = nn.Sequential(
                nn.Linear(features_in, filtration_hidden),
                nn.ReLU(),
                nn.Linear(filtration_hidden, num_filtrations),
                act,
            )
        else:
            self.filtration = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(features_in, filtration_hidden),
                        nn.ReLU(),
                        nn.Linear(filtration_hidden, 1),
                        act,
                    )
                    for _ in range(num_filtrations)
                ]
            )
        self.share_filtration = share_filtration_parameters

        # Output projection
        topo_dim = num_filtrations * total_coord
        if residual_and_bn:
            self.out = nn.Linear(topo_dim, features_out)
            self.bn = nn.BatchNorm1d(features_out)
        else:
            self.out = nn.Linear(features_in + topo_dim, features_out)

    # -- helpers --

    def _filtration_values(self, x: torch.Tensor) -> torch.Tensor:
        if self.share_filtration:
            return self.filtration(x)
        return torch.cat([m(x) for m in self.filtration], dim=1)

    def _coord_fun(self, persistence: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(persistence) for m in self.coord_modules], dim=1)

    def _compute_persistence_diagrams(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        vertex_slices: torch.Tensor,
        edge_slices: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Compute persistence diagrams for all filtrations and all graphs.

        Returns a tensor of shape ``[num_filtrations, n_nodes, 2]``.
        """
        filtered_v = self._filtration_values(x)  # [N, F]
        filtered_e, _ = torch.max(
            torch.stack([filtered_v[edge_index[0]], filtered_v[edge_index[1]]]),
            dim=0,
        )  # [E, F]

        # Move to CPU for GUDHI (keeps gradient-carrying tensors on device)
        fv_cpu = filtered_v.cpu()
        fe_cpu = filtered_e.cpu()
        ei_cpu = edge_index.cpu().transpose(1, 0).contiguous()

        vertex_index = torch.arange(n_nodes, dtype=torch.int)

        persistence_diagrams = torch.empty(
            (self.num_filtrations, n_nodes, 2), dtype=torch.float
        )

        for filt_idx in range(self.num_filtrations):
            for (vi, vj), (ei, ej) in zip(
                pairwise(vertex_slices.tolist()),
                pairwise(edge_slices.tolist()),
            ):
                vertices = vertex_index[vi:vj]
                edges = ei_cpu[ei:ej]

                f_verts = fv_cpu[vi:vj, filt_idx]
                f_edgs = fe_cpu[ei:ej, filt_idx]

                pd = _compute_ph_single_graph(
                    vertices, f_verts, edges, f_edgs, offset=vi
                )
                persistence_diagrams[filt_idx, vi:vj] = pd

        return persistence_diagrams.to(x.device)

    # -- forward --

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        vertex_slices: torch.Tensor,
        edge_slices: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features ``[N, features_in]``.
            edge_index: ``[2, E]``.
            vertex_slices: ``[num_graphs + 1]``.
            edge_slices: ``[num_graphs + 1]``.
            batch_index: ``[N]`` graph membership.

        Returns:
            Updated node features ``[N, features_out]``.
        """
        n_nodes = x.size(0)
        ph = self._compute_persistence_diagrams(
            x, edge_index, vertex_slices, edge_slices, n_nodes
        )  # [F, N, 2]

        # Coordinate activations per filtration, concatenated
        coord_acts = torch.cat(
            [self._coord_fun(ph[f]) for f in range(self.num_filtrations)],
            dim=1,
        )  # [N, F * total_coord]

        if self.residual_and_bn:
            h = self.bn(self.out(coord_acts))
            h = x + h
        else:
            h = F.relu(self.out(torch.cat([x, coord_acts], dim=1)))

        return h


# ---------------------------------------------------------------------------
# Graph Transformer attention block
# ---------------------------------------------------------------------------


class GraphTransformerAttention(nn.Module):
    """Multi-head self-attention over nodes within each graph."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[N, dim]`` node features.
            batch: ``[N]`` graph membership.

        Returns:
            ``[N, dim]`` updated node features.
        """
        x_dense, mask = to_dense_batch(x, batch)  # [B, Nmax, dim], [B, Nmax]
        B, Nmax, D = x_dense.shape

        Q = self.q(x_dense).view(B, Nmax, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(x_dense).view(B, Nmax, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(x_dense).view(B, Nmax, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        attn_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nmax, D)
        out = self.out_proj(out)

        return out[mask]


class GraphTransformerBlock(nn.Module):
    """Pre-norm transformer block with graph-aware self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphTransformerAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), batch)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model: TopoGraphTransformer
# ---------------------------------------------------------------------------


@dataclass
class TopoGraphTransformerHParams:
    """Hyperparameters for :class:`TopoGraphTransformer`."""

    hidden_dim: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 3
    ff_mult: int = 2
    dropout: float = 0.1

    # Topology layer
    num_filtrations: int = 4
    filtration_hidden: int = 32
    coord_funs: dict[str, int] = field(default_factory=lambda: {"triangle": 4, "line": 4})
    topo_residual_bn: bool = True
    share_filtration_parameters: bool = True
    apply_tanh: bool = False
    topo_every_n_layers: int = 1

    # Pooling
    pooling: Literal["add", "mean"] = "add"


class TopoGraphTransformer(nn.Module):
    """Graph Transformer augmented with TOGL-style topological features.

    Architecture
    ------------
    1. Linear embedding of raw node features → ``hidden_dim``.
    2. ``num_transformer_layers`` × (GraphTransformerBlock + TopologyLayer).
       The topology layer is applied every ``topo_every_n_layers`` transformer
       layers (default: every layer).
    3. Graph-level pooling (add or mean).
    4. MLP prediction head.

    The TopologyLayer learns node-level filtrations, computes H0 persistent
    homology via GUDHI (as in ``torch_topological``), and maps the resulting
    (birth, death) pairs through learnable coordinate functions to obtain
    per-node topological descriptors that are fused back into the node
    representations — exactly as in TOGL.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hparams: TopoGraphTransformerHParams | None = None,
    ) -> None:
        super().__init__()
        if hparams is None:
            hparams = TopoGraphTransformerHParams()
        self.hp = hparams
        h = hparams.hidden_dim

        self.node_embed = nn.Linear(input_dim, h)

        self.transformer_blocks = nn.ModuleList()
        self.topo_layers = nn.ModuleList()
        for i in range(hparams.num_transformer_layers):
            self.transformer_blocks.append(
                GraphTransformerBlock(h, hparams.num_heads, hparams.ff_mult, hparams.dropout)
            )
            if (i + 1) % hparams.topo_every_n_layers == 0:
                self.topo_layers.append(
                    TopologyLayer(
                        features_in=h,
                        features_out=h,
                        num_filtrations=hparams.num_filtrations,
                        filtration_hidden=hparams.filtration_hidden,
                        coord_funs=hparams.coord_funs,
                        residual_and_bn=hparams.topo_residual_bn,
                        share_filtration_parameters=hparams.share_filtration_parameters,
                        apply_tanh=hparams.apply_tanh,
                    )
                )
            else:
                self.topo_layers.append(None)  # type: ignore[arg-type]

        self.final_norm = nn.LayerNorm(h)

        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, output_dim),
        )

        self.pool = global_add_pool if hparams.pooling == "add" else global_mean_pool

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Args:
            batch: PyTorch Geometric ``Batch`` with ``.x``, ``.edge_index``,
                   and ``.batch``.

        Returns:
            Predictions ``[batch_size]`` or ``[batch_size, output_dim]``.
        """
        x: torch.Tensor = batch.x
        edge_index: torch.Tensor = batch.edge_index
        batch_index: torch.Tensor = batch.batch

        vertex_slices = self._get_slices(batch, "x")
        edge_slices = self._get_slices(batch, "edge_index")

        x = self.node_embed(x.float())

        for transformer_block, topo_layer in zip(self.transformer_blocks, self.topo_layers):
            x = transformer_block(x, batch_index)
            if topo_layer is not None:
                x = topo_layer(x, edge_index, vertex_slices, edge_slices, batch_index)

        x = self.final_norm(x)
        x = self.pool(x, batch_index)
        out = self.head(x)
        return out.squeeze(-1)

    @staticmethod
    def _get_slices(batch: Data, key: str) -> torch.Tensor:
        """Extract cumulative slice tensor for *key* from a PyG Batch."""
        if hasattr(batch, "_slice_dict") and key in batch._slice_dict:
            slices = batch._slice_dict[key]
        elif hasattr(batch, "__slices__") and key in batch.__slices__:
            slices = batch.__slices__[key]
        else:
            raise ValueError(
                f"Cannot find slice information for '{key}' in batch. "
                "Make sure you are passing a proper PyG Batch object."
            )
        if isinstance(slices, torch.Tensor):
            return slices.long()
        return torch.tensor(slices, dtype=torch.long)
