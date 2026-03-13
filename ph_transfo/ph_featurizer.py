"""Persistence homology Betti number featurizer for graphs.

Implements edge-weight filtration and sublevel-set (degree) filtration to
compute Betti number curves β0 (connected components) and β1 (independent
cycles), which are used as topological features for the graph transformer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Union-Find (Disjoint-Set) data structure
# ---------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set union-find with path compression and union by rank.

    Tracks the number of connected components via ``num_components``.

    Args:
        n: Number of elements (nodes).
    """

    def __init__(self, n: int) -> None:
        self.parent: List[int] = list(range(n))
        self.rank: List[int] = [0] * n
        self.num_components: int = n

    def find(self, x: int) -> int:
        """Return the root of the component containing *x* (path-compressed)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x: int, y: int) -> bool:
        """Merge the components of *x* and *y*.

        Returns:
            ``True`` if the two nodes were in different components and were
            merged; ``False`` if they already shared a component.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.num_components -= 1
        return True


# ---------------------------------------------------------------------------
# Core filtration functions
# ---------------------------------------------------------------------------


def compute_betti_curves(
    edge_index: np.ndarray,
    num_nodes: int,
    edge_weights: Optional[np.ndarray] = None,
    num_steps: int = 50,
) -> np.ndarray:
    """Compute Betti number curves via an edge-weight filtration.

    At each filtration level *t*, only edges with weight ≤ *t* are included.
    The Betti numbers are computed from the resulting subgraph:

    * β0(t) — number of connected components
    * β1(t) — number of independent cycles = |E(t)| − |V| + β0(t)

    Args:
        edge_index: Integer array of shape ``(2, E)`` with source/target indices.
        num_nodes: Total number of nodes in the graph.
        edge_weights: Float array of shape ``(E,)``.  Defaults to all-ones
            (unweighted graph).
        num_steps: Number of evenly-spaced filtration levels.

    Returns:
        betti_curves: Float array of shape ``(num_steps, 2)`` where column 0
            is β0 and column 1 is β1 at each filtration level.
    """
    if edge_weights is None:
        edge_weights = np.ones(edge_index.shape[1], dtype=float)

    edge_weights = np.asarray(edge_weights, dtype=float)

    # Work with undirected edges (upper triangle only, no self-loops)
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    src_u = src[mask]
    dst_u = dst[mask]
    w_u = edge_weights[mask]

    betti_curves = np.zeros((num_steps, 2), dtype=float)

    if len(w_u) == 0:
        # No edges: β0 = num_nodes, β1 = 0 at every level
        betti_curves[:, 0] = float(num_nodes)
        return betti_curves

    min_w, max_w = float(w_u.min()), float(w_u.max())
    if min_w == max_w:
        thresholds = np.full(num_steps, min_w)
    else:
        thresholds = np.linspace(min_w, max_w, num_steps)

    # Sort edges by weight for a single-pass sweep
    sort_idx = np.argsort(w_u)
    src_sorted = src_u[sort_idx]
    dst_sorted = dst_u[sort_idx]
    w_sorted = w_u[sort_idx]

    uf = UnionFind(num_nodes)
    edge_ptr = 0
    num_edges_added = 0

    for i, t in enumerate(thresholds):
        while edge_ptr < len(w_sorted) and w_sorted[edge_ptr] <= t:
            uf.union(int(src_sorted[edge_ptr]), int(dst_sorted[edge_ptr]))
            num_edges_added += 1
            edge_ptr += 1

        beta0 = uf.num_components
        beta1 = max(0, num_edges_added - num_nodes + beta0)
        betti_curves[i, 0] = beta0
        betti_curves[i, 1] = beta1

    return betti_curves


def compute_node_persistence_features(
    edge_index: np.ndarray,
    num_nodes: int,
    node_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute node-level Betti number features via a sublevel-set filtration.

    For each node *v* with scalar value ``f(v)``, the subgraph induced by
    ``{u : f(u) ≤ f(v)}`` is built and its Betti numbers [β0, β1] are
    recorded as the feature vector of *v*.  This captures the local
    topological context of each node in the graph.

    Args:
        edge_index: Integer array of shape ``(2, E)``.
        num_nodes: Total number of nodes.
        node_values: Float array of shape ``(N,)`` used as the filtration
            function.  Defaults to node degree.

    Returns:
        node_features: Float array of shape ``(N, 2)`` with [β0, β1] per node.
    """
    if node_values is None:
        degrees = np.zeros(num_nodes, dtype=float)
        if edge_index.shape[1] > 0:
            np.add.at(degrees, edge_index[0], 1)
        node_values = degrees

    node_features = np.zeros((num_nodes, 2), dtype=float)

    for v in range(num_nodes):
        threshold = node_values[v]
        active = set(int(i) for i in np.where(node_values <= threshold)[0])

        uf = UnionFind(num_nodes)
        num_edges = 0

        for j in range(edge_index.shape[1]):
            u, w = int(edge_index[0, j]), int(edge_index[1, j])
            if u in active and w in active and u < w:
                uf.union(u, w)
                num_edges += 1

        n_active = len(active)
        if n_active == 0:
            continue

        active_roots = {uf.find(x) for x in active}
        beta0 = len(active_roots)
        beta1 = max(0, num_edges - n_active + beta0)
        node_features[v, 0] = beta0
        node_features[v, 1] = beta1

    return node_features


# ---------------------------------------------------------------------------
# High-level featurizer class
# ---------------------------------------------------------------------------


class PHFeaturizer:
    """Persistence Homology featurizer for graphs.

    Computes Betti number curves from graph filtrations and provides
    fixed-size feature vectors for use in graph transformers.

    Args:
        num_filtration_steps: Number of evenly-spaced filtration levels.
        max_homology_dim: Maximum homology dimension to include in the
            feature vector (0 → only β0; 1 → β0 and β1).
    """

    def __init__(
        self,
        num_filtration_steps: int = 50,
        max_homology_dim: int = 1,
    ) -> None:
        if max_homology_dim not in (0, 1):
            raise ValueError("max_homology_dim must be 0 or 1")
        self.num_filtration_steps = num_filtration_steps
        self.max_homology_dim = max_homology_dim

    @property
    def feature_dim(self) -> int:
        """Dimension of the global graph feature vector."""
        return self.num_filtration_steps * (self.max_homology_dim + 1)

    def compute_graph_features(
        self,
        edge_index: Union[np.ndarray, "torch.Tensor"],
        num_nodes: int,
        edge_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> np.ndarray:
        """Return a flattened Betti-curve feature vector for the whole graph.

        Args:
            edge_index: Shape ``(2, E)``.
            num_nodes: Number of nodes.
            edge_weights: Shape ``(E,)`` edge weights; defaults to all-ones.

        Returns:
            features: Float array of shape ``(feature_dim,)``.
        """
        edge_index, edge_weights = _to_numpy(edge_index, edge_weights)
        betti_curves = compute_betti_curves(
            edge_index, num_nodes, edge_weights, self.num_filtration_steps
        )
        # Truncate to requested homology dimensions
        betti_curves = betti_curves[:, : self.max_homology_dim + 1]
        return betti_curves.flatten()

    def compute_node_features(
        self,
        edge_index: Union[np.ndarray, "torch.Tensor"],
        num_nodes: int,
        node_values: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> np.ndarray:
        """Return node-level Betti number features via sublevel-set filtration.

        Args:
            edge_index: Shape ``(2, E)``.
            num_nodes: Number of nodes.
            node_values: Shape ``(N,)`` filtration values; defaults to degree.

        Returns:
            features: Float array of shape ``(N, 2)``.
        """
        edge_index, node_values = _to_numpy(edge_index, node_values)
        return compute_node_persistence_features(edge_index, num_nodes, node_values)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_numpy(
    arr1: Union[np.ndarray, "torch.Tensor"],
    arr2: Optional[Union[np.ndarray, "torch.Tensor"]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert tensors to numpy arrays (no-op if already numpy)."""
    try:
        import torch  # noqa: PLC0415

        if isinstance(arr1, torch.Tensor):
            arr1 = arr1.detach().cpu().numpy()
        if isinstance(arr2, torch.Tensor):
            arr2 = arr2.detach().cpu().numpy()
    except ImportError:
        pass
    return arr1, arr2
