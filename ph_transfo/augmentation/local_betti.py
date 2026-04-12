"""Compute local persistent homology features for molecular graphs.

For each node v, builds a single local Rips complex (at max scale) and extracts:
  1. Dense Betti curves: β₀(ε), β₁(ε) at many scales (from one PH computation)
  2. Local persistence statistics: mean/max/total persistence, entropy, feature count
  3. Edge filtration values: the Rips scale at which each bond edge first appears
"""

from __future__ import annotations

import numpy as np
import gudhi as gd


def _betti_curve_from_pairs(
    pairs: list[tuple[float, float]],
    scales: np.ndarray,
    dim: int = 0,
) -> np.ndarray:
    """Evaluate Betti curve at given scales from finite persistence pairs."""
    curve = np.zeros(len(scales), dtype=np.float32)
    for birth, death in pairs:
        alive = (scales >= birth) & (scales < death)
        curve[alive] += 1
    if dim == 0:
        curve += 1  # essential H0 feature (component that never dies)
    return curve


def _persistence_stats(pairs: list[tuple[float, float]]) -> np.ndarray:
    """Compute [mean_pers, max_pers, total_pers, entropy, count] from pairs."""
    stats = np.zeros(5, dtype=np.float32)
    if not pairs:
        return stats
    pers = np.array([d - b for b, d in pairs], dtype=np.float32)
    stats[0] = pers.mean()
    stats[1] = pers.max()
    stats[2] = pers.sum()
    if stats[2] > 0:
        probs = pers / stats[2]
        stats[3] = -np.sum(probs * np.log(probs + 1e-10))
    stats[4] = len(pers)
    return stats


def local_ph_features(
    positions: np.ndarray,
    n_curve_points: int = 20,
    curve_range: tuple[float, float] = (1.0, 6.0),
) -> np.ndarray:
    """Compute rich local PH features for each atom.

    For each node v, builds ONE Rips complex on B(v, max_scale) and derives
    Betti curves + persistence stats from a single persistence computation.

    Features per node (total = 2*n_curve_points + 10):
      - Betti-0 curve at n_curve_points scales      [n_curve_points]
      - Betti-1 curve at n_curve_points scales      [n_curve_points]
      - H0 stats: mean_pers, max_pers, total_pers, entropy, count  [5]
      - H1 stats: mean_pers, max_pers, total_pers, entropy, count  [5]

    Args:
        positions: Atom 3D coordinates, shape ``[n_atoms, 3]``.
        n_curve_points: Number of evaluation points for Betti curves.
        curve_range: (min, max) scale range for Betti curves.

    Returns:
        Array of shape ``[n_atoms, 2*n_curve_points + 10]``.
    """
    n = len(positions)
    scales = np.linspace(curve_range[0], curve_range[1], n_curve_points)
    max_scale = curve_range[1]
    feat_dim = 2 * n_curve_points + 10
    features = np.zeros((n, feat_dim), dtype=np.float32)

    dist_matrix = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    for v in range(n):
        nbr_mask = dist_matrix[v] <= max_scale
        nbr_indices = np.where(nbr_mask)[0]

        if len(nbr_indices) <= 1:
            features[v, :n_curve_points] = 1.0  # β₀ = 1
            continue

        # One Rips complex per node at max_scale
        local_dist = dist_matrix[np.ix_(nbr_indices, nbr_indices)]
        rips = gd.RipsComplex(distance_matrix=local_dist, max_edge_length=max_scale)
        st = rips.create_simplex_tree(max_dimension=2)
        st.persistence()

        pairs = {0: [], 1: []}
        for dim, (birth, death) in st.persistence():
            if dim in pairs and death != float("inf"):
                pairs[dim].append((birth, death))

        # Betti curves from the single persistence diagram
        features[v, :n_curve_points] = _betti_curve_from_pairs(pairs[0], scales, dim=0)
        features[v, n_curve_points:2*n_curve_points] = _betti_curve_from_pairs(pairs[1], scales, dim=1)

        # Persistence statistics
        features[v, 2*n_curve_points:2*n_curve_points+5] = _persistence_stats(pairs[0])
        features[v, 2*n_curve_points+5:] = _persistence_stats(pairs[1])

    return features


def local_betti_features(
    positions: np.ndarray,
    scales: list[float],
) -> np.ndarray:
    """Compute local Betti-0 and Betti-1 curves at given scales for each atom.

    This is the legacy interface used by the ``betti_scales`` concat mode.
    For each node v, builds a Rips complex on its neighbourhood and evaluates
    the Betti curves at exactly the requested scales.

    Args:
        positions: Atom 3D coordinates, shape ``[n_atoms, 3]``.
        scales: List of filtration scales to evaluate.

    Returns:
        Array of shape ``[n_atoms, 2 * len(scales)]``.
    """
    scales_arr = np.asarray(scales, dtype=np.float32)
    n = len(positions)
    n_scales = len(scales_arr)
    max_scale = float(scales_arr.max())
    features = np.zeros((n, 2 * n_scales), dtype=np.float32)

    dist_matrix = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    for v in range(n):
        nbr_mask = dist_matrix[v] <= max_scale
        nbr_indices = np.where(nbr_mask)[0]

        if len(nbr_indices) <= 1:
            features[v, :n_scales] = 1.0  # β₀ = 1
            continue

        local_dist = dist_matrix[np.ix_(nbr_indices, nbr_indices)]
        rips = gd.RipsComplex(distance_matrix=local_dist, max_edge_length=max_scale)
        st = rips.create_simplex_tree(max_dimension=2)
        st.persistence()

        pairs = {0: [], 1: []}
        for dim, (birth, death) in st.persistence():
            if dim in pairs and death != float("inf"):
                pairs[dim].append((birth, death))

        features[v, :n_scales] = _betti_curve_from_pairs(pairs[0], scales_arr, dim=0)
        features[v, n_scales:] = _betti_curve_from_pairs(pairs[1], scales_arr, dim=1)

    return features


def edge_filtration_values(
    positions: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Compute the Rips filtration value for each edge.

    This is simply the Euclidean distance between the two endpoints.

    Args:
        positions: Atom 3D coordinates, shape ``[n_atoms, 3]``.
        edge_index: Edge index array, shape ``[2, n_edges]``.

    Returns:
        Array of shape ``[n_edges, 1]`` with filtration values.
    """
    src, dst = edge_index[0], edge_index[1]
    dists = np.linalg.norm(positions[src] - positions[dst], axis=-1, keepdims=True)
    return dists.astype(np.float32)
