"""Tests for ph_transfo.ph_featurizer."""

import numpy as np
import pytest

from ph_transfo.ph_featurizer import (
    PHFeaturizer,
    UnionFind,
    compute_betti_curves,
    compute_node_persistence_features,
)


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_initial_components(self):
        uf = UnionFind(5)
        assert uf.num_components == 5

    def test_union_merges_components(self):
        uf = UnionFind(4)
        merged = uf.union(0, 1)
        assert merged is True
        assert uf.num_components == 3

    def test_union_same_component_returns_false(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        merged = uf.union(0, 1)
        assert merged is False
        assert uf.num_components == 3

    def test_find_consistency(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(1) == uf.find(2)
        assert uf.find(3) != uf.find(0)

    def test_full_merge(self):
        n = 5
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        assert uf.num_components == 1


# ---------------------------------------------------------------------------
# compute_betti_curves — edge-weight filtration
# ---------------------------------------------------------------------------


class TestComputeBettiCurves:
    def test_single_node_no_edges(self):
        edge_index = np.empty((2, 0), dtype=int)
        curves = compute_betti_curves(edge_index, num_nodes=1, num_steps=10)
        assert curves.shape == (10, 2)
        np.testing.assert_array_equal(curves[:, 0], 1)  # 1 component
        np.testing.assert_array_equal(curves[:, 1], 0)  # no cycles

    def test_two_nodes_no_edges(self):
        edge_index = np.empty((2, 0), dtype=int)
        curves = compute_betti_curves(edge_index, num_nodes=2, num_steps=5)
        np.testing.assert_array_equal(curves[:, 0], 2)
        np.testing.assert_array_equal(curves[:, 1], 0)

    def test_path_two_nodes(self):
        # One edge connecting node 0 and node 1 (weight=1)
        edge_index = np.array([[0, 1], [1, 0]])  # directed; we take upper tri
        curves = compute_betti_curves(edge_index, num_nodes=2, num_steps=5)
        # At every level (min_w == max_w == 1.0) edge is included
        assert curves[-1, 0] == 1  # 1 component
        assert curves[-1, 1] == 0  # no cycles

    def test_triangle_has_cycle(self):
        # Triangle: 3 nodes, 3 edges
        edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        curves = compute_betti_curves(edge_index, num_nodes=3, num_steps=10)
        # All edges have same weight → final state is everything included
        assert curves[-1, 0] == 1   # 1 connected component
        assert curves[-1, 1] == 1   # 1 independent cycle

    def test_weighted_triangle_filtration(self):
        # Triangle with distinct edge weights: 0.3, 0.5, 0.7
        # Build directed edges (both directions per undirected edge)
        src = np.array([0, 1, 1, 2, 0, 2])
        dst = np.array([1, 0, 2, 1, 2, 0])
        weights = np.array([0.3, 0.3, 0.5, 0.5, 0.7, 0.7])
        edge_index = np.vstack([src, dst])
        curves = compute_betti_curves(edge_index, num_nodes=3, edge_weights=weights, num_steps=3)
        # At the first level (t=0.3): edge(0,1) added → β0=2, β1=0
        assert curves[0, 0] == 2
        assert curves[0, 1] == 0
        # At the last level (t=0.7): all edges → β0=1, β1=1
        assert curves[-1, 0] == 1
        assert curves[-1, 1] == 1

    def test_two_disjoint_edges(self):
        # 4 nodes: edges (0-1) and (2-3)
        edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        curves = compute_betti_curves(edge_index, num_nodes=4, num_steps=5)
        assert curves[-1, 0] == 2  # 2 components
        assert curves[-1, 1] == 0  # no cycles

    def test_self_loops_excluded(self):
        # Only self-loops — treated as no edges
        edge_index = np.array([[0, 1, 2], [0, 1, 2]])
        curves = compute_betti_curves(edge_index, num_nodes=3, num_steps=5)
        np.testing.assert_array_equal(curves[:, 0], 3)
        np.testing.assert_array_equal(curves[:, 1], 0)

    def test_output_shape(self):
        edge_index = np.array([[0, 1], [1, 0]])
        curves = compute_betti_curves(edge_index, num_nodes=3, num_steps=20)
        assert curves.shape == (20, 2)

    def test_default_weights_unweighted(self):
        # No weights supplied → unit weights → all edges added at first step
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        curves_weighted = compute_betti_curves(
            edge_index, num_nodes=3, edge_weights=np.ones(4), num_steps=5
        )
        curves_default = compute_betti_curves(edge_index, num_nodes=3, num_steps=5)
        np.testing.assert_array_equal(curves_weighted, curves_default)


# ---------------------------------------------------------------------------
# compute_node_persistence_features
# ---------------------------------------------------------------------------


class TestComputeNodePersistenceFeatures:
    def test_output_shape(self):
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        feats = compute_node_persistence_features(edge_index, num_nodes=3)
        assert feats.shape == (3, 2)

    def test_isolated_nodes_zero_features(self):
        # Node 2 is isolated (no edges connect to it)
        edge_index = np.array([[0, 1], [1, 0]])
        feats = compute_node_persistence_features(
            edge_index, num_nodes=3, node_values=np.array([1.0, 1.0, 2.0])
        )
        # At threshold=2 (node 2 included): active = {0,1,2}, edge 0-1 present.
        # Two components: {0,1} and {2} → β0=2, β1=0
        assert feats[2, 0] == 2
        assert feats[2, 1] == 0

    def test_beta1_cycle_at_highest_node(self):
        # Triangle: all nodes at threshold=1
        edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        feats = compute_node_persistence_features(
            edge_index, num_nodes=3, node_values=np.ones(3)
        )
        # All three nodes active (same value) → triangle → β0=1, β1=1
        for i in range(3):
            assert feats[i, 0] == 1
            assert feats[i, 1] == 1

    def test_default_degree_values(self):
        # Path 0-1-2: degrees = [1, 2, 1]
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        feats = compute_node_persistence_features(edge_index, num_nodes=3)
        assert feats.shape == (3, 2)
        # All values are non-negative
        assert (feats >= 0).all()


# ---------------------------------------------------------------------------
# PHFeaturizer
# ---------------------------------------------------------------------------


class TestPHFeaturizer:
    def test_feature_dim_both_dimensions(self):
        ph = PHFeaturizer(num_filtration_steps=30, max_homology_dim=1)
        assert ph.feature_dim == 60

    def test_feature_dim_zero_only(self):
        ph = PHFeaturizer(num_filtration_steps=30, max_homology_dim=0)
        assert ph.feature_dim == 30

    def test_compute_graph_features_shape(self):
        ph = PHFeaturizer(num_filtration_steps=20)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        feats = ph.compute_graph_features(edge_index, num_nodes=3)
        assert feats.shape == (ph.feature_dim,)

    def test_compute_node_features_shape(self):
        ph = PHFeaturizer(num_filtration_steps=20)
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        feats = ph.compute_node_features(edge_index, num_nodes=3)
        assert feats.shape == (3, 2)

    def test_invalid_max_homology_dim(self):
        with pytest.raises(ValueError):
            PHFeaturizer(max_homology_dim=2)

    def test_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        ph = PHFeaturizer(num_filtration_steps=10)
        edge_index_t = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        feats = ph.compute_graph_features(edge_index_t, num_nodes=3)
        assert feats.shape == (ph.feature_dim,)

    def test_graph_features_triangle_beta1(self):
        ph = PHFeaturizer(num_filtration_steps=5, max_homology_dim=1)
        # Triangle
        edge_index = np.array([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        feats = ph.compute_graph_features(edge_index, num_nodes=3)
        feats_2d = feats.reshape(5, 2)
        # Last filtration level: β1 should be 1
        assert feats_2d[-1, 1] == 1.0
