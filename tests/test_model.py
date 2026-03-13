"""Tests for ph_transfo.model (PHGraphTransformer)."""

import numpy as np
import pytest
import torch

from ph_transfo.model import PHGraphTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def triangle_graph():
    """Triangle graph: 3 nodes, 3 undirected edges."""
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
    num_nodes = 3
    node_features = torch.randn(num_nodes, 4)
    return node_features, edge_index, num_nodes


@pytest.fixture()
def path_graph():
    """Path graph 0-1-2-3: 4 nodes, 3 undirected edges."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    num_nodes = 4
    node_features = torch.randn(num_nodes, 8)
    return node_features, edge_index, num_nodes


# ---------------------------------------------------------------------------
# Basic forward-pass tests
# ---------------------------------------------------------------------------


class TestPHGraphTransformerForward:
    def test_output_shape_single_class(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        model = PHGraphTransformer(
            node_in_dim=4,
            d_model=16,
            num_layers=2,
            num_heads=2,
            num_filtration_steps=10,
            out_dim=1,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (1,)

    def test_output_shape_multiclass(self, path_graph):
        node_features, edge_index, _ = path_graph
        model = PHGraphTransformer(
            node_in_dim=8,
            d_model=16,
            num_layers=2,
            num_heads=2,
            num_filtration_steps=10,
            out_dim=5,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (5,)

    def test_output_finite(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=2,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert torch.isfinite(out).all()

    def test_no_ph_features(self, triangle_graph):
        """Model should still run when both PH options are disabled."""
        node_features, edge_index, _ = triangle_graph
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
            use_node_ph=False, use_graph_ph=False,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (1,)

    def test_only_node_ph(self, path_graph):
        node_features, edge_index, _ = path_graph
        model = PHGraphTransformer(
            node_in_dim=8, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
            use_node_ph=True, use_graph_ph=False,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (1,)

    def test_only_graph_ph(self, path_graph):
        node_features, edge_index, _ = path_graph
        model = PHGraphTransformer(
            node_in_dim=8, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
            use_node_ph=False, use_graph_ph=True,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (1,)

    def test_readout_modes(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        for mode in ("mean", "sum", "max"):
            model = PHGraphTransformer(
                node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
                num_filtration_steps=5, out_dim=1, readout=mode,
            )
            model.eval()
            out = model(node_features, edge_index)
            assert out.shape == (1,), f"readout={mode} failed"

    def test_invalid_readout(self):
        with pytest.raises(ValueError):
            PHGraphTransformer(node_in_dim=4, readout="invalid")

    def test_edge_weights(self, path_graph):
        node_features, edge_index, _ = path_graph
        edge_weights = torch.rand(edge_index.shape[1])
        model = PHGraphTransformer(
            node_in_dim=8, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        model.eval()
        out = model(node_features, edge_index, edge_weights=edge_weights)
        assert out.shape == (1,)

    def test_precomputed_adj(self, triangle_graph):
        node_features, edge_index, N = triangle_graph
        adj = torch.zeros(N, N)
        adj[edge_index[0], edge_index[1]] = 1.0
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        model.eval()
        out = model(node_features, edge_index, adj=adj)
        assert out.shape == (1,)

    def test_numpy_edge_index(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        ei_np = edge_index.numpy()
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        model.eval()
        out = model(node_features, ei_np)
        assert out.shape == (1,)


# ---------------------------------------------------------------------------
# Gradient / training tests
# ---------------------------------------------------------------------------


class TestPHGraphTransformerGradients:
    def test_backward_pass(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        node_features = node_features.requires_grad_(True)
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=2, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        out = model(node_features, edge_index)
        out.sum().backward()
        # node_features gradient might be None because PH is computed in numpy;
        # but model parameters should have gradients.
        has_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_param_grad

    def test_parameters_update(self, path_graph):
        node_features, edge_index, _ = path_graph
        model = PHGraphTransformer(
            node_in_dim=8, d_model=16, num_layers=2, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        params_before = [p.clone().detach() for p in model.parameters()]
        out = model(node_features, edge_index)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        params_after = list(model.parameters())
        changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(params_before, params_after)
        )
        assert changed


# ---------------------------------------------------------------------------
# encode() (node embeddings)
# ---------------------------------------------------------------------------


class TestPHGraphTransformerEncode:
    def test_encode_shape(self, path_graph):
        node_features, edge_index, N = path_graph
        d_model = 16
        model = PHGraphTransformer(
            node_in_dim=8, d_model=d_model, num_layers=2, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        model.eval()
        emb = model.encode(node_features, edge_index)
        assert emb.shape == (N, d_model)

    def test_encode_finite(self, triangle_graph):
        node_features, edge_index, _ = triangle_graph
        model = PHGraphTransformer(
            node_in_dim=4, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=1,
        )
        model.eval()
        emb = model.encode(node_features, edge_index)
        assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# Batched forward pass
# ---------------------------------------------------------------------------


class TestPHGraphTransformerBatched:
    def test_batched_output_shape(self):
        B, N, d_in, d_out = 3, 5, 4, 2
        node_features = torch.randn(B, N, d_in)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # shared topology
        model = PHGraphTransformer(
            node_in_dim=d_in, d_model=16, num_layers=1, num_heads=2,
            num_filtration_steps=5, out_dim=d_out,
        )
        model.eval()
        out = model(node_features, edge_index)
        assert out.shape == (B, d_out)
