"""Tests for TopoGraphTransformer and its sub-components."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import erdos_renyi_graph

from ph_transfo.algorithms.networks.topo_transformer import (
    COORD_TRANSFORMS,
    GraphTransformerAttention,
    GraphTransformerBlock,
    GaussianTransform,
    LineTransform,
    RationalHatTransform,
    TopoGraphTransformer,
    TopoGraphTransformerHParams,
    TopologyLayer,
    TriangleTransform,
    _compute_ph_single_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_random_graphs(n_graphs: int = 6, n_nodes: int = 12, n_features: int = 8, p: float = 0.3):
    """Create a list of random PyG Data objects."""
    return [
        Data(
            x=torch.randn(n_nodes, n_features),
            edge_index=erdos_renyi_graph(n_nodes, p),
            num_nodes=n_nodes,
        )
        for _ in range(n_graphs)
    ]


@pytest.fixture()
def batch():
    """A small batched PyG graph (batch_size=3)."""
    loader = DataLoader(_make_random_graphs(6), batch_size=3)
    return next(iter(loader))


@pytest.fixture()
def single_batch():
    """A batch of size 1."""
    loader = DataLoader(_make_random_graphs(1, n_nodes=8), batch_size=1)
    return next(iter(loader))


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


class TestCoordinateTransforms:
    """Basic shape and gradient checks for each coordinate function."""

    @pytest.mark.parametrize("cls", [TriangleTransform, GaussianTransform, LineTransform, RationalHatTransform])
    def test_output_shape(self, cls):
        mod = cls(output_dim=5)
        x = torch.randn(10, 2)
        out = mod(x)
        assert out.shape == (10, 5)

    @pytest.mark.parametrize("cls", [TriangleTransform, GaussianTransform, LineTransform, RationalHatTransform])
    def test_gradient_flows(self, cls):
        mod = cls(output_dim=3)
        x = torch.randn(4, 2, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_all_registered(self):
        expected = {"triangle", "gaussian", "line", "rational_hat"}
        assert set(COORD_TRANSFORMS.keys()) == expected


# ---------------------------------------------------------------------------
# Persistent homology (GUDHI)
# ---------------------------------------------------------------------------


class TestPersistentHomology:
    def test_trivial_graph_no_edges(self):
        """Graph with no edges: every node paired with itself."""
        vertices = torch.arange(5)
        f_vertices = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])
        edges = torch.empty(0, 2, dtype=torch.int)
        f_edges = torch.empty(0)

        pd = _compute_ph_single_graph(vertices, f_vertices, edges, f_edges, offset=0)
        assert pd.shape == (5, 2)
        # No edges ⇒ birth == death for every node
        assert torch.allclose(pd[:, 0], pd[:, 1])

    def test_simple_path(self):
        """Path graph 0—1—2: nodes merge along the path."""
        vertices = torch.arange(3)
        f_vertices = torch.tensor([0.1, 0.3, 0.2])
        edges = torch.tensor([[0, 1], [1, 2]])
        f_edges = torch.tensor([0.3, 0.3])

        pd = _compute_ph_single_graph(vertices, f_vertices, edges, f_edges, offset=0)
        assert pd.shape == (3, 2)
        # At least one node should have birth != death (a merge happened)
        assert not torch.allclose(pd[:, 0], pd[:, 1])

    def test_offset_applied(self):
        """Offset shifts vertex indices properly (second graph in a batch)."""
        vertices = torch.arange(5, 8)
        f_vertices = torch.tensor([0.4, 0.1, 0.7])
        edges = torch.tensor([[5, 6], [6, 7]])
        f_edges = torch.tensor([0.4, 0.7])

        pd = _compute_ph_single_graph(vertices, f_vertices, edges, f_edges, offset=5)
        assert pd.shape == (3, 2)

    def test_persistence_diagram_values_between_filtrations(self):
        """Birth/death values should be within the range of filtration values."""
        vertices = torch.arange(4)
        f_vertices = torch.tensor([0.1, 0.5, 0.3, 0.8])
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
        f_edges = torch.tensor([0.5, 0.5, 0.8])

        pd = _compute_ph_single_graph(vertices, f_vertices, edges, f_edges, offset=0)
        assert pd[:, 0].min() >= f_vertices.min()
        assert pd[:, 1].max() <= f_vertices.max()


# ---------------------------------------------------------------------------
# TopologyLayer
# ---------------------------------------------------------------------------


class TestTopologyLayer:
    def test_output_shape_residual(self, batch):
        layer = TopologyLayer(features_in=64, features_out=64, num_filtrations=2, residual_and_bn=True)
        x = torch.randn(batch.num_nodes, 64)
        vs = batch._slice_dict["x"].clone().detach().long()
        es = batch._slice_dict["edge_index"].clone().detach().long()

        out = layer(x, batch.edge_index, vs, es, batch.batch)
        assert out.shape == (batch.num_nodes, 64)

    def test_output_shape_concat(self, batch):
        layer = TopologyLayer(features_in=64, features_out=32, num_filtrations=2, residual_and_bn=False)
        x = torch.randn(batch.num_nodes, 64)
        vs = batch._slice_dict["x"].clone().detach().long()
        es = batch._slice_dict["edge_index"].clone().detach().long()

        out = layer(x, batch.edge_index, vs, es, batch.batch)
        assert out.shape == (batch.num_nodes, 32)

    def test_independent_filtrations(self, batch):
        layer = TopologyLayer(
            features_in=64, features_out=64, num_filtrations=3,
            share_filtration_parameters=False, residual_and_bn=True,
        )
        x = torch.randn(batch.num_nodes, 64)
        vs = batch._slice_dict["x"].clone().detach().long()
        es = batch._slice_dict["edge_index"].clone().detach().long()

        out = layer(x, batch.edge_index, vs, es, batch.batch)
        assert out.shape == (batch.num_nodes, 64)

    def test_all_coord_funs(self, batch):
        layer = TopologyLayer(
            features_in=32, features_out=32, num_filtrations=2,
            coord_funs={"triangle": 2, "gaussian": 2, "line": 2, "rational_hat": 2},
        )
        x = torch.randn(batch.num_nodes, 32)
        vs = batch._slice_dict["x"].clone().detach().long()
        es = batch._slice_dict["edge_index"].clone().detach().long()

        out = layer(x, batch.edge_index, vs, es, batch.batch)
        assert out.shape == (batch.num_nodes, 32)

    def test_tanh_filtrations(self, batch):
        layer = TopologyLayer(features_in=64, features_out=64, apply_tanh=True)
        x = torch.randn(batch.num_nodes, 64)
        vs = batch._slice_dict["x"].clone().detach().long()
        es = batch._slice_dict["edge_index"].clone().detach().long()

        out = layer(x, batch.edge_index, vs, es, batch.batch)
        assert out.shape == (batch.num_nodes, 64)


# ---------------------------------------------------------------------------
# GraphTransformerAttention & Block
# ---------------------------------------------------------------------------


class TestGraphTransformerAttention:
    def test_output_shape(self, batch):
        attn = GraphTransformerAttention(dim=32, num_heads=4, dropout=0.0)
        x = torch.randn(batch.num_nodes, 32)
        out = attn(x, batch.batch)
        assert out.shape == x.shape

    def test_deterministic_eval(self, batch):
        attn = GraphTransformerAttention(dim=32, num_heads=4, dropout=0.1)
        attn.eval()
        x = torch.randn(batch.num_nodes, 32)
        out1 = attn(x, batch.batch)
        out2 = attn(x, batch.batch)
        assert torch.allclose(out1, out2)


class TestGraphTransformerBlock:
    def test_output_shape(self, batch):
        block = GraphTransformerBlock(dim=32, num_heads=4, ff_mult=2, dropout=0.0)
        x = torch.randn(batch.num_nodes, 32)
        out = block(x, batch.batch)
        assert out.shape == x.shape

    def test_residual_connection(self, batch):
        """With zero-init, the residual should make output close to input."""
        block = GraphTransformerBlock(dim=32, num_heads=4, ff_mult=2, dropout=0.0)
        # Zero all weights ⇒ attention and FFN output ≈ 0 ⇒ residual ≈ x
        for p in block.parameters():
            p.data.zero_()
        x = torch.randn(batch.num_nodes, 32)
        out = block(x, batch.batch)
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# TopoGraphTransformer (full model)
# ---------------------------------------------------------------------------


class TestTopoGraphTransformer:
    def test_forward_default_hparams(self, batch):
        model = TopoGraphTransformer(input_dim=8, output_dim=1)
        out = model(batch)
        n_graphs = batch.num_graphs
        assert out.shape == (n_graphs,)

    def test_forward_multiclass(self, batch):
        model = TopoGraphTransformer(input_dim=8, output_dim=5)
        out = model(batch)
        assert out.shape == (batch.num_graphs, 5)

    def test_backward(self, batch):
        model = TopoGraphTransformer(input_dim=8, output_dim=1)
        out = model(batch)
        loss = out.sum()
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_mean_pooling(self, batch):
        hp = TopoGraphTransformerHParams(pooling="mean")
        model = TopoGraphTransformer(input_dim=8, output_dim=1, hparams=hp)
        out = model(batch)
        assert out.shape == (batch.num_graphs,)

    def test_topo_every_2_layers(self, batch):
        hp = TopoGraphTransformerHParams(num_transformer_layers=4, topo_every_n_layers=2)
        model = TopoGraphTransformer(input_dim=8, output_dim=1, hparams=hp)
        out = model(batch)
        assert out.shape == (batch.num_graphs,)
        # Only 2 topo layers should be non-None (layers 2 and 4)
        active = [l for l in model.topo_layers if l is not None]
        assert len(active) == 2

    def test_no_shared_filtrations(self, batch):
        hp = TopoGraphTransformerHParams(share_filtration_parameters=False)
        model = TopoGraphTransformer(input_dim=8, output_dim=1, hparams=hp)
        out = model(batch)
        assert out.shape == (batch.num_graphs,)

    def test_concat_mode(self, batch):
        hp = TopoGraphTransformerHParams(topo_residual_bn=False)
        model = TopoGraphTransformer(input_dim=8, output_dim=1, hparams=hp)
        out = model(batch)
        assert out.shape == (batch.num_graphs,)

    def test_custom_coord_funs(self, batch):
        hp = TopoGraphTransformerHParams(
            coord_funs={"gaussian": 3, "rational_hat": 3},
        )
        model = TopoGraphTransformer(input_dim=8, output_dim=1, hparams=hp)
        out = model(batch)
        assert out.shape == (batch.num_graphs,)

    def test_single_graph_batch(self, single_batch):
        model = TopoGraphTransformer(input_dim=8, output_dim=1)
        out = model(single_batch)
        assert out.shape == (1,)

    def test_eval_mode(self, batch):
        model = TopoGraphTransformer(input_dim=8, output_dim=1)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (batch.num_graphs,)

    def test_different_graph_sizes(self):
        """Batch with graphs of varying node counts."""
        graphs = [
            Data(x=torch.randn(n, 4), edge_index=erdos_renyi_graph(n, 0.4), num_nodes=n)
            for n in [5, 10, 3, 7]
        ]
        loader = DataLoader(graphs, batch_size=4)
        batch = next(iter(loader))

        model = TopoGraphTransformer(input_dim=4, output_dim=1)
        out = model(batch)
        assert out.shape == (4,)
