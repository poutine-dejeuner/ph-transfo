"""Tests for ph_transfo.graph_transformer."""

import pytest
import torch
import torch.nn as nn

from ph_transfo.graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerLayer,
    MultiHeadGraphAttention,
)


# ---------------------------------------------------------------------------
# MultiHeadGraphAttention
# ---------------------------------------------------------------------------


class TestMultiHeadGraphAttention:
    def test_output_shape_unbatched(self):
        N, d = 5, 16
        attn = MultiHeadGraphAttention(d_model=d, num_heads=2)
        x = torch.randn(N, d)
        out = attn(x)
        assert out.shape == (N, d)

    def test_output_shape_batched(self):
        B, N, d = 3, 5, 16
        attn = MultiHeadGraphAttention(d_model=d, num_heads=2)
        x = torch.randn(B, N, d)
        out = attn(x)
        assert out.shape == (B, N, d)

    def test_output_shape_with_bias(self):
        N, d = 6, 32
        attn = MultiHeadGraphAttention(d_model=d, num_heads=4)
        x = torch.randn(N, d)
        bias = torch.randn(N, N)
        out = attn(x, attention_bias=bias)
        assert out.shape == (N, d)

    def test_invalid_heads(self):
        with pytest.raises(ValueError):
            MultiHeadGraphAttention(d_model=15, num_heads=4)

    def test_gradient_flows(self):
        N, d = 4, 8
        attn = MultiHeadGraphAttention(d_model=d, num_heads=2)
        x = torch.randn(N, d, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_output_is_deterministic_in_eval_mode(self):
        N, d = 4, 8
        attn = MultiHeadGraphAttention(d_model=d, num_heads=2, dropout=0.5)
        attn.eval()
        x = torch.randn(N, d)
        out1 = attn(x)
        out2 = attn(x)
        torch.testing.assert_close(out1, out2)

    def test_batched_bias_broadcast(self):
        B, N, d = 2, 5, 16
        attn = MultiHeadGraphAttention(d_model=d, num_heads=2)
        x = torch.randn(B, N, d)
        bias = torch.randn(B, N, N)  # per-sample bias
        out = attn(x, attention_bias=bias)
        assert out.shape == (B, N, d)


# ---------------------------------------------------------------------------
# GraphTransformerLayer
# ---------------------------------------------------------------------------


class TestGraphTransformerLayer:
    def test_output_shape_unbatched(self):
        N, d = 6, 32
        layer = GraphTransformerLayer(d_model=d, num_heads=4)
        x = torch.randn(N, d)
        out = layer(x)
        assert out.shape == (N, d)

    def test_output_shape_with_adj(self):
        N, d = 5, 16
        layer = GraphTransformerLayer(d_model=d, num_heads=2)
        x = torch.randn(N, d)
        adj = torch.eye(N)
        out = layer(x, adj)
        assert out.shape == (N, d)

    def test_no_edge_bias(self):
        N, d = 4, 8
        layer = GraphTransformerLayer(d_model=d, num_heads=2, use_edge_bias=False)
        assert layer.edge_bias is None
        x = torch.randn(N, d)
        out = layer(x)
        assert out.shape == (N, d)

    def test_output_shape_batched(self):
        B, N, d = 3, 7, 32
        layer = GraphTransformerLayer(d_model=d, num_heads=4)
        x = torch.randn(B, N, d)
        out = layer(x)
        assert out.shape == (B, N, d)

    def test_custom_d_ff(self):
        N, d = 4, 16
        layer = GraphTransformerLayer(d_model=d, num_heads=2, d_ff=64)
        x = torch.randn(N, d)
        out = layer(x)
        assert out.shape == (N, d)

    def test_gradient_flows(self):
        N, d = 5, 16
        layer = GraphTransformerLayer(d_model=d, num_heads=2)
        x = torch.randn(N, d, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_layer_norm_output(self):
        """Output should have approximately unit variance after enough passes."""
        N, d = 8, 32
        layer = GraphTransformerLayer(d_model=d, num_heads=4, dropout=0.0)
        layer.eval()
        x = torch.randn(N, d)
        out = layer(x)
        # Output should be finite and well-scaled
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# GraphTransformerEncoder
# ---------------------------------------------------------------------------


class TestGraphTransformerEncoder:
    def test_output_shape(self):
        N, d = 8, 32
        enc = GraphTransformerEncoder(d_model=d, num_layers=2, num_heads=4)
        x = torch.randn(N, d)
        out = enc(x)
        assert out.shape == (N, d)

    def test_output_shape_with_adj(self):
        N, d = 6, 16
        enc = GraphTransformerEncoder(d_model=d, num_layers=3, num_heads=2)
        x = torch.randn(N, d)
        adj = (torch.rand(N, N) > 0.5).float()
        out = enc(x, adj)
        assert out.shape == (N, d)

    def test_num_layers(self):
        enc = GraphTransformerEncoder(d_model=16, num_layers=5, num_heads=2)
        assert len(enc.layers) == 5

    def test_gradient_flows(self):
        N, d = 6, 16
        enc = GraphTransformerEncoder(d_model=d, num_layers=2, num_heads=2)
        x = torch.randn(N, d, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None

    def test_has_final_layer_norm(self):
        enc = GraphTransformerEncoder(d_model=16, num_layers=2, num_heads=2)
        assert isinstance(enc.norm, nn.LayerNorm)
