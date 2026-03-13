# ph-transfo

A **Graph Transformer** with **Persistence Homology Betti number featurization**.

## Overview

`ph-transfo` enriches standard graph transformer representations with
topological features derived from persistence homology.  Two complementary
featurization strategies are combined:

| Feature type | How it works |
|---|---|
| **Global PH features** | Betti-number curves (β0, β1) computed over the entire graph via an edge-weight filtration.  A single feature vector is projected to `d_model` and *added* to every node embedding. |
| **Node PH features** | Per-node [β0, β1] computed from a sublevel-set filtration (default: node degree).  Appended to raw node features before the linear embedding. |

The resulting enriched node representations are then processed by a
pre-norm multi-head graph transformer encoder with learned adjacency bias.

### Architecture

```
Input graph (node features + edge index)
        │
        ├── PHFeaturizer ──► global Betti curve ──► Linear ──► broadcast add
        │
        └── node PH features ──► concat ──► Linear node embed
                                                    │
                                          GraphTransformerEncoder
                                          (L × [MultiHeadGraphAttention + FFN])
                                                    │
                                              graph readout (mean/sum/max)
                                                    │
                                              Linear output head
                                                    │
                                               prediction
```

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.9, NumPy ≥ 1.20.

## Quick start

```python
import torch
from ph_transfo import PHGraphTransformer

# A triangle graph: 3 nodes, 3 undirected edges
edge_index = torch.tensor([[0, 1, 2, 1, 2, 0],
                            [1, 2, 0, 0, 1, 2]])
node_features = torch.randn(3, 16)   # 3 nodes, 16-dim features

model = PHGraphTransformer(
    node_in_dim=16,        # raw node feature dimension
    d_model=64,            # transformer hidden dimension
    num_layers=4,          # stacked transformer layers
    num_heads=4,           # attention heads
    num_filtration_steps=50,  # PH filtration resolution
    out_dim=2,             # e.g. binary classification logits
)

logits = model(node_features, edge_index)  # shape (2,)
```

### Node embeddings

```python
node_emb = model.encode(node_features, edge_index)  # shape (N, d_model)
```

## Components

| Class / function | Description |
|---|---|
| `PHFeaturizer` | Computes Betti-curve global features and node-level PH features |
| `compute_betti_curves` | Edge-weight filtration → (β0, β1) at each level |
| `compute_node_persistence_features` | Sublevel-set filtration → per-node (β0, β1) |
| `MultiHeadGraphAttention` | Multi-head self-attention with optional structural bias |
| `GraphTransformerLayer` | Single pre-norm transformer layer (attention + FFN) |
| `GraphTransformerEncoder` | Stack of `GraphTransformerLayer` modules |
| `PHGraphTransformer` | End-to-end model combining PH features + graph transformer |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
