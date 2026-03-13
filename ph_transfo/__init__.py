"""ph_transfo — Graph Transformer with Persistence Homology Betti number featurization."""

from .graph_transformer import (
    GraphTransformerEncoder,
    GraphTransformerLayer,
    MultiHeadGraphAttention,
)
from .model import PHGraphTransformer
from .ph_featurizer import (
    PHFeaturizer,
    UnionFind,
    compute_betti_curves,
    compute_node_persistence_features,
)

__all__ = [
    # PH featurizer
    "PHFeaturizer",
    "UnionFind",
    "compute_betti_curves",
    "compute_node_persistence_features",
    # Graph transformer building blocks
    "MultiHeadGraphAttention",
    "GraphTransformerLayer",
    "GraphTransformerEncoder",
    # Combined model
    "PHGraphTransformer",
]
