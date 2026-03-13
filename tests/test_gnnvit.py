import torch
from torch_geometric.data import Batch, Data

from reg_transfo.algorithms.networks.gnnvit import GNNViTNetwork


def test_molecule_gnn_transformer_forward():
    model = GNNViTNetwork(
        schnet_hidden_channels=128,
        schnet_num_filters=128,
        schnet_num_interactions=3,
        schnet_num_gaussians=50,
        vit_img_size=32,
        vit_patch_size=4,
        vit_embed_dim=128,
        vit_depth=4,
        vit_num_heads=4,
        fusion_hidden_dim=64,
        output_dim=12,  # QM9 has 12 targets
    )

    # Create fake batch
    # 2 molecules
    # Mol 1: 3 atoms
    # Mol 2: 2 atoms

    # Pos (N, 3)
    pos1 = torch.rand(3, 3)
    pos2 = torch.rand(2, 3)

    # X (one-hot) (N, 8) - 8 atom types: H, C, N, O, F, P, S, Cl
    x1 = torch.zeros(3, 8)
    x1[[0, 1, 2], [0, 1, 2]] = 1.0
    x2 = torch.zeros(2, 8)
    x2[[0, 1], [3, 4]] = 1.0

    y1 = torch.rand(12)
    y2 = torch.rand(12)

    d1 = Data(x=x1, pos=pos1, y=y1)
    d2 = Data(x=x2, pos=pos2, y=y2)

    batch_graph = Batch.from_data_list([d1, d2])

    # Persistence images (B, 32, 32)
    batch_graph.persistence_img = torch.rand(2, 32, 32)

    # Forward
    out = model(batch_graph)

    assert out.shape == (2, 12), f"Expected shape (2, 12), got {out.shape}"

    # Check if features were captured
    assert 'last_interaction' in model.schnet_features
    features = model.schnet_features['last_interaction']
    # Should be (Total atoms, hidden_channels) -> (5, 128)
    assert features.shape[0] == 5
    assert features.shape[1] == 128
