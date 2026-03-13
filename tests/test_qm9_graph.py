"""Test QM9 graph datamodule with one-hot encoded atoms."""

import torch
from torch_geometric.data import Batch

from reg_transfo.datamodules.dchem import QM9DataModule


def test_qm9_graph_datamodule():
    """Test QM9DataModule with graph featurizer."""
    dm = QM9DataModule(featurizer='Raw', batch_size=4)
    dm.prepare_data()
    dm.setup('fit')

    # Check datasets exist
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    # Get a batch from train loader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Check batch is a PyG Batch
    assert isinstance(batch, Batch)
    assert batch.num_graphs == 4

    # Check node features are one-hot + positions (8 + 3 = 11)
    assert batch.x.shape[1] == len(dm.ATOM_TYPES) + 3  # 8 atom types + 3 positions

    # Check that first 8 dimensions are one-hot encoded
    one_hot_part = batch.x[:, :8]
    assert torch.all((one_hot_part == 0) | (one_hot_part == 1))  # Binary values
    assert torch.allclose(one_hot_part.sum(dim=1), torch.ones(one_hot_part.shape[0]))  # One-hot

    # Check positions exist
    assert batch.pos.shape[0] == batch.x.shape[0]
    assert batch.pos.shape[1] == 3

    # Check edges exist
    assert batch.edge_index.shape[0] == 2

    # Check targets exist
    assert batch.y is not None

    print("✓ All tests passed!")
    print(f"  - Batch has {batch.num_graphs} molecules")
    print(f"  - Total {batch.x.shape[0]} atoms")
    print(f"  - Node features: {batch.x.shape} (one-hot[8] + positions[3])")
    print(f"  - Positions: {batch.pos.shape}")
    print(f"  - Edges: {batch.edge_index.shape}")
    print(f"  - Targets: {batch.y.shape}")
    print("  - Example atom feature:")
    print(f"    - One-hot: {batch.x[0, :8].tolist()}")
    print(f"    - Position: {batch.x[0, 8:].tolist()}")


def test_qm9_vector_datamodule():
    """Test QM9DataModule with ECFP featurizer (original behavior)."""
    dm = QM9DataModule(featurizer='ECFP', batch_size=4)
    dm.prepare_data()
    dm.setup('fit')

    # Check datasets exist
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    # Get a batch from train loader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    # Check batch is tuple (X, y)
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    X, y = batch

    # Check shapes
    assert X.shape[0] == 4
    assert y.shape[0] == 4

if __name__ == "__main__":
    test_qm9_graph_datamodule()
    # test_qm9_vector_datamodule()
