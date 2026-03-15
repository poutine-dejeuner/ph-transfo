"""Tests for DeepChem datamodules."""

import pytest

from reg_transfo.datamodules.dchem import (
    PDBbindDataModule,
    QM7DataModule,
    QM8DataModule,
    QM9DataModule,
)


@pytest.mark.parametrize(
    "dm_class,expected_name",
    [
        (QM7DataModule, "qm7"),
        (QM8DataModule, "qm8"),
        (QM9DataModule, "qm9"),
        (PDBbindDataModule, "pdbbind"),
    ],
)
def test_datamodule_creation(dm_class, expected_name):
    """Test that datamodules can be instantiated."""
    dm = dm_class(batch_size=16)
    assert dm.dataset_name == expected_name
    assert dm.batch_size == 16


@pytest.mark.slow
def test_qm7_datamodule():
    """Test QM7 datamodule loading."""
    dm = QM7DataModule(batch_size=8)
    dm.prepare_data()
    dm.setup("fit")

    # Check datasets exist
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    # Check dataloaders
    train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()

    # Get a batch
    batch = next(iter(train_loader))
    assert len(batch) in [2, 3]  # X, y, (optionally w)
    X, y = batch[0], batch[1]
    assert X.shape[0] == 8  # batch size
    assert y.shape[0] == 8


@pytest.mark.slow
def test_qm9_datamodule():
    """Test QM9 datamodule loading."""
    dm = QM9DataModule(batch_size=8)
    dm.prepare_data()
    dm.setup("fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.tasks is not None

    # QM9 has multiple tasks
    assert len(dm.tasks) > 1

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    X, y = batch[0], batch[1]
    assert X.shape[0] == 8
    assert y.shape[0] == 8
