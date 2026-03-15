import abc
import os
import sys
from typing import Generic, TypeVar
from unittest.mock import MagicMock, patch

import hydra_zen
import numpy as np
import omegaconf
import pytest
import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture
from torch.utils.data import DataLoader

from reg_transfo.algorithms.lightning_module_tests import convert_list_and_tuples_to_dicts
from reg_transfo.conftest import algorithm_config
from reg_transfo.datamodules.molecule_data import (
    MoleculeDataModule,
    MoleculePersistenceImageDataset,
)
from reg_transfo.utils.testutils import IN_GITHUB_CLOUD_CI

DataModuleType = TypeVar("DataModuleType", bound=LightningDataModule)

@pytest.fixture
def mock_dataset_class():
    with patch('reg_transfo.datamodules.molecule_data.MoleculePersistenceImageDataset') as mock:
        yield mock

@pytest.fixture
def mock_dataset_instance(mock_dataset_class):
    mock_instance = MagicMock()
    mock_dataset_class.return_value = mock_instance

    # Mock __len__
    mock_instance.__len__.return_value = 10

    # Mock __getitem__
    # Return a fake item: coords (N, 3), one_hot (N, 5), persistence_image (32, 32), energy (1)
    def get_item(idx):
        N = 5 # number of atoms
        coords = np.random.rand(N, 3).astype(np.float32)
        one_hot = np.eye(N, 5).astype(np.float32) # simple one-hot
        if N > 5: # pad if needed or just random
             one_hot = np.random.rand(N, 5).astype(np.float32)

        pimg = np.random.rand(32, 32).astype(np.float32)
        energy = 1.0

        return {
            'one_hot': one_hot,
            'persistence_image': pimg,
            'energy': energy,
            'coords': coords,
        }

    mock_instance.__getitem__.side_effect = get_item
    return mock_instance

def test_molecule_datamodule_setup(mock_dataset_class, mock_dataset_instance):
    dm = MoleculeDataModule(mol_dir="/fake/dir", batch_size=2)
    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None

    # Check sizes
    assert len(dm.train_dataset) == 8
    assert len(dm.val_dataset) == 2

def test_molecule_datamodule_dataloaders(mock_dataset_class, mock_dataset_instance):
    dm = MoleculeDataModule(mol_dir="/fake/dir", batch_size=2)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    # Test batch iteration
    batch = next(iter(train_loader))
    batch_graph, batch_images, batch_energies = batch

    from torch_geometric.data import Batch

    assert isinstance(batch_graph, Batch)
    assert batch_images.shape == (2, 32, 32)
    assert batch_energies.shape == (2,)

def test_molecule_persistence_image_dataset_real_files():
    # Test the Dataset class directly with real files
    mol_dir = '/media/vincent/disque_local/Downloads/pickle/'
    if not os.path.exists(mol_dir):
        pytest.skip(f"Real dataset directory {mol_dir} not found")

    # Use a limited number of files or just check existence
    # We can rely on the default pattern *.pickle

    dataset = MoleculePersistenceImageDataset(mol_dir=mol_dir)

    # Check length (we know there are files there)
    assert len(dataset) > 0

    # Check getitem
    item = dataset[0]

    assert 'one_hot' in item
    assert 'persistence_image' in item
    assert 'energy' in item
    assert 'coords' in item

    assert isinstance(item['one_hot'], np.ndarray)
    assert isinstance(item['persistence_image'], np.ndarray)
    assert isinstance(item['coords'], np.ndarray)
    assert isinstance(item['energy'], float)

    assert item['persistence_image'].shape == (32, 32)
    assert item['one_hot'].shape[1] == 5 # 5 atom types

def test_molecule_datamodule_real_files():
    # Integration test using a few real files from the dataset directory
    mol_dir = '/media/vincent/disque_local/Downloads/pickle/'
    if not os.path.exists(mol_dir):
        pytest.skip(f"Real dataset directory {mol_dir} not found")

    dm = MoleculeDataModule(mol_dir=mol_dir, batch_size=4)
    dm.setup()

    # Just checking if we can get a batch without errors
    loader = dm.train_dataloader()

    # We might need to iterate a bit to make sure collate works
    batch = next(iter(loader))

    batch_graph, batch_images, batch_energies = batch

    from torch_geometric.data import Batch

    assert isinstance(batch_graph, Batch)
    assert batch_images.shape[0] == 4
    assert batch_images.shape[1:] == (32, 32)
    assert batch_energies.shape == (4,)



# Use a dummy, empty algorithm, to keep the datamodule tests independent of the algorithms.
# This is a unit test for the datamodule, so we don't want to involve the algorithm here.


@pytest.mark.skipif(
    IN_GITHUB_CLOUD_CI and sys.platform == "darwin",
    reason="Getting weird bugs with MacOS on GitHub CI.",
)
@pytest.mark.parametrize(algorithm_config.__name__, ["no_op"], indirect=True, ids=[""])
class DataModuleTests(Generic[DataModuleType], abc.ABC):
    @pytest.fixture(
        scope="class",
        params=[
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
            RunningStage.TESTING,
            pytest.param(
                RunningStage.PREDICTING,
                marks=pytest.mark.xfail(
                    reason="Might not be implemented by the datamodule.",
                    raises=MisconfigurationException,
                ),
            ),
        ],
    )
    def stage(self, request: pytest.FixtureRequest):
        return getattr(request, "param", RunningStage.TRAINING)

    @pytest.fixture(scope="class")
    def datamodule(self, dict_config: omegaconf.DictConfig) -> DataModuleType:
        """Fixture that creates the datamodule instance, given the current Hydra config."""
        datamodule = hydra_zen.instantiate(dict_config["datamodule"])
        return datamodule

    @pytest.fixture(scope="class")
    def dataloader(self, datamodule: DataModuleType, stage: RunningStage) -> DataLoader:
        datamodule.prepare_data()
        if stage == RunningStage.TRAINING:
            datamodule.setup("fit")
            dataloader = datamodule.train_dataloader()
        elif stage in [RunningStage.VALIDATING, RunningStage.SANITY_CHECKING]:
            datamodule.setup("validate")
            dataloader = datamodule.val_dataloader()
        elif stage == RunningStage.TESTING:
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        else:
            assert stage == RunningStage.PREDICTING
            datamodule.setup("predict")
            dataloader = datamodule.predict_dataloader()
        return dataloader

    @pytest.fixture(scope="class")
    def batch(self, dataloader: DataLoader):
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch

    def test_first_batch(
        self,
        batch,
        tensor_regression: TensorRegressionFixture,
    ):
        batch = convert_list_and_tuples_to_dicts(batch)
        tensor_regression.check(batch, include_gpu_name_in_stats=False)
