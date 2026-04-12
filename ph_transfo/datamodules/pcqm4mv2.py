"""DataModule for OGB PCQM4Mv2 (3.7M molecules, HOMO-LUMO gap prediction).

Uses PyG's built-in PCQM4Mv2 dataset. Optionally generates 3D conformers
with RDKit and computes local Betti features as node augmentation.
"""

import logging
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ph_transfo.utils.env_vars import DATA_DIR, NUM_WORKERS

logger = logging.getLogger(__name__)


def _smiles_to_data_with_pos(smiles: str) -> "torch_geometric.data.Data":
    """Convert SMILES to PyG Data with 3D coordinates via RDKit ETKDG."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from torch_geometric.utils import from_rdmol

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("")

    mol = Chem.AddHs(mol)
    # Generate 3D conformer
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status == 0:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol = Chem.RemoveHs(mol)

    data = from_rdmol(mol)
    data.smiles = smiles

    # Extract 3D positions if conformer was generated
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
        data.pos = pos

    return data


class PCQM4Mv2DataModule(LightningDataModule):
    """LightningDataModule for PCQM4Mv2 (3.7M molecules, HOMO-LUMO gap).

    When ``betti_scales`` is set, uses a custom ``from_smiles`` that generates
    3D conformers so local Betti features can be computed.  This makes the
    initial processing significantly slower (~hours) but only runs once.
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        batch_size: int = 256,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        shuffle: bool = True,
        data: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / "pcqm4mv2"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data = data or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # PCQM4Mv2 node features: 9 atom descriptors
        self.x_dim = self.data.get("x_dim", 9)
        self.y_dim = self.data.get("y_dim", 1)
        self.betti_scales = self.data.get("betti_scales")

        if self.betti_scales and self.x_dim:
            self.x_dim = self.x_dim + 2 * len(self.betti_scales)

        self.save_hyperparameters()

    def prepare_data(self):
        """Download dataset (runs once on rank 0)."""
        from torch_geometric.datasets import PCQM4Mv2

        from_smiles_fn = _smiles_to_data_with_pos if self.betti_scales else None
        PCQM4Mv2(root=str(self.data_dir), split="train", from_smiles=from_smiles_fn)

    def setup(self, stage: str | None = None):
        """Load splits."""
        from torch_geometric.datasets import PCQM4Mv2

        from_smiles_fn = _smiles_to_data_with_pos if self.betti_scales else None

        if stage == "fit" or stage is None:
            train_ds = PCQM4Mv2(root=str(self.data_dir), split="train", from_smiles=from_smiles_fn)
            val_ds = PCQM4Mv2(root=str(self.data_dir), split="val", from_smiles=from_smiles_fn)

            if self.betti_scales:
                self.train_dataset = _BettiWrapperDataset(train_ds, self.betti_scales)
                self.val_dataset = _BettiWrapperDataset(val_ds, self.betti_scales)
            else:
                self.train_dataset = train_ds
                self.val_dataset = val_ds

        if stage == "test" or stage is None:
            test_ds = PCQM4Mv2(root=str(self.data_dir), split="test-dev", from_smiles=from_smiles_fn)
            if self.betti_scales:
                self.test_dataset = _BettiWrapperDataset(test_ds, self.betti_scales)
            else:
                self.test_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class _BettiWrapperDataset(torch.utils.data.Dataset):
    """Wraps a PyG dataset to lazily compute local Betti features on access."""

    def __init__(self, pyg_dataset, betti_scales: list[float]):
        self.dataset = pyg_dataset
        self.betti_scales = betti_scales

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if data.pos is not None and self.betti_scales:
            from ph_transfo.augmentation.local_betti import local_betti_features

            betti = local_betti_features(data.pos.numpy(), scales=self.betti_scales)
            betti_t = torch.from_numpy(betti).float()
            data.x = torch.cat([data.x.float(), betti_t], dim=1)

        return data
