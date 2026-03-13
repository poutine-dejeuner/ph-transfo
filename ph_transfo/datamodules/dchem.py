"""DataModules for DeepChem molecular datasets (QM7, QM8, QM9, PDBbind)."""

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from deepchem import molnet
from lightning import LightningDataModule
from rdkit import RDLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import Batch, Data

from reg_transfo.utils.env_vars import DATA_DIR, NUM_WORKERS

# Suppress all deepchem warnings and logging BEFORE import
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Suppress RDKit C-level logging (deprecation warnings, sanitization errors, etc.)
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


class DeepChemDataModule(LightningDataModule):
    """Base DataModule for DeepChem molecular datasets with graph representation."""

    ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17]  # H, C, N, O, F, P, S, Cl
    ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']

    def __init__(
        self,
        dataset_name: Literal["qm7", "qm8", "qm9", "pdbbind"],
        data_dir: str | Path = DATA_DIR,
        batch_size: int = 32,
        num_workers: int = NUM_WORKERS,
        data_type: str = "graph",
        persistence_img_size: int = 32,
        splitter: str = "random",
        pin_memory: bool = True,
        shuffle: bool = True,
        data: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            dataset_name: Name of the dataset to load (qm7, qm8, qm9, or pdbbind)
            data_dir: Directory to store/load data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            data_type: graph or vector representation (default: graph)
            persistence_img_size: Size of the persistence image (height=width)
            splitter: Splitting method (random, scaffold, etc.)
            pin_memory: Pin memory for faster GPU transfer
            shuffle: Shuffle training data
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) / "deepchem" / dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type
        self.persistence_img_size = persistence_img_size
        self.splitter = splitter
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data = data or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tasks = None
        self.y_dim = self.data.get("y_dim")

        self.save_hyperparameters()

    def prepare_data(self):
        """Download dataset if needed."""
        if self.data_type == "graph":
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)

        loader_map = {
            "qm7": molnet.load_qm7,
            "qm8": molnet.load_qm8,
            "qm9": molnet.load_qm9,
            "pdbbind": molnet.load_pdbbind,
        }

        loader = loader_map[self.dataset_name]
        loader(
            featurizer="Raw",
            splitter=self.splitter,
            data_dir=str(self.data_dir),
        )

    def setup(self, stage: str | None = None):
        """Load and setup datasets."""
        self._setup_graph_datasets(stage)

    def _setup_graph_datasets(self, stage: str | None = None):
        """Load datasets as molecular graphs with one-hot atom features."""
        from deepchem import molnet

        loader_map = {
            "qm7": molnet.load_qm7,
            "qm8": molnet.load_qm8,
            "qm9": molnet.load_qm9,
            "pdbbind": molnet.load_pdbbind,
        }

        loader = loader_map[self.dataset_name]
        self.tasks, datasets, _ = loader(
            featurizer="Raw",
            splitter=self.splitter,
            data_dir=str(self.data_dir),
        )

        train_dc, valid_dc, test_dc = datasets

        if stage == "fit" or stage is None:
            self.train_dataset = GraphDataset(train_dc, self.ATOM_TYPES, self.persistence_img_size, cache_dir=self.data_dir / "train")
            self.val_dataset = GraphDataset(valid_dc, self.ATOM_TYPES, self.persistence_img_size, cache_dir=self.data_dir / "val")

        if stage == "test" or stage is None:
            self.test_dataset = GraphDataset(test_dc, self.ATOM_TYPES, self.persistence_img_size, cache_dir=self.data_dir / "test")

    def _to_torch_dataset(self, dc_dataset):
        """Convert DeepChem dataset to PyTorch TensorDataset."""
        X = torch.from_numpy(dc_dataset.X).float()
        y = torch.from_numpy(dc_dataset.y).float()

        return TensorDataset(X, y)

    def train_dataloader(self):
        """Return training dataloader."""
        collate_fn = Batch.from_data_list if self.data_type == "graph" else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        collate_fn = Batch.from_data_list if self.data_type == "graph" else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        collate_fn = Batch.from_data_list if self.data_type == "graph" else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


class GraphDataset(Dataset):
    """Convert DeepChem dataset to PyTorch Geometric graph dataset with pre-computed persistence
    images."""

    def __init__(self, dc_dataset, atom_types: list[int], persistence_img_size: int = 32, cache_dir: Path | None = None):
        self.atom_types = atom_types
        self.atom_types_tensor = torch.tensor(atom_types, dtype=torch.long)
        self.persistence_img_size = persistence_img_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.data_list: list[Data] = []

        # Check if we have cached complete graphs (MUCH faster - just 1 file load)
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"graph_data_{persistence_img_size}.pt"
            if cache_file.exists():
                logger.info("Loading cached graphs from %s", cache_file)
                self.data_list = torch.load(cache_file, weights_only=False)
                logger.info("Loaded %d graphs from cache", len(self.data_list))
                return

        # Pre-process: build graphs + persistence images from scratch
        logger.info("Pre-processing %d molecules...", len(dc_dataset))

        # KEY OPTIMIZATION: Load ALL molecules into memory ONCE
        # This avoids 100k+ slow accesses to dc_dataset.X[i]
        logger.info("Loading molecules into memory...")
        molecules = [dc_dataset.X[i] for i in range(len(dc_dataset))]
        targets = dc_dataset.y

        self._preprocess(molecules, targets)
        logger.info("Pre-processing complete.")

        # Save complete graphs to cache
        if self.cache_dir and cache_file:
            logger.info("Saving graphs to cache: %s", cache_file)
            torch.save(self.data_list, cache_file)

    def _preprocess(self, molecules: list, targets: np.ndarray):
        """Build graphs and compute persistence images."""
        from persim import PersistenceImager
        from ripser import Rips
        from tqdm import tqdm

        rips = Rips(maxdim=1, coeff=2, verbose=False)
        pimgr = PersistenceImager(pixel_size=1)
        target_shape = (self.persistence_img_size, self.persistence_img_size)

        # Pass 1: Build graphs + collect H1 diagrams
        diagrams_h1: list[np.ndarray] = []
        logger.info("Building graphs and computing H1 diagrams...")
        for i, mol_obj in enumerate(tqdm(molecules, desc="Graphs + H1", disable=False)):
            y = torch.from_numpy(targets[i]).float()
            positions = torch.from_numpy(mol_obj.GetConformer().GetPositions()).float()

            # One-hot atom features
            one_hots = []
            for atom in mol_obj.GetAtoms():
                z = atom.GetAtomicNum()
                idx_in_types = torch.searchsorted(self.atom_types_tensor, z)
                one_hot = torch.zeros(len(self.atom_types))
                if idx_in_types < len(self.atom_types) and self.atom_types[idx_in_types] == z:
                    one_hot[idx_in_types] = 1.0
                one_hots.append(one_hot)
            x = torch.stack(one_hots)

            # Edge index from bonds
            edge_pairs = []
            for bond in mol_obj.GetBonds():
                bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_pairs.append([bi, bj])
                edge_pairs.append([bj, bi])
            edge_index = (torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
                         if edge_pairs else torch.zeros((2, 0), dtype=torch.long))

            # Persistence diagram H1
            coords = positions.numpy()
            dgm = rips.fit_transform(coords)
            h1 = dgm[1] if len(dgm) > 1 else np.empty((0, 2))
            diagrams_h1.append(h1)

            self.data_list.append(Data(x=x, pos=positions, edge_index=edge_index, y=y))

        # Pass 2: Fit persistence imager + compute images
        non_empty = [d for d in diagrams_h1 if len(d) > 0]
        if non_empty:
            logger.info("Fitting persistence imager on %d diagrams...", len(non_empty))
            pimgr.fit(non_empty)

        logger.info("Computing persistence images...")
        for i, h1 in enumerate(tqdm(diagrams_h1, desc="PI", disable=False)):
            if len(h1) == 0:
                pi = torch.zeros(1, *target_shape)
            else:
                img = pimgr.transform([h1])[0]
                pi = torch.from_numpy(img).float().unsqueeze(0)
                if pi.shape[1:] != target_shape:
                    pi = pi.unsqueeze(0)
                    pi = torch.nn.functional.interpolate(pi, size=target_shape, mode='bilinear', align_corners=False)
                    pi = pi.squeeze(0)
            self.data_list[i].persistence_img = pi

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]



class QM7DataModule(DeepChemDataModule):
    """DataModule for QM7 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm7", **kwargs)


class QM8DataModule(DeepChemDataModule):
    """DataModule for QM8 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm8", **kwargs)


class QM9DataModule(DeepChemDataModule):
    """DataModule for QM9 dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="qm9", **kwargs)


class PDBbindDataModule(DeepChemDataModule):
    """DataModule for PDBbind dataset."""

    def __init__(self, **kwargs):
        super().__init__(dataset_name="pdbbind", **kwargs)
