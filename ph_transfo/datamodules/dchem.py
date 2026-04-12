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

from ph_transfo.utils.env_vars import DATA_DIR, NUM_WORKERS
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
        self.splitter = splitter
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data = data or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tasks = None
        self.x_dim = self.data.get("x_dim")
        self.y_dim = self.data.get("y_dim")
        self.betti_scales = self.data.get("betti_scales")

        # Topology feature config
        self.topo = self.data.get("topo", {})
        self.topo_enabled = bool(self.topo)
        self.n_curve_points = self.topo.get("n_curve_points", 20)
        self.curve_range = tuple(self.topo.get("curve_range", [1.0, 6.0]))

        # Compute topo feature dim: 2*n_curve_points (Betti curves) + 10 (persistence stats)
        if self.topo_enabled and self.x_dim:
            self.topo_dim = 2 * self.n_curve_points + 10
            # topo features are NOT concatenated to x; they go in data.topo_x
            # x_dim stays as base atom features
        elif self.betti_scales and self.x_dim:
            # Legacy: old betti_scales concat mode
            self.x_dim = self.x_dim + 2 * len(self.betti_scales)
        self.topo_dim = 2 * self.n_curve_points + 10 if self.topo_enabled else 0

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

        topo_kwargs = {}
        if self.topo_enabled:
            topo_kwargs = {"topo_config": {"n_curve_points": self.n_curve_points, "curve_range": self.curve_range}}

        if stage == "fit" or stage is None:
            self.train_dataset = GraphDataset(train_dc, self.ATOM_TYPES, cache_dir=self.data_dir / "train", betti_scales=self.betti_scales, **topo_kwargs)
            self.val_dataset = GraphDataset(valid_dc, self.ATOM_TYPES, cache_dir=self.data_dir / "val", betti_scales=self.betti_scales, **topo_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = GraphDataset(test_dc, self.ATOM_TYPES, cache_dir=self.data_dir / "test", betti_scales=self.betti_scales, **topo_kwargs)

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
    """Convert DeepChem dataset to PyTorch Geometric graph dataset with optional
    topology features (rich PH node features + edge filtration values)."""

    def __init__(self, dc_dataset, atom_types: list[int],
                 cache_dir: Path | None = None, betti_scales: list[float] | None = None,
                 topo_config: dict | None = None):
        self.atom_types = atom_types
        self.atom_types_tensor = torch.tensor(atom_types, dtype=torch.long)
        self.betti_scales = betti_scales
        self.topo_config = topo_config
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.data_list: list[Data] = []

        # Cache key
        cache_suffix = ""
        if topo_config:
            nc = topo_config.get("n_curve_points", 20)
            cr = topo_config.get("curve_range", (1.0, 6.0))
            cache_suffix += f"_topo_{nc}_{cr[0]:.1f}_{cr[1]:.1f}"
        elif betti_scales:
            scale_str = "_".join(f"{s:.1f}" for s in betti_scales)
            cache_suffix += f"_betti_{scale_str}"

        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"graph_data{cache_suffix}.pt"
            if cache_file.exists():
                logger.info("Loading cached graphs from %s", cache_file)
                self.data_list = torch.load(cache_file, weights_only=False)
                logger.info("Loaded %d graphs from cache", len(self.data_list))
                return

        logger.info("Pre-processing %d molecules...", len(dc_dataset))
        logger.info("Loading molecules into memory...")
        molecules = [dc_dataset.X[i] for i in range(len(dc_dataset))]
        targets = dc_dataset.y

        self._preprocess(molecules, targets)
        logger.info("Pre-processing complete.")

        if self.cache_dir and cache_file:
            logger.info("Saving graphs to cache: %s", cache_file)
            torch.save(self.data_list, cache_file)

    def _preprocess(self, molecules: list, targets: np.ndarray):
        """Build graphs with optional topology features."""
        from torch_topological.nn import VietorisRipsComplex
        from tqdm import tqdm

        vr = VietorisRipsComplex(dim=1)

        use_topo = self.topo_config is not None
        use_legacy_betti = not use_topo and self.betti_scales is not None and len(self.betti_scales) > 0

        if use_topo:
            from ph_transfo.augmentation.local_betti import edge_filtration_values, local_ph_features
        elif use_legacy_betti:
            from ph_transfo.augmentation.local_betti import local_betti_features

        logger.info("Building graphs and computing persistence diagrams...")
        for i, mol_obj in enumerate(tqdm(molecules, desc="Graphs + PH", disable=False)):
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

            data = Data(x=x, pos=positions, edge_index=edge_index, y=y)

            if use_topo:
                # Rich local PH features → stored separately for PE-style injection
                topo_x = local_ph_features(
                    positions.numpy(),
                    n_curve_points=self.topo_config.get("n_curve_points", 20),
                    curve_range=tuple(self.topo_config.get("curve_range", (1.0, 6.0))),
                )
                data.topo_x = torch.from_numpy(topo_x)

                # Edge filtration values
                if edge_index.numel() > 0:
                    edge_filt = edge_filtration_values(positions.numpy(), edge_index.numpy())
                    data.edge_filt = torch.from_numpy(edge_filt)
                else:
                    data.edge_filt = torch.zeros(0, 1)

            elif use_legacy_betti:
                betti = local_betti_features(positions.numpy(), scales=self.betti_scales)
                betti_t = torch.from_numpy(betti).float()
                data.x = torch.cat([data.x, betti_t], dim=1)

            # Persistence diagrams via VietorisRipsComplex (H0 + H1)
            with torch.no_grad():
                ph_info = vr(positions)
            data.dgm_h0 = ph_info[0].diagram if len(ph_info) > 0 else torch.empty(0, 2)
            data.dgm_h1 = ph_info[1].diagram if len(ph_info) > 1 else torch.empty(0, 2)

            self.data_list.append(data)

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
