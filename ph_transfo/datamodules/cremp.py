import glob
import os
import pickle
from typing import Any

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
from tqdm import tqdm

ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'S']
N_CONFORMERS = 31148756 # estimate
N_MOLECULES = 3258
MOL_DIR = '/media/vincent/disque_local/Downloads/pickle/'


def load_mol(mol_path: str) -> dict[str, Any]:
    """Load a molecule from a pickle file.

    Args:
        mol_path: Path to the pickle file containing the molecule.

    Returns:
        A dictionary containing:
        - 'rd_mol': RDKit molecule object with conformers
        - 'conformers': List of conformer data (e.g., energies)
    """
    with open(mol_path, 'rb') as f:
        mol_dict = pickle.load(f)
    return mol_dict

class MoleculeConformerDataset(Dataset):
    def __init__(self, mol_dir: str = MOL_DIR, mol_pattern: str = "*.pickle"):

        self.mol_dir = mol_dir
        pattern = os.path.join(mol_dir, mol_pattern)
        self.files = sorted(glob.glob(pattern))
        self.atom_types = ATOM_SYMBOLS

    def __len__(self) -> int:
        return N_MOLECULES

    def get_item_coords_onehots_energy(self, idx: int) -> tuple[np.ndarray,
        np.ndarray, np.ndarray]:

        mol_path = self.files[idx]
        mol_dict = load_mol(mol_path)
        rd_mol = mol_dict.get("rd_mol")

        n_conformers = rd_mol.GetNumConformers()
        conf_idx = np.random.randint(n_conformers)
        energy = mol_dict['conformers'][conf_idx]['totalenergy']
        conf = rd_mol.GetConformer(conf_idx)

        coords = []
        atom_types = []
        for atom in rd_mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append((pos.x, pos.y, pos.z))
            atom_types.append(atom.GetSymbol())
        coords = np.asarray(coords, dtype=np.float32)

        atom_index = {s: i for i, s in enumerate(ATOM_SYMBOLS)}
        one_hot_vectors = np.eye(len(ATOM_SYMBOLS), dtype=np.float32)
        one_hot_embed = np.vstack([one_hot_vectors[atom_index.get(s, 0)] for s in atom_types]).astype(np.float32)

        return coords, one_hot_embed, energy

    def __getitem__(self, idx: int) -> dict[str, Any]:

        coords, one_hot_embed, energy = self.get_item_coords_onehots_energy(idx)
        features = np.concatenate([one_hot_embed, coords], axis=1)

        return {
            "features": features,
            "energy": float(energy),
        }


class MoleculePersistenceImageDataset(MoleculeConformerDataset):
    """Subclass that returns one-hot embeddings, a persistence image of coordinates, and energy.

    Uses persim to compute persistence images from pairwise distances of atom coordinates.
    """

    def __init__(self, mol_dir: str = MOL_DIR, mol_pattern: str = "*.pickle", shape:
                 tuple[int, int] = (32, 32), spread: float = 1.0, weight: float = 1.0):

        super().__init__(mol_dir, mol_pattern)
        self.shape = shape
        self.spread = spread
        self.weight = weight

    def _compute_persistence_image(self, data: np.ndarray) -> np.ndarray:
        from persim import PersistenceImager
        from ripser import Rips

        rips = Rips(maxdim=1, coeff=2, verbose=False)
        # diagrams = [rips.fit_transform(datum) for datum in data]
        diagrams_h1 = [rips.fit_transform(datum)[1] for datum in data]

        # Check if diagram is empty
        if len(diagrams_h1[0]) == 0:
            return np.zeros((1, *self.shape), dtype=np.float32)

        pimgr = PersistenceImager(pixel_size=1)
        pimgr.fit(diagrams_h1)
        imgs = pimgr.transform(diagrams_h1)
        imgs_np = np.stack(imgs)

        # Resize to ensure fixed shape (32x32)
        # The persistence imager's resolution depends on data bounds when pixel_size is fixed.
        # We enforce the target shape by interpolation.
        if imgs_np.shape[1] != self.shape[0] or imgs_np.shape[2] != self.shape[1]:
            t = torch.tensor(imgs_np).unsqueeze(1) # (B, 1, H, W)
            t = torch.nn.functional.interpolate(t, size=self.shape, mode='bilinear', align_corners=False)
            imgs_np = t.squeeze(1).numpy()

        return imgs_np.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        coords, one_hot_embed, energy = self.get_item_coords_onehots_energy(idx)
        pimg = self._compute_persistence_image([coords])[0]
        return {
            'one_hot': one_hot_embed,
            'persistence_image': pimg,
            'energy': float(energy),
            'coords': coords,
        }


def _get_n_conformer(mol_path: str) -> int:
    mol_dict = load_mol(mol_path)
    n_conf = mol_dict.get("rd_mol").GetNumConformers()
    return n_conf

def init_num_conformers():
    from multiprocessing import Pool, cpu_count
    mol_dir = '/media/vincent/disque_local/Downloads/pickle/'
    mol_pattern = "*.pickle"
    pattern = os.path.join(mol_dir, mol_pattern)
    files = sorted(glob.glob(pattern))
    n_conformers = 0

    n_workers = max(1, cpu_count() - 1)
    n_processed_files = 0
    predicted_n = 0
    with Pool(n_workers) as pool:
        # for n in tqdm(pool.imap_unordered(_get_n_conformer, files), total=len(files)):
        for n in pool.imap_unordered(_get_n_conformer, files):
            prev_predicted_n = predicted_n
            n_processed_files += 1
            n_conformers += n
            predicted_n = (n_conformers / n_processed_files) * len(files)
            print(predicted_n)
            if np.abs(predicted_n - prev_predicted_n) < 0.1:
                break

    return n_conformers


def _symbols_from_file(path):
    mol_dict = load_mol(path)
    rd_mol = mol_dict.get("rd_mol")
    symbols = set()
    for atom in rd_mol.GetAtoms():
        symbols.add(atom.GetSymbol())
    return symbols


def init_atom_symbols_list():
    """Read all molecules in the dataset to get the list of unique atom symbols."""
    from multiprocessing import Pool, cpu_count

    mol_dir = '/media/vincent/disque_local/Downloads/pickle/'
    mol_pattern = "*.pickle"
    pattern = os.path.join(mol_dir, mol_pattern)
    files = glob.glob(pattern)
    atom_symbols = set()

    n_workers = max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        for symbols in tqdm(pool.imap_unordered(_symbols_from_file, files), total=len(files)):
            atom_symbols.update(symbols)

    return sorted(list(atom_symbols))

class CREMPDataModule(lightning.LightningDataModule):
    def __init__(self, mol_dir: str = MOL_DIR, batch_size: int = 32, num_workers: int = 4, data: dict | None = None):
        super().__init__()
        self.mol_dir = mol_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = data or {}
        self.dataset = None

    def setup(self, stage: str = None):
        if self.dataset is None:
            self.dataset = MoleculePersistenceImageDataset(self.mol_dir)
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        data_list = []
        images = []
        energies = []

        for item in batch:
            # Convert to torch tensors
            pos = torch.tensor(item['coords'], dtype=torch.float)
            x = torch.tensor(item['one_hot'], dtype=torch.float)
            y = torch.tensor([item['energy']], dtype=torch.float)

            # Create PyG Data object
            # Note: We don't have edge_index here, we might need to compute it on the fly (Radius Graph) or in the model.
            # SchNet for example computes distances internally from pos.
            data = Data(x=x, pos=pos, y=y)
            data_list.append(data)

            images.append(torch.tensor(item['persistence_image'], dtype=torch.float))
            energies.append(y)

        # Batch the graphs
        batch_graph = Batch.from_data_list(data_list)

        # Batch the images
        batch_images = torch.stack(images)

        # Batch the energies (redundant as it is in batch_graph.y, but good for check)
        batch_energies = torch.cat(energies)

        return batch_graph, batch_images, batch_energies
