from collections.abc import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, global_add_pool

from reg_transfo.algorithms.molecule_base import MoleculeRegressor

# Atom-type tables used by each data source.
# DeepChem (QM7/QM8/QM9): 8-type one-hot  →  atomic numbers
DCHEM_Z = torch.tensor([1, 6, 7, 8, 9, 15, 16, 17], dtype=torch.long)
# CREMP: 5-type one-hot  →  atomic numbers
CREMP_Z = torch.tensor([1, 6, 7, 8, 16], dtype=torch.long)


class MoleculeSchNet(MoleculeRegressor):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        out_channels: int = 1,
        datamodule=None,
        loss_fn: Callable | None = None,
        metrics: dict[str, Callable] | None = None,
        lr_scheduler=None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay, loss_fn=loss_fn, metrics=metrics, lr_scheduler=lr_scheduler)
        self.save_hyperparameters(ignore=["datamodule", "loss_fn", "metrics"])

        self.out_channels = out_channels
        self.schnet = SchNet(hidden_channels=hidden_channels, num_filters=num_filters,
                             num_interactions=num_interactions, num_gaussians=num_gaussians,
                             cutoff=10.0)

        # Hook the last interaction block to get per-atom embeddings,
        # pool them, then project to out_channels.
        self._node_features: dict = {}
        self.schnet.interactions[-1].register_forward_hook(
            lambda m, i, o: self._node_features.__setitem__("h", o)
        )
        self.output_head = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _unpack_batch(batch):
        """Return the PyG Batch graph regardless of collate format."""
        if isinstance(batch, tuple | list):
            return batch[0]
        return batch

    def training_step(self, batch, batch_idx):
        batch_graph = self._unpack_batch(batch)
        preds = self(batch_graph)
        targets = batch_graph.y.view(preds.shape)
        loss = self.loss_fn(preds, targets)
        self.log("train/loss", loss, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        batch_graph = self._unpack_batch(batch)
        preds = self(batch_graph)
        targets = batch_graph.y.view(preds.shape)
        loss = self.loss_fn(preds, targets)
        self.log("val/loss", loss, batch_size=batch_graph.y.size(0))
        for name, fn in self.metrics.items():
            self.log(f"val/{name}", fn(preds, targets), batch_size=batch_graph.y.size(0))
        return loss

    @staticmethod
    def _z_from_batch(batch_graph) -> torch.Tensor:
        """Derive atomic-number tensor from the batch.

        Supports three cases:
        1. ``batch_graph.z`` already contains atomic numbers
        2. ``batch_graph.x`` is a 5-col one-hot  (CREMP)
        3. ``batch_graph.x`` is an 8-col one-hot  (DeepChem QM7/8/9)
        """
        if getattr(batch_graph, "z", None) is not None:
            return batch_graph.z.long()

        if batch_graph.x is None:
            raise ValueError(
                "batch_graph has neither 'z' nor 'x'. "
                "The dataset must provide one-hot atom features (x) or atomic numbers (z)."
            )

        n_types = batch_graph.x.size(1)
        atom_indices = batch_graph.x.argmax(dim=1)
        z_table = CREMP_Z.to(batch_graph.x.device) if n_types <= 5 else DCHEM_Z.to(batch_graph.x.device)
        return z_table[atom_indices]

    def forward(self, batch_graph, batch_images=None):
        z_values = self._z_from_batch(batch_graph)
        self.schnet(z_values, batch_graph.pos, batch_graph.batch)
        h = self._node_features["h"]                  # [N, hidden_channels]
        pooled = global_add_pool(h, batch_graph.batch) # [B, hidden_channels]
        out = self.output_head(pooled)                 # [B, out_channels]
        if self.out_channels == 1:
            return out.squeeze(-1)                     # [B]
        return out                                     # [B, out_channels]
