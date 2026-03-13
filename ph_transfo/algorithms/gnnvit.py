"""GNN-ViT Algorithm for molecular property prediction."""

import functools

import hydra_zen
import torch
from lightning import LightningDataModule
from torch.optim.optimizer import Optimizer

from reg_transfo.algorithms.molecule_base import MoleculeRegressor
from reg_transfo.utils.typing_utils import HydraConfigFor


class GNNViTAlgorithm(MoleculeRegressor):
    """GNN-ViT algorithm combining graph and image modalities."""

    def __init__(
        self,
        network: HydraConfigFor[torch.nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        datamodule: LightningDataModule,
        init_seed: int = 42,
        loss_fn=None,
        metrics=None,
        lr_scheduler=None,
    ):
        """
        Args:
            network: Hydra config for the network (if None, builds GNNViTNetwork)
            optimizer: Hydra config for optimizer
            datamodule: DataModule (ignored in save_hyperparameters)
            init_seed: Random seed for weight initialization
            loss_fn: Loss function for training (optional)
            metrics: Metrics for validation (optional)
            lr_scheduler: Hydra config for lr scheduler (optional)
        """
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        self.save_hyperparameters(ignore=["datamodule", "network", "optimizer", "loss_fn", "metrics"])

        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.init_seed = init_seed

        # Build network
        self.network = hydra_zen.instantiate(network)

    def forward(self, batch):
        return self.network(batch)

    def configure_optimizers(self):
        """Configure optimizer using Hydra config or default Adam."""
        if self.optimizer_config is not None:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
            optimizer = optimizer_partial(self.parameters())
        else:
            optimizer = torch.optim.Adam(self.parameters())

        if self.lr_scheduler_config is None:
            return optimizer
        # Hydra may have already resolved the config to a functools.partial
        if callable(self.lr_scheduler_config):
            scheduler = self.lr_scheduler_config(optimizer)
        else:
            scheduler = hydra_zen.instantiate(self.lr_scheduler_config)(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
