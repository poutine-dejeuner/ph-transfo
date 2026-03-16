from collections.abc import Callable

import hydra_zen
import lightning.pytorch as pl
import torch
import torch.nn as nn


class MoleculeRegressor(pl.LightningModule):
    def __init__(
        self,
        network=None,
        datamodule=None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_fn: Callable | None = None,
        metrics: dict[str, Callable] | None = None,
        lr_scheduler=None,
        optimizer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule", "loss_fn", "metrics", "network"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or {
            "mae": nn.L1Loss(),
        }
        self.lr_scheduler_config = lr_scheduler
        self.network_config = network
        self.network: nn.Module | None = None

        if self.network_config is not None:
            self.network = hydra_zen.instantiate(self.network_config)

    def forward(self, batch):
        assert self.network is not None, "No network configured"
        return self.network(batch)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.view(preds.shape)
        loss = self.loss_fn(preds, targets)
        self.log('train/loss', loss, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.view(preds.shape)
        loss = self.loss_fn(preds, targets)
        self.log('val/loss', loss, batch_size=batch.y.size(0))

        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(preds, targets)
            self.log(f'val/{metric_name}', metric_value, batch_size=batch.y.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        targets = batch.y.view(preds.shape)
        loss = self.loss_fn(preds, targets)
        self.log('test/loss', loss, batch_size=batch.y.size(0))

        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(preds, targets)
            self.log(f'test/{metric_name}', metric_value, batch_size=batch.y.size(0))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler_config is None:
            return optimizer
        if callable(self.lr_scheduler_config):
            scheduler = self.lr_scheduler_config(optimizer)
        else:
            scheduler = hydra_zen.instantiate(self.lr_scheduler_config)(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
