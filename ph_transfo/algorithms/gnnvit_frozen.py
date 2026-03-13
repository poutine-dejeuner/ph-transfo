"""GNN-ViT with frozen pretrained SchNet for feature extraction."""

import functools
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.optim.optimizer import Optimizer

from reg_transfo.algorithms.gnnvit import GNNViTAlgorithm
from reg_transfo.utils.typing_utils import HydraConfigFor


class GNNViTFrozenGNN(GNNViTAlgorithm):
    """GNN-ViT with frozen pretrained SchNet backbone.

    Loads a pretrained SchNet checkpoint and freezes all GNN parameters. Only trains the ViT
    encoder and fusion MLP.
    """

    def __init__(
        self,
        network: HydraConfigFor[torch.nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        datamodule: LightningDataModule,
        schnet_checkpoint: str | Path,
        init_seed: int = 42,
        loss_fn=None,
        metrics=None,
        lr_scheduler=None,
    ):
        """
        Args:
            network: Hydra config for the GNNViT network
            optimizer: Hydra config for optimizer
            datamodule: DataModule
            schnet_checkpoint: Path to pretrained SchNet checkpoint (.ckpt file)
            init_seed: Random seed for weight initialization
            loss_fn: Loss function
            metrics: Metrics for validation
            lr_scheduler: LR scheduler config
        """
        super().__init__(
            network=network,
            optimizer=optimizer,
            datamodule=datamodule,
            init_seed=init_seed,
            loss_fn=loss_fn,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
        )

        self.schnet_checkpoint = Path(schnet_checkpoint)
        self._load_and_freeze_schnet()

    def _load_and_freeze_schnet(self):
        """Load pretrained SchNet weights and freeze parameters."""
        if not self.schnet_checkpoint.exists():
            raise FileNotFoundError(f"SchNet checkpoint not found: {self.schnet_checkpoint}")

        # Load checkpoint
        checkpoint = torch.load(self.schnet_checkpoint, map_location='cpu', weights_only=False)

        # Extract SchNet state dict from the full algorithm checkpoint
        state_dict = checkpoint['state_dict']
        schnet_state = {
            k.replace('schnet.', ''): v
            for k, v in state_dict.items()
            if k.startswith('schnet.')
        }

        # Load into GNNViT's SchNet component
        missing, unexpected = self.network.schnet.load_state_dict(schnet_state, strict=False)

        if missing:
            print(f"Warning: Missing keys in SchNet: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in SchNet: {unexpected}")

        # Freeze all SchNet parameters
        for param in self.network.schnet.parameters():
            param.requires_grad = False

        print(f"✓ Loaded and froze SchNet from {self.schnet_checkpoint}")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(f"  Frozen params: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")
