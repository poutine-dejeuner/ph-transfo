import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from logging import getLogger

import hydra_zen
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer

from reg_transfo.algorithms.callbacks.classification_metrics import ClassificationMetricsCallback
from reg_transfo.datamodules.image_classification.image_classification import (
    ImageClassificationDataModule,
)
from reg_transfo.utils.typing_utils import HydraConfigFor

logger = getLogger(__name__)


class BaseAlgorithm(LightningModule, ABC):
    """Example learning algorithm for image classification."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: HydraConfigFor[torch.nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        init_seed: int = 42,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        self.save_hyperparameters(ignore=["datamodule"])

        self.network: torch.nn.Module | None = None

    def configure_model(self):
        # Save this for PyTorch-Lightning to infer the input/output shapes of the network.
        if self.network is not None:
            logger.info("Network is already instantiated.")
            return
        self.example_input_array = torch.zeros((self.datamodule.batch_size, *self.datamodule.dims))
        with torch.random.fork_rng():
            # deterministic weight initialization
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)
            if any(torch.nn.parameter.is_lazy(p) for p in self.network.parameters()):
                # Do a forward pass to initialize any lazy weights. This is necessary for
                # distributed training and to infer shapes.
                _ = self.network(self.example_input_array)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the network."""
        assert self.network is not None
        out = self.network(input)
        return out

    @abstractmethod
    def training_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        pass

    @abstractmethod
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        pass

    @abstractmethod
    def test_step(self, batch: tuple[Tensor, Tensor], batch_index: int):
        pass

    def configure_optimizers(self):
        """Creates the optimizers.

        See [`lightning.pytorch.core.LightningModule.configure_optimizers`][] for more information.
        """
        # Instantiate the optimizer config into a functools.partial object.
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        # Call the functools.partial object, passing the parameters as an argument.
        optimizer = optimizer_partial(self.parameters())
        # This then returns the optimizer.
        return optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Creates callbacks to be used by default during training."""
        return [
            ClassificationMetricsCallback.attach_to(self, num_classes=self.datamodule.num_classes)
        ]
