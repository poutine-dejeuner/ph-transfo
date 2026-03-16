"""All the configuration classes for the ph_transfo."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ph_transfo.configs.config import Config
from ph_transfo.utils.env_vars import get_constant

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


OmegaConf.register_new_resolver("constant", get_constant)
OmegaConf.register_new_resolver("eval", eval)


def _run_dir_from_ckpt(ckpt_path: str | None, default_dir: str) -> str:
    """If resuming from a checkpoint, reuse its run directory; otherwise use default."""
    if ckpt_path:
        from pathlib import Path

        ckpt = Path(ckpt_path)
        # Walk up to find the run directory (parent of the checkpoints/ folder)
        for parent in ckpt.parents:
            if parent.name == "checkpoints":
                return str(parent.parent)
        return str(ckpt.parent)
    return default_dir


OmegaConf.register_new_resolver("run_dir_from_ckpt", _run_dir_from_ckpt)


def add_configs_to_hydra_store():
    # from ph_transfo.utils.remote_launcher_plugin import RemoteSlurmQueueConf
    """Adds all configs to the Hydra Config store."""
    # ConfigStore.instance().store(
    #     group="hydra/launcher",
    #     name="remote_submitit_slurm",
    #     node=RemoteSlurmQueueConf,
    #     provider="Mila",
    # )
    pass


__all__ = [
    "Config",
]
