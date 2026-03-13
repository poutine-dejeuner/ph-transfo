import logging
import warnings

# Suppress noisy third-party warnings before any imports trigger them
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Suppress deepchem/tensorflow loggers before they're imported (they log on import)
logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("torch_geometric").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
