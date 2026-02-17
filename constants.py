import torch
import os
from enum import Enum

class DatasetType(Enum):
    LOBSTER = "LOBSTER"
    FI_2010 = "FI_2010"
    BTC = "BTC"
    ENGINE = "ENGINE"
    

class ModelType(Enum):
    MLPLOB = "MLPLOB"
    TLOB = "TLOB"
    BINCTABL = "BINCTABL"
    DEEPLOB = "DEEPLOB"
    
class SamplingType(Enum):
    TIME = "time"
    QUANTITY = "quantity"
    NONE = "none"



# for 15 days of TSLA
TSLA_LOB_MEAN_SIZE_10 = 165.44670902537212
TSLA_LOB_STD_SIZE_10 = 481.7127061897184
TSLA_LOB_MEAN_PRICE_10 = 20180.439318660694
TSLA_LOB_STD_PRICE_10 = 814.8782058033195

TSLA_EVENT_MEAN_SIZE = 88.09459295373463
TSLA_EVENT_STD_SIZE = 86.55913199110894
TSLA_EVENT_MEAN_PRICE = 20178.610720500274
TSLA_EVENT_STD_PRICE = 813.8188032145645
TSLA_EVENT_MEAN_TIME = 0.08644932804905886
TSLA_EVENT_STD_TIME = 0.3512181506722207
TSLA_EVENT_MEAN_DEPTH = 7.365325300819055
TSLA_EVENT_STD_DEPTH = 8.59342838063813

# for 15 days of INTC
INTC_LOB_MEAN_SIZE_10 = 6222.424274871972
INTC_LOB_STD_SIZE_10 = 7538.341086370264
INTC_LOB_MEAN_PRICE_10 = 3635.766219937785
INTC_LOB_STD_PRICE_10 = 44.15649995373795

INTC_EVENT_MEAN_SIZE = 324.6800802006092
INTC_EVENT_STD_SIZE = 574.5781447696605
INTC_EVENT_MEAN_PRICE = 3635.78165265669
INTC_EVENT_STD_PRICE = 43.872407609651184
INTC_EVENT_MEAN_TIME = 0.025201754040915927
INTC_EVENT_STD_TIME = 0.11013627432323592
INTC_EVENT_MEAN_DEPTH = 1.3685517399834501
INTC_EVENT_STD_DEPTH = 2.333747222206966




LOBSTER_HORIZONS = [10, 20, 50, 100]
PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_ORDER = 6
LEN_SMOOTH = 10

def _strtobool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device() -> str:
    force_cpu = _strtobool(os.getenv("TLOB_FORCE_CPU", "0"))
    if force_cpu:
        print("TLOB_FORCE_CPU=1 -> forcing CPU execution")
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    # Proactively verify that this CUDA build supports the current GPU arch.
    try:
        capability = torch.cuda.get_device_capability(0)
        sm = f"sm_{capability[0]}{capability[1]}"
        supported_archs = set(torch.cuda.get_arch_list())
        if supported_archs and sm not in supported_archs:
            print(
                "CUDA available but GPU arch is unsupported by this torch build "
                f"(gpu={sm}, torch_archs={sorted(supported_archs)}). Falling back to CPU."
            )
            return "cpu"
        probe = torch.zeros(1, device="cuda")
        _ = (probe + 1).item()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "no kernel image is available" in msg or "no kernel image available" in msg:
            print("CUDA kernel image unsupported for this GPU. Falling back to CPU.")
            return "cpu"
        raise

    return "cuda"


DEVICE = _resolve_device()
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = "EvolutionData"
SPLIT_RATES = [0.8, 0.1, 0.1]
WANDB_API = ""
WANDB_USERNAME = ""
