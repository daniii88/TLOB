import os
import random
import warnings

import zipfile
warnings.filterwarnings("ignore")
import numpy as np
import torch
import constants as cst
import hydra
from config.config import Config
from run import run_wandb, run, sweep_init
from preprocessing.lobster import LOBSTERDataBuilder
from preprocessing.btc import BTCDataBuilder
from constants import DatasetType


def _btc_dataset_dir(config: Config):
    dataset_dir = getattr(config.dataset, "data_path", "")
    if dataset_dir:
        return dataset_dir
    return os.path.join(cst.DATA_DIR, "BTC")


def _btc_dataset_paths(config: Config):
    btc_dir = _btc_dataset_dir(config)
    return [os.path.join(btc_dir, f"{split}.npy") for split in ("train", "val", "test")]


def _prepare_btc_dataset(config: Config, reason: str):
    btc_dataset_dir = _btc_dataset_dir(config)
    default_btc_dir = os.path.join(cst.DATA_DIR, "BTC")
    if os.path.normpath(btc_dataset_dir) != os.path.normpath(default_btc_dir):
        raise RuntimeError(
            "Automatic BTC preprocessing currently supports only dataset.data_path="
            f"`{default_btc_dir}`. For custom dataset.data_path, place train.npy/val.npy/test.npy "
            "manually in that folder."
        )
    print(f"Preparing BTC dataset ({reason})")
    data_builder = BTCDataBuilder(
        data_dir=cst.DATA_DIR,
        date_trading_days=config.dataset.dates,
        split_rates=cst.SPLIT_RATES,
        sampling_type=config.dataset.sampling_type,
        sampling_time=config.dataset.sampling_time,
        sampling_quantity=config.dataset.sampling_quantity,
    )
    data_builder.prepare_save_datasets()


@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    print("Using device: ", cst.DEVICE)
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    if config.dataset.type == DatasetType.FI_2010:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    elif config.dataset.type == DatasetType.BTC:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    elif config.dataset.type == DatasetType.ENGINE:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 64
    elif config.dataset.type == DatasetType.LOBSTER:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46
    
    if config.dataset.type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        # prepare the datasets, this will save train.npy, val.npy and test.npy in the data directory
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=config.dataset.data_path,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    elif config.dataset.type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        try:
            # take the .zip files name in FI-2010 directory
            dir = config.dataset.data_path + "/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(dir)  # Extracts to the current directory           
            print("Data extracted.")
        except Exception as e:
            raise RuntimeError(f"Error downloading or extracting data: {e}") from e
        
    elif config.dataset.type == cst.DatasetType.BTC:
        missing_btc_files = [path for path in _btc_dataset_paths(config) if not os.path.exists(path)]
        should_prepare_btc = (not config.experiment.is_data_preprocessed) or bool(missing_btc_files)
        if should_prepare_btc:
            reason = (
                "missing preprocessed files: " + ", ".join(missing_btc_files)
                if missing_btc_files
                else "experiment.is_data_preprocessed=false"
            )
            _prepare_btc_dataset(config, reason)

    if config.experiment.is_wandb:
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb is required when experiment.is_wandb=true. "
                "Install it with `pip install wandb` or set experiment.is_wandb=false."
            ) from exc
        if config.experiment.is_sweep:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    else:
        run(config, accelerator)
    

def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')
    
if __name__ == "__main__":
    set_torch()
    hydra_app()
    
