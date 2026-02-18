import lightning as L
import omegaconf
import torch
import os
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from config.config import Config
from models.engine import Engine
from preprocessing.fi_2010 import fi_2010_load
from preprocessing.lobster import lobster_load
from preprocessing.btc import btc_load
from preprocessing.engine import engine_load
from preprocessing.dataset import Dataset, DataModule, safe_collate
from utils.engine_recall import compute_class_weights, compute_sample_weights
import constants as cst
from constants import DatasetType, SamplingType
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])


def class_distribution(labels: torch.Tensor):
    total = labels.shape[0]
    if total == 0:
        return {0: 0.0, 1: 0.0, 2: 0.0}
    classes, counts = torch.unique(labels, return_counts=True)
    counts_by_class = {int(cls.item()): int(cnt.item()) for cls, cnt in zip(classes, counts)}
    return {cls: counts_by_class.get(cls, 0) / total for cls in (0, 1, 2)}


def default_ckpt_dir(config: Config) -> str:
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    dataset = config.dataset.type.value
    horizon = config.experiment.horizon
    if dataset == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        return f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"
    return f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"


def resolve_ckpt_dir(config: Config) -> None:
    if config.experiment.dir_ckpt in {"", "model.ckpt"}:
        config.experiment.dir_ckpt = default_ckpt_dir(config)


def save_metrics_summary(config: Config, model_type: str, model: Engine, test_output: dict) -> str:
    summary_dir = os.path.join(cst.DIR_SAVED_MODEL, model_type, config.experiment.dir_ckpt)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "metrics_summary.json")
    summary = {
        "dataset": config.dataset.type.value,
        "data_path": config.dataset.data_path,
        "model_type": model_type,
        "dir_ckpt": config.experiment.dir_ckpt,
        "seed": int(config.experiment.seed),
        "horizon": int(config.experiment.horizon),
        "loss_name": str(config.experiment.loss_name),
        "use_weighted_sampler": bool(config.dataset.use_weighted_sampler),
        "weighted_sampler_pow": float(config.dataset.weighted_sampler_pow),
        "min_event_precision": float(config.experiment.min_event_precision),
        "val_metrics": getattr(model, "best_val_metrics", {}),
        "test_metrics": test_output,
    }
    flat = {}
    for key, value in summary["val_metrics"].items():
        flat[key] = value
    for key, value in summary["test_metrics"].items():
        flat[key] = value
    summary["flat_metrics"] = flat
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def run(config: Config, accelerator):
    resolve_ckpt_dir(config)

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.experiment.max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=True, min_delta=0.002),
            TQDMProgressBar(refresh_rate=100)
            ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler=None,
        check_val_every_n_epoch=1
    )
    train(config, trainer)


def train(config: Config, trainer: L.Trainer, run=None):
    print_setup(config)
    dataset_type = config.dataset.type.value
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    horizon = config.experiment.horizon
    model_type = config.model.type
    checkpoint_ref = config.experiment.checkpoint_reference
    checkpoint_path = os.path.join(cst.DIR_SAVED_MODEL, model_type.value, checkpoint_ref)
    dataset_type = config.dataset.type.value
    class_weights = None
    if dataset_type == "FI_2010":
        path = config.dataset.data_path
        train_input, train_labels, val_input, val_labels, test_input, test_labels = fi_2010_load(path, seq_size, horizon, config.model.hyperparameters_fixed["all_features"])
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = min(train_set.length, 1000)
            val_set.length = min(val_set.length, 1000)
            test_set.length = min(test_set.length, 10000)
        data_module = DataModule(
            train_set=Dataset(train_input, train_labels, seq_size),
            val_set=Dataset(val_input, val_labels, seq_size),
            test_set=Dataset(test_input, test_labels, seq_size),
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
        test_loaders = [data_module.test_dataloader()]
    
    elif dataset_type == "BTC":
        train_input, train_labels = btc_load(os.path.join(config.dataset.data_path, "train.npy"), cst.LEN_SMOOTH, horizon, seq_size)
        val_input, val_labels = btc_load(os.path.join(config.dataset.data_path, "val.npy"), cst.LEN_SMOOTH, horizon, seq_size)
        test_input, test_labels = btc_load(os.path.join(config.dataset.data_path, "test.npy"), cst.LEN_SMOOTH, horizon, seq_size)
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = min(train_set.length, 1000)
            val_set.length = min(val_set.length, 1000)
            test_set.length = min(test_set.length, 10000)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        ) 

        test_loaders = [data_module.test_dataloader()]
    elif dataset_type == "ENGINE":
        train_input, train_labels = engine_load(
            os.path.join(config.dataset.data_path, "train.npy"), cst.LEN_SMOOTH, horizon, seq_size
        )
        val_input, val_labels = engine_load(
            os.path.join(config.dataset.data_path, "val.npy"), cst.LEN_SMOOTH, horizon, seq_size
        )
        test_input, test_labels = engine_load(
            os.path.join(config.dataset.data_path, "test.npy"), cst.LEN_SMOOTH, horizon, seq_size
        )
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        test_set = Dataset(test_input, test_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = min(train_set.length, 1000)
            val_set.length = min(val_set.length, 1000)
            test_set.length = min(test_set.length, 10000)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size * 4,
            num_workers=4,
        )
        test_loaders = [data_module.test_dataloader()]
        
    elif dataset_type == "LOBSTER":
        lobster_data_root = config.dataset.data_path
        training_stocks = config.dataset.training_stocks
        testing_stocks = config.dataset.testing_stocks
        for i in range(len(training_stocks)):
            if i == 0:
                for j in range(2):
                    if j == 0:
                        path = os.path.join(lobster_data_root, training_stocks[i], "train.npy")
                        train_input, train_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                    if j == 1:
                        path = os.path.join(lobster_data_root, training_stocks[i], "val.npy")
                        val_input, val_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
            else:
                for j in range(2):
                    if j == 0:
                        path = os.path.join(lobster_data_root, training_stocks[i], "train.npy")
                        train_labels = torch.cat((train_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        train_input_tmp, train_labels_tmp = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                        train_input = torch.cat((train_input, train_input_tmp), 0)
                        train_labels = torch.cat((train_labels, train_labels_tmp), 0)
                    if j == 1:
                        path = os.path.join(lobster_data_root, training_stocks[i], "val.npy")
                        val_labels = torch.cat((val_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        val_input_tmp, val_labels_tmp = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                        val_input = torch.cat((val_input, val_input_tmp), 0)
                        val_labels = torch.cat((val_labels, val_labels_tmp), 0)
        test_loaders = []
        for i in range(len(testing_stocks)):
            path = os.path.join(lobster_data_root, testing_stocks[i], "test.npy")
            test_input, test_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
            test_set = Dataset(test_input, test_labels, seq_size)
            test_dataloader = DataLoader(
                dataset=test_set,
                batch_size=config.dataset.batch_size*4,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=4,
                persistent_workers=True,
                collate_fn=safe_collate,
            )
            test_loaders.append(test_dataloader)
        
        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = min(train_set.length, 1000)
            val_set.length = min(val_set.length, 1000)
            test_set.length = min(test_set.length, 10000)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    counts_train = class_distribution(train_labels)
    counts_val = class_distribution(val_labels)
    counts_test = class_distribution(test_labels)
    print()
    print("Train set shape: ", train_input.shape)
    print("Val set shape: ", val_input.shape)
    print("Test set shape: ", test_input.shape)
    print(f"Classes distribution in train set: up {counts_train[0]:.2f} stat {counts_train[1]:.2f} down {counts_train[2]:.2f}")
    print(f"Classes distribution in val set: up {counts_val[0]:.2f} stat {counts_val[1]:.2f} down {counts_val[2]:.2f}")
    print(f"Classes distribution in test set: up {counts_test[0]:.2f} stat {counts_test[1]:.2f} down {counts_test[2]:.2f}")
    print()

    loss_name = str(config.experiment.loss_name).lower()
    if loss_name not in {"ce", "weighted_ce"}:
        raise ValueError(
            f"unsupported experiment.loss_name `{config.experiment.loss_name}`; expected `ce` or `weighted_ce`"
        )

    effective_train_labels = train_set.y[: train_set.length]

    if loss_name == "weighted_ce":
        class_weights = compute_class_weights(effective_train_labels, num_classes=3)
        print(f"Using weighted CE with class weights: {class_weights.tolist()}")

    if config.dataset.use_weighted_sampler:
        sample_weights = compute_sample_weights(
            effective_train_labels,
            num_classes=3,
            sampler_pow=config.dataset.weighted_sampler_pow,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=sample_weights.numel(),
            replacement=True,
        )
        data_module.train_sampler = train_sampler
        data_module.is_shuffle_train = False
        print(
            "Using WeightedRandomSampler with "
            f"pow={config.dataset.weighted_sampler_pow}"
        )
    
    experiment_type = config.experiment.type
    if "FINETUNING" in experiment_type or "EVALUATION" in experiment_type:
        if checkpoint_ref == "":
            raise ValueError(
                "experiment.type includes FINETUNING/EVALUATION but "
                "experiment.checkpoint_reference is empty."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"checkpoint not found at `{checkpoint_path}`. "
                "Set experiment.checkpoint_reference to a valid checkpoint path."
            )
        checkpoint = torch.load(checkpoint_path, map_location=cst.DEVICE, weights_only=True)
            
        print("Loading model from checkpoint: ", config.experiment.checkpoint_reference) 
        lr = checkpoint["hyper_parameters"]["lr"]
        dir_ckpt = checkpoint["hyper_parameters"]["dir_ckpt"]
        hidden_dim = checkpoint["hyper_parameters"]["hidden_dim"]
        num_layers = checkpoint["hyper_parameters"]["num_layers"]
        optimizer = checkpoint["hyper_parameters"]["optimizer"]
        model_type = checkpoint["hyper_parameters"]["model_type"]
        max_epochs = checkpoint["hyper_parameters"]["max_epochs"]
        horizon = checkpoint["hyper_parameters"]["horizon"]
        seq_size = checkpoint["hyper_parameters"]["seq_size"]
        if model_type == "MLPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path, 
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
                map_location=cst.DEVICE,
                )
        elif model_type == "TLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=checkpoint["hyper_parameters"]["num_heads"],
                is_sin_emb=checkpoint["hyper_parameters"]["is_sin_emb"],
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
                )
        elif model_type == "BINCTABL":
            model = Engine.load_from_checkpoint(
                checkpoint_path, 
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
                )
        elif model_type == "DEEPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path, 
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
                map_location=cst.DEVICE,
                len_test_dataloader=len(test_loaders[0])
                )
              
    else:
        if model_type == cst.ModelType.MLPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
            )
        elif model_type == cst.ModelType.TLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                len_test_dataloader=len(test_loaders[0]),
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
            )
        elif model_type == cst.ModelType.BINCTABL:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
            )
        elif model_type == cst.ModelType.DEEPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                loss_name=config.experiment.loss_name,
                class_weights=class_weights,
                min_event_precision=config.experiment.min_event_precision,
            )
    
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))   
    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    
    if "TRAINING" in experiment_type or "FINETUNING" in experiment_type:
        trainer.fit(model, train_dataloader, val_dataloader)
        best_model_path = model.last_path_ckpt
        print("Best model path: ", best_model_path) 
        try:
            best_model = Engine.load_from_checkpoint(best_model_path, map_location=cst.DEVICE)
        except: 
            print("no checkpoints has been saved, selecting the last model")
            best_model = model
        best_model.experiment_type = "EVALUATION"
        primary_test_output = None
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(best_model, test_dataloader)
            if i == 0 and output:
                primary_test_output = output[0]
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": output[0]["f1_score"]}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010 ": output[0]["f1_score"]}, commit=False)
        if primary_test_output is not None:
            summary_path = save_metrics_summary(config, config.model.type.value, model, primary_test_output)
            print("Saved metrics summary: ", summary_path)
    else:
        primary_test_output = None
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(model, test_dataloader)
            if i == 0 and output:
                primary_test_output = output[0]
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": output[0]["f1_score"]}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010 ": output[0]["f1_score"]}, commit=False)
        if primary_test_output is not None:
            summary_path = save_metrics_summary(config, config.model.type.value, model, primary_test_output)
            print("Saved metrics summary: ", summary_path)
            
    

def run_wandb(config: Config, accelerator):
    def wandb_sweep_callback():
        try:
            import wandb  # type: ignore
            from lightning.pytorch.loggers import WandbLogger
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "wandb is required when experiment.is_wandb=true. "
                "Install it with `pip install wandb` or set experiment.is_wandb=false."
            ) from exc
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.experiment.is_sweep:
            run_name = ""
            for param in config.model.keys():
                value = config.model[param]
                if param == "hyperparameters_sweep":
                    continue
                if type(value) == omegaconf.dictconfig.DictConfig:
                    for key in value.keys():
                        run_name += str(key[:2]) + "_" + str(value[key]) + "_"
                else:
                    run_name += str(param[:2]) + "_" + str(value.value) + "_"

        wandb_mode = None
        if not os.getenv("WANDB_API_KEY"):
            wandb_mode = "offline"
            print("WANDB_API_KEY not set. Running Weights & Biases in offline mode.")
        run = wandb.init(
            project=cst.PROJECT_NAME,
            name=run_name,
            entity="",
            mode=wandb_mode,
        ) # set entity to your wandb username
        
        if config.experiment.is_sweep:
            model_params = run.config
        else:
            model_params = config.model.hyperparameters_fixed
        wandb_instance_name = ""
        for param in config.model.hyperparameters_fixed.keys():
            if param in model_params:
                config.model.hyperparameters_fixed[param] = model_params[param]
                wandb_instance_name += str(param) + "_" + str(model_params[param]) + "_"

        run.name = wandb_instance_name
        resolve_ckpt_dir(config)
        wandb_instance_name = config.experiment.dir_ckpt
            
        trainer = L.Trainer(
            accelerator=accelerator,
            precision=cst.PRECISION,
            max_epochs=config.experiment.max_epochs,
            callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=True, min_delta=0.002),
                TQDMProgressBar(refresh_rate=1000)
            ],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            detect_anomaly=False,
            check_val_every_n_epoch=1,
        )

        # log simulation details in WANDB console
        run.log({"model": config.model.type.value}, commit=False)
        run.log({"dataset": config.dataset.type.value}, commit=False)
        run.log({"seed": config.experiment.seed}, commit=False)
        run.log({"all_features": config.model.hyperparameters_fixed["all_features"]}, commit=False)
        if config.dataset.type == cst.DatasetType.LOBSTER:
            for i in range(len(config.dataset.training_stocks)):
                run.log({f"training stock{i}": config.dataset.training_stocks[i]}, commit=False)
            for i in range(len(config.dataset.testing_stocks)):
                run.log({f"testing stock{i}": config.dataset.testing_stocks[i]}, commit=False)
            run.log({"sampling_type": config.dataset.sampling_type.value}, commit=False)
            if config.dataset.sampling_type == SamplingType.TIME:
                run.log({"sampling_time": config.dataset.sampling_time}, commit=False)
            elif config.dataset.sampling_type == SamplingType.QUANTITY:
                run.log({"sampling_quantity": config.dataset.sampling_quantity}, commit=False)
        train(config, trainer, run)
        run.finish()

    return wandb_sweep_callback
  
    
def sweep_init(config: Config):
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb is required when experiment.is_sweep=true. "
            "Install it with `pip install wandb`."
        ) from exc
    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if api_key == "":
        raise RuntimeError(
            "WANDB_API_KEY is required when experiment.is_sweep=true. "
            "Set WANDB_API_KEY in the environment."
        )
    wandb.login(key=api_key)
    parameters = {}
    for key in config.model.hyperparameters_sweep.keys():
        parameters[key] = {'values': list(config.model.hyperparameters_sweep[key])}
    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 1.5
        },
        'run_cap': 100,
        'parameters': {**parameters}
    }
    return sweep_config


def print_setup(config: Config):
    print("Model type: ", config.model.type)
    print("Dataset: ", config.dataset.type)
    print("Seed: ", config.experiment.seed)
    print("Sequence size: ", config.model.hyperparameters_fixed["seq_size"])
    print("Horizon: ", config.experiment.horizon)
    print("All features: ", config.model.hyperparameters_fixed["all_features"])
    print("Is data preprocessed: ", config.experiment.is_data_preprocessed)
    print("Is wandb: ", config.experiment.is_wandb)
    print("Is sweep: ", config.experiment.is_sweep)
    print(config.experiment.type)
    print("Is debug: ", config.experiment.is_debug) 
    print("Loss: ", config.experiment.loss_name)
    print("Min event precision: ", config.experiment.min_event_precision)
    print("Use weighted sampler: ", config.dataset.use_weighted_sampler)
    print("Weighted sampler pow: ", config.dataset.weighted_sampler_pow)
    if config.dataset.type == cst.DatasetType.LOBSTER:
        print("Training stocks: ", config.dataset.training_stocks)
        print("Testing stocks: ", config.dataset.testing_stocks)
