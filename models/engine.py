import random
from lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from utils.utils_model import pick_model
from utils.engine_recall import (
    compute_event_metrics,
    predicted_event_rate,
    precision_at_top_percent,
    threshold_sweep_event_metrics,
    pick_threshold_for_target_rate,
)
import constants as cst
from scipy.stats import mode

try:
    import wandb  # type: ignore
except ModuleNotFoundError:
    wandb = None


class Engine(LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
        loss_name="ce",
        class_weights=None,
        min_event_precision=0.20,
        target_event_rate=0.002,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        self.loss_name = str(loss_name).lower()
        self.min_event_precision = float(min_event_precision)
        self.target_event_rate = float(target_event_rate)
        self.model = pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type) 
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("loss_class_weights", class_weights)
        if self.loss_name == "weighted_ce":
            if self.loss_class_weights is None:
                raise ValueError("loss_name=weighted_ce requires class_weights.")
            self.loss_function = nn.CrossEntropyLoss(weight=self.loss_class_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.val_event_scores = []
        self.min_loss = np.inf
        self.save_hyperparameters()
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        self.best_val_metrics = {}
        self.final_test_metrics = {}
        
    def forward(self, x, batch_idx=None):
        output = self.model(x)
        return output
    
    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f'train loss: {sum(self.train_losses) / len(self.train_losses)}')
        return batch_loss_mean
    
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.val_event_scores.append((1.0 - torch.softmax(y_hat, dim=1)[:, 1]).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.item())
        return batch_loss_mean
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)
        in_training_mode = self.experiment_type == "TRAINING" or (
            isinstance(self.experiment_type, (list, tuple, set))
            and "TRAINING" in self.experiment_type
        )
        # Test: with EMA
        if in_training_mode:
            with self.ema.average_parameters():
                y_hat = self.forward(x, batch_idx)
                batch_loss = self.loss(y_hat, y)
                self.test_targets.append(y.cpu().numpy())
                self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
                self.test_proba.append((1.0 - torch.softmax(y_hat, dim=1)[:, 1]).cpu().numpy())
                batch_loss_mean = torch.mean(batch_loss)
                self.test_losses.append(batch_loss_mean.item())
        else:
            y_hat = self.forward(x, batch_idx)
            batch_loss = self.loss(y_hat, y)
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.test_proba.append((1.0 - torch.softmax(y_hat, dim=1)[:, 1]).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.test_losses.append(batch_loss_mean.item())
        return batch_loss_mean
    
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        # Store train loss for combined plotting
        self.current_train_loss = loss
        print(f'Train loss on epoch {self.current_epoch}: {loss}')
        
    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []

        targets = np.concatenate(self.val_targets)
        predictions = np.concatenate(self.val_predictions)
        val_event_scores = np.concatenate(self.val_event_scores)
        class_report = classification_report(
            targets, predictions, digits=4, output_dict=True, zero_division=0
        )
        event_metrics = compute_event_metrics(targets, predictions)
        event_rate = predicted_event_rate(predictions)
        event_pr_auc = (
            float(average_precision_score((targets != 1).astype(np.int32), val_event_scores))
            if np.unique((targets != 1).astype(np.int32)).size > 1
            else 0.0
        )
        precision_top_01 = precision_at_top_percent(targets, val_event_scores, top_percent=0.001)
        precision_top_02 = precision_at_top_percent(targets, val_event_scores, top_percent=0.002)
        sweep_rows = threshold_sweep_event_metrics(
            targets=targets,
            event_scores=val_event_scores,
            thresholds=np.arange(0.20, 0.951, 0.05),
        )
        threshold_pick = pick_threshold_for_target_rate(
            sweep_rows=sweep_rows,
            target_rate=self.target_event_rate,
        )

        # model checkpointing
        if self.val_loss < self.min_loss:
            # if the improvement is less than 0.0002, we halve the learning rate
            if self.val_loss - self.min_loss > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2
            self.min_loss = self.val_loss
            self.best_val_metrics = {
                "val_loss": float(self.val_loss),
                "val_f1_score": float(class_report["macro avg"]["f1-score"]),
                "val_accuracy": float(class_report["accuracy"]),
                "val_precision": float(class_report["macro avg"]["precision"]),
                "val_recall": float(class_report["macro avg"]["recall"]),
                "val_event_precision": float(event_metrics["event_precision"]),
                "val_event_recall": float(event_metrics["event_recall"]),
                "val_event_f1": float(event_metrics["event_f1"]),
                "val_predicted_event_rate": float(event_rate),
                "val_event_pr_auc": float(event_pr_auc),
                "val_precision_top_0_1pct": float(precision_top_01),
                "val_precision_top_0_2pct": float(precision_top_02),
                "val_threshold_sweep_pick": threshold_pick,
            }
            self.model_checkpointing(self.val_loss)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2

        # Log losses to wandb (both individually and in the same plot)
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)

        # Continue with regular Lightning logging for compatibility
        self.log("val_loss", self.val_loss)
        print(f"Validation loss on epoch {self.current_epoch}: {self.val_loss}")
        print(classification_report(targets, predictions, digits=4, zero_division=0))
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])
        self.log("val_event_precision", event_metrics["event_precision"])
        self.log("val_event_recall", event_metrics["event_recall"])
        self.log("val_event_f1", event_metrics["event_f1"])
        self.log("val_predicted_event_rate", event_rate)
        self.log("val_event_pr_auc", event_pr_auc)
        self.log("val_precision_top_0_1pct", precision_top_01)
        self.log("val_precision_top_0_2pct", precision_top_02)
        print(
            f"Validation event metrics: precision={event_metrics['event_precision']:.4f} "
            f"recall={event_metrics['event_recall']:.4f} f1={event_metrics['event_f1']:.4f} "
            f"pred_rate={event_rate:.4%} pr_auc={event_pr_auc:.4f} "
            f"p@0.1%={precision_top_01:.4f} p@0.2%={precision_top_02:.4f} "
            f"(guard={self.min_event_precision:.2f})"
        )
        if threshold_pick is not None:
            print(
                "Validation threshold pick: "
                f"th={threshold_pick['threshold']:.2f} "
                f"precision={threshold_pick['precision']:.4f} "
                f"recall={threshold_pick['recall']:.4f} "
                f"pred_rate={threshold_pick['predicted_event_rate']:.4%} "
                f"(target_rate={self.target_event_rate:.4%})"
            )
        self.val_targets = []
        self.val_predictions = [] 
        self.val_event_scores = []
    
    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses to wandb in the same plot."""
        if self.is_wandb:
            if wandb is None:
                raise RuntimeError(
                    "wandb is required when experiment.is_wandb=true. "
                    "Install it with `pip install wandb` or set experiment.is_wandb=false."
                )
            # Log combined losses for a single plot
            wandb.log({
                "losses": {
                    "train": train_loss,
                    "validation": val_loss
                },
                "epoch": self.global_step
            })
    
    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)    
        predictions = np.concatenate(self.test_predictions)
        predictions_path = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "predictions")
        np.save(predictions_path, predictions)
        class_report = classification_report(
            targets, predictions, digits=4, output_dict=True, zero_division=0
        )
        event_metrics = compute_event_metrics(targets, predictions)
        event_scores = np.concatenate(self.test_proba)
        event_rate = predicted_event_rate(predictions)
        event_pr_auc = (
            float(average_precision_score((targets != 1).astype(np.int32), event_scores))
            if np.unique((targets != 1).astype(np.int32)).size > 1
            else 0.0
        )
        precision_top_01 = precision_at_top_percent(targets, event_scores, top_percent=0.001)
        precision_top_02 = precision_at_top_percent(targets, event_scores, top_percent=0.002)
        sweep_rows = threshold_sweep_event_metrics(
            targets=targets,
            event_scores=event_scores,
            thresholds=np.arange(0.20, 0.951, 0.05),
        )
        threshold_pick = pick_threshold_for_target_rate(
            sweep_rows=sweep_rows,
            target_rate=self.target_event_rate,
        )
        test_loss = sum(self.test_losses) / len(self.test_losses)
        print(classification_report(targets, predictions, digits=4, zero_division=0))
        self.log("test_loss", test_loss)
        self.log("f1_score", class_report["macro avg"]["f1-score"])
        self.log("accuracy", class_report["accuracy"])
        self.log("precision", class_report["macro avg"]["precision"])
        self.log("recall", class_report["macro avg"]["recall"])
        self.log("test_event_precision", event_metrics["event_precision"])
        self.log("test_event_recall", event_metrics["event_recall"])
        self.log("test_event_f1", event_metrics["event_f1"])
        self.log("test_predicted_event_rate", event_rate)
        self.log("test_event_pr_auc", event_pr_auc)
        self.log("test_precision_top_0_1pct", precision_top_01)
        self.log("test_precision_top_0_2pct", precision_top_02)
        self.final_test_metrics = {
            "test_loss": float(test_loss),
            "f1_score": float(class_report["macro avg"]["f1-score"]),
            "accuracy": float(class_report["accuracy"]),
            "precision": float(class_report["macro avg"]["precision"]),
            "recall": float(class_report["macro avg"]["recall"]),
            "test_event_precision": float(event_metrics["event_precision"]),
            "test_event_recall": float(event_metrics["event_recall"]),
            "test_event_f1": float(event_metrics["event_f1"]),
            "test_predicted_event_rate": float(event_rate),
            "test_event_pr_auc": float(event_pr_auc),
            "test_precision_top_0_1pct": float(precision_top_01),
            "test_precision_top_0_2pct": float(precision_top_02),
            "test_threshold_sweep_pick": threshold_pick,
        }
        print(
            f"Test event metrics: precision={event_metrics['event_precision']:.4f} "
            f"recall={event_metrics['event_recall']:.4f} f1={event_metrics['event_f1']:.4f} "
            f"pred_rate={event_rate:.4%} pr_auc={event_pr_auc:.4f} "
            f"p@0.1%={precision_top_01:.4f} p@0.2%={precision_top_02:.4f}"
        )
        if threshold_pick is not None:
            print(
                "Test threshold pick: "
                f"th={threshold_pick['threshold']:.2f} "
                f"precision={threshold_pick['precision']:.4f} "
                f"recall={threshold_pick['recall']:.4f} "
                f"pred_rate={threshold_pick['predicted_event_rate']:.4%} "
                f"(target_rate={self.target_event_rate:.4%})"
            )
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []  
        self.first_test = False
        test_proba = np.concatenate(self.test_proba)
        precision, recall, _ = precision_recall_curve((targets != 1).astype(np.int32), test_proba, pos_label=1)
        self.plot_pr_curves(recall, precision, self.is_wandb) 
        
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Lion':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer
    
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")

    def model_checkpointing(self, loss):        
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".pt"
                             )
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)
        
        # Save PyTorch checkpoint
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt)
            
            # Save ONNX model
            onnx_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            
            onnx_filename = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".onnx"
                            )
            onnx_path = os.path.join(onnx_dir, onnx_filename)
            
            # Create dummy input with appropriate shape
            dummy_input = torch.randn(1, self.seq_size, self.num_features, device=self.device)
            
            # Export to ONNX
            try:
                torch.onnx.export(
                    self.model,                  # model being run
                    dummy_input,                 # model input (or a tuple for multiple inputs)
                    onnx_path,                   # where to save the model
                    export_params=True,          # store the trained parameter weights inside the model file
                    opset_version=12,            # the ONNX version to export the model to
                    do_constant_folding=True,    # whether to execute constant folding for optimization
                    input_names=['input'],       # the model's input names
                    output_names=['output'],     # the model's output names
                    dynamic_axes={               # variable length axes
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            except Exception as e:
                print(f"Failed to export ONNX model: {e}")
        
        self.last_path_ckpt = path_ckpt  
        
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label='Precision-Recall', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        if is_wandb:
            if wandb is None:
                raise RuntimeError(
                    "wandb is required when experiment.is_wandb=true. "
                    "Install it with `pip install wandb` or set experiment.is_wandb=false."
                )
            wandb.log({f"precision_recall_curve_{self.dataset_type}": wandb.Image(plt)})
        plt.savefig(cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" +f"precision_recall_curve_{self.dataset_type}.svg")
        #plt.show()
        plt.close()
        
def compute_most_attended(att_feature):
    ''' att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features) '''
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)  # Use transpose instead of permute
    ''' att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) '''
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                # Extract the indices for the current layer and sequence element
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                # Find the most frequent index
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                # Store the result
                most_frequent_indices[layer, head, seq] = most_frequent_index
                # Compute the average value for the most frequent index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                # Store the average value
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values
