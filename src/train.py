"""
Training module for the DL-IDS project.
Handles model training with class-weighted loss, early stopping,
checkpointing, and training history tracking.
"""

import os
import time
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_config, get_project_root, format_duration, get_device
from src.model import get_model, count_parameters

logger = logging.getLogger("DL-IDS.train")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Model trainer with full training loop, validation, and checkpointing."""

    def __init__(self, model, config=None, device=None):
        if config is None:
            config = load_config()
        self.config = config
        self.model = model
        self.device = device if device else get_device()
        self.model.to(self.device)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
        self.best_val_loss = float("inf")
        self.training_time = 0

    def _create_optimizer(self):
        """Create optimizer based on config (paper uses RMSprop)."""
        opt_name = self.config["training"].get("optimizer", "rmsprop").lower()
        lr = self.config["training"].get("learning_rate", 0.001)

        if opt_name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr)
        elif opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr)

    def _create_criterion(self, class_weights=None):
        """Create loss function with optional class weights (paper's weighted cross-entropy)."""
        if class_weights and self.config["training"].get("use_class_weights", True):
            num_classes = len(class_weights)
            weights = torch.zeros(num_classes)
            for cls_idx, weight in class_weights.items():
                weights[int(cls_idx)] = weight
            weights = weights.to(self.device)
            logger.info("Using class-weighted cross-entropy loss")
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

    def _create_dataloaders(self, X_train, y_train, X_val, y_val):
        """Create PyTorch DataLoaders."""
        batch_size = self.config["training"].get("batch_size", 64)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )

        pin_mem = (self.device.type == "cuda")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem
        )

        return train_loader, val_loader

    def train(self, data_dict):
        """
        Full training loop.
        
        Args:
            data_dict: Dictionary from DataPreprocessor.prepare_data() containing
                      X_train, X_val, y_train, y_val, class_weights, etc.
        
        Returns:
            Training history dict.
        """
        X_train = data_dict["X_train"]
        X_val = data_dict["X_val"]
        y_train = data_dict["y_train"]
        y_val = data_dict["y_val"]
        class_weights = data_dict.get("class_weights")

        epochs = self.config["training"].get("epochs", 30)
        es_config = self.config["training"].get("early_stopping", {})
        patience = es_config.get("patience", 5)
        min_delta = es_config.get("min_delta", 0.001)

        # Setup
        optimizer = self._create_optimizer()
        criterion = self._create_criterion(class_weights)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        train_loader, val_loader = self._create_dataloaders(X_train, y_train, X_val, y_val)

        # Checkpoint directory
        root = get_project_root()
        ckpt_dir = os.path.join(root, self.config["training"].get("checkpoint_dir", "artifacts/checkpoints"))
        os.makedirs(ckpt_dir, exist_ok=True)

        total_params, trainable_params = count_parameters(self.model)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"Training for {epochs} epochs, batch_size={self.config['training']['batch_size']}")
        logger.info(f"Optimizer: {self.config['training']['optimizer']}, LR: {self.config['training']['learning_rate']}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_losses = []
            train_preds = []
            train_labels = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            # Validation phase
            self.model.eval()
            val_losses = []
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())

            # Compute metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="weighted")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch + 1}/{epochs} [{format_duration(epoch_time)}] - "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                f"Acc: {train_acc:.4f}/{val_acc:.4f} - "
                f"Val F1: {val_f1:.4f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, best_path)
                logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        self.training_time = time.time() - start_time
        logger.info(f"Training complete in {format_duration(self.training_time)}")

        # Load best model
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

        return self.history

    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            root = get_project_root()
            filepath = os.path.join(root, self.config["artifacts"]["model_dir"], "dl_ids_model.pt")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "num_features": getattr(self.model, "num_features", None),
            "num_classes": getattr(self.model, "num_classes", None),
            "training_time": self.training_time,
            "history": self.history,
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def save_history(self, filepath=None):
        """Save training history to JSON."""
        if filepath is None:
            root = get_project_root()
            filepath = os.path.join(root, "reports", "training_history.json")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        history_serializable["training_time_seconds"] = self.training_time

        with open(filepath, "w") as f:
            json.dump(history_serializable, f, indent=2)
        logger.info(f"Training history saved to {filepath}")


def train_model(model_name="DL-IDS", config=None):
    """
    High-level training function that loads data, preprocesses, trains, and saves.
    
    Args:
        model_name: Name of the model to train ('DL-IDS', 'CNN', 'LSTM')
        config: Configuration dict
    
    Returns:
        Trained model, preprocessor, data_dict, history
    """
    from src.data_loader import load_all_csvs
    from src.preprocessing import DataPreprocessor

    if config is None:
        config = load_config()

    device = get_device()
    logger.info(f"Training {model_name} model...")

    # Load and preprocess data
    logger.info("Loading dataset...")
    df = load_all_csvs(config)

    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.prepare_data(df)

    # Create model
    model = get_model(
        model_name,
        num_features=data_dict["num_features"],
        num_classes=data_dict["num_classes"],
        config=config.get("model", {}),
    )

    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(data_dict)

    # Save
    root = get_project_root()
    model_filename = f"{model_name.lower().replace('-', '_')}_model.pt"
    trainer.save_model(os.path.join(root, "artifacts", model_filename))
    trainer.save_history(os.path.join(root, "reports", f"{model_name.lower().replace('-', '_')}_history.json"))
    preprocessor.save()

    return model, preprocessor, data_dict, history
