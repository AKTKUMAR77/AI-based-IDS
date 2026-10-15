"""
Inference module for the DL-IDS project.
Handles single-record and batch predictions using the trained model
and saved preprocessing artifacts.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from src.utils import load_config, get_project_root, get_device
from src.model import get_model
from src.preprocessing import DataPreprocessor

logger = logging.getLogger("DL-IDS.inference")


class IDSPredictor:
    """
    Inference engine for the DL-IDS.
    Loads saved model and preprocessing artifacts for prediction.
    """

    def __init__(self, config=None, model_name="DL-IDS", device=None):
        if config is None:
            config = load_config()
        self.config = config
        self.model_name = model_name
        self.device = device if device else get_device()
        self.model = None
        self.preprocessor = None
        self.is_loaded = False

    def load(self, model_path=None, artifacts_dir=None):
        """
        Load the trained model and preprocessing artifacts.
        
        Args:
            model_path: Path to the saved model .pt file.
            artifacts_dir: Directory containing preprocessing artifacts.
        """
        root = get_project_root()
        
        if artifacts_dir is None:
            artifacts_dir = os.path.join(root, self.config["artifacts"]["model_dir"])
        
        if model_path is None:
            model_filename = f"{self.model_name.lower().replace('-', '_')}_model.pt"
            model_path = os.path.join(artifacts_dir, model_filename)
        
        # Load preprocessing artifacts
        self.preprocessor = DataPreprocessor(self.config)
        self.preprocessor.load(artifacts_dir)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        num_features = checkpoint["num_features"]
        num_classes = checkpoint["num_classes"]
        
        self.model = get_model(
            self.model_name, num_features, num_classes,
            config=self.config.get("model", {})
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.is_loaded = True
        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Features: {num_features}, Classes: {num_classes}")

    def predict_single(self, features_dict):
        """
        Predict for a single record given as a dictionary of feature values.
        
        Args:
            features_dict: Dict mapping feature names to values.
        
        Returns:
            Dict with predicted class, confidence, and all probabilities.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Create a single-row DataFrame
        df = pd.DataFrame([features_dict])
        
        # Apply preprocessing
        X = self.preprocessor.transform_new_data(df)
        
        # Predict
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_class_idx = int(np.argmax(probs))
        predicted_label = self.preprocessor.label_encoder.classes_[predicted_class_idx]
        confidence = float(probs[predicted_class_idx])
        
        # All class probabilities
        class_probs = {
            self.preprocessor.label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "is_attack": predicted_label != "Normal",
        }

    def predict_batch(self, df, batch_size=256):
        """
        Predict for a batch of records from a DataFrame.
        
        Args:
            df: pandas DataFrame with feature columns.
            batch_size: Batch size for inference.
        
        Returns:
            DataFrame with added prediction columns.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        result_df = df.copy()
        
        # Apply preprocessing
        try:
            X = self.preprocessor.transform_new_data(df)
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
        
        # Predict in batches
        all_preds = []
        all_probs = []
        
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size].to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_preds.extend(preds)
                all_probs.extend(np.max(probs, axis=1))
        
        # Decode predictions
        predicted_labels = self.preprocessor.label_encoder.inverse_transform(all_preds)
        
        result_df["Predicted_Class"] = predicted_labels
        result_df["Confidence"] = all_probs
        result_df["Is_Attack"] = [label != "Normal" for label in predicted_labels]
        
        return result_df

    def get_feature_names(self):
        """Get the list of feature names required for prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.preprocessor.feature_names

    def get_class_names(self):
        """Get the list of class names."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return list(self.preprocessor.label_encoder.classes_)

    def get_sample_input(self):
        """Get a sample input dictionary with default values for all features."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        return {name: 0.0 for name in self.preprocessor.feature_names}
