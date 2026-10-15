"""
Preprocessing module for the DL-IDS project.
Handles feature engineering, scaling, encoding, and train/val/test splitting.
Ensures no data leakage: transformations are fit only on training data.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import load_config, get_project_root, set_seed

logger = logging.getLogger("DL-IDS.preprocessing")


class DataPreprocessor:
    """
    Reusable preprocessing pipeline for the DL-IDS.
    Fits on training data only, applies consistently to val/test/inference data.
    """

    def __init__(self, config=None):
        if config is None:
            config = load_config()
        self.config = config
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.label_mapping = None
        self.class_weights = None
        self.is_fitted = False

    def clean_data(self, df):
        """
        Clean the raw dataframe: handle missing values, infinities,
        drop constant/irrelevant columns, and remove duplicates.
        
        Args:
            df: Raw pandas DataFrame.
        
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning data...")
        initial_rows = len(df)

        # Strip column names
        df.columns = df.columns.str.strip()

        # Drop specified constant/irrelevant columns
        drop_cols = self.config["preprocessing"].get("drop_columns", [])
        existing_drop = [c for c in drop_cols if c in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop)
            logger.info(f"  Dropped {len(existing_drop)} constant/irrelevant columns")

        # Also drop Fwd Header Length.1 (duplicate of Fwd Header Length)
        if "Fwd Header Length.1" in df.columns and "Fwd Header Length" in df.columns:
            df = df.drop(columns=["Fwd Header Length.1"])
            logger.info("  Dropped duplicate column: Fwd Header Length.1")

        # Replace infinite values with NaN
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[num_cols]).any(axis=1)
        inf_count = inf_mask.sum()
        if inf_count > 0:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            logger.info(f"  Replaced {inf_count} rows with infinite values -> NaN")

        # Drop rows with NaN (they are very few in CICIDS2017)
        nan_count = df.isnull().any(axis=1).sum()
        if nan_count > 0:
            df = df.dropna()
            logger.info(f"  Dropped {nan_count} rows with NaN values")

        # Remove duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
            logger.info(f"  Removed {dup_count:,} duplicate rows")

        final_rows = len(df)
        logger.info(
            f"  Cleaning complete: {initial_rows:,} -> {final_rows:,} rows "
            f"({initial_rows - final_rows:,} removed)"
        )

        return df.reset_index(drop=True)

    def encode_labels(self, labels):
        """
        Encode string labels to integers.
        
        Args:
            labels: pandas Series of string labels.
        
        Returns:
            numpy array of encoded labels.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(labels)
            self.label_mapping = {
                label: int(idx)
                for idx, label in enumerate(self.label_encoder.classes_)
            }
            logger.info(f"Label mapping: {self.label_mapping}")
        else:
            encoded = self.label_encoder.transform(labels)

        return encoded

    def compute_class_weights(self, labels):
        """
        Compute class weights using balanced formula adapted from the paper:
        w_i = total_samples / (num_classes * n_i)
        
        This gives higher weight to underrepresented classes while capping
        extreme ratios to prevent loss explosion on very small classes.
        
        Args:
            labels: numpy array of encoded integer labels.
        
        Returns:
            Dictionary mapping class index to weight.
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = float(counts.sum())
        num_classes = len(unique)
        weights = {}
        for cls, count in zip(unique, counts):
            w = float(total / (num_classes * count))
            # Cap maximum class weight to 25.0 to ensure gradient stability
            weights[int(cls)] = min(w, 25.0)

        self.class_weights = weights
        logger.info(f"Class weights: { {self.label_encoder.classes_[k]: f'{v:.3f}' for k, v in weights.items()} }")
        return weights

    def fit_scaler(self, X):
        """
        Fit the feature scaler on training data only.
        
        Args:
            X: numpy array of training features.
        """
        scaler_type = self.config["preprocessing"].get("scaler", "standard")
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.scaler.fit(X)
        logger.info(f"Fitted {scaler_type} scaler on {X.shape[0]:,} samples")

    def transform_features(self, X):
        """
        Transform features using the fitted scaler.
        
        Args:
            X: numpy array of features.
        
        Returns:
            Scaled numpy array.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.transform(X)

    def prepare_data(self, df):
        """
        Full preprocessing pipeline: clean, split, encode, scale.
        Ensures no data leakage (fit only on training data).
        
        Args:
            df: Raw combined DataFrame with Label column.
        
        Returns:
            dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                           feature_names, label_mapping, class_weights
        """
        seed = self.config["preprocessing"].get("random_seed", 42)
        set_seed(seed)

        # Clean data
        df = self.clean_data(df)

        # Separate features and labels
        label_col = self.config["dataset"].get("label_column", "Label")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        # Optional balanced sampling / class capping for memory/runtime management
        max_samples = self.config["preprocessing"].get("max_samples_per_class", 0)
        if max_samples and max_samples > 0:
            logger.info(f"Subsampling: capping classes to max {max_samples:,} samples per class...")
            sampled_dfs = []
            for _, grp in df.groupby(label_col):
                if len(grp) > max_samples:
                    sampled_dfs.append(grp.sample(n=max_samples, random_state=seed))
                else:
                    sampled_dfs.append(grp)
            df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Dataset size after class capping: {len(df):,} rows")

        y = df[label_col].values
        X = df.drop(columns=[label_col])
        
        # Ensure all feature columns are numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            logger.warning(f"Dropping non-numeric feature columns: {list(non_numeric)}")
            X = X.select_dtypes(include=[np.number])

        self.feature_names = list(X.columns)
        logger.info(f"Feature count: {len(self.feature_names)}")

        X = X.values.astype(np.float32)

        # Encode labels
        y_encoded = self.encode_labels(y)

        # Split: train / (val + test)
        train_ratio = self.config["preprocessing"]["train_ratio"]
        val_ratio = self.config["preprocessing"]["val_ratio"]
        test_ratio = self.config["preprocessing"]["test_ratio"]

        # First split: train vs (val+test)
        val_test_ratio = val_ratio + test_ratio
        _, class_counts_all = np.unique(y_encoded, return_counts=True)
        stratify_1 = y_encoded if (class_counts_all.min() >= 2) else None
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=val_test_ratio, random_state=seed, stratify=stratify_1
        )

        # Second split: val vs test
        test_fraction = test_ratio / val_test_ratio
        _, class_counts_temp = np.unique(y_temp, return_counts=True)
        stratify_2 = y_temp if (class_counts_temp.min() >= 2) else None
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_fraction, random_state=seed, stratify=stratify_2
        )

        logger.info(
            f"Split sizes - Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}"
        )

        # Fit scaler on training data ONLY
        self.fit_scaler(X_train)

        # Transform all splits
        X_train = self.transform_features(X_train)
        X_val = self.transform_features(X_val)
        X_test = self.transform_features(X_test)

        # Compute class weights on training data
        if self.config["training"].get("use_class_weights", True):
            self.compute_class_weights(y_train)

        self.is_fitted = True

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "label_mapping": self.label_mapping,
            "class_weights": self.class_weights,
            "num_features": X_train.shape[1],
            "num_classes": len(self.label_mapping),
        }

    def transform_new_data(self, df):
        """
        Apply the fitted preprocessing pipeline to new data (inference).
        
        Args:
            df: DataFrame with the same feature columns as training data.
        
        Returns:
            Scaled numpy array of features.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call prepare_data() first or load().")

        # Strip column names
        df.columns = df.columns.str.strip()
        
        # Drop label column if present
        label_col = self.config["dataset"].get("label_column", "Label")
        if label_col in df.columns:
            df = df.drop(columns=[label_col])
        
        # Keep only the features used during training
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        X = df[self.feature_names].values.astype(np.float32)
        
        # Handle inf and nan
        X = np.where(np.isinf(X), np.nan, X)
        # Replace NaN with 0 (will be scaled)
        X = np.nan_to_num(X, nan=0.0)
        
        return self.transform_features(X)

    def save(self, directory=None):
        """Save all preprocessing artifacts."""
        if directory is None:
            root = get_project_root()
            directory = os.path.join(root, self.config["artifacts"]["model_dir"])
        
        os.makedirs(directory, exist_ok=True)

        # Save scaler
        scaler_path = os.path.join(directory, "preprocessor.joblib")
        joblib.dump(self.scaler, scaler_path)

        # Save label encoder
        le_path = os.path.join(directory, "label_encoder.joblib")
        joblib.dump(self.label_encoder, le_path)

        # Save feature names
        feat_path = os.path.join(directory, "feature_names.json")
        with open(feat_path, "w") as f:
            json.dump(self.feature_names, f)

        # Save label mapping
        lm_path = os.path.join(directory, "label_mapping.json")
        with open(lm_path, "w") as f:
            json.dump(self.label_mapping, f)

        # Save class weights
        if self.class_weights:
            cw_path = os.path.join(directory, "class_weights.json")
            with open(cw_path, "w") as f:
                json.dump(self.class_weights, f)

        logger.info(f"Preprocessing artifacts saved to {directory}")

    def load(self, directory=None):
        """Load all preprocessing artifacts."""
        if directory is None:
            root = get_project_root()
            directory = os.path.join(root, self.config["artifacts"]["model_dir"])

        self.scaler = joblib.load(os.path.join(directory, "preprocessor.joblib"))
        self.label_encoder = joblib.load(os.path.join(directory, "label_encoder.joblib"))

        with open(os.path.join(directory, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)

        with open(os.path.join(directory, "label_mapping.json"), "r") as f:
            self.label_mapping = json.load(f)

        cw_path = os.path.join(directory, "class_weights.json")
        if os.path.exists(cw_path):
            with open(cw_path, "r") as f:
                self.class_weights = json.load(f)
                # Convert string keys back to int
                self.class_weights = {int(k): v for k, v in self.class_weights.items()}

        self.is_fitted = True
        logger.info(f"Preprocessing artifacts loaded from {directory}")

    def decode_labels(self, encoded_labels):
        """Convert encoded integer labels back to string labels."""
        return self.label_encoder.inverse_transform(encoded_labels)
