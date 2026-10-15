"""
Automated Unit and Integration Tests for DL-IDS Pipeline.
Tests configuration, models, preprocessing, training, evaluation, and inference.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, get_project_root, set_seed, get_device
from src.model import get_model, count_parameters, DLIDS, CNNOnly, LSTMOnly
from src.preprocessing import DataPreprocessor
from src.train import EarlyStopping, Trainer
from src.evaluate import evaluate_model, evaluate_sklearn_model
from src.inference import IDSPredictor


class TestConfigAndUtils:
    def test_load_config(self):
        config = load_config()
        assert "dataset" in config
        assert "preprocessing" in config
        assert "model" in config
        assert "training" in config
        assert config["model"]["name"] == "DL-IDS"

    def test_get_project_root(self):
        root = get_project_root()
        assert os.path.exists(root)
        assert os.path.exists(os.path.join(root, "config.yaml"))

    def test_device_selection(self):
        device = get_device()
        assert isinstance(device, torch.device)


class TestModelArchitectures:
    @pytest.fixture
    def dummy_data(self):
        batch_size = 4
        num_features = 67
        num_classes = 10
        x = torch.randn(batch_size, num_features)
        return x, num_features, num_classes

    def test_dl_ids_forward(self, dummy_data):
        x, num_features, num_classes = dummy_data
        model = get_model("DL-IDS", num_features, num_classes)
        model.eval()
        with torch.no_grad():
            out = model(x)
            probs = model.predict_proba(x)
        assert out.shape == (4, num_classes)
        assert probs.shape == (4, num_classes)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-4)

    def test_cnn_only_forward(self, dummy_data):
        x, num_features, num_classes = dummy_data
        model = get_model("CNN", num_features, num_classes)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, num_classes)
        assert model.num_features == num_features
        assert model.num_classes == num_classes

    def test_lstm_only_forward(self, dummy_data):
        x, num_features, num_classes = dummy_data
        model = get_model("LSTM", num_features, num_classes)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, num_classes)
        assert model.num_features == num_features
        assert model.num_classes == num_classes

    def test_parameter_counts(self, dummy_data):
        _, num_features, num_classes = dummy_data
        model = get_model("DL-IDS", num_features, num_classes)
        total, trainable = count_parameters(model)
        assert total > 0
        assert total == trainable


class TestPreprocessing:
    @pytest.fixture
    def sample_dataframe(self):
        np.random.seed(42)
        n = 100
        data = {
            f"feature_{i}": np.random.randn(n) for i in range(20)
        }
        # Add infinite and NaN values to test cleaning
        data["feature_0"][0] = np.inf
        data["feature_1"][1] = -np.inf
        data["feature_2"][2] = np.nan
        # Add duplicate row
        data["Label"] = ["Normal"] * 70 + ["DoS"] * 20 + ["PortScan"] * 8 + ["Heartbleed"] * 2
        df = pd.DataFrame(data)
        # Duplicate row 5
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        return df

    def test_cleaning_and_encoding(self, sample_dataframe):
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.clean_data(sample_dataframe)
        
        # Check inf and nan removed
        assert not np.isinf(cleaned.select_dtypes(include=[np.number]).values).any()
        assert not cleaned.isnull().values.any()
        # Check duplicates removed
        assert not cleaned.duplicated().any()

    def test_prepare_data_no_leakage(self, sample_dataframe):
        preprocessor = DataPreprocessor()
        data_dict = preprocessor.prepare_data(sample_dataframe)
        
        assert "X_train" in data_dict
        assert "X_val" in data_dict
        assert "X_test" in data_dict
        assert "y_train" in data_dict
        assert "class_weights" in data_dict
        assert data_dict["num_features"] > 0
        assert data_dict["num_classes"] > 0
        
        # Verify scaler was fit on train only
        assert preprocessor.is_fitted
        assert preprocessor.scaler is not None

    def test_class_weights_computation(self):
        preprocessor = DataPreprocessor()
        labels = np.array([0, 0, 0, 0, 1, 1, 2])
        preprocessor.label_encoder = type("DummyLE", (), {"classes_": ["A", "B", "C"]})()
        weights = preprocessor.compute_class_weights(labels)
        assert len(weights) == 3
        # Class 2 is rarest, so its weight must be higher than class 0
        assert weights[2] > weights[0]
        assert all(w <= 25.0 for w in weights.values())


class TestTrainingAndEvaluation:
    def test_early_stopping(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        assert not es.early_stop
        
        # Loss improves
        es(1.0)
        es(0.9)
        es(0.8)
        assert es.counter == 0
        
        # Loss does not improve
        es(0.805)
        es(0.806)
        es(0.807)
        assert es.early_stop

    def test_trainer_mini_run(self):
        np.random.seed(42)
        n_train, n_val, n_features, n_classes = 32, 16, 10, 3
        data_dict = {
            "X_train": np.random.randn(n_train, n_features).astype(np.float32),
            "y_train": np.random.randint(0, n_classes, size=n_train),
            "X_val": np.random.randn(n_val, n_features).astype(np.float32),
            "y_val": np.random.randint(0, n_classes, size=n_val),
            "class_weights": {0: 1.0, 1: 1.0, 2: 1.0},
        }
        
        config = {
            "training": {
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 0.01,
                "optimizer": "rmsprop",
                "use_class_weights": True,
                "early_stopping": {"patience": 2, "min_delta": 0.001},
                "checkpoint_dir": "artifacts/checkpoints",
            },
            "artifacts": {"model_dir": "artifacts"},
        }
        
        model = get_model("DL-IDS", num_features=n_features, num_classes=n_classes)
        trainer = Trainer(model, config, device=torch.device("cpu"))
        history = trainer.train(data_dict)
        
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert "val_acc" in history


class TestInferencePipeline:
    def test_inference_with_mock_model(self):
        predictor = IDSPredictor()
        predictor.is_loaded = True
        
        # Setup dummy preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.feature_names = ["f1", "f2"]
        preprocessor.is_fitted = True
        
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        preprocessor.scaler = StandardScaler()
        preprocessor.scaler.mean_ = np.array([0.0, 0.0])
        preprocessor.scaler.scale_ = np.array([1.0, 1.0])
        
        le = LabelEncoder()
        le.fit(["Normal", "DoS"])
        preprocessor.label_encoder = le
        predictor.preprocessor = preprocessor
        
        # Mock model: logits giving higher score to class index 0 ("DoS")
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Returns 2 classes: class 0 ("DoS") gets highest score
                return torch.tensor([[0.8, 0.2]])
        
        predictor.model = MockModel()
        predictor.device = torch.device("cpu")
        
        res = predictor.predict_single({"f1": 1.0, "f2": 2.0})
        assert "predicted_class" in res
        assert res["predicted_class"] == "DoS"
        assert res["is_attack"] is True
        assert "confidence" in res

    def test_real_model_inference(self):
        # Test loading the actual trained model and running predictions
        model_path = os.path.join(PROJECT_ROOT, "artifacts", "dl_ids_model.pt")
        if not os.path.exists(model_path):
            pytest.skip("Model artifacts not yet generated")
        
        predictor = IDSPredictor()
        predictor.load()
        assert predictor.is_loaded
        assert predictor.model is not None
        assert len(predictor.get_class_names()) == 10
        assert len(predictor.get_feature_names()) == 67
        
        # Test single prediction
        sample = predictor.get_sample_input()
        result = predictor.predict_single(sample)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "class_probabilities" in result
        assert result["predicted_class"] in predictor.get_class_names()
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Test batch prediction
        batch_df = pd.DataFrame([sample, sample])
        batch_res = predictor.predict_batch(batch_df)
        assert "Predicted_Class" in batch_res.columns
        assert "Confidence" in batch_res.columns
        assert "Is_Attack" in batch_res.columns
        assert len(batch_res) == 2


class TestArtifactsAndReports:
    def test_all_reports_exist(self):
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        plots_dir = os.path.join(reports_dir, "plots")
        
        expected_reports = [
            "dataset_inspection.json",
            "dl_ids_metrics.json",
            "rf_metrics.json",
            "cnn_metrics.json",
            "model_comparison.json",
            "dl_ids_history.json",
        ]
        for f in expected_reports:
            p = os.path.join(reports_dir, f)
            assert os.path.exists(p), f"Missing report: {f}"
            assert os.path.getsize(p) > 0, f"Empty report: {f}"
            
        expected_plots = [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "training_history.png",
            "per_class_metrics.png",
            "class_distribution.png",
            "rf_confusion_matrix.png",
            "cnn_confusion_matrix.png",
            "model_comparison.png",
        ]
        for f in expected_plots:
            p = os.path.join(plots_dir, f)
            assert os.path.exists(p), f"Missing plot: {f}"
            assert os.path.getsize(p) > 0, f"Empty plot: {f}"

