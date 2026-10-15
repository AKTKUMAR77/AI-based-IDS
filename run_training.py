"""
Main training and evaluation pipeline for DL-IDS.
Runs the complete pipeline: load data, preprocess, train models, evaluate, and save results.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import logging
import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils import load_config, setup_directories, set_seed, setup_logging, save_json, get_device, format_duration
from src.data_loader import load_all_csvs, inspect_dataset, print_inspection_report
from src.preprocessing import DataPreprocessor
from src.model import get_model, count_parameters
from src.train import Trainer
from src.evaluate import (
    evaluate_model, evaluate_sklearn_model,
    generate_all_plots, plot_model_comparison,
    plot_confusion_matrix, save_metrics
)


def main():
    """Run the complete DL-IDS training and evaluation pipeline."""
    
    # Setup
    config = load_config(os.path.join(project_root, "config.yaml"))
    root = setup_directories(config)
    logger = setup_logging(os.path.join(root, "reports", "training.log"))
    set_seed(config["preprocessing"]["random_seed"])
    device = get_device()
    
    logger.info("=" * 60)
    logger.info("DL-IDS: AI-Based Intrusion Detection System")
    logger.info("=" * 60)
    
    # ========== STEP 1: Load and Inspect Dataset ==========
    logger.info("\n[STEP 1] Loading and inspecting dataset...")
    start = time.time()
    
    df = load_all_csvs(config)
    
    # Generate inspection report
    report = inspect_dataset(df)
    print_inspection_report(report)
    save_json(report, os.path.join(root, "reports", "dataset_inspection.json"))
    
    logger.info(f"Dataset loaded in {format_duration(time.time() - start)}")
    
    # ========== STEP 2: Preprocess Data ==========
    logger.info("\n[STEP 2] Preprocessing data...")
    start = time.time()
    
    preprocessor = DataPreprocessor(config)
    data_dict = preprocessor.prepare_data(df)
    
    # Free memory from raw dataframe
    del df
    
    logger.info(f"Preprocessing complete in {format_duration(time.time() - start)}")
    logger.info(f"Features: {data_dict['num_features']}, Classes: {data_dict['num_classes']}")
    logger.info(f"Train: {len(data_dict['y_train']):,}, Val: {len(data_dict['y_val']):,}, Test: {len(data_dict['y_test']):,}")
    
    # Get label names for reporting
    label_names = list(preprocessor.label_encoder.classes_)
    logger.info(f"Classes: {label_names}")
    
    # Save label distribution for plots
    unique, counts = np.unique(data_dict["y_train"], return_counts=True)
    label_dist = {label_names[i]: int(c) for i, c in zip(unique, counts)}
    
    # ========== STEP 3: Train DL-IDS (CNN-LSTM) Model ==========
    logger.info("\n[STEP 3] Training DL-IDS (CNN-LSTM) model...")
    start = time.time()
    
    dl_ids_model = get_model(
        "DL-IDS",
        num_features=data_dict["num_features"],
        num_classes=data_dict["num_classes"],
        config=config.get("model", {}),
    )
    
    total_p, train_p = count_parameters(dl_ids_model)
    logger.info(f"DL-IDS parameters: {total_p:,} total, {train_p:,} trainable")
    logger.info(f"Model architecture:\n{dl_ids_model}")
    
    trainer = Trainer(dl_ids_model, config, device)
    dl_ids_history = trainer.train(data_dict)
    
    dl_ids_train_time = trainer.training_time
    logger.info(f"DL-IDS training complete in {format_duration(dl_ids_train_time)}")
    
    # Save DL-IDS model
    trainer.save_model(os.path.join(root, "artifacts", "dl_ids_model.pt"))
    trainer.save_history(os.path.join(root, "reports", "dl_ids_history.json"))
    preprocessor.save()
    
    # ========== STEP 4: Evaluate DL-IDS ==========
    logger.info("\n[STEP 4] Evaluating DL-IDS model...")
    
    dl_ids_metrics = evaluate_model(
        dl_ids_model, data_dict["X_test"], data_dict["y_test"],
        label_names=label_names, device=device
    )
    dl_ids_metrics["training_time_seconds"] = dl_ids_train_time
    dl_ids_metrics["model_name"] = "DL-IDS (CNN-LSTM)"
    
    save_metrics(dl_ids_metrics, os.path.join(root, "reports", "dl_ids_metrics.json"))
    
    # Generate DL-IDS plots
    generate_all_plots(
        dl_ids_metrics, dl_ids_history, label_names, label_dist,
        plots_dir=os.path.join(root, "reports", "plots")
    )
    
    # ========== STEP 5: Train and Evaluate Baseline - Random Forest ==========
    logger.info("\n[STEP 5] Training Random Forest baseline...")
    start = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=config["preprocessing"]["random_seed"],
        class_weight="balanced",
    )
    rf_model.fit(data_dict["X_train"], data_dict["y_train"])
    rf_train_time = time.time() - start
    logger.info(f"Random Forest training complete in {format_duration(rf_train_time)}")
    
    rf_metrics = evaluate_sklearn_model(
        rf_model, data_dict["X_test"], data_dict["y_test"], label_names=label_names
    )
    rf_metrics["training_time_seconds"] = rf_train_time
    rf_metrics["model_name"] = "Random Forest"
    
    save_metrics(rf_metrics, os.path.join(root, "reports", "rf_metrics.json"))
    
    # Save RF model
    joblib.dump(rf_model, os.path.join(root, "artifacts", "rf_model.joblib"))
    logger.info("Random Forest model saved")
    
    # RF confusion matrix
    cm_rf = np.array(rf_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_rf, label_names,
                         save_path=os.path.join(root, "reports", "plots", "rf_confusion_matrix.png"),
                         title="Random Forest Confusion Matrix")
    
    # ========== STEP 6: Train and Evaluate Baseline - CNN Only ==========
    logger.info("\n[STEP 6] Training CNN-only baseline...")
    start = time.time()
    
    cnn_model = get_model(
        "CNN",
        num_features=data_dict["num_features"],
        num_classes=data_dict["num_classes"],
        config=config.get("model", {}),
    )
    
    cnn_trainer = Trainer(cnn_model, config, device)
    cnn_history = cnn_trainer.train(data_dict)
    cnn_train_time = cnn_trainer.training_time
    
    cnn_trainer.save_model(os.path.join(root, "artifacts", "cnn_model.pt"))
    cnn_trainer.save_history(os.path.join(root, "reports", "cnn_history.json"))
    
    cnn_metrics = evaluate_model(
        cnn_model, data_dict["X_test"], data_dict["y_test"],
        label_names=label_names, device=device
    )
    cnn_metrics["training_time_seconds"] = cnn_train_time
    cnn_metrics["model_name"] = "CNN Only"
    
    save_metrics(cnn_metrics, os.path.join(root, "reports", "cnn_metrics.json"))
    
    cm_cnn = np.array(cnn_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_cnn, label_names,
                         save_path=os.path.join(root, "reports", "plots", "cnn_confusion_matrix.png"),
                         title="CNN-Only Confusion Matrix")
    
    # ========== STEP 7: Model Comparison ==========
    logger.info("\n[STEP 7] Generating model comparison...")
    
    all_metrics = {
        "DL-IDS (CNN-LSTM)": dl_ids_metrics,
        "CNN Only": cnn_metrics,
        "Random Forest": rf_metrics,
    }
    
    # Comparison plot
    plot_model_comparison(
        all_metrics,
        save_path=os.path.join(root, "reports", "plots", "model_comparison.png")
    )
    
    # Comparison table
    comparison_table = {}
    for model_name, metrics in all_metrics.items():
        comparison_table[model_name] = {
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision (macro)": f"{metrics['precision_macro']:.4f}",
            "Recall (macro)": f"{metrics['recall_macro']:.4f}",
            "F1 (macro)": f"{metrics['f1_macro']:.4f}",
            "F1 (weighted)": f"{metrics['f1_weighted']:.4f}",
            "Training Time": format_duration(metrics.get("training_time_seconds", 0)),
            "Inference Time (ms/sample)": f"{metrics.get('avg_inference_latency_ms', 0):.3f}",
        }
    
    save_json(comparison_table, os.path.join(root, "reports", "model_comparison.json"))
    
    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    for model_name, vals in comparison_table.items():
        logger.info(f"\n{model_name}:")
        for metric, value in vals.items():
            logger.info(f"  {metric}: {value}")
    logger.info("=" * 80)
    
    # ========== STEP 8: Save Training Configuration ==========
    training_config = {
        "model_architecture": "CNN-LSTM Hybrid (DL-IDS)",
        "num_features": int(data_dict["num_features"]),
        "num_classes": int(data_dict["num_classes"]),
        "class_names": label_names,
        "train_samples": int(len(data_dict["y_train"])),
        "val_samples": int(len(data_dict["y_val"])),
        "test_samples": int(len(data_dict["y_test"])),
        "batch_size": config["training"]["batch_size"],
        "epochs_trained": len(dl_ids_history["train_loss"]),
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "use_class_weights": config["training"]["use_class_weights"],
        "device": str(device),
    }
    save_json(training_config, os.path.join(root, "artifacts", "training_config.json"))
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Artifacts saved to: {os.path.join(root, 'artifacts')}")
    logger.info(f"Reports saved to: {os.path.join(root, 'reports')}")
    logger.info(f"Plots saved to: {os.path.join(root, 'reports', 'plots')}")
    
    return all_metrics


if __name__ == "__main__":
    main()
