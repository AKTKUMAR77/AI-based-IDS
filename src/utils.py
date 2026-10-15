"""
Utility functions for the DL-IDS project.
Provides config loading, logging, directory creation, and reproducibility setup.
"""

import os
import sys
import json
import yaml
import random
import logging
import numpy as np
from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load project configuration from YAML file."""
    # Try to find config relative to project root
    if not os.path.isabs(config_path):
        # Check if we're running from project root or src/
        if os.path.exists(config_path):
            pass
        elif os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root():
    """Get the project root directory."""
    # Navigate up from src/ to project root
    current = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current) == "src":
        return os.path.dirname(current)
    return current


def setup_directories(config):
    """Create necessary directories for the project."""
    dirs = [
        config.get("artifacts", {}).get("model_dir", "artifacts"),
        config.get("training", {}).get("checkpoint_dir", "artifacts/checkpoints"),
        config.get("evaluation", {}).get("results_dir", "reports"),
        config.get("evaluation", {}).get("plots_dir", "reports/plots"),
    ]
    
    root = get_project_root()
    for d in dirs:
        full_path = os.path.join(root, d)
        os.makedirs(full_path, exist_ok=True)
    
    return root


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(log_file=None, level=logging.INFO):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    
    return logging.getLogger("DL-IDS")


def save_json(data, filepath):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath):
    """Load dictionary from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device():
    """Get the best available device (CUDA > CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    except ImportError:
        return "cpu"


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
