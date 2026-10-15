"""
Data loader module for the DL-IDS project.
Handles loading and combining all 8 CICIDS2017 CSV files,
performing initial cleaning, and creating the combined dataset.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
from src.utils import load_config, get_project_root

logger = logging.getLogger("DL-IDS.data_loader")


def load_single_csv(filepath, label_groups=None):
    """
    Load a single CSV file and apply initial cleaning.
    
    Args:
        filepath: Path to the CSV file.
        label_groups: Optional dict mapping raw labels to grouped labels.
    
    Returns:
        pandas DataFrame with cleaned column names.
    """
    logger.info(f"Loading: {os.path.basename(filepath)}")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Clean and normalize label column
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
        # Consolidate any Web Attack variants (handling en-dash/encoding variations)
        df["Label"] = df["Label"].apply(lambda x: "Web Attack" if "Web Attack" in x else x)
        # Apply label grouping if provided
        if label_groups:
            df["Label"] = df["Label"].map(label_groups).fillna(df["Label"])
    
    rows = len(df)
    logger.info(f"  Loaded {rows:,} rows from {os.path.basename(filepath)}")
    
    return df


def load_all_csvs(config=None, apply_label_groups=True):
    """
    Load and combine all CSV files from the dataset directory.
    
    Args:
        config: Project configuration dict. Loaded from config.yaml if None.
        apply_label_groups: Whether to apply label grouping from config.
    
    Returns:
        Combined pandas DataFrame.
    """
    if config is None:
        config = load_config()
    
    root = get_project_root()
    dataset_path = os.path.join(root, config["dataset"]["path"])
    
    # Get label groups
    label_groups = None
    if apply_label_groups:
        label_groups = config["dataset"].get("label_groups", None)
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    logger.info(f"Found {len(csv_files)} CSV files in {dataset_path}")
    
    # Load and combine
    dfs = []
    for f in csv_files:
        df = load_single_csv(f, label_groups=label_groups)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    
    return combined


def inspect_dataset(df):
    """
    Generate a comprehensive inspection report of the dataset.
    
    Args:
        df: pandas DataFrame to inspect.
    
    Returns:
        Dictionary containing inspection results.
    """
    report = {}
    
    # Basic info
    report["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    report["columns"] = list(df.columns)
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = {
        col: int(count) for col, count in missing.items() if count > 0
    }
    report["total_missing"] = int(missing.sum())
    
    # Infinite values (numeric columns only)
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in num_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = int(inf_count)
    report["infinite_values"] = inf_counts
    
    # Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    report["constant_columns"] = constant_cols
    
    # Label distribution
    if "Label" in df.columns:
        label_dist = df["Label"].value_counts().to_dict()
        report["label_distribution"] = {k: int(v) for k, v in label_dist.items()}
        report["num_classes"] = df["Label"].nunique()
    
    # Numeric statistics summary
    report["numeric_stats"] = {
        "mean_range": [float(df[num_cols].mean().min()), float(df[num_cols].mean().max())],
        "std_range": [float(df[num_cols].std().min()), float(df[num_cols].std().max())],
    }
    
    return report


def print_inspection_report(report):
    """Print a formatted inspection report."""
    print("\n" + "=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    
    print(f"\nShape: {report['shape']['rows']:,} rows × {report['shape']['columns']} columns")
    print(f"Total missing values: {report['total_missing']:,}")
    print(f"Duplicate rows: {report['duplicate_rows']:,}")
    print(f"Constant columns ({len(report['constant_columns'])}): {report['constant_columns']}")
    
    if report.get("infinite_values"):
        print(f"\nInfinite values:")
        for col, count in report["infinite_values"].items():
            print(f"  {col}: {count:,}")
    
    if report.get("label_distribution"):
        print(f"\nLabel distribution ({report['num_classes']} classes):")
        for label, count in sorted(
            report["label_distribution"].items(), key=lambda x: -x[1]
        ):
            pct = count / report["shape"]["rows"] * 100
            # Sanitize label for Windows console encoding
            safe_label = label.encode('ascii', errors='replace').decode('ascii')
            print(f"  {safe_label}: {count:,} ({pct:.2f}%)")
    
    print("=" * 60)
