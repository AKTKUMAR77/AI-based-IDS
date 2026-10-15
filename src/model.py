"""
DL-IDS Model Architecture.

Implements the CNN-LSTM hybrid model from the paper:
"DL-IDS: Extracting Features Using CNN-LSTM Hybrid Network for Intrusion Detection System"
Sun et al., 2020.

ADAPTATION NOTE:
The original paper processes raw packet bytes (8 packets × 40 bytes each),
converting them to traffic images. Our dataset provides flow-level tabular
features (78 numeric features per flow record) from CICFlowMeter.

Our adaptation:
- Reshape the 1D feature vector into a pseudo-2D representation for CNN processing
- CNN extracts local feature patterns (analogous to spatial features in the paper)
- LSTM processes sequential patterns across feature groups
- The architecture preserves the paper's key design choices:
  * Two conv+pool combinations with sigmoid and ReLU activations
  * Two LSTM layers with 85 hidden units
  * Class-weighted cross-entropy loss
  * RMSprop optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DLIDS(nn.Module):
    """
    CNN-LSTM Hybrid Model for Intrusion Detection.
    
    Architecture (adapted from paper):
    1. Input: 1D feature vector → reshaped to 2D for CNN
    2. CNN Block 1: Conv1D + Sigmoid + MaxPool (local feature extraction)
    3. CNN Block 2: Conv1D + ReLU + MaxPool (broader pattern extraction)
    4. LSTM: 2 layers with 85 hidden units (sequential pattern learning)
    5. FC: Fully connected classification head
    6. Output: Softmax over num_classes
    """

    def __init__(self, num_features, num_classes, config=None):
        super(DLIDS, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Default hyperparameters from config/paper
        if config is None:
            config = {}
        
        cnn_cfg = config.get("cnn", {})
        lstm_cfg = config.get("lstm", {})
        fc_cfg = config.get("fc", {})
        
        conv1_filters = cnn_cfg.get("conv1_filters", 32)
        conv2_filters = cnn_cfg.get("conv2_filters", 64)
        conv1_kernel = cnn_cfg.get("conv1_kernel", 3)
        conv2_kernel = cnn_cfg.get("conv2_kernel", 3)
        pool_size = cnn_cfg.get("pool_size", 2)
        cnn_dropout = cnn_cfg.get("dropout_cnn", 0.3)
        
        lstm_hidden = lstm_cfg.get("hidden_size", 85)
        lstm_layers = lstm_cfg.get("num_layers", 2)
        lstm_dropout = lstm_cfg.get("dropout_lstm", 0.3)
        
        fc_hidden = fc_cfg.get("hidden_size", 128)
        fc_dropout = fc_cfg.get("dropout_fc", 0.5)
        
        # ========== CNN Block ==========
        # We treat the feature vector as a 1D signal with 1 channel
        # Conv1D processes the feature vector to extract local patterns
        
        # Combination I: Conv1D + Sigmoid + MaxPool (paper: small kernel for local features)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=conv1_filters,
            kernel_size=conv1_kernel, padding=conv1_kernel // 2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        self.cnn_dropout1 = nn.Dropout(cnn_dropout)
        
        # Combination II: Conv1D + ReLU + MaxPool (paper: larger kernel for broader patterns)
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters, out_channels=conv2_filters,
            kernel_size=conv2_kernel, padding=conv2_kernel // 2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)
        self.cnn_dropout2 = nn.Dropout(cnn_dropout)
        
        # Calculate the output size after CNN
        # After conv1 (same padding) -> pool1 (÷2) -> conv2 (same padding) -> pool2 (÷2)
        cnn_out_len = num_features // (pool_size * pool_size)
        if cnn_out_len < 1:
            cnn_out_len = 1
        
        # ========== LSTM Block ==========
        # Reshape CNN output to sequence format for LSTM
        # Each position in the CNN output becomes a timestep
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )
        
        # ========== Classification Head ==========
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.bn_fc = nn.BatchNorm1d(fc_hidden)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, num_features)
        
        Returns:
            Tensor of shape (batch_size, num_classes) - logits
        """
        batch_size = x.size(0)
        
        # Reshape to (batch, 1, features) for Conv1D
        x = x.unsqueeze(1)  # (batch, 1, num_features)
        
        # CNN Block 1: Conv + Sigmoid + Pool (paper specification)
        x = self.conv1(x)           # (batch, conv1_filters, num_features)
        x = self.bn1(x)
        x = torch.sigmoid(x)        # Paper uses sigmoid after first conv
        x = self.pool1(x)           # (batch, conv1_filters, num_features/2)
        x = self.cnn_dropout1(x)
        
        # CNN Block 2: Conv + ReLU + Pool (paper specification)
        x = self.conv2(x)           # (batch, conv2_filters, num_features/2)
        x = self.bn2(x)
        x = F.relu(x)               # Paper uses ReLU after second conv
        x = self.pool2(x)           # (batch, conv2_filters, num_features/4)
        x = self.cnn_dropout2(x)
        
        # Prepare for LSTM: (batch, sequence_len, features)
        # Transpose from (batch, channels, seq_len) to (batch, seq_len, channels)
        x = x.permute(0, 2, 1)     # (batch, seq_len, conv2_filters)
        
        # LSTM Block
        # Paper: Layer 6 (linear activation), Layer 7 (ReLU activation)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        x = h_n[-1]                 # (batch, lstm_hidden)
        
        # Classification Head
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)            # (batch, num_classes)
        
        return x

    def predict_proba(self, x):
        """Get prediction probabilities using softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class CNNOnly(nn.Module):
    """
    CNN-only baseline model (for comparison as in the paper).
    """

    def __init__(self, num_features, num_classes, config=None):
        super(CNNOnly, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        if config is None:
            config = {}
        cnn_cfg = config.get("cnn", {})
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LSTMOnly(nn.Module):
    """
    LSTM-only baseline model (for comparison as in the paper).
    """

    def __init__(self, num_features, num_classes, config=None):
        super(LSTMOnly, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Treat each feature as a timestep with 1 feature
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=85,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        
        self.fc1 = nn.Linear(85, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape to (batch, features, 1) - each feature is a timestep
        x = x.unsqueeze(2)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(model_name, num_features, num_classes, config=None):
    """
    Factory function to create a model by name.
    
    Args:
        model_name: One of 'DL-IDS', 'CNN', 'LSTM'
        num_features: Number of input features
        num_classes: Number of output classes
        config: Model configuration dict
    
    Returns:
        PyTorch model instance
    """
    models = {
        "DL-IDS": DLIDS,
        "CNN": CNNOnly,
        "LSTM": LSTMOnly,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](num_features, num_classes, config)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
