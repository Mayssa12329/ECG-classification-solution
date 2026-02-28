"""
PyTorch Dataset for ECG signals with demographics
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class ECGDataset(Dataset):
    """
    Multimodal ECG Dataset combining signals and demographics.
    
    Args:
        X_signals: ECG signals (N, 1000, 12)
        Y_meta_df: Pandas DataFrame with metadata
        superclasses: List of diagnostic classes
        meta_features: List of demographic feature names
    """
    
    def __init__(self, X_signals, Y_meta_df, superclasses, meta_features):
        # Signal: (N, 1000, 12) → transpose to (N, 12, 1000) for Conv1D
        self.signals = torch.tensor(
            X_signals.transpose(0, 2, 1), 
            dtype=torch.float32
        )
        
        # Demographics: (N, n_features)
        self.demographics = torch.tensor(
            Y_meta_df[meta_features].values, 
            dtype=torch.float32
        )
        
        # Labels: (N, n_classes) - multi-label binary matrix
        self.labels = torch.tensor(
            Y_meta_df[superclasses].values, 
            dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            signal: (12, 1000) - ECG signal
            demographics: (n_features,) - demographic features
            label: (n_classes,) - binary labels
        """
        return self.signals[idx], self.demographics[idx], self.labels[idx]