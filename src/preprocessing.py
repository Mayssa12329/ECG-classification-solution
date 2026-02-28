"""
ECG Signal Preprocessing Pipeline
Bandpass filtering + Z-score normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple

class ECGPreprocessor:
    """
    Preprocessor for ECG signals (PTB-XL dataset).
    
    Pipeline:
    1. Bandpass filter (0.5-40 Hz) - removes baseline drift & high-freq noise
    2. Z-score normalization per lead per sample
    """
    
    def __init__(self, lowcut: float = 0.5, highcut: float = 40.0, 
                 fs: int = 100, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        
        Args:
            signal: (1000, 12) - single ECG record
            
        Returns:
            Filtered signal (1000, 12)
        """
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=0)
    
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Z-score normalization per lead.
        
        Args:
            signal: (1000, 12)
            
        Returns:
            Normalized signal (1000, 12)
        """
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-8
        return (signal - mean) / std
    
    def process_batch(self, X: np.ndarray, apply_filter: bool = True) -> np.ndarray:
        """
        Process batch of ECG signals.
        
        Args:
            X: (N, 1000, 12) - batch of ECG records
            apply_filter: whether to apply bandpass filter
            
        Returns:
            Processed batch (N, 1000, 12) as float32
        """
        X_out = np.empty_like(X, dtype=np.float32)
        
        for i in range(len(X)):
            sig = X[i].copy().astype(np.float64)
            
            if apply_filter:
                sig = self.bandpass_filter(sig)
            
            sig = self.normalize(sig)
            X_out[i] = sig.astype(np.float32)
        
        return X_out


def prepare_metadata(df, features: list) -> Tuple[np.ndarray, dict]:
    """
    Prepare metadata features (imputation + standardization).
    
    Args:
        df: DataFrame with metadata
        features: list of feature names
        
    Returns:
        Processed metadata (N, n_features), normalization stats
    """
    import pandas as pd
    
    df_clean = df.copy()
    
    # Imputation
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['weight'] = df_clean['weight'].fillna(df_clean['weight'].median())
    df_clean['nurse'] = df_clean['nurse'].fillna(df_clean['nurse'].mode()[0])
    df_clean['site'] = df_clean['site'].fillna(df_clean['site'].mode()[0])
    df_clean['device'] = df_clean['device'].astype('category').cat.codes
    
    # Fit normalization on numeric features
    numeric_features = ['age', 'weight']
    meta_mean = df_clean[numeric_features].mean()
    meta_std = df_clean[numeric_features].std() + 1e-8
    
    df_clean[numeric_features] = (df_clean[numeric_features] - meta_mean) / meta_std
    
    stats = {
        'mean': meta_mean.to_dict(),
        'std': meta_std.to_dict()
    }
    
    return df_clean[features].values.astype(np.float32), stats

