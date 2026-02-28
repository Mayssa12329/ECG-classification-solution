"""
LightECGNet - Lightweight multimodal ECG classifier for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSepConv1d(nn.Module):
    """Depthwise-separable convolution (8-9x fewer params than standard Conv1D)"""
    
    def __init__(self, in_ch, out_ch, kernel_size, padding='same'):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class ResBlock1d(nn.Module):
    """Residual block with depthwise-separable convolutions"""
    
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1 = DepthwiseSepConv1d(channels, channels, kernel_size)
        self.conv2 = DepthwiseSepConv1d(channels, channels, kernel_size)
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        return F.relu(self.bn(x + self.conv2(self.conv1(x))))


class LightECGNet(nn.Module):
    """
    Lightweight multimodal ECG classifier optimized for Raspberry Pi 5.
    
    Architecture:
    - CNN branch: Processes 12-lead ECG signal (12, 1000)
    - MLP branch: Processes demographics (6 features)
    - Fusion head: Combines both modalities
    
    Args:
        n_leads: Number of ECG leads (default: 12)
        n_meta: Number of demographic features (default: 6)
        n_classes: Number of output classes (default: 5)
        base_channels: Base channel width (default: 64)
        dropout: Dropout rate (default: 0.3)
    """
    
    def __init__(self, n_leads=12, n_meta=6, n_classes=5, base_channels=64, dropout=0.3):
        super().__init__()
        
        # CNN Branch (Signal)
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 1000 → 500
        )
        
        self.block1 = nn.Sequential(
            ResBlock1d(base_channels, kernel_size=7),
            nn.MaxPool1d(2)  # 500 → 250
        )
        
        self.block2 = nn.Sequential(
            ResBlock1d(base_channels * 2, kernel_size=5),
            nn.MaxPool1d(2)  # 250 → 125
        )
        
        self.block3 = nn.Sequential(
            ResBlock1d(base_channels * 4, kernel_size=3),
            nn.MaxPool1d(2)  # 125 → 62
        )
        
        # Transition layers (channel expansion)
        self.trans1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, 1, bias=False),
            nn.BatchNorm1d(base_channels * 2), nn.ReLU()
        )
        self.trans2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, 1, bias=False),
            nn.BatchNorm1d(base_channels * 4), nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        cnn_out_dim = base_channels * 4
        
        # MLP Branch (Demographics)
        self.meta_mlp = nn.Sequential(
            nn.Linear(n_meta, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion Head
        fusion_in = cnn_out_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, signal, meta):
        """
        Args:
            signal: (B, 12, 1000) - ECG signals
            meta: (B, 6) - Demographics
            
        Returns:
            logits: (B, 5) - Raw logits (apply sigmoid for probabilities)
        """
        # CNN branch
        x = self.stem(signal)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        
        # MLP branch
        m = self.meta_mlp(meta)
        
        # Fusion
        x = torch.cat([x, m], dim=1)
        return self.head(x)


