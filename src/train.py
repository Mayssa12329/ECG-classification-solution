"""
Training script for LightECGNet
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import yaml
import time
from typing import Dict

from .model import LightECGNet
from .dataset import ECGDataset
from .evaluate import compute_macro_auc


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for signals, meta, labels in loader:
        signals = signals.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(signals, meta)
        loss = criterion(logits, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    for signals, meta, labels in loader:
        signals = signals.to(device)
        meta = meta.to(device)
        labels = labels.to(device)
        
        logits = model(signals, meta)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * len(labels)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    
    avg_loss = total_loss / len(loader.dataset)
    auc = compute_macro_auc(all_labels, all_logits)
    
    return avg_loss, auc


def train(config_path: str = "config/config.yaml"):
    """Main training function"""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    
    # Load data (assumes preprocessed data exists)
    from torch.utils.data import DataLoader
    import numpy as np
    import pandas as pd
    
    # TODO: Adjust paths
    x_train = np.load("data/processed/x_train.npy")
    y_train = pd.read_csv("data/processed/y_train.csv")
    x_valid = np.load("data/processed/x_valid.npy")
    y_valid = pd.read_csv("data/processed/y_valid.csv")
    
    train_ds = ECGDataset(x_train, y_train)
    valid_ds = ECGDataset(x_valid, y_valid)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model = LightECGNet(**config['model']).to(device)
    
    # Loss (with class weights)
    pos_weight = torch.tensor(config['pos_weight']).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_auc = validate(model, valid_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid AUC: {valid_auc:.4f} | "
              f"Time: {time.time()-t0:.0f}s")
        
        # Save best
        if valid_auc > best_auc:
            best_auc = valid_auc
            patience_counter = 0
            torch.save(model.state_dict(), "models/checkpoints/best_model.pth")
            print(f"  ✅ Saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"⏹ Early stopping (best AUC: {best_auc:.4f})")
            break
    
    print(f"\n🏆 Training complete! Best Val AUC: {best_auc:.4f}")


