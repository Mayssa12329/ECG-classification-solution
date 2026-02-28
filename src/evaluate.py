"""
Evaluation metrics for multi-label ECG classification
Computes Macro-AUC, per-class AUC, confusion matrices, and optimal thresholds
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    multilabel_confusion_matrix,
    classification_report,
    auc as sk_auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from pathlib import Path


def compute_macro_auc(y_true: np.ndarray, y_pred_logits: np.ndarray) -> float:
    """
    Compute Macro-AUC (main metric for PhysioNet Challenge).
    
    Args:
        y_true: Ground truth labels (N, 5)
        y_pred_logits: Model output logits (N, 5) - NOT probabilities
        
    Returns:
        Macro-averaged AUC across all classes
    """
    # Convert logits to probabilities
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    
    try:
        return roc_auc_score(y_true, y_prob, average='macro')
    except ValueError as e:
        print(f"⚠️  Warning: Could not compute AUC - {e}")
        return 0.0


def compute_per_class_auc(y_true: np.ndarray, y_pred_logits: np.ndarray, 
                          class_names: List[str]) -> Dict[str, float]:
    """
    Compute AUC for each class individually.
    
    Args:
        y_true: Ground truth labels (N, 5)
        y_pred_logits: Model output logits (N, 5)
        class_names: List of class names
        
    Returns:
        Dictionary {class_name: auc_score}
    """
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    
    per_class_auc = {}
    for i, cls in enumerate(class_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            per_class_auc[cls] = auc
        except ValueError:
            per_class_auc[cls] = 0.0
    
    return per_class_auc


def find_optimal_thresholds(y_true: np.ndarray, y_pred_logits: np.ndarray, 
                           class_names: List[str]) -> Dict[str, float]:
    """
    Find optimal classification threshold per class using Youden's J statistic.
    J = max(TPR - FPR)
    
    Args:
        y_true: Ground truth labels (N, 5)
        y_pred_logits: Model output logits (N, 5)
        class_names: List of class names
        
    Returns:
        Dictionary {class_name: optimal_threshold}
    """
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    
    optimal_thresholds = {}
    
    for i, cls in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_prob[:, i])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_thresholds[cls] = float(thresholds[best_idx])
    
    return optimal_thresholds


def evaluate_model(model, dataloader, device: str, class_names: List[str],
                   save_dir: str = "results") -> Dict:
    """
    Complete evaluation of the model on a test set.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        device: 'cuda' or 'cpu'
        class_names: List of diagnostic class names
        save_dir: Directory to save plots
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model = model.to(device)
    
    all_logits = []
    all_labels = []
    
    print("🔬 Running evaluation...")
    
    with torch.no_grad():
        for signals, meta, labels in dataloader:
            signals = signals.to(device)
            meta = meta.to(device)
            
            logits = model(signals, meta)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    
    # Compute metrics
    macro_auc = compute_macro_auc(all_labels, all_logits)
    per_class_auc = compute_per_class_auc(all_labels, all_logits, class_names)
    optimal_thresholds = find_optimal_thresholds(all_labels, all_logits, class_names)
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"\n🎯 Macro-AUC: {macro_auc:.4f}")
    
    print(f"\n📈 Per-Class AUC:")
    for cls, auc in per_class_auc.items():
        print(f"   {cls:6s}: {auc:.4f}")
    
    print(f"\n🎚️  Optimal Thresholds (Youden's J):")
    for cls, thr in optimal_thresholds.items():
        print(f"   {cls:6s}: {thr:.3f}")
    
    # Generate predictions with optimal thresholds
    y_prob = 1.0 / (1.0 + np.exp(-all_logits))
    y_pred = np.zeros_like(all_labels)
    
    for i, cls in enumerate(class_names):
        y_pred[:, i] = (y_prob[:, i] >= optimal_thresholds[cls]).astype(int)
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, y_pred, target_names=class_names, digits=3))
    
    # Save plots
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plot_roc_curves(all_labels, all_logits, class_names, save_path=f"{save_dir}/roc_curves.png")
    plot_confusion_matrices(all_labels, y_pred, class_names, save_path=f"{save_dir}/confusion_matrices.png")
    
    results = {
        'macro_auc': macro_auc,
        'per_class_auc': per_class_auc,
        'optimal_thresholds': optimal_thresholds,
        'y_true': all_labels,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"\n✅ Plots saved to {save_dir}/")
    
    return results


def plot_roc_curves(y_true: np.ndarray, y_pred_logits: np.ndarray, 
                    class_names: List[str], save_path: str = None):
    """Plot ROC curves for all classes"""
    
    y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
    
    colors = ['#2196F3', '#F44336', '#FF9800', '#9C27B0', '#4CAF50']
    
    plt.figure(figsize=(10, 8))
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = sk_auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linewidth=2.5, 
                 label=f'{cls} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - LightECGNet', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ ROC curves saved: {save_path}")
    
    plt.close()


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], save_path: str = None):
    """Plot confusion matrices for each class"""
    
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    colors = ['#2196F3', '#F44336', '#FF9800', '#9C27B0', '#4CAF50']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        tn, fp, fn, tp = mcm[i].ravel()
        cm = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                    cbar=False, linewidths=1, linecolor='white',
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        
        axes[i].set_title(cls, color=color, fontweight='bold', fontsize=13)
        axes[i].set_xlabel('Predicted', fontsize=10)
        axes[i].set_ylabel('True', fontsize=10)
    
    plt.suptitle('Confusion Matrices (Optimal Thresholds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Confusion matrices saved: {save_path}")
    
    plt.close()


