"""
EdgeCardio - Complete Pipeline
Supports both local PTB-XL dataset and Kaggle download
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import wfdb
import ast
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Import custom modules
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.preprocessing import ECGPreprocessor, prepare_metadata
from src.model import LightECGNet
from src.dataset import ECGDataset
from src.train import train_epoch, validate
from src.evaluate import evaluate_model
from src.export import export_to_onnx


def download_ptbxl_from_kaggle():
    """
    Download PTB-XL dataset from Kaggle using kagglehub.
    
    Returns:
        Path to downloaded dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Installing...")
        os.system("pip install kagglehub")
        import kagglehub
    
    print("="*70)
    print("📥 DOWNLOADING PTB-XL FROM KAGGLE")
    print("="*70)
    print("\n⚠️  First-time download may take 5-10 minutes (~2 GB)")
    print("   Subsequent runs will use cached version\n")
    
    # Download dataset
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    
    print(f"\n✅ Dataset downloaded to: {path}")
    
    # Find the actual PTB-XL directory
    ptbxl_dir = None
    
    # Check common patterns
    possible_paths = [
        path,
        os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"),
        os.path.join(path, "ptb-xl"),
    ]
    
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "ptbxl_database.csv")):
            ptbxl_dir = p
            break
    
    if ptbxl_dir is None:
        # List directory structure to help debug
        print("\n📂 Directory structure:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files)-5} more files')
            if level > 2:  # Don't go too deep
                break
        
        raise FileNotFoundError(
            f"Could not find ptbxl_database.csv in {path}\n"
            f"Please check the directory structure above and update the path."
        )
    
    print(f"   ✓ PTB-XL dataset found at: {ptbxl_dir}")
    
    return ptbxl_dir


def load_ptbxl_data(path: str, sampling_rate: int = 100):
    """
    Load PTB-XL dataset from PhysioNet.
    
    Args:
        path: Path to PTB-XL dataset root
        sampling_rate: 100 or 500 Hz
        
    Returns:
        X: ECG signals (N, 1000, 12)
        Y: Metadata DataFrame
    """
    print("="*70)
    print("📥 LOADING PTB-XL DATASET")
    print("="*70)
    
    # Verify path exists
    metadata_path = os.path.join(path, 'ptbxl_database.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"PTB-XL metadata not found at: {metadata_path}\n"
            f"Please check the dataset path: {path}"
        )
    
    # Load metadata
    print(f"\n📊 Loading metadata from: {metadata_path}")
    Y = pd.read_csv(metadata_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    print(f"   ✓ Loaded {len(Y)} records")
    
    # Load raw ECG signals
    print(f"\n📈 Loading ECG signals (sampling rate: {sampling_rate} Hz)...")
    
    def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_lr, desc="Loading signals")]
        else:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_hr, desc="Loading signals")]
        data = np.array([signal for signal, meta in data])
        return data
    
    X = load_raw_data(Y, sampling_rate, path)
    print(f"   ✓ Loaded signals: {X.shape}")
    
    return X, Y


def prepare_labels(Y: pd.DataFrame, path: str, superclasses: list):
    """
    Prepare diagnostic labels from SCP codes.
    
    Args:
        Y: Metadata DataFrame
        path: Path to PTB-XL dataset
        superclasses: List of diagnostic superclasses
        
    Returns:
        Y_labeled: Filtered DataFrame with binary labels
        mask: Boolean mask for filtering X
    """
    print("\n" + "="*70)
    print("🏷️  PREPARING DIAGNOSTIC LABELS")
    print("="*70)
    
    # Load SCP statements
    scp_path = os.path.join(path, 'scp_statements.csv')
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    print(f"\n📋 Diagnostic SCP codes: {len(agg_df)}")
    
    # Aggregate to superclasses
    def aggregate_superclass(y_dic):
        return list(set(
            agg_df.loc[key].diagnostic_class
            for key in y_dic.keys()
            if key in agg_df.index
        ))
    
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_superclass)
    Y['n_superclass'] = Y['diagnostic_superclass'].apply(len)
    
    # Filter records with no label
    no_label = (Y['n_superclass'] == 0).sum()
    print(f"   ⚠️  Records with no label: {no_label} ({100*no_label/len(Y):.1f}%) → will be excluded")
    
    mask = Y['n_superclass'].values > 0
    Y_labeled = Y[mask].copy()
    
    # Build binary label matrix
    for cls in superclasses:
        Y_labeled[cls] = Y_labeled['diagnostic_superclass'].apply(lambda x: int(cls in x))
    
    print(f"\n✅ Label distribution:")
    for cls in superclasses:
        n = Y_labeled[cls].sum()
        print(f"   {cls:6s}: {n:5d} ({100*n/len(Y_labeled):.1f}%)")
    
    return Y_labeled, mask


def prepare_data_splits(Y: pd.DataFrame, X: np.ndarray, superclasses: list, meta_features: list):
    """
    Prepare train/valid/test splits and clean metadata.
    
    Args:
        Y: Labeled DataFrame
        X: ECG signals
        superclasses: List of diagnostic classes
        meta_features: List of demographic features
        
    Returns:
        Dictionary with train/valid/test data
    """
    print("\n" + "="*70)
    print("✂️  DATA SPLITTING & CLEANING")
    print("="*70)
    
    # Clean metadata
    print("\n🧹 Cleaning metadata...")
    COLS_TO_DROP = [
        'height', 'heart_axis', 'infarction_stadium1', 'infarction_stadium2',
        'baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems',
        'extra_beats', 'pacemaker', 'report', 'recording_date',
        'validated_by', 'second_opinion', 'initial_autogenerated_report',
        'validated_by_human', 'filename_lr', 'filename_hr',
        'scp_codes', 'diagnostic_superclass', 'n_superclass', 'patient_id'
    ]
    COLS_TO_DROP = [c for c in COLS_TO_DROP if c in Y.columns]
    Y_clean = Y.drop(columns=COLS_TO_DROP).copy()
    
    # Imputation
    Y_clean['age'] = Y_clean['age'].fillna(Y_clean['age'].median())
    Y_clean['weight'] = Y_clean['weight'].fillna(Y_clean['weight'].median())
    Y_clean['nurse'] = Y_clean['nurse'].fillna(Y_clean['nurse'].mode()[0])
    Y_clean['site'] = Y_clean['site'].fillna(Y_clean['site'].mode()[0])
    Y_clean['device'] = Y_clean['device'].astype('category').cat.codes
    
    print("   ✓ Metadata cleaned and imputed")
    
    # Split by strat_fold
    def get_split(Y_df, X_arr, folds):
        mask = np.isin(Y_df.strat_fold.values, folds)
        return X_arr[mask], Y_df[mask].copy()
    
    x_train_raw, y_train = get_split(Y_clean, X, list(range(1, 9)))
    x_valid_raw, y_valid = get_split(Y_clean, X, [9])
    x_test_raw, y_test = get_split(Y_clean, X, [10])
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(y_train):5d} ({100*len(y_train)/len(Y_clean):.1f}%)")
    print(f"   Valid: {len(y_valid):5d} ({100*len(y_valid)/len(Y_clean):.1f}%)")
    print(f"   Test:  {len(y_test):5d} ({100*len(y_test)/len(Y_clean):.1f}%)")
    
    # Preprocess signals
    print(f"\n🔧 Preprocessing ECG signals...")
    preprocessor = ECGPreprocessor(lowcut=0.5, highcut=40.0, fs=100, order=4)
    
    x_train = preprocessor.process_batch(x_train_raw)
    x_valid = preprocessor.process_batch(x_valid_raw)
    x_test = preprocessor.process_batch(x_test_raw)
    print("   ✓ Signals preprocessed (bandpass filter + z-score normalization)")
    
    # Normalize metadata
    NUM_META = ['age', 'weight']
    meta_mean = y_train[NUM_META].mean()
    meta_std = y_train[NUM_META].std() + 1e-8
    
    for df in [y_train, y_valid, y_test]:
        df[NUM_META] = (df[NUM_META] - meta_mean) / meta_std
    
    print("   ✓ Metadata normalized")
    
    # Compute class weights
    pos_counts = y_train[superclasses].sum()
    neg_counts = len(y_train) - pos_counts
    pos_weight = (neg_counts / pos_counts).values
    
    print(f"\n⚖️  Class weights (pos_weight):")
    for cls, weight in zip(superclasses, pos_weight):
        print(f"   {cls:6s}: {weight:.3f}")
    
    return {
        'train': {'X': x_train, 'Y': y_train},
        'valid': {'X': x_valid, 'Y': y_valid},
        'test': {'X': x_test, 'Y': y_test},
        'pos_weight': pos_weight,
        'meta_stats': {'mean': meta_mean, 'std': meta_std}
    }


def train_model(data_dict: dict, config: dict, device: str):
    """
    Train LightECGNet model.
    
    Args:
        data_dict: Dictionary with train/valid/test data
        config: Model and training configuration
        device: 'cuda' or 'cpu'
        
    Returns:
        Trained model, best validation AUC
    """
    print("\n" + "="*70)
    print("🚀 TRAINING LIGHTECGNET")
    print("="*70)
    
    # Create datasets
    train_ds = ECGDataset(
        data_dict['train']['X'],
        data_dict['train']['Y'],
        superclasses=config['superclasses'],
        meta_features=config['meta_features']
    )
    valid_ds = ECGDataset(
        data_dict['valid']['X'],
        data_dict['valid']['Y'],
        superclasses=config['superclasses'],
        meta_features=config['meta_features']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = LightECGNet(
        n_leads=12,
        n_meta=6,
        n_classes=5,
        base_channels=config['base_channels'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {total_params:,} parameters (~{total_params*4/1e6:.2f} MB)")
    
    # Loss and optimizer
    pos_weight = torch.tensor(data_dict['pos_weight'], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Valid Loss':>10} | {'Valid AUC':>10}")
    print("-" * 50)
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    os.makedirs('models/checkpoints', exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_auc = validate(model, valid_loader, criterion, device)
        
        scheduler.step()
        
        flag = ''
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'models/checkpoints/best_model.pth')
            flag = ' ✅'
        else:
            patience_counter += 1
        
        print(f"{epoch:>6} | {train_loss:>10.4f} | {valid_loss:>10.4f} | {valid_auc:>10.4f}{flag}")
        
        if patience_counter >= config['patience']:
            print(f"\n⏹  Early stopping (best epoch: {best_epoch}, AUC: {best_auc:.4f})")
            break
    
    print(f"\n🏆 Best Validation AUC: {best_auc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('models/checkpoints/best_model.pth', map_location=device))
    
    return model, best_auc


def main():
    parser = argparse.ArgumentParser(description="EdgeCardio - Complete Pipeline")
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to PTB-XL dataset root (if None, downloads from Kaggle)')
    parser.add_argument('--use-kaggle', action='store_true',
                       help='Force download from Kaggle even if --data-path is provided')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and load existing model')
    
    args = parser.parse_args()
    
    # Configuration
    SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    META_FEATURES = ['age', 'sex', 'weight', 'nurse', 'site', 'device']
    
    config = {
        'superclasses': SUPERCLASSES,
        'meta_features': META_FEATURES,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'patience': 10,
        'base_channels': 64,
        'dropout': 0.3
    }
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\n🖥️  Device: {device}")
    
    # 1. Get dataset path
    if args.use_kaggle or args.data_path is None:
        data_path = download_ptbxl_from_kaggle()
    else:
        data_path = args.data_path
        if not os.path.exists(os.path.join(data_path, 'ptbxl_database.csv')):
            print(f"❌ PTB-XL dataset not found at: {data_path}")
            print(f"   Falling back to Kaggle download...")
            data_path = download_ptbxl_from_kaggle()
    
    # 2. Load data
    X, Y = load_ptbxl_data(data_path, sampling_rate=100)
    
    # 3. Prepare labels
    Y_labeled, mask = prepare_labels(Y, data_path, SUPERCLASSES)
    X_labeled = X[mask]
    
    # 4. Prepare splits
    data_dict = prepare_data_splits(Y_labeled, X_labeled, SUPERCLASSES, META_FEATURES)
    
    # 5. Train model
    if not args.skip_training:
        model, best_valid_auc = train_model(data_dict, config, device)
    else:
        print("\n⏭️  Skipping training, loading existing model...")
        model = LightECGNet(n_leads=12, n_meta=6, n_classes=5,
                           base_channels=64, dropout=0.3)
        model.load_state_dict(torch.load('models/checkpoints/best_model.pth', map_location=device))
        model = model.to(device)
        best_valid_auc = None
    
    # 6. Evaluate on test set
    print("\n" + "="*70)
    print("📊 EVALUATION ON TEST SET")
    print("="*70)
    
    test_ds = ECGDataset(
        data_dict['test']['X'],
        data_dict['test']['Y'],
        superclasses=SUPERCLASSES,
        meta_features=META_FEATURES
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    os.makedirs('results', exist_ok=True)
    results = evaluate_model(model, test_loader, device, SUPERCLASSES, save_dir='results')
    
    print(f"\n🎯 Test Macro-AUC: {results['macro_auc']:.4f}")
    
    # 7. Export to ONNX
    print("\n" + "="*70)
    print("📦 EXPORTING TO ONNX")
    print("="*70)
    
    # Save complete checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_leads': 12, 'n_meta': 6, 'n_classes': 5,
            'base_channels': 64, 'dropout': 0.3
        },
        'superclasses': SUPERCLASSES,
        'meta_features': META_FEATURES,
        'optimal_thresholds': results['optimal_thresholds'],
        'meta_mean': data_dict['meta_stats']['mean'].to_dict(),
        'meta_std': data_dict['meta_stats']['std'].to_dict(),
        'test_macro_auc': results['macro_auc'],
        'best_valid_auc': best_valid_auc,
    }, 'models/checkpoints/lightecgnet_final.pth')
    
    print("   ✓ Complete checkpoint saved: models/checkpoints/lightecgnet_final.pth")
    
    # Export to ONNX
    os.makedirs('models/onnx', exist_ok=True)
    onnx_path = export_to_onnx(
        checkpoint_path='models/checkpoints/best_model.pth',
        output_path='models/onnx/lightecgnet.onnx',
        opset_version=14,
        verify=True
    )
    
    # Save a sample ECG for testing
    os.makedirs('data', exist_ok=True)
    sample_ecg = data_dict['test']['X'][0]
    np.save('data/sample_ecg.npy', sample_ecg)
    print("\n   ✓ Sample ECG saved: data/sample_ecg.npy")
    
    # Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📌 Summary:")
    print(f"   - Dataset source:     {'Kaggle' if args.use_kaggle or args.data_path is None else 'Local'}")
    print(f"   - Training records:   {len(data_dict['train']['Y'])}")
    print(f"   - Validation records: {len(data_dict['valid']['Y'])}")
    print(f"   - Test records:       {len(data_dict['test']['Y'])}")
    if best_valid_auc:
        print(f"   - Best Valid AUC:     {best_valid_auc:.4f}")
    print(f"   - Test Macro-AUC:     {results['macro_auc']:.4f}")
    print(f"   - Model saved:        models/checkpoints/best_model.pth")
    print(f"   - ONNX model:         {onnx_path}")
    print(f"\n🎬 Next step: Run Raspberry Pi benchmark")
    print(f"   python raspberry_pi/benchmark.py")


if __name__ == "__main__":
    main()