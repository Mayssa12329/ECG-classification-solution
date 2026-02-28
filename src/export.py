"""
Export trained PyTorch model to ONNX format for edge deployment
Optimized for Raspberry Pi 5 inference with ONNX Runtime
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import yaml
from .model import LightECGNet


def export_to_onnx(checkpoint_path: str, 
                   output_path: str = "models/onnx/lightecgnet.onnx",
                   opset_version: int = 14,
                   verify: bool = True):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (14 recommended for Pi)
        verify: Whether to verify ONNX model after export
    """
    
    print("="*70)
    print("📦 EXPORTING TO ONNX")
    print("="*70)
    
    # Load model config
    config_path = "config/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = LightECGNet(**config['model'])
    model.eval()
    model.cpu()
    
    # Load trained weights
    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"   ✓ Weights loaded")
    
    # Create dummy inputs
    batch_size = 1
    dummy_signal = torch.randn(batch_size, 12, 1000)
    dummy_meta = torch.randn(batch_size, 6)
    
    # Export to ONNX
    print(f"\n🔄 Exporting to ONNX (opset {opset_version})...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_signal, dummy_meta),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimization
        input_names=['ecg_signal', 'demographics'],
        output_names=['logits'],
        dynamic_axes={
            'ecg_signal': {0: 'batch_size'},
            'demographics': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    model_size_mb = Path(output_path).stat().st_size / (1024**2)
    
    print(f"   ✓ ONNX model saved: {output_path}")
    print(f"   ✓ Model size: {model_size_mb:.2f} MB")
    
    # Verify size constraint
    if model_size_mb > 100:
        print(f"   ❌ WARNING: Model size > 100 MB (PhysioNet constraint)")
    else:
        print(f"   ✅ Model size < 100 MB (constraint satisfied)")
    
    # Verify ONNX model
    if verify:
        print(f"\n🔍 Verifying ONNX model...")
        verify_onnx_model(output_path, dummy_signal, dummy_meta, model)
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETE")
    print("="*70)
    
    return output_path


def verify_onnx_model(onnx_path: str, 
                     dummy_signal: torch.Tensor, 
                     dummy_meta: torch.Tensor,
                     pytorch_model: torch.nn.Module):
    """
    Verify ONNX model produces same output as PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        dummy_signal: Test ECG signal
        dummy_meta: Test demographics
        pytorch_model: Original PyTorch model
    """
    
    # Check ONNX validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ ONNX model is valid")
    
    # Load ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_signal, dummy_meta).numpy()
    
    # ONNX Runtime inference
    onnx_inputs = {
        'ecg_signal': dummy_signal.numpy(),
        'demographics': dummy_meta.numpy()
    }
    onnx_output = ort_session.run(None, onnx_inputs)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"   ✓ Output verification:")
    print(f"     - Max difference:  {max_diff:.6f}")
    print(f"     - Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print(f"   ✅ ONNX and PyTorch outputs match!")
    else:
        print(f"   ⚠️  Warning: Large difference between ONNX and PyTorch")
    
    # Benchmark latency
    print(f"\n⏱️  Latency benchmark (CPU)...")
    
    import time
    n_runs = 100
    latencies = []
    
    # Warm-up
    for _ in range(10):
        _ = ort_session.run(None, onnx_inputs)
    
    # Measure
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, onnx_inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    
    latencies = np.array(latencies)
    
    print(f"   ✓ Latency (batch=1, {n_runs} runs):")
    print(f"     - Mean:   {latencies.mean():.2f} ms")
    print(f"     - Median: {np.median(latencies):.2f} ms")
    print(f"     - P95:    {np.percentile(latencies, 95):.2f} ms")
    
    if latencies.mean() < 200:
        print(f"   ✅ Latency < 200 ms (constraint satisfied)")
    else:
        print(f"   ❌ Latency > 200 ms (optimization needed)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='models/onnx/lightecgnet.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify
    )