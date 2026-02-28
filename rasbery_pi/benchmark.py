"""
Raspberry Pi 5 Benchmark for EdgeCardio
Tests ONNX model inference performance
"""

import os
import sys
import time
import platform
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def benchmark_raspberry_pi(
    model_path='models/onnx/lightecgnet.onnx',
    sample_ecg_path='data/sample_ecg.npy',
    n_warmup=10,
    n_runs=100
):
    """
    Benchmark ONNX model on Raspberry Pi 5 (or any CPU).
    
    Args:
        model_path: Path to ONNX model
        sample_ecg_path: Path to sample ECG signal
        n_warmup: Number of warmup iterations
        n_runs: Number of benchmark iterations
    """
    
    print("="*70)
    print("🫀 EdgeCardio - Raspberry Pi 5 Benchmark")
    print("="*70)
    
    # System info
    print(f"\n📌 System Info:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Machine:  {platform.machine()}")
    print(f"   Python:   {platform.python_version()}")
    
    # Fix paths relative to project root
    model_path = project_root / model_path
    sample_ecg_path = project_root / sample_ecg_path
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ ONNX model not found at: {model_path}\n"
            f"   Please run: python main.py --use-kaggle --epochs 50"
        )
    
    if not sample_ecg_path.exists():
        raise FileNotFoundError(
            f"❌ Sample ECG not found at: {sample_ecg_path}\n"
            f"   Please run main.py first to generate sample data"
        )
    
    # Load ONNX model
    print(f"\n📦 Loading ONNX model...")
    print(f"   Path: {model_path}")
    
    session = ort.InferenceSession(
        str(model_path), 
        providers=['CPUExecutionProvider']
    )
    
    print(f"   ✅ Model loaded successfully")
    
    # Get model info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"\n📊 Model Info:")
    print(f"   Inputs:  {input_names}")
    print(f"   Outputs: {output_names}")
    
    # Load sample ECG
    print(f"\n📈 Loading sample ECG...")
    ecg_signal = np.load(sample_ecg_path).astype(np.float32)
    
    # Prepare inputs
    # ECG: (1000, 12) → (1, 12, 1000)
    ecg_input = ecg_signal.T[np.newaxis, :, :]  # (1, 12, 1000)
    
    # Demographics: dummy values (age, sex, weight, nurse, site, device)
    demographics = np.array([[50.0, 1.0, 75.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    print(f"   ECG shape:          {ecg_input.shape}")
    print(f"   Demographics shape: {demographics.shape}")
    
    # Warmup
    print(f"\n🔥 Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        _ = session.run(
            output_names,
            {
                'ecg_signal': ecg_input,
                'demographics': demographics
            }
        )
    print(f"   ✅ Warmup complete")
    
    # Benchmark
    print(f"\n⏱️  Benchmarking ({n_runs} iterations)...")
    latencies = []
    
    for i in range(n_runs):
        start = time.perf_counter()
        outputs = session.run(
            output_names,
            {
                'ecg_signal': ecg_input,
                'demographics': demographics
            }
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{n_runs}")
    
    # Get predictions
    logits = outputs[0][0]  # (5,)
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    
    # Statistics
    latencies = np.array(latencies)
    mean_lat = latencies.mean()
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    min_lat = latencies.min()
    max_lat = latencies.max()
    std_lat = latencies.std()
    
    # Results
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n⏱️  Latency Statistics ({n_runs} runs):")
    print(f"   Mean:   {mean_lat:.2f} ms")
    print(f"   Median: {median_lat:.2f} ms")
    print(f"   Std:    {std_lat:.2f} ms")
    print(f"   Min:    {min_lat:.2f} ms")
    print(f"   Max:    {max_lat:.2f} ms")
    print(f"   P95:    {p95_lat:.2f} ms")
    print(f"   P99:    {p99_lat:.2f} ms")
    
    # Throughput
    throughput = 1000.0 / mean_lat  # samples per second
    print(f"\n🚀 Throughput:")
    print(f"   {throughput:.1f} samples/second")
    
    # Check constraints
    print(f"\n✅ Constraint Validation:")
    if mean_lat < 200:
        print(f"   ✅ Latency < 200 ms (actual: {mean_lat:.2f} ms)")
    else:
        print(f"   ❌ Latency > 200 ms (actual: {mean_lat:.2f} ms)")
    
    # Model size
    model_size_mb = model_path.stat().st_size / (1024**2)
    if model_size_mb < 100:
        print(f"   ✅ Model size < 100 MB (actual: {model_size_mb:.2f} MB)")
    else:
        print(f"   ❌ Model size > 100 MB (actual: {model_size_mb:.2f} MB)")
    
    # Sample prediction
    print(f"\n🔬 Sample Prediction:")
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    print(f"   {'Class':6s} | {'Probability':>12s}")
    print(f"   {'-'*6}-+-{'-'*12}")
    for cls, prob in zip(classes, probs):
        print(f"   {cls:6s} | {prob:>12.4f}")
    
    # Raspberry Pi specific info
    if platform.machine() in ['aarch64', 'armv7l']:
        print(f"\n🍓 Raspberry Pi Detected!")
        print(f"   Architecture: {platform.machine()}")
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'BCM2712' in cpuinfo:
                    print(f"   Model: Raspberry Pi 5")
                elif 'BCM2711' in cpuinfo:
                    print(f"   Model: Raspberry Pi 4")
        except:
            pass
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    
    return {
        'mean_latency_ms': mean_lat,
        'median_latency_ms': median_lat,
        'p95_latency_ms': p95_lat,
        'throughput_sps': throughput,
        'model_size_mb': model_size_mb
    }


def main():
    """Main benchmark script"""
    
    # Default paths
    results = benchmark_raspberry_pi(
        model_path='models/onnx/lightecgnet.onnx',
        sample_ecg_path='data/sample_ecg.npy',
        n_warmup=10,
        n_runs=100
    )
    
    # Save results
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    import json
    results_file = results_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/onnx/lightecgnet.onnx")
    parser.add_argument("--sample_ecg_path", default="data/sample_ecg.npy")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()

    benchmark_raspberry_pi(
        model_path=args.model_path,
        sample_ecg_path=args.sample_ecg_path,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs
    )