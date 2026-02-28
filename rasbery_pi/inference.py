"""
Real-time ECG inference on Raspberry Pi
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ECGInferenceEngine:
    """
    Lightweight inference engine for ECG classification.
    Optimized for Raspberry Pi 5.
    """
    
    def __init__(self, model_path='models/onnx/lightecgnet.onnx'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model (relative to project root)
        """
        # Fix path
        model_path = project_root / model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        print(f"✅ Model loaded from: {model_path}")
    
    def predict(self, ecg_signal, demographics):
        """
        Run inference on ECG signal.
        
        Args:
            ecg_signal: (1000, 12) ECG signal
            demographics: (6,) demographic features [age, sex, weight, nurse, site, device]
            
        Returns:
            Dictionary with predictions
        """
        # Prepare inputs
        ecg_input = ecg_signal.T[np.newaxis, :, :].astype(np.float32)  # (1, 12, 1000)
        demo_input = demographics[np.newaxis, :].astype(np.float32)     # (1, 6)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {
                'ecg_signal': ecg_input,
                'demographics': demo_input
            }
        )
        
        # Parse outputs
        logits = outputs[0][0]  # (5,)
        probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
        
        # Get predictions
        predictions = {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        
        return {
            'probabilities': predictions,
            'primary_diagnosis': max(predictions, key=predictions.get),
            'confidence': max(predictions.values())
        }


def demo_inference():
    """Demo inference on sample ECG"""
    
    print("="*70)
    print("🫀 EdgeCardio - Real-time Inference Demo")
    print("="*70)
    
    # Load sample ECG
    sample_path = project_root / 'data' / 'sample_ecg.npy'
    
    if not sample_path.exists():
        print(f"\n❌ Sample ECG not found: {sample_path}")
        print(f"   Run main.py first to generate sample data")
        return
    
    ecg_signal = np.load(sample_path)
    
    # Dummy demographics (age=50, male, weight=75kg)
    demographics = np.array([50.0, 1.0, 75.0, 0.0, 0.0, 0.0])
    
    # Initialize engine
    engine = ECGInferenceEngine()
    
    # Run inference
    print(f"\n🔬 Running inference...")
    import time
    start = time.perf_counter()
    result = engine.predict(ecg_signal, demographics)
    end = time.perf_counter()
    
    # Display results
    print(f"\n📊 Prediction Results:")
    print(f"   Latency: {(end-start)*1000:.2f} ms")
    print(f"\n   {'Class':6s} | {'Probability':>12s}")
    print(f"   {'-'*6}-+-{'-'*12}")
    for cls, prob in result['probabilities'].items():
        marker = ' ←' if cls == result['primary_diagnosis'] else ''
        print(f"   {cls:6s} | {prob:>12.4f}{marker}")
    
    print(f"\n   Primary Diagnosis: {result['primary_diagnosis']}")
    print(f"   Confidence:        {result['confidence']:.2%}")
    
    print("\n" + "="*70)
    print("✅ Inference Complete")
    print("="*70)


if __name__ == "__main__":
    demo_inference()