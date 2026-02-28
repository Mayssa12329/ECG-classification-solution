import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model (dynamic INT8)")
    parser.add_argument("--input", default="models/onnx/lightecgnet.onnx", help="Input ONNX model path")
    parser.add_argument("--output", default="models/onnx/lightecgnet_int8.onnx", help="Output quantized ONNX path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError(
            "onnxruntime quantization is not available. "
            "Install 'onnxruntime-tools' or ensure you have a full onnxruntime package.\n"
            "Try: pip install onnxruntime-tools"
        ) from e

    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8
    )

    print(f"✅ Quantized model saved to: {out_path}")
    print(f"   Input size:  {in_path.stat().st_size / (1024**2):.2f} MB")
    print(f"   Output size: {out_path.stat().st_size / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()