#!/usr/bin/env python3
"""
fuse_and_export_onnx.py - Merge MLX LoRA adapters and export to ONNX

This script:
1. Fuses MLX LoRA adapters with the base FunctionGemma model
2. Saves the merged model in Hugging Face format
3. Exports the merged model to ONNX using optimum-cli

Usage:
    python fuse_and_export_onnx.py \
        --adapter ./training-output/adapters \
        --base-model mlx-community/functiongemma-270m-it-4bit \
        --output ./merged-model \
        --onnx-output ./onnx-model

Requirements:
    pip install mlx mlx-lm transformers optimum[exporters,onnxruntime]
"""

import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse MLX LoRA adapters and export to ONNX"
    )
    
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to MLX LoRA adapter directory (contains adapters.safetensors)",
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="mlx-community/functiongemma-270m-it-4bit",
        help="Base model to merge adapters into",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for fused model (HF format)",
    )
    
    parser.add_argument(
        "--onnx-output",
        type=str,
        help="Optional: Output directory for ONNX model. If not specified, uses {output}_onnx",
    )
    
    parser.add_argument(
        "--quantize",
        choices=["int8", "int4", "fp16", "none"],
        default="none",
        help="ONNX quantization level",
    )
    
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export, only fuse adapters",
    )
    
    return parser.parse_args()


def fuse_adapters(adapter_path: str, base_model: str, output_path: str):
    """
    Fuse MLX LoRA adapters with base model using mlx_lm.fuse
    """
    print("=" * 60)
    print("Step 1: Fusing LoRA adapters with base model")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_path}")
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--dequantize",  # Convert from 4-bit back to fp16 for ONNX export
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        print(f"\n✅ Fused model saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Fusion failed: {e}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return False


def export_to_onnx(model_path: str, onnx_output: str, quantize: str):
    """
    Export Hugging Face model to ONNX using optimum Python API
    """
    print("\n" + "=" * 60)
    print("Step 2: Exporting to ONNX")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"ONNX output: {onnx_output}")
    print(f"Quantization: {quantize}")
    
    try:
        from optimum.exporters.onnx import main_export
        from pathlib import Path
        
        # Create output directory
        Path(onnx_output).mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting model to ONNX...")
        
        # Use the Python API instead of CLI
        # main_export expects:
        # - model_name_or_path: str or Path
        # - output: str or Path  
        # - task:  str (optional, auto-detected)
        # - opset: int (optional)
        # - device: str (optional)
        main_export(
            model_name_or_path=model_path,
            output=onnx_output,
            task="text-generation-with-past",  # Optimized for chat
        )
        
        print(f"\n✅ ONNX model saved to: {onnx_output}")
        return True
        
    except ImportError as e:
        print(f"\n❌ ONNX export failed: {e}")
        print("\n⚠️  optimum not installed correctly. Install with:")
        print("   pip install optimum onnxruntime")
        return False
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MLX LoRA Adapter Fusion and ONNX Export")
    print("=" * 60)
    
    # Validate paths
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"❌ Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)
    
    # Check for adapters.safetensors
    adapter_file = adapter_path / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"❌ Error: adapters.safetensors not found in: {adapter_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fuse adapters
    success = fuse_adapters(
        str(adapter_path),
        args.base_model,
        str(output_path)
    )
    
    if not success:
        print("\n❌ Fusion failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Export to ONNX (optional)
    if not args.skip_onnx:
        onnx_output = args.onnx_output or f"{args.output}_onnx"
        onnx_output_path = Path(onnx_output)
        onnx_output_path.mkdir(parents=True, exist_ok=True)
        
        success = export_to_onnx(
            str(output_path),
            str(onnx_output_path),
            args.quantize
        )
        
        if not success:
            print("\n⚠️  ONNX export failed, but fused model is still available")
            print(f"   Fused model: {output_path}")
            sys.exit(1)
    
    # Success!
    print("\n" + "=" * 60)
    print("✅ All steps completed successfully!")
    print("=" * 60)
    print(f"\nFused model:  {output_path}")
    if not args.skip_onnx:
        print(f"ONNX model:   {onnx_output_path}")
    
    print("\nNext steps:")
    if not args.skip_onnx:
        print("  1. Test ONNX model with Transformers.js")
        print("  2. Upload to Hugging Face Hub (optional)")
        print("  3. Use in your application via config.models.transformers")
    else:
        print("  1. Export to ONNX using this script (without --skip-onnx)")
        print("  2. Or use the fused model with MLX directly")


if __name__ == "__main__":
    main()
