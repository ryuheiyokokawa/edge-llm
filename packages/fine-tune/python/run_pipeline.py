#!/usr/bin/env python3
"""
Complete fine-tuning pipeline for FunctionGemma tool-calling models.

This script runs the full pipeline:
1. Generate training data (FunctionGemma format)
2. Train LoRA adapters (using mlx-lm)
3. Test adapters (pre-fusion)
4. Fuse adapters into base model
5. Test fused model (post-fusion)
6. Export to ONNX
7. Export to MLC (WebLLM)
8. Test ONNX model
9. Quantize ONNX (optional)
10. Test quantized model

Usage:
  # Full pipeline with defaults
  python run_pipeline.py

  # Custom data
  python run_pipeline.py --tools-file my-tools.json --examples-file my-examples.json

  # Skip steps
  python run_pipeline.py --skip-training --skip-onnx

  # Adjust training
  python run_pipeline.py --iters 500 --batch-size 4
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"\n🔧 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return None
    return result


def step_generate_data(args) -> bool:
    """Step 1: Generate training data."""
    print("\n" + "=" * 60)
    print("STEP 1: Generate Training Data")
    print("=" * 60)
    
    cmd = [
        sys.executable, "python/generate_training_data.py",
        "--tools-file", str(args.tools_file),
        "--examples-file", str(args.examples_file),
        "--output-dir", str(args.data_dir),
    ]
    
    return run_cmd(cmd, cwd=args.base_dir) is not None


def step_train(args) -> bool:
    """Step 2: Train LoRA adapters."""
    print("\n" + "=" * 60)
    print("STEP 2: Train LoRA Adapters")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.base_model,
        "--data", str(args.data_dir),
        "--train",
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--steps-per-report", "30",
        "--steps-per-eval", "50",
        "--adapter-path", str(args.adapter_dir),
    ]
    
    return run_cmd(cmd, cwd=args.base_dir) is not None


def step_test_adapters(args) -> bool:
    """Step 3: Test adapters (pre-fusion)."""
    print("\n" + "=" * 60)
    print("STEP 3: Test Adapters (Pre-Fusion)")
    print("=" * 60)
    
    test_code = f'''
from mlx_lm import load, generate

model, tokenizer = load("{args.base_model}", adapter_path="{args.adapter_dir}")

prompt = """<bos><start_of_turn>developer
You are a model that can do function calling.
Must use: <start_function_call>call:name{{arg:<escape>val<escape>}}<end_function_call>
<start_function_declaration>declaration:calculate{{description:<escape>Math<escape>,parameters:{{}},required:[],type:<escape>OBJECT<escape>}}<end_function_declaration>
<end_of_turn>
<start_of_turn>user
What is 5 * 12?<end_of_turn>
<start_of_turn>model
"""

response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
print("Response:", response)

# Validate format
if "<start_function_call>call:" in response and "<escape>" in response:
    print("✅ Adapter test PASSED - correct FunctionGemma format")
    exit(0)
else:
    print("❌ Adapter test FAILED - incorrect format")
    exit(1)
'''
    
    cmd = [sys.executable, "-c", test_code]
    result = run_cmd(cmd, cwd=args.base_dir, check=False)
    return result is not None and result.returncode == 0


def step_fuse(args) -> bool:
    """Step 4: Fuse adapters into base model."""
    print("\n" + "=" * 60)
    print("STEP 4: Fuse Adapters")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", args.base_model,
        "--adapter-path", str(args.adapter_dir),
        "--save-path", str(args.fused_dir),
        "--dequantize",
    ]
    
    return run_cmd(cmd, cwd=args.base_dir) is not None


def step_test_fused(args) -> bool:
    """Step 5: Test fused model (post-fusion)."""
    print("\n" + "=" * 60)
    print("STEP 5: Test Fused Model (Post-Fusion)")
    print("=" * 60)
    
    test_code = f'''
from mlx_lm import load, generate

model, tokenizer = load("{args.fused_dir}")

prompt = """<bos><start_of_turn>developer
You are a model that can do function calling.
Must use: <start_function_call>call:name{{arg:<escape>val<escape>}}<end_function_call>
<start_function_declaration>declaration:calculate{{description:<escape>Math<escape>,parameters:{{}},required:[],type:<escape>OBJECT<escape>}}<end_function_declaration>
<end_of_turn>
<start_of_turn>user
What is 5 * 12?<end_of_turn>
<start_of_turn>model
"""

response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
print("Response:", response)

if "<start_function_call>call:" in response and "<escape>" in response:
    print("✅ Fused model test PASSED")
    exit(0)
else:
    print("❌ Fused model test FAILED")
    exit(1)
'''
    
    cmd = [sys.executable, "-c", test_code]
    result = run_cmd(cmd, cwd=args.base_dir, check=False)
    return result is not None and result.returncode == 0


def step_export_onnx(args) -> bool:
    """Step 6: Export to ONNX."""
    print("\n" + "=" * 60)
    print("STEP 6: Export to ONNX")
    print("=" * 60)
    
    cmd = [
        sys.executable, "python/export_and_test_onnx.py",
        "--input", str(args.fused_dir),
        "--output", str(args.onnx_dir),
    ]
    
    # Add quantization (FP16 by default for quality)
    if args.quantize:
        cmd.extend(["--quantize", "--quantize-type", "fp16"])
    
    # Always prepare for browser deployment
    cmd.append("--prepare-browser")
    
    return run_cmd(cmd, cwd=args.base_dir) is not None


def step_export_mlc(args) -> bool:
    """Step 7: Export to MLC."""
    print("\n" + "=" * 60)
    print("STEP 7: Export to MLC (WebLLM)")
    print("=" * 60)
    
    cmd = [
        sys.executable, "python/export_mlc.py",
        "--input", str(args.fused_dir),
        "--output", str(args.mlc_dir),
    ]
    
    if args.mlc_quant:
        cmd.extend(["--quantization", args.mlc_quant])
    if args.compile_mlc:
        cmd.append("--compile")
        
    return run_cmd(cmd, cwd=args.base_dir) is not None


def main():
    parser = argparse.ArgumentParser(
        description="Complete fine-tuning pipeline for FunctionGemma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input data options
    parser.add_argument(
        "--tools-file", type=Path,
        default=Path("examples/tool-definitions"),
        help="Path to tool definitions (default: examples/tool-definitions/)"
    )
    parser.add_argument(
        "--examples-file", type=Path,
        default=Path("examples/training-examples.json"),
        help="Path to training examples (default: examples/training-examples.json)"
    )
    
    # Model options
    parser.add_argument(
        "--base-model", type=str,
        default="google/functiongemma-270m-it",
        help="Base model to fine-tune (use full precision, not quantized)"
    )
    
    # Training options
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Training iterations (default: 100, optimal for this dataset)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Training batch size (default: 2)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=Path, default=Path("working"),
        help="Base output directory (default: working/)"
    )
    parser.add_argument(
        "--quantize", "-q", action="store_true",
        help="Quantize ONNX model (default: FP16 for quality)"
    )
    
    # Skip options
    parser.add_argument("--skip-data", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-tests", action="store_true", help="Skip all tests")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--skip-mlc", action="store_true", help="Skip MLC export")
    parser.add_argument("--mlc-quant", type=str, default="q0f16", help="MLC quantization (default: q0f16 for quality)")
    parser.add_argument("--compile-mlc", action="store_true", help="Compile MLC model to WASM (requires toolchain/Docker)")
    
    args = parser.parse_args()
    
    # Set up paths
    args.base_dir = Path(__file__).parent.parent
    args.data_dir = args.base_dir / "training-data"
    args.adapter_dir = args.base_dir / "training-output" / "adapters"
    args.fused_dir = args.output_dir / "fused-model"
    args.onnx_dir = args.output_dir / "onnx-model"
    args.mlc_dir = args.output_dir / "mlc-model"
    
    print("=" * 60)
    print("🚀 FunctionGemma Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Tools: {args.tools_file}")
    print(f"Examples: {args.examples_file}")
    print(f"Iterations: {args.iters}")
    print(f"Output: {args.output_dir}")
    
    # Run pipeline
    steps = [
        ("Generate Data", step_generate_data, args.skip_data),
        ("Train Adapters", step_train, args.skip_training),
        ("Test Adapters", step_test_adapters, args.skip_tests or args.skip_training),
        ("Fuse Model", step_fuse, args.skip_training),
        ("Test Fused", step_test_fused, args.skip_tests or args.skip_training),
        ("Export ONNX", step_export_onnx, args.skip_onnx),
        ("Export MLC", step_export_mlc, args.skip_mlc),
    ]
    
    for name, step_fn, skip in steps:
        if skip:
            print(f"\n⏭️  Skipping: {name}")
            continue
        
        if not step_fn(args):
            print(f"\n❌ Pipeline failed at: {name}")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Adapters: {args.adapter_dir}")
    print(f"  Fused model: {args.fused_dir}")
    if not args.skip_onnx:
        print(f"  ONNX model: {args.onnx_dir}")
        if args.quantize:
            print(f"  Quantized: {args.onnx_dir}-int8")
    if not args.skip_mlc:
        print(f"  MLC model: {args.mlc_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
