#!/usr/bin/env python3
"""
Export fused MLX model to MLC (WebLLM) format.

This script:
1. Converts a fused MLX model (safetensors) to MLC format using mlc_llm
2. Generates necessary configuration files for WebLLM deployment
3. Organizes output into a structure compatible with the Edge-LLM framework

Usage:
  python export_mlc.py --input working/fused-model --output working/mlc-model
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str], check: bool = True) -> bool:
    """Run a command and print output."""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    return True

def export_to_mlc(input_path: Path, output_path: Path, quantization: str = "q4f16_1") -> bool:
    """Export model to MLC format using mlc_llm CLI."""
    print(f"\n📦 Exporting to MLC: {input_path} → {output_path} ({quantization})")
    
    try:
        # mlc_llm convert-weight --model <input> --quantization <quant> --output <output>
        cmd = [
            sys.executable, "-m", "mlc_llm", "convert_weight",
            str(input_path),
            "--quantization", quantization,
            "--output", str(output_path)
        ]
        
        if not run_cmd(cmd):
            return False
            
        # mlc_llm gen_config --model <input> --quantization <quant> --output <output>
        # This generates mlc-chat-config.json
        cmd = [
            sys.executable, "-m", "mlc_llm", "gen_config",
            str(input_path),
            "--quantization", quantization,
            "--output", str(output_path),
            "--conv-template", "gemma_instruction"  # Standard Gemma IT template for MLC
        ]
        
        if not run_cmd(cmd):
            return False
            
        print(f"✅ MLC conversion complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ MLC export failed: {e}")
        return False

def compile_model(config_path: Path, output_path: Path) -> bool:
    """Compile the model to WASM/WebGPU format."""
    print(f"\n🔨 Compiling MLC model to WASM: {config_path} → {output_path}")
    
    # Check for emcc (Emscripten)
    if shutil.which("emcc") is None:
        print("⚠️  Warning: emcc (Emscripten) not found in PATH.")
        print("   WASM compilation requires Emscripten. Skip to weight conversion only.")
        return True # Don't fail the whole script, just skip compilation
        
    try:
        # mlc_llm compile <config> --device webgpu --output <output>
        cmd = [
            sys.executable, "-m", "mlc_llm", "compile",
            str(config_path),
            "--device", "webgpu",
            "--output", str(output_path)
        ]
        
        return run_cmd(cmd)
        
    except Exception as e:
        print(f"❌ MLC compilation failed: {e}")
        return False

def finalize_structure(output_path: Path):
    """Finalize the directory structure for deployment."""
    print(f"\n📂 Finalizing MLC structure...")
    
    # Ensure params/ directory exists
    # If mlc_llm convert_weight puts files in the root, we might want to move them
    # to a params/ subdirectory later if WebLLM strictly requires it.
    
    # Copy tokenizer files if missing from output but present in input
    # (Sometimes mlc_llm skips them)
    pass

def main():
    parser = argparse.ArgumentParser(description="Export fused MLX model to MLC format")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input fused model directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output MLC model directory")
    parser.add_argument("--quantization", "-q", type=str, default="q0f16", help="MLC quantization (default: q0f16 for quality)")
    parser.add_argument("--compile", action="store_true", help="Also compile model to WASM (requires toolchain)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input model not found: {args.input}")
        sys.exit(1)
        
    # 1. Convert Weights & Gen Config
    if not export_to_mlc(args.input, args.output, args.quantization):
        sys.exit(1)
        
    # 2. Optional Compilation
    if args.compile:
        config_path = args.output / "mlc-chat-config.json"
        wasm_output = args.output / f"model-webgpu.wasm"
        if not compile_model(config_path, wasm_output):
            print("⚠️  Compilation failed, but weights were converted.")
        
    finalize_structure(args.output)
    
    print("\n✅ MLC Export Complete!")

if __name__ == "__main__":
    main()
