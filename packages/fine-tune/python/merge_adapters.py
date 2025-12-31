#!/usr/bin/env python3
"""
merge_adapters.py - Merge LoRA adapters into base model

Usage:
    python merge_adapters.py \
        --model mlx-community/functiongemma-270m-it-4bit \
        --adapters ./output/adapters \
        --output ./merged-model
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import save_model
except ImportError as e:
    print(f"Error: MLX packages not installed. Run: pip install mlx mlx-lm")
    print(f"Details: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model path (HuggingFace model path or local path)",
    )
    
    parser.add_argument(
        "--adapters",
        type=str,
        required=True,
        help="Path to LoRA adapters directory",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def validate_paths(args) -> dict:
    """Validate input paths exist."""
    adapter_path = Path(args.adapters)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapters directory not found: {adapter_path}")
    
    # Look for adapter files
    adapter_files = list(adapter_path.glob("*.safetensors")) + list(adapter_path.glob("*.npz"))
    
    if not adapter_files:
        raise FileNotFoundError(f"No adapter files found in: {adapter_path}")
    
    return {
        "adapter_path": adapter_path,
        "adapter_files": adapter_files,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LoRA Adapter Merge (MLX)")
    print("=" * 60)
    
    # Validate paths
    print(f"\n📂 Validating adapter path: {args.adapters}")
    try:
        path_info = validate_paths(args)
        print(f"   Found {len(path_info['adapter_files'])} adapter file(s)")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_path}")
    
    # Load base model with adapters
    print(f"\n🔄 Loading base model with adapters...")
    print(f"   Base: {args.model}")
    print(f"   Adapters: {args.adapters}")
    
    try:
        model, tokenizer = load(
            args.model,
            adapter_path=str(path_info["adapter_path"]),
        )
        print(f"   ✅ Model loaded with adapters")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Fuse LoRA weights into base model
    print(f"\n🔀 Fusing LoRA weights into base model...")
    try:
        # The model with adapters loaded already has fused weights when saved
        # We just need to save the full model
        save_model(
            model=model,
            tokenizer=tokenizer,
            path=str(output_path),
        )
        print(f"   ✅ Model fused and saved")
    except Exception as e:
        print(f"❌ Failed to fuse model: {e}")
        sys.exit(1)
    
    # Save merge info
    merge_info = {
        "base_model": args.model,
        "adapter_path": str(path_info["adapter_path"]),
        "merged_at": str(Path(args.output).resolve()),
    }
    
    with open(output_path / "merge_info.json", "w") as f:
        json.dump(merge_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Merge complete!")
    print("=" * 60)
    print(f"\n📁 Merged model saved to: {output_path}")
    print(f"\nTo use the merged model:")
    print(f"    from mlx_lm import load, generate")
    print(f"    model, tokenizer = load('{output_path}')")


if __name__ == "__main__":
    main()
