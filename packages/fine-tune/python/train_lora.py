#!/usr/bin/env python3
"""
train_lora.py - LoRA fine-tuning script for FunctionGemma using MLX

Usage:
    python train_lora.py \
        --model mlx-community/functiongemma-270m-it-4bit \
        --data ./training-data \
        --output ./output \
        --epochs 3 \
        --batch-size 4 \
        --lora-rank 8 \
        --learning-rate 2e-4
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load
    import subprocess
except ImportError as e:
    print(f"Error: MLX packages not installed. Run: pip install mlx mlx-lm")
    print(f"Details: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune FunctionGemma with LoRA using MLX"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/functiongemma-270m-it-4bit",
        help="Base model to fine-tune (HuggingFace model path or local path)",
    )
    
    # Data configuration
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data directory (must contain train.jsonl, valid.jsonl)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained adapters",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (scaling factor) - Note: MLX-LM calculates this internally",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to apply LoRA (default: 16)",
    )
    
    # Advanced options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def validate_data_dir(data_dir: str) -> dict:
    """Validate that required data files exist."""
    data_path = Path(data_dir)
    
    train_file = data_path / "train.jsonl"
    valid_file = data_path / "valid.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")
    
    # Count examples
    train_count = sum(1 for _ in open(train_file))
    valid_count = sum(1 for _ in open(valid_file))
    
    return {
        "train": str(train_file),
        "valid": str(valid_file),
        "train_count": train_count,
        "valid_count": valid_count,
    }


def create_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_training_config(output_dir: Path, args: argparse.Namespace):
    """Save training configuration for reproducibility."""
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_path


def main():
    args = parse_args()
    
    print("=" * 60)
    print("FunctionGemma LoRA Fine-Tuning (MLX)")
    print("=" * 60)
    
    # Validate data directory
    print(f"\n📂 Validating data directory: {args.data}")
    try:
        data_info = validate_data_dir(args.data)
        print(f"   Train examples: {data_info['train_count']}")
        print(f"   Valid examples: {data_info['valid_count']}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = create_output_dir(args.output)
    print(f"\n📁 Output directory: {output_path}")
    
    # Save config
    config_path = save_training_config(output_path, args)
    print(f"   Config saved: {config_path}")
    
    # Calculate iterations from epochs
    iters = args.epochs * data_info["train_count"] // args.batch_size
    
    print(f"\n🎯 LoRA Configuration:")
    print(f"   Rank: {args.lora_rank}")
    print(f"   Layers: {args.lora_layers}")
    
    print(f"\n📊 Training Configuration:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Total iterations: {iters}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max sequence length: {args.max_tokens}")
    
    # Build command for mlx_lm.lora
    adapter_path = str(output_path / "adapters")
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", args.data,
        "--iters", str(iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", adapter_path,
        "--num-layers", str(args.lora_layers),
        "--max-seq-length", str(args.max_tokens),
        "--seed", str(args.seed),
        "--steps-per-report", "10",
        "--steps-per-eval", str(iters // 3) if iters > 3 else "1",
    ]
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 Starting LoRA fine-tuning...")
    print("=" * 60 + "\n")
    
    try:
        # Run mlx_lm.lora as subprocess
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ Training complete!")
            print("=" * 60)
            print(f"\n📁 Adapters saved to: {adapter_path}")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
