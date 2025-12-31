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
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.tuner import train as lora_train
    from mlx_lm.tuner.trainer import TrainingArgs
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
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
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
        help="LoRA alpha (scaling factor)",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=None,
        help="Number of layers to apply LoRA (default: all layers)",
    )
    
    # Advanced options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory",
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
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
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
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Load model and tokenizer
    print(f"\n🔄 Loading model: {args.model}")
    try:
        model, tokenizer = load(args.model)
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Configure LoRA
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "dropout": 0.0,
        "scale": args.lora_alpha / args.lora_rank,
    }
    
    if args.lora_layers:
        lora_config["num_layers"] = args.lora_layers
    
    print(f"\n🎯 LoRA Configuration:")
    print(f"   Rank: {args.lora_rank}")
    print(f"   Alpha: {args.lora_alpha}")
    print(f"   Scale: {lora_config['scale']:.4f}")
    
    # Training arguments
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.epochs * data_info["train_count"] // args.batch_size,
        val_batches=min(25, data_info["valid_count"]),
        steps_per_report=10,
        steps_per_eval=100,
        steps_per_save=500,
        adapter_path=str(output_path / "adapters"),
        max_seq_length=args.max_tokens,
        grad_checkpoint=args.grad_checkpoint,
    )
    
    print(f"\n📊 Training Configuration:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Total iterations: {training_args.iters}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max sequence length: {args.max_tokens}")
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 Starting LoRA fine-tuning...")
    print("=" * 60 + "\n")
    
    try:
        lora_train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=data_info["train"],
            val_dataset=data_info["valid"],
            lora_config=lora_config,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print("=" * 60)
        print(f"\n📁 Adapters saved to: {output_path / 'adapters'}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
