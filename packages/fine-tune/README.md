# @edge-llm/fine-tune

Python fine-tuning pipeline for FunctionGemma models, wrapped for npm users.

> 💡 **AI Assistant?** See [PROMPT_RUN_PIPELINE.md](./PROMPT_RUN_PIPELINE.md) for prompts to run this pipeline.

## Quick Start

### 1. Setup Python Environment

```bash
cd packages/fine-tune

# macOS/Linux
npm run setup

# Windows
npm run setup:win
```

This creates a Python virtual environment in `python/.venv/` and installs all dependencies.

### 2. Run the Pipeline

```bash
# Full pipeline (generates data → trains → exports ONNX)
npm run pipeline

# With FP16 quantization (832MB, recommended for browser)
npm run pipeline:quantize
```

The pipeline will:
- Generate training data from `examples/`
- Train LoRA adapters (100 iterations)
- Fuse adapters into base model
- Export to ONNX (optionally quantized)
- Prepare for browser deployment

### 3. Deploy to Example App

```bash
cp -r working/onnx-model-fp16/* ../../examples/app/public/models/custom-functiongemma/
```

## Custom Data

### Using Your Own Tools & Examples

```bash
# Activate venv first
source python/.venv/bin/activate

# Run with custom data
python python/run_pipeline.py \
  --tools-file path/to/my-tools.json \
  --examples-file path/to/my-examples.json \
  --quantize
```

### Tool Definitions Format

```json
[
  {
    "name": "calculate",
    "description": "Evaluate a math expression",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "Math expression"}
      },
      "required": ["expression"]
    }
  }
]
```

### Training Examples Format

```json
[
  {
    "userQuery": "What is 5 * 12?",
    "expectedToolCalls": [{"name": "calculate", "arguments": {"expression": "5 * 12"}}]
  }
]
```

## Pipeline Options

| Option | Default | Description |
|--------|---------|-------------|
| `--iters` | 100 | Training iterations |
| `--batch-size` | 2 | Training batch size |
| `--quantize` | off | Enable ONNX quantization |
| `--skip-training` | off | Skip training step |
| `--skip-onnx` | off | Skip ONNX export |

```bash
# Example: More training iterations
python python/run_pipeline.py --iters 500 --quantize

# Example: Resume with existing training
python python/run_pipeline.py --skip-training --quantize
```

## Outputs

After running the pipeline:

```
working/
├── fused-model/        # MLX model with adapters merged
├── onnx-model/         # ONNX model (FP32, ~1.6GB)
└── onnx-model-fp16/    # ONNX model (FP16, ~832MB) ← Use this!
```

## Quantization Options

| Type | Size | Quality | Browser Compatible |
|------|------|---------|-------------------|
| FP32 | 1.6GB | Best | ⚠️ Too large |
| **FP16** | **832MB** | **Excellent** | **✅ Recommended** |
| INT8 | 417MB | Degrades | ❌ Not recommended |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `npm run setup` fails | Run manually: `python3 -m venv python/.venv && source python/.venv/bin/activate && pip install -r python/requirements.txt` |
| MLX not found | Ensure you're on Apple Silicon. MLX only works on M1/M2/M3 Macs. |
| ONNX export fails | Install: `pip install optimum onnxruntime` |
| Model too large | Use `--quantize` flag (FP16, ~832MB) |
| Browser can't load | Ensure `onnx/` subdirectory structure exists |

## File Structure

```
packages/fine-tune/
├── python/                     # Core Python scripts
│   ├── .venv/                  # Virtual environment (gitignored)
│   ├── requirements.txt        # Python dependencies
│   ├── run_pipeline.py         # Main entry point
│   ├── generate_training_data.py
│   └── export_and_test_onnx.py
├── examples/                   # Example training data
│   ├── tool-definitions/       # Tool JSON schemas
│   └── training-examples.json  # Training examples
├── training-data/              # Generated JSONL (gitignored)
├── training-output/            # Trained adapters (gitignored)
└── working/                    # Pipeline outputs (gitignored)
```

## Requirements

- **Python 3.10+** with pip
- **Apple Silicon Mac** (M1/M2/M3) for MLX training
- **Node.js 18+** for ONNX testing

## License

MIT
