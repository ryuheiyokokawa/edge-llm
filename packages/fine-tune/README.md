# @edge-llm/fine-tune

Fine-tuning system for FunctionGemma models targeting edge deployment.

## Features

- 📝 **Dataset Preparation**: Convert training examples to FunctionGemma format
- 🤖 **Synthetic Generation**: Generate training data using local LLMs (Ollama)
- 🏋️ **LoRA Training**: Parameter-efficient fine-tuning with MLX (Apple Silicon) or CUDA
- 📦 **Model Export**: Export to safetensors, GGUF, or ONNX formats
- 🔧 **CLI Tools**: Complete command-line interface for all operations

> 💡 **AI Assistant?** See [PROMPT_RUN_PIPELINE.md](./PROMPT_RUN_PIPELINE.md) for prompts to run this pipeline.

## Quick Start

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Set up Python environment for training
cd packages/fine-tune
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install mlx mlx-lm  # For Apple Silicon
# OR
# pip install torch transformers peft  # For CUDA
```

### 2. Prepare Training Data

```bash
npx fine-tune dataset \
  --input examples/training-examples.json \
  --output ./training-data \
  --tools examples/tool-definitions/common-tools.json
```

### 3. Train Model

```bash
# Set Python path to use virtual environment
export PYTHON_PATH=python/.venv/bin/python

npx fine-tune train \
  --data ./training-data \
  --output ./training-output \
  --epochs 3 \
  --batch-size 4
```

### 4. Export Model

```bash
npx fine-tune export \
  --model ./training-output/adapters \
  --output ./models \
  --format safetensors
```

## CLI Commands

### `dataset` - Prepare Training Dataset

Convert training examples to FunctionGemma format and split into train/valid/test sets.

```bash
fine-tune dataset [options]

Options:
  -i, --input <file>     Input file with training examples (JSON)
  -o, --output <dir>     Output directory for dataset files
  -t, --tools <file>     Tool definitions file (JSON)
  --split <ratio>        Train/valid/test split ratio (default: 0.8,0.1,0.1)
```

### `train` - Train LoRA Adapter

Fine-tune FunctionGemma using LoRA for parameter-efficient training.

```bash
fine-tune train [options]

Options:
  -d, --data <dir>       Training data directory
  -o, --output <dir>     Output directory for trained adapters
  -m, --model <name>     Base model (default: mlx-community/functiongemma-270m-it-4bit)
  --epochs <n>           Number of training epochs (default: 3)
  --batch-size <n>       Batch size (default: 4)
  --lr <rate>            Learning rate (default: 0.0002)
  --lora-rank <n>        LoRA rank (default: 8)
  --check-env            Check Python environment and exit
```

### `export` - Export Trained Model

Export model to various deployment formats.

```bash
fine-tune export [options]

Options:
  -m, --model <path>     Path to trained adapters or model
  -o, --output <dir>     Output directory
  -f, --format <fmt>     Output format: safetensors, gguf (default: safetensors)
  -q, --quantization     Quantization: 4bit, 8bit, none (default: 4bit)
  --info                 Show model information
```

### `validate` - Validate Model

Check model files and environment configuration.

```bash
fine-tune validate [options]

Options:
  -m, --model <path>     Path to trained model or adapters
  -t, --test <file>      Test dataset file (JSONL) - optional
```

## Python Environment Setup

### Apple Silicon (MLX)

```bash
cd packages/fine-tune
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install mlx mlx-lm
```

Set the Python path when running CLI commands:
```bash
export PYTHON_PATH=python/.venv/bin/python
# OR inline:
PYTHON_PATH=python/.venv/bin/python npx fine-tune train --data ./data ...
```

### Linux/Windows (CUDA)

```bash
cd packages/fine-tune
python3 -m venv python/.venv
source python/.venv/bin/activate  # On Windows: python\.venv\Scripts\activate
pip install torch transformers peft
```

## Training Data Format

### Training Examples (JSON)

```json
{
  "userQuery": "What's 15 times 23?",
  "expectedToolCalls": [
    {
      "name": "calculate",
      "arguments": {
        "expression": "15 * 23"
      }
    }
  ],
  "toolResponse": "345",
  "assistantResponse": "15 times 23 equals 345."
}
```

### Tool Definitions (JSON)

Standard JSON Schema format compatible with `@edge-llm/core`:

```json
{
  "name": "calculate",
  "description": "Evaluate a mathematical expression",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Mathematical expression to evaluate"
      }
    },
    "required": ["expression"]
  }
}
```

## Example Workflow

```bash
# 1. Set up Python environment (one-time)
cd packages/fine-tune
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install mlx mlx-lm

# 2. Prepare dataset
export PYTHON_PATH=python/.venv/bin/python
npx fine-tune dataset \
  -i examples/training-examples.json \
  -o ./data \
  -t examples/tool-definitions/common-tools.json

# 3. Check environment
npx fine-tune train --check-env

# 4. Train model (1 epoch for testing)
npx fine-tune train \
  -d ./data \
  -o ./output \
  --epochs 1 \
  --batch-size 2

# 5. Validate
npx fine-tune validate -m ./output/adapters

# 6. Export if needed
npx fine-tune export \
  -m ./output/adapters \
  -o ./models \
  -f safetensors
```

## TypeScript API

```typescript
import { 
  DatasetBuilder,
  MLXTrainer,
  ModelExporter 
} from "@edge-llm/fine-tune";

// Prepare dataset
const builder = new DatasetBuilder({
  tools: myTools,
  outputDir: "./data",
});
builder.addExamples(examples);
await builder.build();

// Train
const trainer = new MLXTrainer({
  datasetPath: "./data",
  outputPath: "./output",
  epochs: 3,
});

trainer.on("progress", (progress) => {
  console.log(`Epoch ${progress.epoch}, Loss: ${progress.loss}`);
});

const result = await trainer.train();

// Export
const exporter = new ModelExporter();
await exporter.export({
  adapterPath: "./output/adapters",
  outputFormats: ["safetensors"],
  outputDir: "./models",
});
```

## Directory Structure

```
packages/fine-tune/
├── src/              # TypeScript source
├── python/           # Python training scripts
│   ├── .venv/        # Virtual environment (not in git)
│   ├── train_lora.py
│   └── requirements.txt
├── examples/         # Example data
│   ├── training-examples.json
│   └── tool-definitions/
├── training-data/    # Generated datasets (not in git)
└── training-output/  # Trained models (not in git)
```

## Troubleshooting

### MLX not available
```bash
# Ensure you're using the venv Python
which python  # Should point to .venv
pip install mlx mlx-lm
```

### Sequences longer than max_seq_length
Increase `--max-tokens` or pre-split your data:
```bash
fine-tune train --data ./data --max-tokens 1024
```

### Out of memory
Reduce batch size:
```bash
fine-tune train --data ./data --batch-size 1
```

## License

MIT
