# WebGPU Enablement with Transformers.js

This guide documents the complete path from fine-tuning a model to deploying it with WebGPU acceleration in the browser using Transformers.js.

## Overview

Edge-LLM uses Transformers.js for browser-based inference with WebGPU acceleration. This provides near-native performance for running LLMs locally without a GPU server.

### Architecture

```
Fine-tuning → ONNX Export → Q4 Quantization → Browser Runtime
    ↓              ↓              ↓                ↓
 MLX-LM       optimum-cli    MatMul4Bits     Transformers.js
                               Quantizer         + WebGPU
```

---

## Prerequisites

- Python 3.10+ with virtual environment
- Node.js 18+
- Browser with WebGPU support (Chrome 113+, Edge 113+)
- ~16GB RAM for model conversion

---

## Step 1: Fine-Tuning (Optional)

If starting from a pre-trained model, skip to Step 2.

### Install Dependencies

```bash
cd packages/fine-tune
python -m venv python/.venv
source python/.venv/bin/activate
pip install -r python/requirements.txt
```

### Run Fine-Tuning

```bash
python -m mlx_lm lora \
  --model google/functiongemma-270m-it \
  --data working/training-data \
  --train \
  --fine-tune-type full \
  --batch-size 2 \
  --iters 100 \
  --adapter-path working/adapters
```

### Fuse Adapters

```bash
python -m mlx_lm fuse \
  --model google/functiongemma-270m-it \
  --adapter-path working/adapters \
  --save-path working/fused-model \
  --dequantize
```

---

## Step 2: Export to ONNX

### Option A: Automated Pipeline (Recommended)

The `packages/fine-tune` package includes an automated pipeline that handles export and quantization:

```bash
cd packages/fine-tune
npm run setup          # One-time setup
npm run pipeline:quantize  # Exports with Q4-WebGPU quantization
```

This automatically:
- Exports to ONNX format
- Applies Q4-WebGPU quantization (~800MB output)
- Prepares model for browser deployment (onnx/ subdirectory, chat_template)

### Option B: Manual Export

Clone the Transformers.js repo (one-time setup) and use its conversion scripts:

```bash
# Clone transformers.js (do this once, outside your project)
git clone https://github.com/huggingface/transformers.js.git ~/transformers.js
cd ~/transformers.js
pip install -r scripts/requirements.txt

# Run the conversion
python -m scripts.convert \
  --model_id google/functiongemma-270m-it \
  --output_parent_dir ~/my-onnx-output \
  --task text-generation
```

For fine-tuned models, replace `google/functiongemma-270m-it` with your fused model path.

---

## Step 3: Q4 Quantization

Apply 4-bit quantization using MatMul4BitsQuantizer for WebGPU performance:

```python
import onnx
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
from optimum.onnx.graph_transformations import check_and_save_model

# Load the ONNX model
model = onnx.load('working/onnx-output/model.onnx')

# Apply q4 quantization
quantizer = MatMul4BitsQuantizer(
    model=model,
    block_size=32,
    is_symmetric=True,
    accuracy_level=0,
)
quantizer.process()

# Save as embedded monolith (CRITICAL: no external data)
check_and_save_model(
    quantizer.model.model,
    'working/onnx-q4/model.onnx'
)
```

> **Important**: The output file MUST be a single embedded `.onnx` file with all weights included. External data files (`.onnx_data`) do NOT work with local browser serving.

---

## Step 4: Deploy Model Files

Copy the model and config files to your app's public directory:

```
public/models/your-model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
└── onnx/
    └── model.onnx    # Must be single file with embedded weights
```

---

## Step 5: Configure Runtime

### LLMProvider Config

```tsx
<LLMProvider
  config={{
    preferredRuntime: "transformers",
    models: {
      // Local model path (starts with "/")
      transformers: "/models/your-model",
    },
  }}
>
  <YourApp />
</LLMProvider>
```

### How It Works

The Transformers.js runtime automatically detects local vs remote models:

| Model Path | Behavior |
|------------|----------|
| `/models/...` | Local: loads `model.onnx` directly, no dtype specified |
| `org/model-ONNX` | HuggingFace: loads `model_q4.onnx` with dtype=q4 |

---

## Troubleshooting

### "Failed to load external data file... Module.MountedFiles not available"

**Cause**: Model has external data files that browser can't mount.

**Fix**: Re-export as embedded monolith:
```python
onnx.save_model(model, 'model.onnx', save_as_external_data=False)
```

### "Protobuf parsing failed"

**Cause**: Model file exceeds 2GB protobuf limit.

**Fix**: Use q4 quantization to reduce size below 2GB.

### Model file loads but session creation fails

**Cause**: Specifying `dtype` for local models causes it to look for `model_q4.onnx` instead of `model.onnx`.

**Fix**: Ensure your model is named `model.onnx` in the `onnx/` folder. The runtime automatically omits dtype for local paths.

---

## Configuration Reference

### Environment Settings (Automatic)

For local models, these are set automatically:

```typescript
env.localModelPath = "/models/";
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.useCustomCache = false;
```

### Vite MIME Types

If using Vite, add this plugin to correctly serve ONNX files:

```typescript
// vite.config.ts
function onnxMimeTypePlugin(): Plugin {
  return {
    name: 'onnx-mime-type',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.endsWith('.onnx')) {
          res.setHeader('Content-Type', 'application/octet-stream');
        }
        next();
      });
    },
  };
}
```

---

## Performance Notes

| Quantization | File Size | WebGPU | WASM | Recommended |
|--------------|-----------|--------|------|-------------|
| fp32 | ~1.6GB | ⚠️ Large | ✅ | No |
| fp16 | ~800MB | ⚠️ | ✅ | WASM only |
| q4 | ~400-800MB | ✅ Fast | ⚠️ | **Yes** |

**WebGPU with q4 quantization** provides the best balance of size and performance for browser deployment.

---

## Example: Full Pipeline

```bash
# 1. Fine-tune (optional)
python -m mlx_lm lora --model google/functiongemma-270m-it --data data/ --train

# 2. Fuse adapters
python -m mlx_lm fuse --model google/functiongemma-270m-it --adapter-path adapters/ --save-path fused/

# 3. Export to ONNX
python -m scripts.convert --model_id ./fused/ --output_parent_dir output/ --task text-generation

# 4. Quantize
python quantize_q4.py  # Script using MatMul4BitsQuantizer

# 5. Deploy
cp output/model.onnx public/models/my-model/onnx/
cp fused/*.json public/models/my-model/
```

Then in your app:
```tsx
models: { transformers: "/models/my-model" }
```
