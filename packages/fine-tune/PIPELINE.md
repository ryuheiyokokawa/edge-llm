# Fine-Tuning Pipeline - What Works

This document captures the verified working approaches for fine-tuning and exporting FunctionGemma models for browser deployment.

## 🚀 Quick Start

```bash
cd packages/fine-tune

# Setup Python environment
python -m venv python/.venv
source python/.venv/bin/activate  
pip install -r python/requirements.txt

# Run full pipeline (with default examples)
python python/run_pipeline.py

# Or with custom data
python python/run_pipeline.py \
  --tools-file path/to/my-tools.json \
  --examples-file path/to/my-examples.json
```

---

## ✅ Verified Working Pipeline

| Step | Status | Output |
|------|--------|--------|
| Training (100 iters) | ✅ | Loss 4.718→0.069 (optimal) |
| Pre-fusion test | ✅ | `call:name{arg:<escape>val<escape>}` |
| Fusion | ✅ | Same output quality |
| ONNX export | ✅ | 1.6GB FP32 model |
| **FP16 quantization** | ✅ | **832MB (recommended)** |
| Browser deployment | ✅ | Function calling works! |

---

## 🔍 Browser Deployment Requirements

**Critical findings from comparing `onnx-community/functiongemma-270m-it-ONNX` (working) vs our custom export (initially failing):**

| Requirement | Working Model | Our Initial Export | Fix |
|-------------|--------------|-------------------|-----|
| ONNX file location | `onnx/model.onnx` | `model.onnx` (root) | Move to `onnx/` subdirectory |
| Chat template | Embedded in `tokenizer_config.json` | Missing | Copy from reference model |
| External data format | Config specifies split files | Config expects split but has single file | Remove `transformers.js_config.use_external_data_format` for single-file models |
| `eos_token_id` | `1` (single int) | `[1, 49]` (array) | Use single int for compatibility |

**Required files for browser deployment:**
```
custom-functiongemma/
├── config.json                 # Model config (no external data format if single file)
├── generation_config.json      # Generation params
├── tokenizer_config.json       # MUST include chat_template!
├── tokenizer.json              # Tokenizer vocabulary
├── special_tokens_map.json     # Special token mappings
├── chat_template.jinja         # Optional (can be embedded in tokenizer_config)
└── onnx/
    └── model.onnx              # ONNX model (in subdirectory!)
```

---

## 📜 Scripts

### 1. `python/generate_training_data.py`
Converts JSON examples to FunctionGemma format training data.

```bash
# Use default examples (from examples/ folder)
python python/generate_training_data.py

# Use custom data
python python/generate_training_data.py \
  --tools-file my-tools.json \
  --examples-file my-examples.json \
  --output-dir training-data
```

**Input formats:**

`tools-file.json`:
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

`examples-file.json`:
```json
[
  {
    "userQuery": "What is 5 * 12?",
    "expectedToolCalls": [{"name": "calculate", "arguments": {"expression": "5 * 12"}}]
  }
]
```

### 2. `python/export_and_test_onnx.py`
Exports fused MLX model to ONNX and tests with onnxruntime.

```bash
# Export only
python python/export_and_test_onnx.py \
  --input working/fused-model \
  --output working/onnx-model

# Export + quantize (INT8) + test
python python/export_and_test_onnx.py \
  --input working/fused-model \
  --output working/onnx-model \
  --quantize

# Export + FP16 quantization (better quality, ~50% size reduction)
python python/export_and_test_onnx.py \
  --input working/fused-model \
  --output working/onnx-model \
  --quantize --quantize-type fp16

# Export + Node.js Transformers.js validation
python python/export_and_test_onnx.py \
  --input working/fused-model \
  --output working/onnx-model \
  --quantize --test-node
```

**Quantization Options:**

| Type | Size | Quality | Browser Compatibility |
|------|------|---------|----------------------|
| `fp32` | 1.6GB | Best | ⚠️ Too large |
| **`fp16`** | **832MB** | **Excellent** | **✅ Recommended** |
| `int8` | 417MB | Degrades output | ❌ Not recommended |
| `q4` | ~200MB | Experimental | ⚠️ May not work |

### 3. `python/run_pipeline.py`
Master script that runs the complete pipeline with testing.

```bash
# Full pipeline
python python/run_pipeline.py

# Skip steps
python python/run_pipeline.py --skip-training --skip-onnx

# Custom settings
python python/run_pipeline.py \
  --iters 500 \
  --batch-size 4 \
  --quantize
```

---

## 📝 Manual Pipeline Steps

```
1. Generate Training Data
   python python/generate_training_data.py
     ↓
2. Train LoRA Adapters (mlx-lm)
   python -m mlx_lm lora --model ... --data training-data --train
     ↓
3. Test Adapters ← CHECKPOINT
   python -c "from mlx_lm import load, generate; ..."
     ↓
4. Fuse Model
   python -m mlx_lm fuse --dequantize --save-path working/fused-model
     ↓
5. Test Fused ← CHECKPOINT
   python -c "from mlx_lm import load, generate; ..."
     ↓
6. Export ONNX
   python python/export_and_test_onnx.py --input ... --output ...
     ↓
7. Quantize (optional)
   python python/export_and_test_onnx.py --input ... --output ... --quantize [--quantize-type fp16|int8|q4]
     ↓
8. Test ONNX ← CHECKPOINT
   (Automatic in export script)
     ↓
9. Test with Node.js Transformers.js ← CHECKPOINT
   node test_onnx_node.mjs working/onnx-model-int8
   (Or use --test-node flag in export script)
     ↓
10. Deploy to Browser
    cp -r working/onnx-model-int8/* examples/app/public/models/custom-functiongemma/
```

---

## ⚠️ Known Issues

| Issue | Fix |
|-------|-----|
| Training garbage output | Use 300+ iterations |
| Chat template missing | Copy `tokenizer_config.json` from reference model |
| Model too large | Use FP16 (~800MB) or INT8 (~417MB) quantization |
| INT8 quality degradation | Try FP16 instead (better quality, larger size) |
| ONNX loading in browser fails | Ensure `onnx/` subdirectory structure |
| Mistral tokenizer warning | Benign, ignore |

---

## 📁 File Structure

```
packages/fine-tune/
├── examples/                   # Example data (defaults)
│   ├── tool-definitions/       # Tool JSON schemas
│   └── training-examples.json  # Training examples
├── python/
│   ├── .venv/                  # Python venv
│   ├── requirements.txt
│   ├── generate_training_data.py
│   ├── export_and_test_onnx.py
│   └── run_pipeline.py         # Master script
├── training-data/              # Generated JSONL (gitignored)
├── training-output/            # Adapters (gitignored)
└── working/                    # All exploration outputs (gitignored)
    ├── fused-model/
    ├── onnx-model/
    └── onnx-model-int8/
```

---

## 🔧 Dependencies

```
# Python (requirements.txt)
mlx>=0.20.0
mlx-lm>=0.19.0
huggingface_hub>=0.20.0
safetensors>=0.4.0
optimum-onnx
onnxruntime
transformers

# Node.js
@huggingface/transformers >= 3.8.1
```
