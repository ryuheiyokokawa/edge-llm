# Prompt: Run Fine-Tuning Pipeline

Use this prompt with a frontier AI model (Claude, Gemini, GPT) in your IDE to run the fine-tuning pipeline.

---

## Default Prompt

```
Run the FunctionGemma fine-tuning pipeline in packages/fine-tune.

Steps:
1. If venv doesn't exist, run: `npm run setup`
2. Run: `npm run pipeline:quantize`

This will:
- Generate training data from examples/
- Train LoRA adapters (100 iterations)
- Fuse adapters into base model
- Export to ONNX with FP16 quantization (~830MB)
- Prepare for browser deployment

Wait for each step to complete before proceeding.
```

---

## With Custom Data

```
Run the fine-tuning pipeline with my custom data:
- Tools: path/to/my-tools.json
- Examples: path/to/my-examples.json

Steps:
1. If venv doesn't exist, run: `npm run setup`
2. Activate venv: `source python/.venv/bin/activate`
3. Run:
   python python/run_pipeline.py \
     --tools-file path/to/my-tools.json \
     --examples-file path/to/my-examples.json \
     --quantize
```

---

## Deploy to Example App

```
After the pipeline completes, deploy the model to the example app:

cp -r working/onnx-model-fp16/* ../../examples/app/public/models/custom-functiongemma/
```

---

## NPM Scripts

| Script | Description |
|--------|-------------|
| `npm run setup` | Create venv and install Python dependencies |
| `npm run pipeline` | Run full pipeline (no quantization) |
| `npm run pipeline:quantize` | Run full pipeline with FP16 quantization |
| `npm run test:onnx` | Test ONNX model with Node.js |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `npm run setup` fails | Run manually: `python3 -m venv python/.venv && source python/.venv/bin/activate && pip install -r python/requirements.txt` |
| ONNX export fails | Ensure `optimum` and `onnxruntime` are installed |
| Model too large | Already using FP16 (~830MB), can try INT8 but quality degrades |
| Browser can't load | Ensure `onnx/` subdirectory structure and proper `tokenizer_config.json` |
