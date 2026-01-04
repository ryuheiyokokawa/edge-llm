# Prompt: Run Fine-Tuning Pipeline

Use this prompt with a frontier AI model (Claude, Gemini, GPT) in your IDE to run the fine-tuning pipeline.

---

## Default Prompt

```
Run the FunctionGemma fine-tuning pipeline in packages/fine-tune.

Steps:
1. Activate Python venv: `source python/.venv/bin/activate`
2. Run: `python python/run_pipeline.py --quantize`

This will:
- Generate training data from examples/
- Train LoRA adapters (100 iterations)
- Fuse adapters into base model
- Export to ONNX with FP16 quantization (832MB)
- Prepare for browser deployment

Wait for each step to complete before proceeding.
```

---

## With Custom Data

```
Run the fine-tuning pipeline with my custom data:
- Tools: path/to/my-tools.json
- Examples: path/to/my-examples.json

Command:
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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Python venv not found | `python -m venv python/.venv && pip install -r python/requirements.txt` |
| ONNX export fails | Ensure `optimum` and `onnxruntime` are installed |
| Model too large | Already using FP16 (832MB), can try INT8 but quality degrades |
| Browser can't load | Ensure `onnx/` subdirectory structure and proper `tokenizer_config.json` |
