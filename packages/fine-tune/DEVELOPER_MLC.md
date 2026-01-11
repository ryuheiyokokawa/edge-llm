# MLC LLM Model Export Guide

This guide explains how to export your fine-tuned FunctionGemma models to MLC (WebLLM) format for browser-based WebGPU inference.

## The Two Stages of MLC Export

1.  **Weight Conversion**: Converts safetensors to MLC's high-quality FP16 format (default) or 4-bit quantized format. This is fast and runs on most machines.
2.  **Compilation (WASM)**: Compiles the model logic into a `.wasm` file using TVM and Emscripten. This requires a complex toolchain.

---

## Quantization & Quality

For small models like **Gemma 2 270M**, we default to **FP16 (`q0f16`)**. 

| Quantization | Size | Recommendation |
|--------------|------|----------------|
| **q0f16** (FP16) | ~830MB | **Default**. Best quality, ensures tool-calling logic remains intact. |
| **q4f16_1** (4-bit) | ~70MB | Optional. High risk of "garbage" output/logic failure for models < 1B params. |

To use 4-bit quantization, override the default:
```bash
npm run pipeline -- --mlc-quant q4f16_1
```

---

## Path A: Using Docker (Recommended for M-chip Macs)

Use the provided Dockerfile to run the compilation in a isolated environment.

1.  **Build the Image**:
    ```bash
    docker build -t mlc-compiler -f docker/Dockerfile.mlc .
    ```

2.  **Run the Export**:
    ```bash
    docker run -it --rm \
      -v $(pwd):/workspace \
      mlc-compiler \
      python3 python/export_mlc.py \
      --input working/fused-model \
      --output working/mlc-model \
      --compile
    ```

---

## Path B: Native Installation (macOS Silicon)

If you prefer to run natively, you need to install the MLC toolchain.

### 1. Prerequisites
- **arm64 Conda**: Verify with `python -c "import platform; print(platform.machine())"`. Must be `arm64`.
- **Emscripten**: Installed and in your PATH.

### 2. Install MLC LLM & TVM
Follow the official [MLC LLM Installation Guide](https://llm.mlc.ai/docs/install/index.html). We recommend the nightly wheels for the runtime, but weight conversion often requires the source build for specific model architectures like Gemma 2.

```bash
# Example setup using nightly wheels
pip install --pre --extra-index-url https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

---

## Integration with Pipeline

You can trigger the MLC export as part of the full pipeline:

```bash
# Just weight conversion (safe for any machine)
npm run pipeline -- --skip-onnx

# Full conversion + WASM compilation (requires toolchain/Docker)
npm run pipeline -- --compile-mlc
```

## Output Structure

The `working/mlc-model` directory will contain:
- `mlc-chat-config.json`: Model configuration.
- `ndarray-cache.json`: Weight metadata.
- `params/`: Sharded quantized weights.
- `model-webgpu.wasm`: The compiled binary (only if `--compile` was used).
- `tokenizer.json`: Tokenizer config.
