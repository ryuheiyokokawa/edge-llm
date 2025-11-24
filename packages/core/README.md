# @edge-llm/core

Core runtime for edge-first LLM tool calling.

## Installation

```bash
npm install @edge-llm/core
```

## Usage

```typescript
import { RuntimeManager, RuntimeConfig } from "@edge-llm/core";

// 1. Configure the runtime
const config: RuntimeConfig = {
  preferredRuntime: "auto", // "webllm" | "transformers" | "api" | "auto"
  debug: true, // Enable debug logging
  models: {
    webllm: { modelId: "Llama-3-8B-Instruct-q4f16_1-MLC" },
    transformers: { modelId: "Xenova/Qwen2.5-0.5B-Instruct" },
  }
};

// 2. Initialize manager
const manager = new RuntimeManager(config);
await manager.initialize();

// 3. Get the runtime and chat
const runtime = manager.getRuntime();
const response = await runtime.chat([
  { role: "user", content: "Hello world!" }
], []);

console.log(response);
```

## Configuration

The `RuntimeConfig` object supports the following options:

- `preferredRuntime`: Force a specific runtime (`"webllm"`, `"transformers"`, `"api"`) or use `"auto"` to select the best available.
- `debug`: Enable verbose console logging for debugging.
- `models`: Configuration for specific runtimes.
  - `webllm`: `{ modelId: string, modelLib?: string }`
  - `transformers`: `{ modelId: string, quantized?: boolean }`
  - `api`: `{ modelId: string, baseUrl?: string }`
- `fallbackStrategy`: Strategy for fallback (`"quality"`, `"speed"`, `"cost"`).

## Architecture

Edge LLM uses a fallback mechanism to ensure the best possible experience:

1.  **WebLLM (WebGPU)**: High-performance, hardware-accelerated inference. Requires WebGPU support.
2.  **Transformers.js (WASM)**: CPU-based inference using WebAssembly. Works on most modern browsers.
3.  **API Fallback**: Optional fallback to a remote API if local inference is not possible.

The `RuntimeManager` automatically detects available capabilities and selects the appropriate runtime.

