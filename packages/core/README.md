# @edge-llm/core

The intelligent runtime manager for hybrid AI inference in the browser.

This package handles the complex logic of orchestrating LLMs: choosing the best available runtime (WebGPU vs WASM vs API), managing downloads, executing tool calls, and enabling seamless hot-mapping.

## Features

- **🧠 Hybrid Inference Engine**: Starts instantly with a cloud API, then silently hot-swaps to a local model (WebLLM/Transformers.js) once downloaded.
- **🛠 Universal Tool Calling**: Write tools once using standard schemas, and run them on any backend. Handles JSON/XML format differences automatically.
- **🔄 Smart Fallback**:
  - **Tier 1**: WebLLM (WebGPU) - Fastest, local.
  - **Tier 2**: Transformers.js (WASM) - CPU fallback, local.
  - **Tier 3**: API (OpenAI/Ollama) - Universal fallback, immediate start.
- **⚡ Hot-Swap Ready**: Applications remain interactive during the 1-2GB model download. The switching happens in the background without user interruption.

## Installation

```bash
npm install @edge-llm/core
```

## Basic Usage

```typescript
import { LLMClient } from "@edge-llm/core";

// 1. Initialize the client
const client = new LLMClient();

await client.initialize({
  // "auto" prioritizes: API (for speed) -> WebGPU -> WASM
  preferredRuntime: "auto", 
  
  // Optional: API fallback (e.g., local Ollama bridge)
  apiUrl: "http://localhost:3001/v1/chat/completions",
  
  models: {
    // Local models to download in background
    webllm: "Llama-3-8B-Instruct-q4f16_1-MLC",
    transformers: "onnx-community/functiongemma-270m-it-ONNX",
  }
});

// 2. Chat with tools
const response = await client.chat([
  { role: "user", content: "Calculate 5 * 12" }
], [
  {
    name: "calculate",
    description: "Evaluate a math expression",
    parameters: { type: "object", properties: { expression: { type: "string" } } },
    handler: async ({ expression }) => eval(expression)
  }
]);
```

## Hybrid Inference & Hot-Swapping

The core philosophy of `@edge-llm/core` is **"Reliability First"**.

1.  **Instant Start**: If an `apiUrl` is provided, the client is "Ready" immediately. It routes initial requests to the API.
2.  **Background Download**: If a local runtime (WebLLM/Transformers.js) is configured, it begins downloading model weights in the background.
3.  **Seamless Switch**: Once the local model is fully loaded and compiled, the client automatically "hot-swaps" the active backend. Subsequent requests run locally on-device.

### Configuration

```typescript
type RuntimeConfig = {
  // Strategy
  preferredRuntime?: "webllm" | "transformers" | "api" | "auto";
  fallbackStrategy?: "quality" | "speed" | "cost";

  // API Config (for API Runtime)
  apiUrl?: string;
  apiKey?: string;
  
  // Model Config
  models?: {
    webllm?: string;       // e.g. "Llama-3-8B-Instruct-q4f16_1-MLC"
    transformers?: string; // e.g. "onnx-community/functiongemma-270m-it-ONNX"
    api?: string;          // e.g. "llama3" (for Ollama) or "gpt-4o"
  };

  // Tool Format Override
  // "json" (standard) or "xml" (FunctionGemma)
  toolCallFormat?: "json" | "xml";
}
```
