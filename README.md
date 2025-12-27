# Edge-LLM: Hybrid AI Framework

**The progressive enhancement framework for running generic LLM tool-calling directly in the browser.**

Edge-LLM enables you to build AI applications that start instantly using cloud APIs and seamlessly transition to local device execution (WebGPU/WASM) for zero-latency, private, and free inference.

![Hybrid Inference Demo](./docs/hybrid-demo.gif)

## Key Features

- **🚀 Hybrid Inference Engine**: Prioritizes immediate availability. Apps start strictly with an API (OpenAI/Ollama) and silently hot-swap to a local model (Llama 3, FunctionGemma) once downloaded in the background.
- **🛠 Universal Tool Calling**: Define tools once (Zod/JSON Schema) and run them on any runtime. The framework automatically handles format variations (JSON for Llama 3, XML for FunctionGemma).
- **📱 Edge-Optimized Runtimes**: 
  - **WebLLM**: Hardware-accelerated WebGPU inference for Llama 3, Phi-3, etc.
  - **Transformers.js**: WASM-based execution for smaller models like FunctionGemma.
- **🔌 Standardized API Bridge**: Use `@edge-llm/server` to proxy local requests to Ollama or compatible APIs, enabling a "Local-First" dev experience.
- **⚛️ React Hooks**: Simple `useLLM` hooks to manage downloading, inference, and tool execution state.

## Architecture

The framework is composed of 4 packages:

| Package | Description |
|Args|---|
| **`@edge-llm/core`** | The brain. Manages `RuntimeManager`, hot-swapping logic, and the `LLMClient` tool execution loop. |
| **`@edge-llm/react`** | React bindings. Provides `LLMProvider` and `useLLM` for easy integration. |
| **`@edge-llm/server`** | A lightweight Node.js/Express bridge that translates browser API calls to Ollama/OpenAI formats. |
| **`@edge-llm/fine-tune`** | (**Experimental**) Utilities for preparing datasets and fine-tuning models for edge-specific tool calling. |

## Quick Start

### 1. Prerequisites
- Node.js 18+
- (Optional) [Ollama](https://ollama.com/) running locally for API fallback dev.

### 2. Installation
```bash
# Clone the repo
git clone https://github.com/ryuheiyokokawa/edge-llm.git
cd edge-llm

# Install dependencies
npm install

# Build all packages
npm run build
```

### 3. Running the Example App
The example app demonstrates the full hybrid flow (API -> WebLLM).

```bash
# Terminal 1: Start the API Bridge (connects to local Ollama)
npm run dev --workspace=@edge-llm/server

# Terminal 2: Start the Web App
npm run dev --workspace=@edge-llm/app
```
Open [http://localhost:3000](http://localhost:3000).

## Configuration

Initialize the client with your preferred strategy:

```typescript
// App.tsx
<LLMProvider
  config={{
    // Auto-detect best available runtime
    preferredRuntime: "auto", 
    
    // Fallback API (e.g., local Ollama bridge)
    apiUrl: "http://localhost:3001/v1/chat/completions",
    
    // Models to use
    models: {
      webllm: "Llama-3-8B-Instruct-q4f16_1-MLC",
      transformers: "onnx-community/functiongemma-270m-it-ONNX",
      api: "llama3", 
    }
  }}
>
  <YourApp />
</LLMProvider>
```

## Contributing

This project is a monorepo managed by NPM Workspaces.

- `packages/core`: Main logic
- `packages/react`: UI bindings
- `packages/server`: API proxy
- `packages/fine-tune`: Training scripts
- `examples/app`: Demo application

See [docs/plan.md](./docs/plan.md) for the architecture roadmap.
