# Edge-First LLM Tool Calling Framework

## Technical Specification & Reference

**Version:** 0.1.0  
**Status:** Research & Design Phase  
**Target:** Frontend developers building tool-calling workflows with local inference

---

## Executive Summary

A progressive enhancement framework for running LLM inference with tool calling capabilities directly in the browser. Automatically selects optimal runtime (WebLLM → Transformers.js → API fallback) based on platform capabilities, with service worker orchestration for zero-cost edge inference.

**Core Value Proposition:** `npm install` + configure tools → instant local LLM tool calling with automatic API fallback.

---

## Problem Space

### Current State

- **Backend dependency:** Every tool-calling workflow requires server infrastructure
- **Cost scaling:** API inference costs multiply with user base
- **Latency overhead:** Network round-trips add 150-350ms minimum
- **Context limits:** Small models (1-3B) sufficient for tool routing but underutilized
- **Framework gap:** No "just works" package for edge LLM + tool calling in React

### Why This Works Now

- Phi-3.5 Mini: 3.8B params with 128K context window
- Qwen2.5 0.5B/1.5B: proven tool calling at sub-2B scale
- WebGPU shipping in Chrome/Edge desktop
- Transformers.js provides WASM fallback
- Service Worker APIs stable across browsers

---

## Design Principles

### 1. Explicit Over Automatic

Multi-turn tool calling is developer-controlled by default. The framework provides the building blocks (response types, tool registry, execution helpers) but the developer orchestrates the loop. This makes state flow obvious and debugging straightforward.

**Rationale:** Small models are less reliable than frontier models. Giving developers explicit control prevents mysterious failures and makes the system predictable.

### 2. Progressive Enhancement

Runtime selection adapts to platform capabilities: WebGPU → WASM → API. Applications work everywhere, with better performance on capable hardware.

**Rationale:** Not all users have WebGPU-capable devices. Graceful degradation ensures universal accessibility while leveraging cutting-edge hardware when available.

### 3. Simple Primitives, Complex Applications

The framework provides two response types (tool_calls, content) and two modes (streaming, complete). Developers compose these into sophisticated workflows.

**Rationale:** Simple, well-understood primitives are easier to learn, debug, and extend than complex abstractions.

### 4. Developer Owns Tool Execution

Tool handlers run in main thread, not service worker. Developers have full control over execution context, error handling, and side effects.

**Rationale:** Tools might need DOM access, React state, or other main-thread features. Service worker isolation would be too restrictive.

### 5. Zero-Cost Edge Inference

The entire framework is built around making local inference viable for production use. Tool routing doesn't require frontier model intelligence.

**Rationale:** 80% of tool calling is pattern matching. A 1.5B model routing between 10 tools is more cost-effective than GPT-4 API calls.

---

## Technical Architecture

### Runtime Selection Strategy

```
Platform Detection:
┌─────────────────────────────────────────┐
│ 1. Check WebGPU availability           │
│    └─> WebLLM (Phi-3.5, Qwen2.5)       │
│                                         │
│ 2. Fallback: Check WASM support        │
│    └─> Transformers.js (Qwen2.5 0.5B)  │
│                                         │
│ 3. Fallback: External API              │
│    └─> OpenAI/Anthropic/OpenRouter     │
└─────────────────────────────────────────┘
```

### Service Worker Architecture

```typescript
// Service Worker handles:
// - Model loading/caching
// - Inference execution
// - Tool call decision making
// - Conversation context management

interface ServiceWorkerController {
  initialize(config: RuntimeConfig): Promise<void>;
  chat(messages: Message[], tools: Tool[]): AsyncIterator<Response>;
  registerTool(tool: ToolDefinition): void;
  getStatus(): RuntimeStatus;
}
```

### Main Thread Interface

```typescript
// React hooks provide clean abstraction
import { useLLM } from "@edge-llm/react";

function Component() {
  const { send, registerTool, registerTools, status } = useLLM({
    preferredRuntime: "auto", // or 'webllm' | 'transformers' | 'api'
    fallbackStrategy: "quality", // or 'speed' | 'cost'
  });

  // Register single tool - executed in main thread
  registerTool({
    name: "getWeather",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string" },
      },
      required: ["city"],
    },
    handler: async ({ city }) => {
      return fetch(`/api/weather/${city}`).then((r) => r.json());
    },
  });

  // Or register multiple tools at once
  registerTools([
    {
      name: "getCalendar",
      description: "Get calendar events",
      parameters: {
        /* ... */
      },
      handler: async (args) => {
        /* ... */
      },
    },
    {
      name: "sendEmail",
      description: "Send an email",
      parameters: {
        /* ... */
      },
      handler: async (args) => {
        /* ... */
      },
    },
  ]);

  return <ChatInterface onSend={send} status={status} />;
}
```

---

## Model Selection & Capabilities

### Tier 1: WebLLM (Desktop, WebGPU)

**Phi-3.5 Mini (3.8B parameters)**

- Context: 128K tokens
- Quantized size: ~2.4GB (Q4_K_M)
- Performance: 350-500ms inference on desktop GPU
- Strengths: Excellent reasoning, multilingual, instruction following
- Limitations: Large initial download, GPU memory requirements
- Use case: Rich tool routing with complex decision trees

**Qwen2.5 1.5B Instruct**

- Context: 32K tokens
- Quantized size: ~1.2GB
- Performance: 250-400ms inference
- Strengths: Native tool calling support, multilingual, compact
- Limitations: Less reasoning capability than Phi-3.5
- Use case: Standard tool routing, faster cold start

### Tier 2: Transformers.js (Universal WASM)

**Qwen2.5 0.5B Instruct**

- Context: 32K tokens
- Quantized size: ~400MB
- Performance: 800-1200ms inference (CPU-bound)
- Strengths: Runs anywhere, smallest footprint, proven tool calling
- Limitations: Slower inference, less capable reasoning
- Use case: Maximum compatibility, mobile/low-power devices

### Tier 3: API Fallback

**When to trigger:**

- Model load fails or times out (>30s)
- Inference exceeds quality threshold (multiple retries)
- User explicitly selects API mode
- Platform capabilities insufficient (no WebGPU, no WASM)

**Cost optimization:**

- Cache model decisions (tool selection patterns)
- Batch similar queries when possible
- Use smallest viable API model (GPT-4o-mini, Claude Haiku)

---

## Tool Calling Protocol

### Tool Registry & Management

Tools are registered in main thread and maintained in a `Map<string, ToolDefinition>`:

```typescript
interface ToolDefinition {
  name: string;
  description: string;
  parameters: JSONSchema;
  handler: (args: any) => Promise<any>;
}

// Single tool registration
registerTool(toolDef);

// Batch registration (recommended for setup)
registerTools([tool1, tool2, tool3]);

// Tool updates (replaces existing)
registerTool({ name: "getWeather" /* new definition */ });

// Tool removal
unregisterTool("getWeather");
```

### Standard Format (OpenAI-compatible)

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

### Response Types

The framework returns one of two response types:

```typescript
type ModelResponse = ToolCallsResponse | ContentResponse;

interface ToolCallsResponse {
  type: "tool_calls";
  calls: ToolCall[];
}

interface ContentResponse {
  type: "content";
  stream?: AsyncIterator<string>; // When stream: true
  text?: string; // When stream: false
}

interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}
```

### Single-Turn Flow (Simple)

```typescript
// User asks something that doesn't need tools
const response = await send("What is 2+2?");

if (response.type === "content") {
  // Stream response
  for await (const chunk of response.stream) {
    appendToUI(chunk);
  }
}
```

### Multi-Turn Flow (Tool Calling)

Developer explicitly handles the tool execution loop:

```typescript
async function handleMessage(userMsg: string) {
  let response = await send(userMsg);

  // Loop until model stops calling tools
  while (response.type === "tool_calls") {
    // Show UI indicators for each tool
    response.calls.forEach((call) => {
      showToolIndicator(call.name);
    });

    // Execute all tools in parallel
    const results = await Promise.all(
      response.calls.map(async (call) => ({
        tool_call_id: call.id,
        result: await executeTool(call),
      }))
    );

    // Send results back to model
    response = await send({ toolResults: results });
  }

  // Final response - stream to user
  for await (const chunk of response.stream) {
    appendToUI(chunk);
  }
}

async function executeTool(call: ToolCall) {
  const tool = toolRegistry.get(call.name);
  if (!tool) throw new Error(`Tool ${call.name} not found`);

  try {
    return await tool.handler(call.arguments);
  } catch (error) {
    return { error: error.message };
  }
}
```

### Streaming vs Complete Responses

Two modes for different use cases:

```typescript
// Streaming (default) - better UX, typewriter effect
const response = await send(userMsg, { stream: true });
for await (const chunk of response.stream) {
  appendToUI(chunk);
}

// Complete (simpler) - easier testing, all-at-once
const response = await send(userMsg, { stream: false });
appendToUI(response.text);
```

**Why both modes:**

- **Testing:** Assert on complete strings vs consuming async iterators
- **Simple UIs:** Not all interfaces need typewriter effects
- **Tool loops:** Non-streaming cleaner for intermediate tool call responses
- **Performance:** Skip streaming overhead for short responses

**Implementation note:** Service worker buffers tokens in non-streaming mode before returning.

### User Input as a Tool

User interaction can be modeled as a special tool:

```typescript
registerTool({
  name: "requestUserInput",
  description: "Ask the user for additional information",
  parameters: {
    type: "object",
    properties: {
      prompt: {
        type: "string",
        description: "Question to ask the user",
      },
      options: {
        type: "array",
        items: { type: "string" },
        description: "Optional: multiple choice options",
      },
    },
    required: ["prompt"],
  },
  handler: async ({ prompt, options }) => {
    // Pause execution, show UI
    if (options) {
      return await showChoiceModal(prompt, options);
    }
    return await showInputModal(prompt);
  },
});
```

**Example flow:**

```
User: "Book me dinner tonight"
Model: calls search_restaurants()
Execute: Returns 5 options
Model: calls requestUserInput("Which restaurant?", options=[...])
Execute: Shows modal, waits for user selection
User: Selects "Italian place"
Model: calls book_reservation(restaurant_id: 3)
Execute: Books reservation
Model: "Booked at 7pm!"
```

### Execution Limits & Safety

```typescript
interface ToolExecutionConfig {
  maxIterations: number; // Default: 5
  executionTimeout: number; // Per tool, default: 30000ms
  totalTimeout: number; // Entire flow, default: 120000ms
  retryStrategy: "exponential" | "linear" | "none";
  maxRetries: number; // Default: 2
}

// Framework enforces:
// - Max tool calling iterations (prevent infinite loops)
// - Per-tool execution timeout
// - Total conversation timeout
// - Automatic error handling and retry
```

### Error Handling

```typescript
interface ToolExecutionError {
  tool: string;
  error: Error;
  retryable: boolean;
  fallbackStrategy?: "skip" | "mock" | "api";
}

// Framework handles:
// - Malformed tool calls → retry with clarification
// - Handler failures → return error to model
// - Timeout → cancel and inform model
// - Parse failures → log and skip
// - Max iterations exceeded → force final response
```

**Error flow:**

1. Tool execution fails
2. Error returned to model as tool result
3. Model decides: retry with different args OR inform user OR call different tool
4. If repeated failures, framework caps at `maxRetries`

---

## Configuration & Defaults

### Minimal Setup

```typescript
// app.tsx
import { LLMProvider } from "@edge-llm/react";

function App() {
  return (
    <LLMProvider>
      <YourApp />
    </LLMProvider>
  );
}
```

### Advanced Configuration

```typescript
<LLMProvider
  config={{
    // Runtime selection
    runtimeStrategy: {
      prefer: "webllm",
      fallbackChain: ["transformers", "api"],
      apiKey: process.env.OPENAI_API_KEY,
    },

    // Model selection
    models: {
      webllm: "Phi-3.5-mini-instruct-q4",
      transformers: "Qwen2.5-0.5B-Instruct-q8",
      api: "gpt-4o-mini",
    },

    // Performance tuning
    performance: {
      maxConcurrentInference: 1,
      inferenceTimeout: 10000, // ms
      modelCacheTTL: 7 * 24 * 60 * 60 * 1000, // 7 days
    },

    // Tool execution
    tools: {
      maxIterations: 5, // Max tool calling loops
      executionTimeout: 30000, // Per tool timeout (ms)
      totalTimeout: 120000, // Entire conversation timeout (ms)
      retryStrategy: "exponential",
      maxRetries: 2,
    },

    // Response options
    streaming: {
      default: true, // Stream by default
      bufferSize: 1, // Chunks to buffer before sending
    },
  }}
>
  <YourApp />
</LLMProvider>
```

---

## Implementation Roadmap

**Current Status:** Phase 1 and Phase 2 are complete. The framework architecture is in place with all types, interfaces, and abstractions. Runtime implementations (WebLLM, Transformers.js) are fully integrated. The React hooks and provider are functional, and the build system is configured. Phase 3.5 (Testing & Example App) is next to validate everything works together.

### Phase 1: Core Infrastructure (MVP)

- [x] Service worker runtime abstraction
  - ✅ ServiceWorkerController for managing inference
  - ✅ LLMClient for main-thread communication
  - ✅ Service worker script template
- [x] WebLLM integration (Phi-3.5)
  - ✅ Full implementation complete with @mlc-ai/web-llm
  - ✅ WebGPU detection and initialization
  - ✅ Tool calling support with OpenAI-compatible format
  - ✅ Streaming and non-streaming response modes
  - ✅ Model loading with progress callbacks
- [x] Transformers.js integration (Qwen2.5 0.5B)
  - ✅ Full implementation complete with @xenova/transformers
  - ✅ WASM detection and initialization
  - ✅ Chat template support for Qwen2.5 models
  - ✅ Tool calling with JSON parsing
  - ✅ Streaming support (simulated via character chunking)
  - ✅ Model loading with progress callbacks
- [x] Tool registry (Map-based, add/remove/update)
  - ✅ ToolRegistry class with full CRUD operations
  - ✅ OpenAI-compatible format conversion
- [x] Basic tool calling protocol (single turn)
  - ✅ Infrastructure ready, runtime implementations pending
- [x] Response types (tool_calls vs content)
  - ✅ TypeScript types defined
  - ✅ ModelResponse union type
- [x] Streaming and complete response modes
  - ✅ ChatOptions with stream flag
  - ✅ AsyncIterator support for streaming
- [x] React hooks (useLLM with send/registerTool/registerTools)
  - ✅ useLLM hook implemented
  - ✅ LLMProvider component
  - ✅ Context API integration
- [x] Model caching strategy
  - ✅ ModelCache class using Cache API
  - ✅ TTL support

**Additional Phase 1 Items (not in original plan):**

- [x] RuntimeManager for automatic runtime selection with fallback chain
- [x] BaseRuntime abstract class for runtime implementations
- [x] Complete TypeScript type system with JSONSchema support
- [x] Build system setup (tsup for libraries, Vite for example)
- [x] Testing infrastructure (Vitest with jsdom for React)
- [x] Monorepo structure with npm workspaces
- [x] Publishing configuration (.npmignore, publishConfig)

### Phase 2: Multi-Turn & Safety

- [x] Multi-turn tool calling flow (developer-controlled loop)
  - ✅ ToolLoop class for managing multi-turn conversations
  - ✅ Automatic conversation history management
  - ✅ Developer-controlled execution with callbacks
- [x] Tool execution timeout & retry logic
  - ✅ ToolExecutor with per-tool timeout support
  - ✅ Exponential and linear retry strategies
  - ✅ Total execution timeout protection
- [x] Max iteration limits (prevent infinite loops)
  - ✅ Configurable max iterations (default: 5)
  - ✅ Automatic loop termination on limit
- [x] Error recovery & error messages to model
  - ✅ Error messages passed back to model as tool results
  - ✅ Retryable vs non-retryable error detection
  - ✅ Graceful error handling in tool loop
- [x] Tool result formatting & validation
  - ✅ ToolValidator for argument validation against JSONSchema
  - ✅ Result formatting for model consumption
  - ✅ Type checking and enum validation
- [x] User-input-as-tool pattern
  - ✅ createUserInputTool helper function
  - ✅ Custom handler support for UI integration
  - ✅ Multiple choice option support
- [x] Performance monitoring
  - ✅ PerformanceMonitor class for metrics collection
  - ✅ Inference time, tool execution time, total time tracking
  - ✅ Iteration and tool call counting

### Phase 3.5: Testing & Example App (Validation)

- [x] Unit tests for core functionality
  - [x] ToolRegistry tests (Jest) - ✅ 8 tests passing
  - [x] ToolExecutor tests (Jest) - ✅ 6 tests passing
  - [ ] ToolLoop tests (Jest) - Pending
  - [x] ToolValidator tests (Jest) - ✅ 7 tests passing
  - [ ] Runtime tests (Jest with mocks) - Pending
- [ ] React component tests
  - [ ] LLMProvider tests (RTL) - Pending
  - [ ] useLLM hook tests (RTL) - Pending
- [x] Example app implementation
  - [x] Basic chat interface - ✅ Full chat UI with message history
  - [x] Tool registration examples - ✅ 3 example tools (calculate, getCurrentTime, searchWeb)
  - [x] Multi-turn tool calling demo - ✅ Automatic tool execution loop
  - [ ] User input tool example - Can be added
  - [ ] Performance metrics display - Can be added
- [ ] Integration testing
  - [ ] End-to-end tool calling flow - Pending
  - [ ] Error handling scenarios - Pending
  - [ ] Timeout and retry behavior - Pending
- [ ] Documentation examples
  - [ ] Working code examples in README - Pending
  - [ ] Common patterns and use cases - Pending

**Status:** Core unit tests complete (21 tests passing). Example app fully functional with chat interface and multi-turn tool calling. Ready for integration testing and React component tests.

### Phase 3: Production Readiness

- [ ] API fallback implementation (Ollama, OpenAI, etc.)
- [ ] Seamless hybrid inference (Start with API, background load local, hot-swap when ready)
- [ ] Memory management (context pruning)
- [ ] Tool execution sandboxing
- [ ] Rate limiting & cost controls (API mode)
- [ ] DevTools panel for debugging tool flows
- [ ] Logging & observability

### Phase 4: Developer Experience

- [ ] CLI for model management
- [ ] Tool library/registry (common tools: calendar, email, search)
- [ ] Prompt templates library
- [ ] TypeScript type generation for tools
- [ ] Example integrations & starter kits
- [ ] Migration guides from other frameworks
- [ ] Auto mode (framework-managed tool execution loop)

### Phase 5: Advanced Features

- [ ] Multi-model routing (different tools → different models)
- [ ] Tool composition (chain multiple tools)
- [ ] Parallel tool execution optimization
- [ ] Memory persistence (IndexedDB)
- [ ] Collaborative filtering (share tool usage patterns)
- [ ] Custom model fine-tuning guides

---

## Performance Targets

### Cold Start (First Load)

- WebLLM: 3-5 seconds (model download + init)
- Transformers.js: 1-2 seconds (smaller model)
- API: <100ms (no local model)

### Warm Start (Cached)

- WebLLM: 200-400ms
- Transformers.js: 100-200ms
- API: <100ms

### Inference Latency

- WebLLM: 250-500ms (desktop GPU)
- Transformers.js: 800-1500ms (CPU-bound)
- API: 150-300ms (network dependent)

### Memory Footprint

- WebLLM: 2-4GB GPU memory
- Transformers.js: 500MB-1GB RAM
- Service Worker: ~50MB overhead

---

## Known Limitations & Tradeoffs

### Technical Constraints

1. **WebGPU adoption:** Safari/iOS support pending (use Transformers.js fallback)
2. **Context windows:** Small models cap at 32K-128K vs unlimited API
3. **Model size:** Even quantized models require significant downloads
4. **First-run UX:** 3-5 second model load on first visit

### When Edge Inference Fails

- Complex multi-step reasoning (use API)
- Domain-specific knowledge requirements (use RAG + API)
- Regulatory/compliance needs (use controlled API)
- Very high query volume from single user (consider API)

### Frontend Limitations

- Tool execution must be safe (no arbitrary code)
- User navigation kills service worker state
- Mobile thermal throttling impacts inference
- Browser memory limits (especially mobile)

---

## Success Metrics

### Developer Adoption

- Lines of code to implement: <50 LOC for basic tool calling
- Time to first inference: <5 minutes from `npm install`
- Tool registration: Single function call (`registerTools([...])`)
- Multi-turn implementation: <20 LOC for complete tool loop
- Framework compatibility: Works with any React setup

### Performance

- Cost reduction: 95%+ for tool routing tasks
- Latency improvement: 40%+ vs API (warm start)
- Tool execution overhead: <50ms per tool call
- Multi-turn loop: <2s for 3-tool sequence
- Reliability: 99%+ tool call parsing accuracy
- User experience: Imperceptible inference delay (<500ms p95)

### Tool Calling Quality

- Tool selection accuracy: >95% correct tool chosen
- Argument parsing: >98% valid JSON
- Multi-turn completion: >90% reach successful conclusion
- Error recovery: >80% successful retry after tool failure
- User input tools: <3s modal response time

### Business Impact

- Enables new use cases (offline-first, high-frequency queries)
- Removes infrastructure scaling concerns
- Privacy-preserving (no data leaves device)
- Competitive differentiation (instant, free inference)

---

## References & Prior Art

### Models

- Phi-3.5: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Function calling benchmarks: berkeley-function-calling-leaderboard

### Infrastructure

- WebLLM: https://github.com/mlc-ai/web-llm
- Transformers.js: https://huggingface.co/docs/transformers.js
- Service Workers: https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API

### Inspiration

- Ollama (desktop LLM runtime)
- LangChain.js (backend orchestration)
- Vercel AI SDK (API abstraction)

---

## Open Questions

1. **Tool versioning:** How to handle tool schema evolution across sessions?
2. **Shared workers:** Can multiple tabs share one model instance?
3. **Parallel tool execution:** Should we allow tools to declare dependencies/ordering?
4. **Cost hybrid:** When to route parts of query to API vs edge (quality thresholds)?
5. **Privacy controls:** How to audit what stays local vs goes to API?
6. **Tool sandboxing:** How strict should we be about tool execution safety?
7. **Context management:** Should we auto-prune conversation history or leave to developer?
8. **Progressive enhancement:** Should we show degraded UI when models aren't loaded yet?
9. **Streaming + tools:** Should we support streaming tool execution results?
10. **Auto mode timing:** If we add auto mode later, how do we expose tool execution events?

---

## Getting Started (Once Released)

```bash
npm install @edge-llm/core @edge-llm/react
```

### Simple Example (No Tools)

```typescript
import { LLMProvider, useLLM } from "@edge-llm/react";

function App() {
  return (
    <LLMProvider>
      <Chat />
    </LLMProvider>
  );
}

function Chat() {
  const { send } = useLLM();

  const handleMessage = async (userMsg: string) => {
    const response = await send(userMsg, { stream: false });

    if (response.type === "content") {
      console.log(response.text);
    }
  };

  return <ChatInterface onSend={handleMessage} />;
}
```

### Tool Calling Example

```typescript
function Chat() {
  const { send, registerTools } = useLLM();

  // Register tools once on mount
  useEffect(() => {
    registerTools([
      {
        name: "calculate",
        description: "Evaluate a mathematical expression",
        parameters: {
          type: "object",
          properties: {
            expression: { type: "string" },
          },
          required: ["expression"],
        },
        handler: async ({ expression }) => {
          return { result: eval(expression) };
        },
      },
      {
        name: "getWeather",
        description: "Get current weather for a city",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
          required: ["city"],
        },
        handler: async ({ city }) => {
          const data = await fetch(`/api/weather/${city}`).then((r) =>
            r.json()
          );
          return data;
        },
      },
    ]);
  }, []);

  const handleMessage = async (userMsg: string) => {
    let response = await send(userMsg);

    // Handle multi-turn tool calling
    while (response.type === "tool_calls") {
      // Execute all tools
      const results = await Promise.all(
        response.calls.map(async (call) => ({
          tool_call_id: call.id,
          result: await executeToolByName(call.name, call.arguments),
        }))
      );

      // Send results back
      response = await send({ toolResults: results });
    }

    // Stream final response
    for await (const chunk of response.stream) {
      appendToUI(chunk);
    }
  };

  return <ChatInterface onSend={handleMessage} />;
}
```

### User Input as Tool Example

```typescript
registerTool({
  name: "askUser",
  description: "Ask the user for clarification or additional information",
  parameters: {
    type: "object",
    properties: {
      question: { type: "string" },
      options: { type: "array", items: { type: "string" } },
    },
    required: ["question"],
  },
  handler: async ({ question, options }) => {
    // Show modal and wait for user input
    const answer = await showInputModal(question, options);
    return { userResponse: answer };
  },
});
```

---

**Status:** Phase 1 core infrastructure complete. Runtime implementations (WebLLM, Transformers.js) are stubbed and ready for integration.  
**Feedback:** Open to collaboration and input from the community.  
**License:** TBD (likely MIT or Apache 2.0)

---

_Last updated: November 2025_  
_Phase 1 Status: Core infrastructure complete, runtime integrations pending_
