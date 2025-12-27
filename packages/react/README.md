# @edge-llm/react

React bindings for the Edge-LLM framework. Provides a simple hook-based interface to build hybrid AI applications.

## Installation

```bash
npm install @edge-llm/react @edge-llm/core
```

## Quick Start

### 1. Wrap your app with `LLMProvider`

This component holds the global state (model loading progress, active runtime) and manages the `LLMClient` lifecycle.

```tsx
// App.tsx
import { LLMProvider } from "@edge-llm/react";

function App() {
  return (
    <LLMProvider
      config={{
        preferredRuntime: "auto",
        apiUrl: "http://localhost:3001/v1/chat/completions",
        models: {
          webllm: "Llama-3-8B-Instruct-q4f16_1-MLC",
        }
      }}
    >
      <ChatInterface />
    </LLMProvider>
  );
}
```

### 2. Use the `useLLM` hook

Access the chat function, current status, and active runtime info.

```tsx
// ChatComponent.tsx
import { useLLM } from "@edge-llm/react";

export function ChatInterface() {
  const { send, status, activeRuntime, registerTools } = useLLM();

  // 1. Register tools (optional)
  useEffect(() => {
    registerTools([{
      name: "get_time",
      description: "Get current time",
      parameters: { type: "object", properties: {} },
      handler: async () => new Date().toISOString()
    }]);
  }, []);

  const handleSend = async () => {
    // "send" automatically handles the full tool-execution loop
    const response = await send([
      { role: "user", content: "What time is it?" }
    ]);
    
    console.log("Response:", response.text);
  };

  return (
    <div>
      <div className="status-bar">
        Status: {status} {/* "initializing" | "loading" | "ready" */}
        Runtime: {activeRuntime} {/* "api" -> "webllm" */}
      </div>
      
      <button onClick={handleSend} disabled={status !== "ready"}>
        Send
      </button>
    </div>
  );
}
```

## Hybrid UI Patterns

Because Edge-LLM supports "Hybrid Inference" (starting with API, upgrading to Local), your UI should reflect this state:

```tsx
const { status, activeRuntime } = useLLM();

if (status === "initializing") return <Spinner />;

return (
  <div>
    {activeRuntime === "api" && (
      <Badge color="yellow">Cloud Mode (Downloading Local Model...)</Badge>
    )}
    {activeRuntime === "webllm" && (
      <Badge color="green">Local Mode (Private & Free)</Badge>
    )}
    <Chat />
  </div>
);
```

## Service Workers (Optional)

For production apps, you can enable Service Worker support to run inference off the main thread, preventing UI jank during heavy generation.

```tsx
<LLMProvider 
  enableServiceWorker={true} 
  serviceWorkerPath="/sw.js" 
  // ...
>
```
