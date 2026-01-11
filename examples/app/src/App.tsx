import { LLMProvider } from "@edge-llm/react";
import { ChatInterface } from "./components/ChatInterface";
import {
  ChatStoreProvider,
  useRuntimeState,
  useRuntimeActions,
} from "./store";

/**
 * Inner component that reads runtime from store
 */
function AppContent() {
  const { runtime } = useRuntimeState();
  const { setRuntime } = useRuntimeActions();

  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      <div
        style={{
          padding: "1rem",
          background: "#333",
          color: "white",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h2 style={{ margin: 0 }}>Edge LLM Playground</h2>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <label>
            Runtime:
            <select
              value={runtime}
              onChange={(e) => setRuntime(e.target.value as any)}
              style={{
                marginLeft: "0.5rem",
                padding: "0.25rem",
                borderRadius: "4px",
              }}
            >
              <option value="transformers">Transformers.js (FunctionGemma)</option>
              <option value="webllm">WebLLM (FunctionGemma-MLC)</option>
              <option value="api">API (Ollama Bridge)</option>
            </select>
          </label>
        </div>
      </div>

      <LLMProvider
        key={runtime} // Force re-initialization when runtime changes
        config={{
          preferredRuntime: runtime,
          apiUrl: "http://localhost:3001/v1/chat/completions",
          // Use XML format for both FunctionGemma runtimes
          toolCallFormat: runtime === "api" ? "json" : "xml",
          debug: true,
          models: {
            // Custom FunctionGemma compiled to MLC/WebGPU format
            webllm: "/models/custom-functiongemma-mlc",
            // Custom FunctionGemma in ONNX format for Transformers.js
            transformers: "/models/custom-functiongemma",
            api: "llama3.2",
          },
        }}
        enableServiceWorker={false}
      >
        <ChatInterface />
      </LLMProvider>
    </div>
  );
}

/**
 * App root with store provider
 */
function App() {
  return (
    <ChatStoreProvider>
      <AppContent />
    </ChatStoreProvider>
  );
}

export default App;
