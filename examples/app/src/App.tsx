import { useState } from "react";
import { LLMProvider } from "@edge-llm/react";
import { ChatInterface } from "./components/ChatInterface";

function App() {
  const [runtime, setRuntime] = useState<"webllm" | "transformers" | "api">("transformers");

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
              style={{ marginLeft: "0.5rem", padding: "0.25rem", borderRadius: "4px" }}
            >
              <option value="transformers">Transformers.js (FunctionGemma)</option>
              <option value="webllm">WebLLM (Llama 3)</option>
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
          toolCallFormat: runtime === "api" ? "json" : "xml",
          debug: true,
          models: {
            webllm: "Llama-3-8B-Instruct-q4f16_1-MLC",
            transformers: "onnx-community/functiongemma-270m-it-ONNX-GQA",
            api: "llama3.2", // Default model for Ollama
          },
        }}
        enableServiceWorker={false}
      >
        <ChatInterface />
      </LLMProvider>
    </div>
  );
}

export default App;
