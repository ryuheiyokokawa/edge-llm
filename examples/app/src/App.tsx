import { useState, useEffect } from "react";
import { LLMProvider, useLLM } from "@edge-llm/react";
import type { ToolDefinition, Message } from "@edge-llm/core";

// Example tools
const exampleTools: ToolDefinition[] = [
  {
    name: "calculate",
    description: "Evaluate a mathematical expression",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description:
            "Mathematical expression to evaluate (e.g., '2+2', '10*5')",
        },
      },
      required: ["expression"],
    },
    handler: async (args: Record<string, unknown>) => {
      const expression = args.expression as string;
      try {
        // Safe evaluation - in production, use a proper math parser
        const result = Function(`"use strict"; return (${expression})`)();
        return { result, expression };
      } catch (error) {
        return { error: `Invalid expression: ${expression}` };
      }
    },
  },
  {
    name: "getCurrentTime",
    description: "Get the current date and time",
    parameters: {
      type: "object",
      properties: {},
      required: [],
    },
    handler: async () => {
      return {
        time: new Date().toISOString(),
        timestamp: Date.now(),
      };
    },
  },
  {
    name: "searchWeb",
    description: "Search the web (mock implementation)",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
    },
    handler: async (args: Record<string, unknown>) => {
      const query = args.query as string;
      // Mock search results
      return {
        query,
        results: [
          { title: `Result 1 for "${query}"`, url: "https://example.com/1" },
          { title: `Result 2 for "${query}"`, url: "https://example.com/2" },
        ],
      };
    },
  },
];

function ChatInterface() {
  const { send, registerTools, status, initialized } = useLLM();
  const [messages, setMessages] = useState<
    Array<{ role: string; content: string }>
  >([]);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Register tools on mount
  useEffect(() => {
    if (initialized) {
      registerTools(exampleTools);
    }
  }, [initialized, registerTools]);

  const handleSend = async () => {
    if (!input.trim() || loading || !initialized) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);
    setError(null);

    // Add user message to history
    const userMsg: Message = { role: "user", content: userMessage };
    setConversationHistory((prev) => [...prev, userMsg]);
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      let currentHistory = [...conversationHistory, userMsg];
      let keepGoing = true;
      let iterations = 0;
      const MAX_ITERATIONS = 5;

      while (keepGoing && iterations < MAX_ITERATIONS) {
        iterations++;
        console.log(`[App] Iteration ${iterations}, sending history length: ${currentHistory.length}`);
        
        const response = await send(currentHistory as any);
        console.log("[App] Received response:", response);

        if (response.type === "content") {
          // Final answer
          const assistantMsg: Message = { 
            role: "assistant", 
            content: response.text || "" 
          };
          setConversationHistory((prev) => [...prev, assistantMsg]);
          setMessages((prev) => [...prev, { role: "assistant", content: response.text || "" }]);
          keepGoing = false;
        } else if (response.type === "tool_calls") {
          // Handle tool calls
          const toolCalls = response.calls;
          const assistantMsg: Message = {
            role: "assistant",
            content: "", // JSON mode usually puts content in the message, but for history we can leave empty or put the JSON
            tool_calls: toolCalls.map(tc => ({
              id: tc.id,
              name: tc.name,
              arguments: tc.arguments
            }))
          };
          
          // Add assistant message with tool calls to history
          currentHistory.push(assistantMsg);
          setConversationHistory((prev) => [...prev, assistantMsg]);

          // Show tool execution in UI
          const toolCallUIMessages = toolCalls.map((call) => ({
            role: "assistant",
            content: `🔧 Calling tool: ${call.name}(${JSON.stringify(call.arguments)})`,
          }));
          setMessages((prev) => [...prev, ...toolCallUIMessages as any]);

          // Execute tools
          const toolResults = [];
          for (const call of toolCalls) {
            console.log(`[App] Executing tool: ${call.name}`);
            try {
              // Find tool
              const tool = exampleTools.find(t => t.name === call.name);
              if (!tool) throw new Error(`Tool ${call.name} not found`);

              const result = await tool.handler(call.arguments);
              toolResults.push({
                tool_call_id: call.id,
                result: result
              });
              setMessages((prev) => [...prev, { role: "assistant", content: `✅ Tool ${call.name} result: ${JSON.stringify(result)}` }]);
            } catch (e) {
              console.error(`[App] Tool execution failed:`, e);
              const errorResult = { error: String(e) };
              toolResults.push({
                tool_call_id: call.id,
                result: errorResult
              });
              setMessages((prev) => [...prev, { role: "assistant", content: `❌ Tool ${call.name} failed: ${JSON.stringify(errorResult)}` }]);
            }
          }

          // Add tool results to history
          const toolResultMessages: Message[] = toolResults.map(tr => ({
            role: "tool",
            content: JSON.stringify(tr.result),
            tool_call_id: tr.tool_call_id
          }));
          
          currentHistory.push(...toolResultMessages);
          setConversationHistory((prev) => [...prev, ...toolResultMessages]);
          
          // Loop continues to send tool results back to model
        } else {
          keepGoing = false;
        }
      }
    } catch (err) {
      console.error("[App] Error in chat loop:", err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(errorMessage);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMessage}` },
      ]);
      setConversationHistory((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, an error occurred." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "2rem" }}>
      <h1>Edge LLM Example App</h1>

      <div
        style={{
          marginBottom: "1rem",
          padding: "0.5rem",
          background: "#f0f0f0",
          borderRadius: "4px",
        }}
      >
        <strong>Status:</strong> {status} {initialized ? "✓" : "..."}
      </div>

      {error && (
        <div
          style={{
            marginBottom: "1rem",
            padding: "0.5rem",
            background: "#ffebee",
            color: "#c62828",
            borderRadius: "4px",
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      <div
        style={{
          border: "1px solid #ddd",
          borderRadius: "4px",
          padding: "1rem",
          marginBottom: "1rem",
          minHeight: "400px",
          maxHeight: "600px",
          overflowY: "auto",
          background: "#fafafa",
        }}
      >
        {messages.length === 0 ? (
          <div style={{ color: "#666", fontStyle: "italic" }}>
            Start a conversation! Try asking:
            <ul>
              <li>"What is 15 * 23?"</li>
              <li>"What time is it?"</li>
              <li>"Search for React hooks"</li>
            </ul>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: "1rem",
                padding: "0.5rem",
                background: msg.role === "user" ? "#e3f2fd" : "#fff",
                borderRadius: "4px",
                borderLeft: `3px solid ${
                  msg.role === "user" ? "#2196f3" : "#4caf50"
                }`,
              }}
            >
              <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong>{" "}
              {msg.content}
            </div>
          ))
        )}
        {loading && (
          <div style={{ color: "#666", fontStyle: "italic" }}>Thinking...</div>
        )}
      </div>

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your message..."
          disabled={!initialized || loading}
          style={{
            flex: 1,
            padding: "0.5rem",
            border: "1px solid #ddd",
            borderRadius: "4px",
            fontSize: "1rem",
          }}
        />
        <button
          onClick={handleSend}
          disabled={!initialized || loading || !input.trim()}
          style={{
            padding: "0.5rem 1rem",
            background:
              initialized && !loading && input.trim() ? "#2196f3" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor:
              initialized && !loading && input.trim()
                ? "pointer"
                : "not-allowed",
            fontSize: "1rem",
          }}
        >
          Send
        </button>
      </div>

      <div
        style={{
          marginTop: "2rem",
          padding: "1rem",
          background: "#f5f5f5",
          borderRadius: "4px",
        }}
      >
        <h3>Available Tools:</h3>
        <ul>
          {exampleTools.map((tool) => (
            <li key={tool.name}>
              <strong>{tool.name}</strong>: {tool.description}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function App() {
  return (
    <LLMProvider
      config={{
        preferredRuntime: "webllm",
        models: {
          webllm: "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
          transformers: "Xenova/Qwen2.5-0.5B-Instruct",
        },
      }}
      enableServiceWorker={false}
    >
      <ChatInterface />
    </LLMProvider>
  );
}

export default App;
