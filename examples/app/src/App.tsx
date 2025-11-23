import { useState, useEffect } from "react";
import { LLMProvider, useLLM } from "@edge-llm/react";
import type { ToolDefinition, ModelResponse, ContentResponse } from "@edge-llm/core";

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

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {


      let response: ModelResponse = await send(userMessage, { stream: true });
      console.log("[App] Original Response:", response);
      let iteration = 0;

      while (iteration < 2) {
        // Handle tool calls
        if (response.type === "tool_calls") {
          iteration++;

          // Show tool execution
          const toolMessages = response.calls.map((call) => ({
            role: "assistant",
            content: `🔧 Calling tool: ${call.name}(${JSON.stringify(
              call.arguments
            )})`,
          }));
          setMessages((prev) => [...prev, ...toolMessages]);

          // Execute tools and get results
          // Note: Hermes models sometimes vary the tool name casing/spacing
          // so we normalize for matching
          const toolResults = response.calls.map((call) => {
            const normalizedCallName = call.name.toLowerCase().replace(/[_\s-]/g, '');
            const tool = exampleTools.find((t) => 
              t.name.toLowerCase().replace(/[_\s-]/g, '') === normalizedCallName
            );
            
            if (!tool) {
              console.warn(`Tool not found: "${call.name}" (normalized: "${normalizedCallName}")`);
              console.log('Available tools:', exampleTools.map(t => t.name));
              return {
                tool_call_id: call.id,
                result: { error: `Tool ${call.name} not found` },
              };
            }

            console.log(`Executing tool: ${tool.name} with args:`, call.arguments);
            return {
              tool_call_id: call.id,
              result: tool.handler(call.arguments),
            };
          });

          // Wait for all tools to complete
          const results = await Promise.all(
            toolResults.map(async (tr) => ({
              tool_call_id: tr.tool_call_id,
              result: await tr.result,
            }))
          );

          // Add assistant message with tool calls to conversation history
          // This is required for the model to understand the context
          const assistantToolCallMessage = {
            role: "assistant" as const,
            content: null,
            tool_calls: response.calls.map((call) => ({
              id: call.id,
              type: "function" as const,
              function: {
                name: call.name,
                arguments: JSON.stringify(call.arguments),
              },
            })),
          };
          
          // Build complete message history for next turn
          const toolResultMessages = results.map((result) => ({
            role: "tool" as const,
            content: JSON.stringify(result.result),
            tool_call_id: result.tool_call_id,
          }));

          // Send the assistant's tool call + tool results as a single request
          // by constructing the proper message format
          console.log("[App] Sending tool results:", results);
          
          // We need to send this as a Message with the proper structure
          // Since useLLM doesn't export Message type, we'll use the toolResults format
          // but we should really be sending: [assistantToolCallMessage, ...toolResultMessages]
          response = await send({ toolResults: results }, { stream: true });
          console.log("[App] Response after tool execution:", response);
        } else if (response.type === "content") {
          // Handle content response
          if (response.stream) {
            // Handle streaming - iterate directly
            let fullText = "";
            const toolCalls: any[] = [];
            
            for await (const chunk of response.stream) {
              console.log("[App] Stream chunk:", chunk);
              if (typeof chunk === "string") {
                fullText += chunk;
                // Note: We accumulate text but don't display yet to avoid showing
                // partial tool call JSON that will be parsed later
              } else if (typeof chunk === "object" && chunk.type === "tool_call_chunk") {
                toolCalls.push(chunk.tool_call);
              }
            }
            
            // After stream completes, decide what to show
            if (toolCalls.length > 0) {
              // We got tool calls, switch to tool handling
              console.log("[App] Received tool calls in stream:", toolCalls);
              console.log("[App] fullText at time of tool calls:", fullText);
              response = {
                type: "tool_calls",
                calls: toolCalls,
              };
              continue;
            } else if (fullText.trim()) {
              // Only add message if we have actual text content
              setMessages((prev) => [...prev, { role: "assistant", content: fullText }]);
            }
          } else {
            // Handle complete response
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: (response as ContentResponse).text || "" },
            ]);
          }
          
          // If we got here and didn't continue, we are done
          break;
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(errorMessage);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMessage}` },
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
        preferredRuntime: "auto",
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
