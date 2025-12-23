/**
 * API fallback runtime implementation
 * Will be implemented in Phase 1 with OpenAI/Anthropic/OpenRouter support
 */
import { BaseRuntime } from "./base.js";
import type {
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
} from "../types.js";

export class APIRuntime extends BaseRuntime {
  async initialize(config: RuntimeConfig): Promise<void> {
    this.config = config;
    this.setStatus("initializing");
    this.log(`[API Runtime] Initializing with endpoint: ${config.apiUrl || "none"}`);

    if (!config.apiUrl) {
       // If no URL, we might still be "ready" if we want to default to something, 
       // but user said we need to point to an endpoint.
       this.log("[API Runtime] Warning: No apiUrl provided during initialization");
    }

    this.setStatus("ready");
  }

  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse> {
    if (this.status !== "ready" || !this.config) {
      throw new Error("API runtime not initialized");
    }

    const { apiUrl, apiKey, debug } = this.config;
    if (!apiUrl) {
      throw new Error("API URL is required for API runtime");
    }

    if (debug) {
      console.log("[API Runtime] Sending request to:", apiUrl);
    }

    const body = {
      messages,
      tools: tools.length > 0 ? tools.map(t => ({
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters
        }
      })) : undefined,
      stream: options?.stream ?? false,
      model: this.config.modelId
    };

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { "Authorization": `Bearer ${apiKey}` } : {})
      },
      body: JSON.stringify(body),
      signal: this.config.signal // Use the abort signal from config
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed (${response.status}): ${errorText}`);
    }

    if (options?.stream) {
      // Basic streaming implementation for fetch
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Response body is null");

      return {
        type: "content",
        stream: (async function* () {
          const decoder = new TextDecoder();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              const chunk = decoder.decode(value, { stream: true });
              
              // Handle SSE format "data: {...}"
              const lines = chunk.split("\n").filter(l => l.trim().startsWith("data: "));
              for (const line of lines) {
                const dataStr = line.replace(/^data: /, "").trim();
                if (dataStr === "[DONE]") break;
                try {
                  const data = JSON.parse(dataStr);
                  const content = data.choices?.[0]?.delta?.content || "";
                  if (content) yield content;
                } catch (e) {
                  // Fallback for non-standard streams
                  yield dataStr;
                }
              }
            }
          } finally {
            reader.releaseLock();
          }
        })()
      };
    } else {
      const data = await response.json();
      
      // Handle tool calls in response
      const toolCalls = data.choices?.[0]?.message?.tool_calls;
      if (toolCalls && toolCalls.length > 0) {
        return {
          type: "tool_calls",
          calls: toolCalls.map((tc: any) => ({
            id: tc.id || `call_${Math.random().toString(36).substr(2, 9)}`,
            name: tc.function?.name || tc.name,
            arguments: typeof tc.function?.arguments === "string" 
              ? JSON.parse(tc.function.arguments) 
              : (tc.function?.arguments || tc.arguments)
          }))
        };
      }

      return {
        type: "content",
        text: data.choices?.[0]?.message?.content || data.content || ""
      };
    }
  }
}
