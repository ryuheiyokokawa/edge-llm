/**
 * WebLLM runtime implementation
 */
import { BaseRuntime } from "./base.js";
import type {
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
  ToolCallsResponse,
  ContentResponse,
  ToolCall,
  ToolCallChunk,
} from "../types.js";
import { RuntimeManager } from "./manager.js";
import type {
  MLCEngineInterface,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
} from "@mlc-ai/web-llm";

export class WebLLMRuntime extends BaseRuntime {
  private engine: MLCEngineInterface | null = null;
  private modelName: string;

  constructor() {
    super();
    // Default model - Hermes-2-Pro-Mistral is verified for tool calling
    this.modelName = "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC";
  }

  async initialize(config: RuntimeConfig): Promise<void> {
    this.config = config;
    this.setStatus("initializing");
    console.log("[WebLLM] Starting initialization...");

    // Check WebGPU availability
    const hasWebGPU = await RuntimeManager.checkWebGPUSupport();
    if (!hasWebGPU) {
      console.log("[WebLLM] WebGPU not available, skipping");
      throw new Error("WebGPU not available. WebLLM requires WebGPU support.");
    }

    console.log("[WebLLM] WebGPU available, proceeding...");

    // Determine model name from config
    this.modelName = config.models?.webllm || this.modelName;
    console.log("[WebLLM] Loading model:", this.modelName);

    try {
      // Dynamically import WebLLM to avoid issues if not available
      console.log("[WebLLM] Importing @mlc-ai/web-llm...");
      const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
      console.log("[WebLLM] Creating engine (this may download the model)...");

      // Initialize WebLLM engine
      this.engine = await CreateMLCEngine(this.modelName, {
        initProgressCallback: (report: { progress: number; text: string }) => {
          // Report loading progress
          console.log(
            `[WebLLM] Progress: ${(report.progress * 100).toFixed(1)}% - ${
              report.text
            }`
          );
          if (report.progress < 1) {
            this.setStatus("loading");
          }
        },
      });

      console.log("[WebLLM] Engine created successfully");
      this.setStatus("ready");
    } catch (error) {
      this.setStatus("error");
      const errorMsg = `Failed to initialize WebLLM: ${
        error instanceof Error ? error.message : String(error)
      }`;
      console.error("[WebLLM]", errorMsg);
      throw new Error(errorMsg);
    }
  }

  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse> {
    if (this.status !== "ready" || !this.engine) {
      throw new Error("WebLLM runtime not initialized");
    }

    try {
      // Convert messages to WebLLM format
      const webllmMessages = this.convertMessages(messages);

      // Convert tools to OpenAI format (WebLLM uses OpenAI-compatible API)
      const toolsFormat =
        tools.length > 0 ? this.convertTools(tools) : undefined;
    
      console.log("[WebLLM] Tools being sent:", tools.length > 0 ? toolsFormat : "No tools");
      console.log("[WebLLM] First tool structure:", tools.length > 0 ? JSON.stringify(toolsFormat![0], null, 2) : "N/A");

      // Create chat completion request
      if (options?.stream) {
        const streamingRequest: ChatCompletionRequestStreaming = {
          messages: webllmMessages,
          stream: true,
          ...(toolsFormat && { tools: toolsFormat, tool_choice: "auto" }),
          ...(options?.temperature !== undefined && {
            temperature: options.temperature,
          }),
          ...(options?.maxTokens !== undefined && {
            max_tokens: options.maxTokens,
          }),
        };
        return this.handleStreamingResponse(streamingRequest);
      } else {
        const nonStreamingRequest: ChatCompletionRequestNonStreaming = {
          messages: webllmMessages,
          stream: false,
          ...(toolsFormat && { tools: toolsFormat, tool_choice: "auto" }),
          ...(options?.temperature !== undefined && {
            temperature: options.temperature,
          }),
          ...(options?.maxTokens !== undefined && {
            max_tokens: options.maxTokens,
          }),
        };
        return this.handleCompleteResponse(nonStreamingRequest);
      }
    } catch (error) {
      throw new Error(
        `WebLLM chat error: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  /**
   * Convert our Message format to WebLLM format
   */
  private convertMessages(messages: Message[]): Array<
    | { role: "system"; content: string }
    | { role: "user"; content: string }
    | {
        role: "assistant";
        content: string | null;
        tool_calls?: Array<{
          id: string;
          type: "function";
          function: { name: string; arguments: string };
        }>;
      }
    | { role: "tool"; content: string; tool_call_id: string; name?: string }
  > {
    return messages.map((msg) => {
      if (msg.role === "system") {
        return { role: "system" as const, content: msg.content };
      } else if (msg.role === "user") {
        return { role: "user" as const, content: msg.content };
      } else if (msg.role === "assistant") {
        const converted: {
          role: "assistant";
          content: string | null;
          tool_calls?: Array<{
            id: string;
            type: "function";
            function: { name: string; arguments: string };
          }>;
        } = {
          role: "assistant" as const,
          content: msg.content || null,
        };

        // Handle tool calls
        if (msg.tool_calls && msg.tool_calls.length > 0) {
          converted.tool_calls = msg.tool_calls.map((tc) => ({
            id: tc.id,
            type: "function" as const,
            function: {
              name: tc.name,
              arguments: JSON.stringify(tc.arguments),
            },
          }));
        }

        return converted;
      } else if (msg.role === "tool") {
        return {
          role: "tool" as const,
          content: msg.content,
          tool_call_id: msg.tool_call_id!,
          ...(msg.name && { name: msg.name }),
        };
      } else {
        // Fallback - shouldn't happen
        return { role: "user" as const, content: msg.content };
      }
    });
  }

  /**
   * Convert tools to OpenAI format
   */
  private convertTools(tools: ToolDefinition[]): Array<{
    type: "function";
    function: {
      name: string;
      description: string;
      parameters: any;
    };
  }> {
    return tools.map((tool) => ({
      type: "function" as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }

  /**
   * Handle streaming response
   */
  private async handleStreamingResponse(
    requestOptions: ChatCompletionRequestStreaming
  ): Promise<ContentResponse> {
    if (!this.engine) {
      throw new Error("Engine not initialized");
    }

    const chunks = await this.engine.chat.completions.create(requestOptions);

    // Create async iterator for streaming
    const stream = this.createStreamIterator(chunks);

    return {
      type: "content",
      stream,
    };
  }

  /**
   * Create async iterable from WebLLM chunks
   * Uses OpenAI-compatible tool calling format
   */
  private async *createStreamIterator(
    chunks: AsyncIterable<ChatCompletionChunk>
  ): AsyncIterable<string | ToolCallChunk> {
    let currentToolCalls: Map<number, {
      id?: string;
      name?: string;
      arguments: string;
    }> = new Map();
    
    let textBuffer = "";
    let hasToolCalls = false;

    for await (const chunk of chunks) {
      console.log("[WebLLM] Stream chunk:", JSON.stringify(chunk));
      const choice = chunk.choices?.[0];
      const delta = choice?.delta;
      const finishReason = choice?.finish_reason;

      // Accumulate text content (but don't yield yet if tool calls are present)
      if (delta?.content) {
        textBuffer += delta.content;
      }

      // Handle OpenAI-style tool calls in delta
      if (delta?.tool_calls && delta.tool_calls.length > 0) {
        hasToolCalls = true;
        for (const tc of delta.tool_calls) {
          const index = tc.index ?? 0;
          const current = currentToolCalls.get(index) || { arguments: "" };

          if (tc.id) current.id = tc.id;
          if (tc.function?.name) current.name = tc.function.name;
          if (tc.function?.arguments) {
            current.arguments += tc.function.arguments;
          }

          currentToolCalls.set(index, current);
        }
      }

      // When stream finishes with tool calls, yield them
      if (finishReason === "tool_calls" && currentToolCalls.size > 0) {
        hasToolCalls = true;
        for (const [_, toolCall] of currentToolCalls) {
          if (toolCall.name && toolCall.arguments) {
            try {
              const args = JSON.parse(toolCall.arguments);
              yield {
                type: "tool_call_chunk",
                tool_call: {
                  id: toolCall.id || `call_${Date.now()}_${Math.random()}`,
                  name: toolCall.name,
                  arguments: args,
                },
              };
            } catch (e) {
              console.error("[WebLLM] Failed to parse tool arguments:", e);
            }
          }
        }
        currentToolCalls.clear();
        // Don't yield text buffer when we have tool calls
        textBuffer = "";
      }
    }
    
    // Only yield text if we didn't have tool calls
    if (!hasToolCalls && textBuffer.trim()) {
      yield textBuffer;
    }
  }

  /**
   * Handle complete (non-streaming) response
   */
  private async handleCompleteResponse(
    requestOptions: ChatCompletionRequestNonStreaming
  ): Promise<ModelResponse> {
    if (!this.engine) {
      throw new Error("Engine not initialized");
    }

    const response = await this.engine.chat.completions.create(requestOptions);

    // Type guard to check if it's a ChatCompletion (not AsyncIterable)
    if (Symbol.asyncIterator in response) {
      throw new Error("Unexpected streaming response in non-streaming mode");
    }

    const completion = response as ChatCompletion;

    // Check if response contains tool calls
    const choice = completion.choices?.[0];
    if (!choice) {
      throw new Error("No response from WebLLM");
    }

    const message = choice.message;

    // Check for tool calls
    if (message.tool_calls && message.tool_calls.length > 0) {
      const toolCalls: ToolCall[] = message.tool_calls.map((tc) => ({
        id: tc.id || `call_${Date.now()}_${Math.random()}`,
        name: tc.function?.name || "",
        arguments: this.parseToolArguments(tc.function?.arguments || "{}"),
      }));

      return {
        type: "tool_calls",
        calls: toolCalls,
      } as ToolCallsResponse;
    }

    // Regular content response
    return {
      type: "content",
      text: message.content || "",
    } as ContentResponse;
  }

  /**
   * Parse tool arguments from JSON string
   */
  private parseToolArguments(args: string): Record<string, unknown> {
    try {
      return JSON.parse(args);
    } catch {
      // If parsing fails, return empty object
      return {};
    }
  }

  /**
   * Dispose resources
   */
  async dispose(): Promise<void> {
    if (this.engine) {
      // WebLLM doesn't have explicit dispose, but we can clear the reference
      this.engine = null;
    }
    await super.dispose();
  }
}
