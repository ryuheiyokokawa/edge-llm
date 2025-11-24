/**
 * Transformers.js runtime implementation
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

// Transformers.js types
type TextGenerationPipeline = any;

export class TransformersRuntime extends BaseRuntime {
  private pipeline: TextGenerationPipeline | null = null;
  private modelName: string;
  private tokenizer: any = null;

  constructor() {
    super();
    // Default model, can be overridden by config
    this.modelName = "Xenova/Qwen2.5-0.5B-Instruct";
  }

  async initialize(config: RuntimeConfig): Promise<void> {
    this.config = config;
    this.setStatus("initializing");
    if (config.debug) {
      console.log("[Transformers.js] Starting initialization...");
    }

    // Check WASM support
    const hasWASM = RuntimeManager.checkWASMSupport();
    if (!hasWASM) {
      if (config.debug) {
        console.log("[Transformers.js] WASM not available, skipping");
      }
      throw new Error(
        "WASM not available. Transformers.js requires WASM support."
      );
    }

    if (config.debug) {
      console.log("[Transformers.js] WASM available, proceeding...");
    }

    // Determine model name from config
    this.modelName = config.models?.transformers || this.modelName;
    if (config.debug) {
      console.log("[Transformers.js] Loading model:", this.modelName);
    }

    try {
      // Dynamically import Transformers.js to avoid issues if not available
      if (config.debug) {
        console.log("[Transformers.js] Importing @xenova/transformers...");
      }
      const { pipeline, AutoTokenizer } = await import("@xenova/transformers");

      // Initialize tokenizer
      if (config.debug) {
        console.log("[Transformers.js] Loading tokenizer...");
      }
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelName, {
        progress_callback: (progress: {
          status: string;
          progress?: number;
        }) => {
          if (config.debug) {
            console.log(
              `[Transformers.js] Tokenizer: ${progress.status}${
                progress.progress
                  ? ` (${(progress.progress * 100).toFixed(1)}%)`
                  : ""
              }`
            );
          }
          if (progress.status === "loading") {
            this.setStatus("loading");
          }
        },
      });

      // Initialize text generation pipeline
      if (config.debug) {
        console.log(
          "[Transformers.js] Loading pipeline (this may download the model)..."
        );
      }
      this.pipeline = await pipeline("text-generation", this.modelName, {
        progress_callback: (progress: {
          status: string;
          progress?: number;
        }) => {
          if (config.debug) {
            console.log(
              `[Transformers.js] Pipeline: ${progress.status}${
                progress.progress
                  ? ` (${(progress.progress * 100).toFixed(1)}%)`
                  : ""
              }`
            );
          }
          if (progress.status === "loading") {
            this.setStatus("loading");
          }
        },
      });

      if (config.debug) {
        console.log("[Transformers.js] Pipeline loaded successfully");
      }
      this.setStatus("ready");
    } catch (error) {
      this.setStatus("error");
      const errorMsg = `Failed to initialize Transformers.js: ${
        error instanceof Error ? error.message : String(error)
      }`;
      console.warn("[Transformers.js]", errorMsg);
      throw new Error(errorMsg);
    }
  }

  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse> {
    if (this.status !== "ready" || !this.pipeline || !this.tokenizer) {
      throw new Error("Transformers.js runtime not initialized");
    }

    try {
      // Format messages for chat template
      const prompt = this.formatChatPrompt(messages, tools);

      // Generate response
      const generationOptions: any = {
        max_new_tokens: options?.maxTokens || 512,
        temperature: options?.temperature || 0.7,
        do_sample: options?.temperature ? options.temperature > 0 : true,
        return_full_text: false,
      };

      if (options?.stream) {
        return this.handleStreamingResponse(prompt, generationOptions);
      } else {
        return this.handleCompleteResponse(prompt, generationOptions, tools);
      }
    } catch (error) {
      throw new Error(
        `Transformers.js chat error: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  /**
   * Format messages into a chat prompt using the model's chat template
   */
  private formatChatPrompt(
    messages: Message[],
    tools: ToolDefinition[]
  ): string {
    // Build system message with tool definitions if present
    let systemMessage = "";
    if (tools.length > 0) {
      const toolsDescription = tools
        .map(
          (tool) =>
            `- ${tool.name}: ${
              tool.description
            }\n  Parameters: ${JSON.stringify(tool.parameters)}`
        )
        .join("\n");
      systemMessage = `You have access to the following tools:\n${toolsDescription}\n\nWhen you need to use a tool, respond with a JSON object in this format:\n{"tool": "tool_name", "arguments": {...}}\n\nOtherwise, respond normally.\n\n`;
    }

    // Apply chat template if available
    if (
      this.tokenizer &&
      typeof this.tokenizer.apply_chat_template === "function"
    ) {
      try {
        // Convert messages to format expected by chat template
        const chatMessages = messages.map((msg) => {
          if (msg.role === "tool") {
            // Transform tool messages to assistant messages with content
            return {
              role: "assistant" as const,
              content: msg.content,
            };
          }
          return {
            role: msg.role as "system" | "user" | "assistant",
            content: msg.content,
          };
        });

        // Add system message if we have tools
        if (systemMessage) {
          chatMessages.unshift({
            role: "system" as const,
            content: systemMessage,
          });
        }

        return this.tokenizer.apply_chat_template(chatMessages, {
          tokenize: false,
          add_generation_prompt: true,
        }) as string;
      } catch (error) {
        // Fallback to manual formatting if chat template fails
        console.warn(
          "Chat template application failed, using fallback:",
          error
        );
      }
    }

    // Fallback: manual prompt formatting
    let prompt = systemMessage;
    for (const msg of messages) {
      if (msg.role === "system") {
        prompt += `System: ${msg.content}\n\n`;
      } else if (msg.role === "user") {
        prompt += `User: ${msg.content}\n\n`;
      } else if (msg.role === "assistant") {
        prompt += `Assistant: ${msg.content}\n\n`;
      } else if (msg.role === "tool") {
        prompt += `Tool result (${msg.name || "unknown"}): ${msg.content}\n\n`;
      }
    }
    prompt += "Assistant: ";

    return prompt;
  }

  /**
   * Handle streaming response
   */
  private async handleStreamingResponse(
    prompt: string,
    options: any
  ): Promise<ContentResponse> {
    if (!this.pipeline) {
      throw new Error("Pipeline not initialized");
    }

    // Create async iterator for streaming
    const stream = this.createStreamIterator(prompt, options);

    return {
      type: "content",
      stream,
    };
  }

  /**
   * Create async iterable from Transformers.js generation
   */
  private async *createStreamIterator(
    prompt: string,
    options: any
  ): AsyncIterable<string | ToolCallChunk> {
    if (!this.pipeline) {
      throw new Error("Pipeline not initialized");
    }

    // Transformers.js doesn't have native streaming, so we'll simulate it
    // by generating in chunks or using a callback-based approach
    const output = await this.pipeline(prompt, {
      ...options,
    });

    // For now, yield the full text (Transformers.js streaming is limited)
    // In a real implementation, we might need to use a different approach
    const text = Array.isArray(output)
      ? output[0]?.generated_text || ""
      : output?.generated_text || "";

    // Yield the generated text (after removing the prompt)
    const generatedText = text.replace(prompt, "").trim();
    
    // Check for tool calls first
    // In this simulated streaming, we can just parse at the end, but we should yield text first
    // However, if the text IS a tool call, we shouldn't yield it as content.
    
    // Let's try to parse tool calls
    // We need access to tools to validate, but we don't have them here easily.
    // But we can use the same regex logic as handleCompleteResponse if we had the tools.
    // For now, let's just yield text. 
    // TODO: Pass tools to createStreamIterator to properly detect tool calls.
    
    if (generatedText) {
      // Simulate streaming by yielding character by character
      // In production, you might want to use token-based chunking
      for (const char of generatedText) {
        yield char;
      }
    }
  }

  /**
   * Handle complete (non-streaming) response
   */
  private async handleCompleteResponse(
    prompt: string,
    options: any,
    tools: ToolDefinition[]
  ): Promise<ModelResponse> {
    if (!this.pipeline) {
      throw new Error("Pipeline not initialized");
    }

    const output = await this.pipeline(prompt, options);

    // Extract generated text
    const generatedText = Array.isArray(output)
      ? output[0]?.generated_text || ""
      : (output?.generated_text as string | undefined) || "";

    // Remove the prompt from the generated text
    const responseText = generatedText
      ? generatedText.replace(prompt, "").trim()
      : "";

    // Try to parse tool calls from the response
    const toolCalls = this.parseToolCallsFromResponse(responseText, tools);

    if (toolCalls.length > 0) {
      return {
        type: "tool_calls",
        calls: toolCalls,
      } as ToolCallsResponse;
    }

    // Regular content response
    return {
      type: "content",
      text: responseText,
    } as ContentResponse;
  }

  /**
   * Parse tool calls from model response
   * Looks for JSON objects in the format: {"tool": "tool_name", "arguments": {...}}
   */
  private parseToolCallsFromResponse(
    response: string,
    tools: ToolDefinition[]
  ): ToolCall[] {
    const toolCalls: ToolCall[] = [];

    // Try to find JSON objects in the response
    // Look for patterns like {"tool": "...", "arguments": {...}}
    const jsonPattern =
      /\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}/g;
    let match;

    while ((match = jsonPattern.exec(response)) !== null) {
      const toolName = match[1];
      const argumentsStr = match[2];

      if (!toolName || !argumentsStr) {
        continue; // Skip invalid matches
      }

      // Check if tool exists
      const tool = tools.find((t) => t.name === toolName);
      if (!tool) {
        continue; // Skip unknown tools
      }

      try {
        const argumentsObj = JSON.parse(argumentsStr);
        toolCalls.push({
          id: `call_${Date.now()}_${Math.random()}`,
          name: toolName,
          arguments: argumentsObj,
        });
      } catch {
        // Invalid JSON, skip
        continue;
      }
    }

    // Alternative: Try to parse the entire response as a tool call
    if (toolCalls.length === 0) {
      try {
        const parsed = JSON.parse(response.trim());
        if (parsed.tool && parsed.arguments) {
          const tool = tools.find((t) => t.name === parsed.tool);
          if (tool) {
            toolCalls.push({
              id: `call_${Date.now()}_${Math.random()}`,
              name: parsed.tool,
              arguments: parsed.arguments,
            });
          }
        }
      } catch {
        // Not a JSON tool call, that's fine
      }
    }

    return toolCalls;
  }

  /**
   * Dispose resources
   */
  async dispose(): Promise<void> {
    if (this.pipeline) {
      // Transformers.js doesn't have explicit dispose, but we can clear references
      this.pipeline = null;
    }
    if (this.tokenizer) {
      this.tokenizer = null;
    }
    await super.dispose();
  }
}
