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
    console.log("[API Runtime] Starting initialization...");

    if (!config.apiKey) {
      console.log("[API Runtime] No API key provided, skipping");
      throw new Error("API key required for API runtime");
    }

    // TODO: Implement API client initialization
    // - Validate API key
    // - Set up client (OpenAI/Anthropic/OpenRouter)

    console.log("[API Runtime] Initialized (stub implementation)");
    this.setStatus("ready");
  }

  async chat(
    _messages: Message[],
    _tools: ToolDefinition[],
    _options?: ChatOptions
  ): Promise<ModelResponse> {
    if (this.status !== "ready") {
      throw new Error("API runtime not initialized");
    }

    // TODO: Implement API inference
    // - Format messages for API
    // - Include tool definitions in OpenAI format
    // - Make API request
    // - Parse response (tool_calls or content)
    // - Handle streaming if requested

    throw new Error("API runtime not yet implemented");
  }
}
