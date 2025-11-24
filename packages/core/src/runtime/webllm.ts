/**
 * WebLLM runtime implementation
 */
import {
  ChatOptions,
  Runtime,
  Message,
  ModelResponse,
  RuntimeConfig,
  ToolDefinition,
} from "../types";
import * as webllm from "@mlc-ai/web-llm";
import { generateSystemPrompt } from "../prompt/system";
import { extractJSON } from "../utils/json-parser";

const defaultConfig: RuntimeConfig = {
  modelId: "Llama-3-8B-Instruct-q4f16_1-MLC",
  temperature: 0.7,
  maxTokens: 2048,
};

export class WebLLMRuntime implements Runtime {
  private engine: webllm.MLCEngineInterface | null = null;
  private config: RuntimeConfig;
  private initPromise: Promise<void> | null = null;

  constructor(config: Partial<RuntimeConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  async initialize(config: RuntimeConfig): Promise<void> {
    if (this.initPromise) return this.initPromise;
    
    // Update config with passed values
    this.config = { ...this.config, ...config };

    this.initPromise = (async () => {
      try {
        console.log("[WebLLM] Initializing engine...");
        const modelId = this.config.modelId || defaultConfig.modelId!;
        
        this.engine = await webllm.CreateMLCEngine(
          modelId,
          {
            initProgressCallback: (report) => {
              console.log("[WebLLM] Progress:", report.text);
            },
          }
        );
        console.log("[WebLLM] Engine created successfully");
      } catch (error) {
        console.error("[WebLLM] Initialization failed:", error);
        this.initPromise = null;
        throw error;
      }
    })();

    return this.initPromise;
  }

  getStatus(): "idle" | "initializing" | "loading" | "ready" | "error" {
    if (this.engine) return "ready";
    if (this.initPromise) return "initializing";
    return "idle";
  }

  async dispose(): Promise<void> {
    if (this.engine) {
      await this.engine.unload();
      this.engine = null;
    }
  }

  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse> {
    if (!this.engine) {
      throw new Error("Engine not initialized");
    }

    console.log("[WebLLM] Chat request:", { messages, tools });

    // Prepare messages with system prompt if tools are available
    let finalMessages = [...messages];
    if (tools && tools.length > 0) {
      const systemPrompt = generateSystemPrompt(tools);
      // Check if there is already a system message
      const systemIndex = finalMessages.findIndex(m => m.role === "system");
      if (systemIndex >= 0) {
        const existingSystemMsg = finalMessages[systemIndex];
        if (existingSystemMsg) {
          finalMessages[systemIndex] = {
            ...existingSystemMsg,
            content: existingSystemMsg.content + "\n\n" + systemPrompt
          };
        }
      } else {
        finalMessages.unshift({ role: "system", content: systemPrompt });
      }
    }

    // Convert messages to WebLLM format
    const webllmMessages: webllm.ChatCompletionMessageParam[] = finalMessages.map((m) => {
      const role = m.role;
      if (role === "user" || role === "system" || role === "assistant") {
        return {
          role: role,
          content: m.content || "",
        };
      }
      // Map tool messages to user messages for JSON mode simplicity
      return {
        role: "user",
        content: `[Tool Result] ${m.content}`,
      };
    });

    const requestOptions: webllm.ChatCompletionRequestNonStreaming = {
      stream: false, // Force non-streaming
      messages: webllmMessages,
      temperature: options?.temperature ?? this.config.temperature,
      max_tokens: options?.maxTokens ?? this.config.maxTokens,
    };

    console.log("[WebLLM] Sending request:", requestOptions);
    const response = await this.engine.chat.completions.create(requestOptions);
    console.log("[WebLLM] Received response:", response);

    const content = response.choices[0]?.message?.content || "";

    // Try to parse tool call from content
    if (tools && tools.length > 0) {
      const json = extractJSON(content);
      if (json && json.tool && json.arguments) {
        console.log("[WebLLM] Detected JSON tool call:", json);
        return {
          type: "tool_calls",
          calls: [
            {
              id: `call_${Date.now()}`,
              name: json.tool,
              arguments: json.arguments,
            },
          ],
        };
      }
    }

    return {
      type: "content",
      text: content,
    };
  }
}
