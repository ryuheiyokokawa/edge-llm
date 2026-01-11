/**
 * WebLLM runtime implementation
 */
import {
  ChatOptions,
  Runtime,
  RuntimeType,
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
} from "../types.js";
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
    const signal = config.signal;

    this.initPromise = (async () => {
      try {
        if (signal?.aborted) throw new Error("Aborted");

        if (this.config.debug) {
          console.log("[WebLLM] Initializing engine...");
        }
        const modelId = this.config.modelId || defaultConfig.modelId!;

        // Check if this is a local model path (starts with /)
        const isLocalModel = modelId.startsWith("/");
        
        let engineConfig: webllm.MLCEngineConfig = {
          initProgressCallback: (report) => {
            if (signal?.aborted) {
              return;
            }
            if (this.config.debug) {
              console.log("[WebLLM] Progress:", report.text);
            }
          },
        };

        if (isLocalModel) {
          // For local models, create a custom model record
          // WebLLM needs absolute URLs, so prepend the origin
          const origin = typeof window !== 'undefined' ? window.location.origin : '';
          const absoluteModelUrl = origin + modelId + '/';
          const customModelId = "custom-functiongemma-mlc";
          
          if (this.config.debug) {
            console.log("[WebLLM] Loading local model from:", absoluteModelUrl);
          }

          // Create custom app config with our local model
          // Note: WebLLM's cleanModelUrl() adds resolve/main/ to URLs (HuggingFace pattern)
          // So our files must be in that subdirectory
          const customAppConfig: webllm.AppConfig = {
            model_list: [
              {
                model: absoluteModelUrl,  // WebLLM will add resolve/main/
                model_id: customModelId,
                // model_lib must point to actual WASM location (with resolve/main/)
                model_lib: absoluteModelUrl + "resolve/main/model-webgpu.wasm",
                overrides: {
                  context_window_size: 8192,
                },
              },
            ],
          };

          engineConfig.appConfig = customAppConfig;
          this.engine = await webllm.CreateMLCEngine(customModelId, engineConfig);
        } else {
          // For prebuilt models, use the model ID directly
          this.engine = await webllm.CreateMLCEngine(modelId, engineConfig);
        }

        if (signal?.aborted) {
          if (this.engine) {
            await this.engine.unload();
            this.engine = null;
          }
          throw new Error("Aborted");
        }

        if (this.config.debug) {
          console.log("[WebLLM] Engine created successfully");
        }
      } catch (error) {
        if (error instanceof Error && error.message === "Aborted") {
           this.initPromise = null;
           throw error;
        }
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

  getType(): RuntimeType {
    return "webllm";
  }

  async dispose(): Promise<void> {
    if (this.engine) {
      await this.engine.unload();
      this.engine = null;
    }
  }

  async clearCache(): Promise<void> {
    if (typeof caches === "undefined") return;
    
    const cacheNames = await caches.keys();
    for (const name of cacheNames) {
      // MLC Chat uses caches for models
      if (name.includes("mlc-chat") || name.includes("web-llm")) {
        await caches.delete(name);
      }
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

    if (this.config.debug) {
      console.log("[WebLLM] Chat request:", { messages, tools });
    }

    // Prepare messages with system prompt if tools are available
    let finalMessages = [...messages];
    if (tools && tools.length > 0) {
      const format = this.config?.toolCallFormat || "json";
      let systemPrompt = "";

      if (format === "xml") {
        systemPrompt = `You are a model that can do function calling with the following functions. 
Must use the EXACT format: <start_function_call>call:name{arg:<escape>val<escape>}<end_function_call>
Example: <start_function_call>call:calculate{expression:<escape>5*12<escape>}<end_function_call>`;
      } else {
        systemPrompt = generateSystemPrompt(tools);
      }
      // Check if there is already a system message
      const systemIndex = finalMessages.findIndex((m) => m.role === "system");
      if (systemIndex >= 0) {
        const existingSystemMsg = finalMessages[systemIndex];
        if (existingSystemMsg) {
          finalMessages[systemIndex] = {
            ...existingSystemMsg,
            content: existingSystemMsg.content + "\n\n" + systemPrompt,
          };
        }
      } else {
        finalMessages.unshift({ role: "system", content: systemPrompt });
      }
    }

    // Convert messages to WebLLM format
    const webllmMessages: webllm.ChatCompletionMessageParam[] =
      finalMessages.map((m) => {
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

    if (this.config.debug) {
      console.log("[WebLLM] Sending request:", requestOptions);
    }
    const response = await this.engine.chat.completions.create(requestOptions);
    if (this.config.debug) {
      console.log("[WebLLM] Received response:", response);
    }

    const content = response.choices[0]?.message?.content || "";

    // Try to parse tool call from content
    if (tools && tools.length > 0) {
      const format = this.config?.toolCallFormat || "json";
      
      if (format === "xml") {
        // FunctionGemma style XML parsing logic
        const functionCallPattern = /[<(]start_function_call[>)](?:call:)?([a-zA-Z0-9_-]+)\{(.*?)\}[<(]end_function_call[>)]/gs;
        let match = functionCallPattern.exec(content);
        if (match) {
          const toolName = match[1]?.trim();
          let argsString = match[2]?.trim();
          if (toolName && toolName !== 'error' && argsString !== undefined) {
             argsString = argsString.replace(/<escape>/g, '"').replace(/([a-zA-Z0-9_]+):/g, '"$1":');
             try {
               const args = JSON.parse(`{${argsString}}`);
               return {
                 type: "tool_calls",
                 calls: [{ id: `call_${Date.now()}`, name: toolName, arguments: args }],
                 text: content,
               };
             } catch (e) {
               console.warn("[WebLLM] Failed to parse XML-like args:", argsString, e);
             }
          }
        }
      } else {
        const json = extractJSON(content);
        if (json && json.tool && json.arguments) {
          if (this.config.debug) {
            console.log("[WebLLM] Detected JSON tool call:", json);
          }
          return {
            type: "tool_calls",
            calls: [
              {
                id: `call_${Date.now()}`,
                name: json.tool,
                arguments: json.arguments,
              },
            ],
            text: content,
          };
        }
      }
    }

    return {
      type: "content",
      text: content,
    };
  }
}
