/**
 * Transformers.js runtime implementation
 */
import { BaseRuntime } from "./base.js";
import type {
  RuntimeType,
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
  ToolCallsResponse,
  ContentResponse,
  ToolCall,
} from "../types.js";
import { RuntimeManager } from "./manager.js";
import { IndexedDBCache } from "../utils/indexeddb-cache.js";
import { generateSystemPrompt } from "../prompt/system.js";
import { extractJSON } from "../utils/json-parser.js";

// Transformers.js types
type TextGenerationPipeline = any;

export class TransformersRuntime extends BaseRuntime {
  private pipeline: TextGenerationPipeline | null = null;
  private modelName: string;
  private tokenizer: any = null;

  constructor() {
    super();
    // Default model: FunctionGemma (specialized for tool calling)
    this.modelName = "onnx-community/functiongemma-270m-it-ONNX";
  }

  private lastProgressMap = new Map<string, number>();
  private abortController: AbortController | null = null;

  async initialize(config: RuntimeConfig): Promise<void> {
    this.config = config;
    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    // Link external signal to our internal controller
    if (config.signal) {
      if (config.signal.aborted) {
        this.abortController.abort();
      } else {
        config.signal.addEventListener("abort", () => {
          this.abortController?.abort();
        });
      }
    }
    
    this.setStatus("initializing");
    this.log("[Transformers.js] Starting initialization...");

    // Check WASM support
    const hasWASM = RuntimeManager.checkWASMSupport();
    if (!hasWASM) {
      this.log("[Transformers.js] WASM not available, skipping");
      throw new Error(
        "WASM not available. Transformers.js requires WASM support."
      );
    }

    if (signal.aborted) throw new Error("Aborted");

    this.log("[Transformers.js] WASM available, proceeding...");

    // Determine model name from config
    this.modelName = config.models?.transformers || this.modelName;
    this.log("[Transformers.js] Loading model:", this.modelName);

    let pipeline, AutoTokenizer, env;

    try {
      // Dynamically import Transformers.js to avoid issues if not available
      this.log("[Transformers.js] Importing @huggingface/transformers...");
      const transformers = await import("@huggingface/transformers");
      ({ pipeline, AutoTokenizer, env } = transformers);

      if (signal.aborted) throw new Error("Aborted");

      // Configure environment if available
      if (env) {
        this.log("[Transformers.js] Configuring environment...");
        env.allowLocalModels = false;
        env.allowRemoteModels = true; // Explicitly allow remote models for initial download
        
        // Use our custom IndexedDB cache to bypass Cache API limitations
        // Only if we are in a browser environment with IndexedDB support
        if (typeof indexedDB !== 'undefined') {
          env.useBrowserCache = false;
          env.useCustomCache = true;
          (env as any).customCache = new IndexedDBCache();
        } else {
          // In Node/Tests, stick to defaults or disable browser cache
          env.useBrowserCache = false;
          env.useCustomCache = false;
        }
        
        this.log("[Transformers.js] Env config:", {
          allowLocalModels: env.allowLocalModels,
          allowRemoteModels: env.allowRemoteModels,
          useCustomCache: env.useCustomCache,
        });
      }
    } catch (error) {
       this.setStatus("error");
       throw new Error(`Import/Config failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (signal.aborted) throw new Error("Aborted");

    try {
      // Initialize tokenizer
      this.log("[Transformers.js] Loading tokenizer...");
      const tokenizerOptions: any = {
        signal, // Pass abort signal
        progress_callback: (progress: any) => {
          if (signal.aborted) throw new Error("Aborted");
          
          const file = progress.file || "tokenizer";
          if (progress.loaded !== undefined && progress.total !== undefined) {
            const percent = Math.floor((progress.loaded / progress.total) * 100);
            const lastPercent = this.lastProgressMap.get(file) ?? -1;
            
            if (percent !== lastPercent) {
              this.lastProgressMap.set(file, percent);
              // Only log every 5% for files to reduce noise, or 100%
              if (percent % 5 === 0 || percent === 100) {
                const sizeInfo = `(${(progress.loaded / 1024 / 1024).toFixed(1)}MB / ${(progress.total / 1024 / 1024).toFixed(1)}MB)`;
                this.log(`[Transformers.js] [${file}] ${progress.status}: ${percent}% ${sizeInfo}`);
              }
            }
          } else if (progress.progress !== undefined) {
            const percent = Math.floor(progress.progress < 1 ? progress.progress * 100 : progress.progress);
            const lastPercent = this.lastProgressMap.get(file) ?? -1;
            
            if (percent !== lastPercent) {
              this.lastProgressMap.set(file, percent);
              this.log(`[Transformers.js] [${file}] ${progress.status}: ${percent}%`);
            }
          } else if (progress.status === "initiate" || progress.status === "download" || progress.status === "done") {
            this.log(`[Transformers.js] [${file}] ${progress.status}`);
          }

          if (progress.status === "loading") {
            this.setStatus("loading");
          }
        },
      };
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelName, tokenizerOptions);
    } catch (error) {
       if (signal.aborted) throw new Error("Aborted");
       this.setStatus("error");
       throw new Error(`Tokenizer initialization failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (signal.aborted) throw new Error("Aborted");

    try {
      // Initialize text generation pipeline
      this.log(
        "[Transformers.js] Loading pipeline (this may download the model)..."
      );
      this.lastProgressMap.clear(); // Reset for pipeline
      const pipelineOptions: any = {
        // device: "wasm", // Removed to allow WebGPU (auto-detect)
        dtype: "fp32", // Switch to full precision (1.1GB) for maximum fidelity
        signal, // Pass abort signal
        progress_callback: (progress: any) => {
          if (signal.aborted) throw new Error("Aborted");

          const file = progress.file || "model";
          if (progress.loaded !== undefined && progress.total !== undefined) {
            const percent = Math.floor((progress.loaded / progress.total) * 100);
            const lastPercent = this.lastProgressMap.get(file) ?? -1;

            if (percent !== lastPercent) {
              this.lastProgressMap.set(file, percent);
              // Only log every 5% for files to reduce noise, or 100%
              if (percent % 5 === 0 || percent === 100) {
                const sizeInfo = `(${(progress.loaded / 1024 / 1024).toFixed(1)}MB / ${(progress.total / 1024 / 1024).toFixed(1)}MB)`;
                this.log(`[Transformers.js] [${file}] ${progress.status}: ${percent}% ${sizeInfo}`);
              }
            }
          } else if (progress.progress !== undefined) {
            const percent = Math.floor(progress.progress < 1 ? progress.progress * 100 : progress.progress);
            const lastPercent = this.lastProgressMap.get(file) ?? -1;

            if (percent !== lastPercent) {
              this.lastProgressMap.set(file, percent);
              this.log(`[Transformers.js] [${file}] ${progress.status}: ${percent}%`);
            }
          } else if (progress.status === "initiate" || progress.status === "download" || progress.status === "done") {
            this.log(`[Transformers.js] [${file}] ${progress.status}`);
          }

          if (progress.status === "loading") {
            this.setStatus("loading");
          }
        },
      };
      this.pipeline = await pipeline("text-generation", this.modelName, pipelineOptions);
      this.log("[Transformers.js] Pipeline loaded successfully");
    } catch (error) {
       if (signal.aborted) throw new Error("Aborted");
       this.setStatus("error");
       throw new Error(`Pipeline initialization failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (signal.aborted) throw new Error("Aborted");

    this.setStatus("ready");
  }

  getType(): RuntimeType {
    return "transformers";
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
      // Format messages using chat template
      const inputs = this.formatChatPrompt(messages, tools);

      this.log("[Transformers.js] Formatted inputs:", inputs);

      // Generate response
      const generationOptions: any = {
        max_new_tokens: options?.maxTokens || 512,
        temperature: options?.temperature ?? 0.0, // Function calling usually needs low temp
        repetition_penalty: 1.1, // Prevent character loops ($$$$$)
        do_sample: false, // Deterministic for function calling
        return_full_text: false,
        signal: (this as any).abortController?.signal,
        ...inputs // Spread tokenized inputs
      };

      // FunctionGemma doesn't support streaming well with tools in this setup yet
      // so we default to complete response
      return this.handleCompleteResponse(inputs, generationOptions, tools);
      
    } catch (error) {
      throw new Error(
        `Transformers.js chat error: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  /**
   * Format messages into a chat prompt using the tokenizer's chat template
   */
  private formatChatPrompt(
    messages: Message[],
    tools: ToolDefinition[]
  ): any {
    // Map internal ToolDefinition to FunctionGemma schema
    const functionGemmaTools = tools.map(tool => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters
      }
    }));

    // Convert messages to chat format
    const chatMessages = messages.map((msg) => {
      if (msg.role === "tool") {
        return {
          role: "tool",
          tool_call_id: msg.tool_call_id,
          name: msg.name,
          content: msg.content,
        };
      }
      return {
        role: msg.role === "system" ? "developer" : msg.role, // FunctionGemma uses 'developer' for system
        content: msg.content,
      };
    });

    // Add required system prompt if not present
    // This is essential for the model to understand its function calling capabilities
    const hasSystem = chatMessages.some(m => m.role === 'developer');
    if (!hasSystem && tools.length > 0) {
      const format = this.config?.toolCallFormat || "xml";
      let systemContent = "";
      
      if (format === "xml") {
        systemContent = `You are a model that can do function calling with the following functions. 
Must use the EXACT format: <start_function_call>call:name{arg:<escape>val<escape>}<end_function_call>
Example: <start_function_call>call:calculate{expression:<escape>5*12<escape>}<end_function_call>`;
      } else {
        // Use standard JSON mode system prompt
        systemContent = generateSystemPrompt(tools);
      }

      chatMessages.unshift({
        role: 'developer',
        content: systemContent
      });
    }

    // Use apply_chat_template with tools
    if (this.tokenizer?.apply_chat_template) {
      try {
        // Log tools being sent to template
        if (this.config?.debug) {
          this.log("[Transformers.js] Tools for template:", JSON.stringify(functionGemmaTools, null, 2));
        }

        const formatted = this.tokenizer.apply_chat_template(chatMessages, {
          tools: functionGemmaTools.length > 0 ? functionGemmaTools : undefined,
          tokenize: true,
          add_generation_prompt: true,
          return_dict: true,
        });

      // Log the decoded prompt for debugging
      if (this.config?.debug && (formatted as any).input_ids) {
        const decodedPrompt = this.tokenizer.decode((formatted as any).input_ids, { skip_special_tokens: false });
        this.log("[Transformers.js] Full decoded prompt:\n", decodedPrompt);
      }

      return formatted;
    } catch (error) {
      console.warn("[Transformers.js] Chat template failed:", error);
      throw error;
    }
  }

  throw new Error("Tokenizer does not support apply_chat_template");
}

  /**
   * Handle complete (non-streaming) response
   */
  private async handleCompleteResponse(
    inputs: any,
    options: any,
    tools: ToolDefinition[]
  ): Promise<ModelResponse> {
    if (!this.pipeline || !this.tokenizer) {
      throw new Error("Pipeline not initialized");
    }

    // Generate
    const output = await this.pipeline.model.generate({ ...inputs, ...options });
    
    // Decode - handle offset for prompt tokens
    // Fallback: if dims are missing (mocked tests), just decode everything
    const offset = inputs.input_ids?.dims?.[1] || 0;
    const decoded = this.tokenizer.decode(output.slice(0, [offset, null]), { skip_special_tokens: false });

  this.log("[Transformers.js] Raw output:", decoded);

  // Parse and filter tool calls
  const format = this.config?.toolCallFormat || "xml";
  let toolCalls: ToolCall[] = [];

  if (format === "xml") {
    toolCalls = this.parseToolCallsFromResponse(decoded);
  } else {
    // JSON mode parsing
    const json = extractJSON(decoded);
    if (json && json.tool && json.arguments) {
      toolCalls = [
        {
          id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
          name: json.tool,
          arguments: json.arguments,
        },
      ];
    }
  }
  
  // Filter out unknown tools
  toolCalls = toolCalls.filter(call => tools.some(t => t.name === call.name));

  if (toolCalls.length > 0) {
    this.log("[Transformers.js] Detected tool calls:", toolCalls);
    return {
      type: "tool_calls",
      calls: toolCalls,
      text: decoded, // Pass original text for history preservation
    } as ToolCallsResponse;
  }

  // Clean up special tokens for text response
  // We strip all angle-bracketed tokens like <end_of_turn>, <bos>, <|im_end|>, etc.
  const cleanText = decoded.replace(/<[^>]+>/g, "").trim();

  this.log("[Transformers.js] Cleaned text:", cleanText);

  return {
    type: "content",
    text: cleanText,
  } as ContentResponse;
}

  /**
   * Parse tool calls from model response (FunctionGemma format)
   */
  private parseToolCallsFromResponse(
    response: string
  ): ToolCall[] {
    const toolCalls: ToolCall[] = [];

    // FunctionGemma format: <start_function_call>call:name{args}<end_function_call>
    // Robust regex: allows optional 'call:', handles internal spaces/newlines, ignores 'error' hallucinations
    const functionCallPattern = /<start_function_call>(?:call:)?([a-zA-Z0-9_-]+)\{(.*?)\}<end_function_call>/gs;
    
    let match;
    while ((match = functionCallPattern.exec(response)) !== null) {
      const toolName = match[1]?.trim();
      let argsString = match[2]?.trim();

      if (!toolName || toolName === 'error' || argsString === undefined) continue;

      // Clean up args: FunctionGemma uses <escape>...<escape> for strings sometimes?
      // Or just standard JSON-like but without quotes on keys?
      // The example shows: location:<escape>London<escape>
      // Let's try to normalize it to JSON.
      
      // 1. Replace <escape> with "
      argsString = argsString.replace(/<escape>/g, '"');
      
      // 2. Quote keys if missing (simple heuristic)
      // location:"London" -> "location":"London"
      argsString = argsString.replace(/([a-zA-Z0-9_]+):/g, '"$1":');

      // 3. Wrap in braces if not already (it was inside braces in the regex match)
      const jsonString = `{${argsString}}`;

      try {
        const args = JSON.parse(jsonString);
        toolCalls.push({
          id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
          name: toolName,
          arguments: args,
        });
      } catch (e) {
        console.warn("Failed to parse function args:", jsonString, e);
      }
    }

    return toolCalls;
  }

  /**
   * Dispose resources
   */
  async dispose(): Promise<void> {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    if (this.pipeline) {
      this.pipeline = null;
    }
    if (this.tokenizer) {
      this.tokenizer = null;
    }
    await super.dispose();
  }

  async clearCache(): Promise<void> {
    this.log("[Transformers.js] Clearing IndexedDB cache...");
    const cache = new IndexedDBCache();
    await cache.clear();
    this.log("[Transformers.js] Cache cleared");
  }
}
