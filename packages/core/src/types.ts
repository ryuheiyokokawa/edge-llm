/**
 * Core types for edge-llm framework
 */

/**
 * JSON Schema type for tool parameters
 */
export type JSONSchema = {
  type: "object";
  properties: Record<string, JSONSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
};

export type JSONSchemaProperty = {
  type: "string" | "number" | "boolean" | "array" | "object";
  description?: string;
  enum?: (string | number)[];
  items?: JSONSchemaProperty;
  properties?: Record<string, JSONSchemaProperty>;
  required?: string[];
};

/**
 * Tool definition registered by developer
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: JSONSchema;
  handler: (args: Record<string, unknown>) => Promise<unknown>;
}

/**
 * Tool call from model
 */
export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * Tool result sent back to model
 */
export interface ToolResult {
  tool_call_id: string;
  result: unknown;
  error?: string;
}

/**
 * Message in conversation
 */
export interface Message {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

/**
 * Response types from model
 */
export type ModelResponse = ToolCallsResponse | ContentResponse;

export interface ToolCallsResponse {
  type: "tool_calls";
  calls: ToolCall[];
}

export interface ToolCallChunk {
  type: "tool_call_chunk";
  tool_call: ToolCall;
}

export interface ContentResponse {
  type: "content";
  stream?: AsyncIterable<string | ToolCallChunk>;
  text?: string;
}

/**
 * Runtime types
 */
export type RuntimeType = "webllm" | "transformers" | "api" | "auto";

export type RuntimeStatus =
  | "idle"
  | "initializing"
  | "loading"
  | "ready"
  | "error";

/**
 * Runtime configuration
 */
export interface RuntimeConfig {
  preferredRuntime?: RuntimeType;
  debug?: boolean;
  fallbackStrategy?: "quality" | "speed" | "cost";
  apiKey?: string;
  apiBaseUrl?: string;
  modelId?: string;
  temperature?: number;
  maxTokens?: number;
  models?: {
    webllm?: string;
    transformers?: string;
    api?: string;
  };
  performance?: {
    maxConcurrentInference?: number;
    inferenceTimeout?: number;
    modelCacheTTL?: number;
  };
}

/**
 * Runtime interface that all runtimes must implement
 */
export interface Runtime {
  /**
   * Initialize the runtime
   */
  initialize(config: RuntimeConfig): Promise<void>;

  /**
   * Send messages and get response
   */
  chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse>;

  /**
   * Get current status
   */
  getStatus(): RuntimeStatus;

  /**
   * Cleanup resources
   */
  dispose(): Promise<void>;
}

/**
 * Chat options
 */
export interface ChatOptions {
  stream?: boolean;
  temperature?: number;
  maxTokens?: number;
}

/**
 * Tool execution configuration
 */
export interface ToolExecutionConfig {
  maxIterations?: number; // Default: 5
  executionTimeout?: number; // Per tool, default: 30000ms
  totalTimeout?: number; // Entire flow, default: 120000ms
  retryStrategy?: "exponential" | "linear" | "none"; // Default: "exponential"
  maxRetries?: number; // Default: 2
}

/**
 * Tool execution error
 */
export interface ToolExecutionError {
  tool: string;
  error: Error;
  retryable: boolean;
  fallbackStrategy?: "skip" | "mock" | "api";
}

/**
 * Tool execution result
 */
export interface ToolExecutionResult {
  tool_call_id: string;
  result: unknown;
  error?: string;
  executionTime?: number; // milliseconds
}

/**
 * Service worker message types
 */
export type ServiceWorkerMessage =
  | InitializeMessage
  | ChatMessage
  | StatusMessage
  | ToolRegistryUpdateMessage;

export interface InitializeMessage {
  type: "INITIALIZE";
  config: RuntimeConfig;
}

export interface ChatMessage {
  type: "CHAT";
  messages: Message[];
  tools: ToolDefinition[];
  options?: ChatOptions;
  requestId: string;
}

export interface StatusMessage {
  type: "STATUS";
  requestId: string;
}

export interface ToolRegistryUpdateMessage {
  type: "TOOL_REGISTRY_UPDATE";
  tools: ToolDefinition[];
}

/**
 * Service worker response types
 */
export type ServiceWorkerResponse =
  | InitializeResponse
  | ChatResponse
  | StatusResponse
  | ErrorResponse;

export interface InitializeResponse {
  type: "INITIALIZE_RESPONSE";
  success: boolean;
  error?: string;
}

export interface ChatResponse {
  type: "CHAT_RESPONSE";
  requestId: string;
  response: ModelResponse;
  error?: string;
}

export interface StatusResponse {
  type: "STATUS_RESPONSE";
  requestId: string;
  status: RuntimeStatus;
}

export interface ErrorResponse {
  type: "ERROR";
  requestId?: string;
  error: string;
}
