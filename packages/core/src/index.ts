/**
 * @edge-llm/core
 * Core runtime for edge-first LLM tool calling
 */

// Types
export type {
  ToolDefinition,
  ToolCall,
  ToolResult,
  Message,
  ModelResponse,
  ToolCallsResponse,
  ContentResponse,
  ToolCallChunk,
  RuntimeType,
  RuntimeStatus,
  RuntimeConfig,
  Runtime,
  ChatOptions,
  JSONSchema,
  JSONSchemaProperty,
  ToolExecutionConfig,
  ToolExecutionError,
  ToolExecutionResult,
} from "./types.js";

// Tool Registry
export { ToolRegistry } from "./tool-registry.js";

// Runtimes
export { WebLLMRuntime } from "./runtime/webllm.js";
export { TransformersRuntime } from "./runtime/transformers.js";
export { APIRuntime } from "./runtime/api.js";
export { RuntimeManager } from "./runtime/manager.js";
export { BaseRuntime } from "./runtime/base.js";

// Service Worker
export { ServiceWorkerController } from "./service-worker.js";

// Client
export { LLMClient } from "./client.js";

// Cache
export { ModelCache } from "./cache.js";

// Tool Execution
export { ToolExecutor } from "./tool-executor.js";
export { ToolLoop } from "./tool-loop.js";
export type { ToolLoopOptions } from "./tool-loop.js";

// Performance Monitoring
export { PerformanceMonitor } from "./performance-monitor.js";
export type { PerformanceMetrics } from "./performance-monitor.js";

// Tool Validation
export { ToolValidator } from "./tool-validator.js";
export type { ValidationResult } from "./tool-validator.js";

// User Input Tool
export {
  createUserInputTool,
  createUserInputToolWithHandler,
} from "./user-input-tool.js";
export type {
  UserInputRequest,
  UserInputResponse,
} from "./user-input-tool.js";
