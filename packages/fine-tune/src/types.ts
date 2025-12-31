/**
 * @edge-llm/fine-tune
 * Types for fine-tuning models for edge-first tool calling
 */

import type { ToolDefinition } from "@edge-llm/core";

/**
 * Configuration for the DatasetBuilder
 */
export interface DatasetBuilderConfig {
  /** Tool definitions to create training data for */
  tools: ToolDefinition[];
  /** Output directory for generated dataset files */
  outputDir: string;
  /** Split ratio for train/valid/test sets (must sum to 1.0) */
  splitRatio?: {
    train: number;
    valid: number;
    test: number;
  };
}

/**
 * A single training example for tool calling
 */
export interface TrainingExample {
  /** The user's input query */
  userQuery: string;
  /** Expected tool calls the model should make */
  expectedToolCalls: ExpectedToolCall[];
  /** Mock responses from tools (for multi-turn examples) */
  toolResponses?: Record<string, unknown>;
  /** Expected final text response after tool execution */
  expectedFinalResponse?: string;
}

/**
 * Expected tool call in training data
 */
export interface ExpectedToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * FunctionGemma control tokens for formatting
 */
export const FUNCTION_GEMMA_TOKENS = {
  START_OF_TURN: "<start_of_turn>",
  END_OF_TURN: "<end_of_turn>",
  START_FUNCTION_DECLARATION: "<start_function_declaration>",
  END_FUNCTION_DECLARATION: "<end_function_declaration>",
  START_FUNCTION_CALL: "<start_function_call>",
  END_FUNCTION_CALL: "<end_function_call>",
  START_FUNCTION_RESPONSE: "<start_function_response>",
  END_FUNCTION_RESPONSE: "<end_function_response>",
} as const;

/**
 * Roles in FunctionGemma conversation format
 */
export type FunctionGemmaRole = "developer" | "user" | "model" | "tool";

/**
 * A formatted training sample ready for JSONL output
 */
export interface FormattedTrainingSample {
  /** The complete formatted text for training */
  text: string;
}

/**
 * Dataset split containing training samples
 */
export interface DatasetSplit {
  train: FormattedTrainingSample[];
  valid: FormattedTrainingSample[];
  test: FormattedTrainingSample[];
}

/**
 * Configuration for synthetic data generation
 */
export interface SyntheticGenerationConfig {
  /** LLM provider for generating examples */
  provider: SyntheticDataProvider;
  /** Number of examples to generate per tool */
  examplesPerTool: number;
  /** Custom prompt for diversity */
  diversityPrompt?: string;
  /** Whether to validate generated JSON */
  validateOutput?: boolean;
}

/**
 * Synthetic data provider configuration
 */
export interface SyntheticDataProvider {
  type: "ollama" | "openai" | "claude";
  /** Model name (e.g., 'llama3.2', 'gpt-4o-mini') */
  model: string;
  /** Base URL for API (e.g., 'http://localhost:11434' for Ollama) */
  baseUrl?: string;
  /** API key (for OpenAI/Claude) */
  apiKey?: string;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  /** Path to training dataset directory */
  datasetPath: string;
  /** Output path for trained model/adapters */
  outputPath: string;
  /** Base model to fine-tune */
  baseModel: string;
  /** Number of training epochs */
  epochs?: number;
  /** Learning rate */
  learningRate?: number;
  /** Batch size */
  batchSize?: number;
  /** LoRA rank */
  loraRank?: number;
  /** LoRA alpha (scaling factor) */
  loraAlpha?: number;
  /** Use QLoRA (4-bit quantization) */
  useQLoRA?: boolean;
}

/**
 * Training progress event
 */
export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  loss: number;
  learningRate: number;
  eta?: string;
}

/**
 * Export configuration
 */
export interface ExportConfig {
  /** Path to trained LoRA adapters */
  adapterPath: string;
  /** Base model identifier */
  baseModel: string;
  /** Output formats to export */
  outputFormats: ExportFormat[];
  /** Output directory */
  outputDir: string;
  /** Quantization level */
  quantization?: "4bit" | "8bit" | "none";
}

/**
 * Supported export formats
 */
export type ExportFormat = "gguf" | "safetensors" | "onnx";

/**
 * Tool definition without handler (for dataset purposes)
 */
export interface ToolSchema {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, ToolParameterSchema>;
    required?: string[];
  };
}

/**
 * Tool parameter schema
 */
export interface ToolParameterSchema {
  type: "string" | "number" | "boolean" | "array" | "object";
  description?: string;
  enum?: (string | number)[];
  items?: ToolParameterSchema;
  properties?: Record<string, ToolParameterSchema>;
  required?: string[];
}
