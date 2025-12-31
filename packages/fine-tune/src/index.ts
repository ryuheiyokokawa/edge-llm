/**
 * @edge-llm/fine-tune
 * Library for fine-tuning models for edge-first tool calling
 */

// Types
export * from "./types";

// Dataset preparation
export { DatasetBuilder, FormatConverter } from "./dataset";

// Synthetic data generation
export { OllamaProvider, SyntheticGenerator } from "./synthetic";

// Training
export {
  MLXTrainer,
  DEFAULT_TRAINING_CONFIG,
  type MLXTrainingConfig,
  type MLXTrainingProgress,
  type TrainingResult,
} from "./training";

// Re-export for convenience
export type { ToolDefinition, Message, ToolCall } from "@edge-llm/core";
