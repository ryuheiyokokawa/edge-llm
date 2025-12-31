/**
 * TrainingConfig - Configuration types for training
 */

import type { TrainingConfig as BaseTrainingConfig } from "../types";

/**
 * Default training configuration
 */
export const DEFAULT_TRAINING_CONFIG: Required<Omit<BaseTrainingConfig, "datasetPath" | "outputPath">> = {
  baseModel: "mlx-community/functiongemma-270m-it-4bit",
  epochs: 3,
  learningRate: 2e-4,
  batchSize: 4,
  loraRank: 8,
  loraAlpha: 16,
  useQLoRA: false,
};

/**
 * MLX-specific training options
 */
export interface MLXTrainingOptions {
  /** Maximum sequence length */
  maxTokens?: number;
  /** Use gradient checkpointing to reduce memory */
  gradientCheckpointing?: boolean;
  /** Random seed */
  seed?: number;
  /** Number of layers to apply LoRA (undefined = all layers) */
  loraLayers?: number;
  /** Weight decay */
  weightDecay?: number;
}

/**
 * Full training configuration for MLX
 */
export interface MLXTrainingConfig extends BaseTrainingConfig, MLXTrainingOptions {}

/**
 * Training progress from Python script
 */
export interface MLXTrainingProgress {
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  loss: number;
  learningRate: number;
  evalLoss?: number;
  status: "loading" | "training" | "evaluating" | "saving" | "complete" | "error";
  message?: string;
}

/**
 * Training result
 */
export interface TrainingResult {
  success: boolean;
  adapterPath?: string;
  configPath?: string;
  error?: string;
  trainingTime?: number;
  finalLoss?: number;
}
