/**
 * DatasetBuilder - Main class for preparing training datasets
 */

import * as fs from "fs/promises";
import * as path from "path";
import { FormatConverter } from "./FormatConverter";
import type {
  DatasetBuilderConfig,
  TrainingExample,
  FormattedTrainingSample,
  DatasetSplit,
  ToolSchema,
} from "../types";

/**
 * Default split ratio for train/valid/test
 */
const DEFAULT_SPLIT_RATIO = {
  train: 0.8,
  valid: 0.1,
  test: 0.1,
};

/**
 * DatasetBuilder prepares training data for fine-tuning
 */
export class DatasetBuilder {
  private config: Required<DatasetBuilderConfig>;
  private tools: ToolSchema[];
  private examples: TrainingExample[] = [];

  constructor(config: DatasetBuilderConfig) {
    this.config = {
      ...config,
      splitRatio: config.splitRatio ?? DEFAULT_SPLIT_RATIO,
    };

    // Convert ToolDefinitions to ToolSchemas (strip handlers)
    this.tools = config.tools.map((tool) =>
      FormatConverter.toolDefinitionToSchema(tool as unknown as {
        name: string;
        description: string;
        parameters: {
          type: "object";
          properties: Record<string, unknown>;
          required?: string[];
        };
      })
    );

    // Validate split ratio
    const { train, valid, test } = this.config.splitRatio;
    const sum = train + valid + test;
    if (Math.abs(sum - 1.0) > 0.001) {
      throw new Error(
        `Split ratio must sum to 1.0, got ${sum} (train: ${train}, valid: ${valid}, test: ${test})`
      );
    }
  }

  /**
   * Add a single training example
   */
  addExample(example: TrainingExample): this {
    this.validateExample(example);
    this.examples.push(example);
    return this;
  }

  /**
   * Add multiple training examples
   */
  addExamples(examples: TrainingExample[]): this {
    for (const example of examples) {
      this.addExample(example);
    }
    return this;
  }

  /**
   * Validate that a training example references valid tools
   */
  private validateExample(example: TrainingExample): void {
    const toolNames = new Set(this.tools.map((t) => t.name));

    for (const call of example.expectedToolCalls) {
      if (!toolNames.has(call.name)) {
        throw new Error(
          `Training example references unknown tool: "${call.name}". ` +
            `Available tools: ${Array.from(toolNames).join(", ")}`
        );
      }
    }
  }

  /**
   * Get the current number of examples
   */
  getExampleCount(): number {
    return this.examples.length;
  }

  /**
   * Get the tool schemas
   */
  getTools(): ToolSchema[] {
    return [...this.tools];
  }

  /**
   * Shuffle an array in place (Fisher-Yates algorithm)
   */
  private shuffle<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = shuffled[i]!;
      shuffled[i] = shuffled[j]!;
      shuffled[j] = temp;
    }
    return shuffled;
  }

  /**
   * Split examples into train/valid/test sets
   */
  private splitExamples(): DatasetSplit {
    const { train: trainRatio, valid: validRatio } = this.config.splitRatio;

    // Shuffle examples
    const shuffled = this.shuffle(this.examples);

    // Convert all examples to formatted samples
    const formatted = FormatConverter.convertExamples(shuffled, this.tools);

    // Calculate split indices
    const trainEnd = Math.floor(formatted.length * trainRatio);
    const validEnd = Math.floor(formatted.length * (trainRatio + validRatio));

    return {
      train: formatted.slice(0, trainEnd),
      valid: formatted.slice(trainEnd, validEnd),
      test: formatted.slice(validEnd),
    };
  }

  /**
   * Write samples to a JSONL file
   */
  private async writeJsonl(
    samples: FormattedTrainingSample[],
    filePath: string
  ): Promise<void> {
    const lines = samples.map((sample) => JSON.stringify(sample));
    await fs.writeFile(filePath, lines.join("\n") + "\n", "utf-8");
  }

  /**
   * Build and save the dataset
   * 
   * @returns Object with paths to generated files and counts
   */
  async build(): Promise<{
    trainPath: string;
    validPath: string;
    testPath: string;
    counts: { train: number; valid: number; test: number };
  }> {
    if (this.examples.length === 0) {
      throw new Error("No training examples added. Call addExample() first.");
    }

    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });

    // Split examples
    const split = this.splitExamples();

    // Define output paths
    const trainPath = path.join(this.config.outputDir, "train.jsonl");
    const validPath = path.join(this.config.outputDir, "valid.jsonl");
    const testPath = path.join(this.config.outputDir, "test.jsonl");

    // Write files
    await Promise.all([
      this.writeJsonl(split.train, trainPath),
      this.writeJsonl(split.valid, validPath),
      this.writeJsonl(split.test, testPath),
    ]);

    return {
      trainPath,
      validPath,
      testPath,
      counts: {
        train: split.train.length,
        valid: split.valid.length,
        test: split.test.length,
      },
    };
  }

  /**
   * Preview formatted samples without saving
   */
  preview(count: number = 3): FormattedTrainingSample[] {
    const samples = FormatConverter.convertExamples(
      this.examples.slice(0, count),
      this.tools
    );
    return samples;
  }

  /**
   * Load examples from a JSON file
   */
  async loadExamplesFromFile(filePath: string): Promise<this> {
    const content = await fs.readFile(filePath, "utf-8");
    const examples = JSON.parse(content) as TrainingExample[];
    this.addExamples(examples);
    return this;
  }

  /**
   * Save tool schemas to a JSON file
   */
  async saveToolSchemas(outputPath?: string): Promise<string> {
    const filePath = outputPath ?? path.join(this.config.outputDir, "tools.json");
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, JSON.stringify(this.tools, null, 2), "utf-8");
    return filePath;
  }
}
