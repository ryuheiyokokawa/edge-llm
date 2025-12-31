/**
 * SyntheticGenerator - Orchestrates synthetic training data generation
 */

import { OllamaProvider } from "./OllamaProvider";
import { DatasetBuilder } from "../dataset/DatasetBuilder";
import type {
  SyntheticGenerationConfig,
  TrainingExample,
  ToolSchema,
} from "../types";

/**
 * Generates synthetic training data using LLM providers
 */
export class SyntheticGenerator {
  private config: SyntheticGenerationConfig;

  constructor(config: SyntheticGenerationConfig) {
    this.config = {
      ...config,
      validateOutput: config.validateOutput ?? true,
    };
  }

  /**
   * Create the appropriate provider based on config
   */
  private createProvider(): OllamaProvider {
    const { provider } = this.config;

    if (provider.type !== "ollama") {
      throw new Error(
        `Provider type "${provider.type}" not yet implemented. Use "ollama" for now.`
      );
    }

    return new OllamaProvider({
      type: "ollama",
      model: provider.model,
      baseUrl: provider.baseUrl,
    });
  }

  /**
   * Validate that the provider is available
   */
  async validateProvider(): Promise<{
    valid: boolean;
    error?: string;
    availableModels?: string[];
  }> {
    try {
      const provider = this.createProvider();
      const availability = await provider.checkAvailability();

      if (!availability.available) {
        return {
          valid: false,
          error:
            availability.error ??
            `Model "${this.config.provider.model}" not found. Available models: ${availability.models.join(", ")}`,
          availableModels: availability.models,
        };
      }

      return { valid: true, availableModels: availability.models };
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Generate synthetic training examples
   * 
   * @param tools - Tool schemas to generate examples for
   * @param onProgress - Progress callback
   * @returns Array of training examples
   */
  async generate(
    tools: ToolSchema[],
    onProgress?: (status: {
      tool: string;
      generated: number;
      total: number;
      phase: "generating" | "validating" | "complete";
    }) => void
  ): Promise<TrainingExample[]> {
    const provider = this.createProvider();
    const allExamples: TrainingExample[] = [];

    for (const tool of tools) {
      onProgress?.({
        tool: tool.name,
        generated: 0,
        total: this.config.examplesPerTool,
        phase: "generating",
      });

      const examples = await provider.generateExamples(
        tool,
        this.config.examplesPerTool
      );

      // Optionally validate output
      if (this.config.validateOutput) {
        onProgress?.({
          tool: tool.name,
          generated: examples.length,
          total: this.config.examplesPerTool,
          phase: "validating",
        });

        const validated = this.validateExamples(examples, tool);
        allExamples.push(...validated);
      } else {
        allExamples.push(...examples);
      }

      onProgress?.({
        tool: tool.name,
        generated: examples.length,
        total: this.config.examplesPerTool,
        phase: "complete",
      });
    }

    return allExamples;
  }

  /**
   * Validate generated examples against tool schema
   */
  private validateExamples(
    examples: TrainingExample[],
    tool: ToolSchema
  ): TrainingExample[] {
    return examples.filter((example) => {
      // Check that expectedToolCalls reference the correct tool
      for (const call of example.expectedToolCalls) {
        if (call.name !== tool.name) {
          console.warn(
            `Example references wrong tool: expected "${tool.name}", got "${call.name}"`
          );
          return false;
        }

        // Check required parameters are present
        const required = tool.parameters.required ?? [];
        for (const param of required) {
          if (!(param in call.arguments)) {
            console.warn(`Missing required parameter "${param}" in example`);
            return false;
          }
        }

        // Basic type validation for string parameters
        for (const [key, value] of Object.entries(call.arguments)) {
          const paramDef = tool.parameters.properties[key];
          if (!paramDef) continue;

          if (paramDef.type === "string" && typeof value !== "string") {
            console.warn(`Parameter "${key}" should be string, got ${typeof value}`);
            return false;
          }
          if (paramDef.type === "number" && typeof value !== "number") {
            console.warn(`Parameter "${key}" should be number, got ${typeof value}`);
            return false;
          }
        }
      }

      return true;
    });
  }

  /**
   * Generate and directly add to a DatasetBuilder
   */
  async generateAndAddToBuilder(
    builder: DatasetBuilder,
    onProgress?: (status: {
      tool: string;
      generated: number;
      total: number;
      phase: "generating" | "validating" | "complete";
    }) => void
  ): Promise<number> {
    const tools = builder.getTools();
    const examples = await this.generate(tools, onProgress);
    
    builder.addExamples(examples);
    
    return examples.length;
  }
}
