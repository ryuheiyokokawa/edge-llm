/**
 * OllamaProvider - Generates synthetic training examples using local Ollama
 */

import type {
  SyntheticDataProvider,
  TrainingExample,
  ToolSchema,
  ExpectedToolCall,
} from "../types";

/**
 * Default Ollama configuration
 */
const DEFAULT_OLLAMA_CONFIG = {
  baseUrl: "http://localhost:11434",
  model: "llama3.2",
};

/**
 * OllamaProvider generates diverse training examples using a local Ollama instance
 */
export class OllamaProvider {
  private baseUrl: string;
  private model: string;

  constructor(config?: Partial<SyntheticDataProvider>) {
    this.baseUrl = config?.baseUrl ?? DEFAULT_OLLAMA_CONFIG.baseUrl;
    this.model = config?.model ?? DEFAULT_OLLAMA_CONFIG.model;
  }

  /**
   * Generate the prompt for creating diverse training examples
   */
  private buildPrompt(tool: ToolSchema, existingQueries: string[]): string {
    const existingExamples =
      existingQueries.length > 0
        ? `\n\nExisting queries (create different variations):\n${existingQueries.map((q) => `- "${q}"`).join("\n")}`
        : "";

    return `You are generating training data for a tool-calling AI model.

Given this tool definition:
\`\`\`json
${JSON.stringify(tool, null, 2)}
\`\`\`

Generate 5 diverse user queries that would trigger this tool, along with the expected function call arguments.

Requirements:
- Make queries natural and varied (questions, commands, casual requests)
- Use different phrasings and vocabulary
- Include edge cases when appropriate
- Arguments must be valid for the tool's parameter schema
${existingExamples}

Respond ONLY with a JSON array in this exact format (no markdown, no explanation):
[
  {
    "userQuery": "natural user query here",
    "toolCall": {"name": "${tool.name}", "arguments": {...}}
  }
]`;
  }

  /**
   * Parse the LLM response into structured examples
   */
  private parseResponse(
    response: string,
    tool: ToolSchema
  ): Array<{ userQuery: string; toolCall: ExpectedToolCall }> {
    try {
      // Extract JSON from response (handle markdown code blocks)
      let jsonStr = response.trim();
      
      // Remove markdown code blocks if present
      if (jsonStr.startsWith("```")) {
        const match = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (match) {
          jsonStr = match[1]!.trim();
        }
      }

      const parsed = JSON.parse(jsonStr);

      if (!Array.isArray(parsed)) {
        console.warn("Response is not an array, wrapping");
        return [];
      }

      // Validate and filter valid examples
      return parsed.filter((item: unknown) => {
        if (typeof item !== "object" || item === null) return false;
        const obj = item as Record<string, unknown>;
        
        if (typeof obj.userQuery !== "string") return false;
        if (typeof obj.toolCall !== "object" || obj.toolCall === null) return false;
        
        const toolCall = obj.toolCall as Record<string, unknown>;
        if (toolCall.name !== tool.name) return false;
        if (typeof toolCall.arguments !== "object") return false;

        return true;
      }).map((item: { userQuery: string; toolCall: { name: string; arguments: Record<string, unknown> } }) => ({
        userQuery: item.userQuery,
        toolCall: {
          name: item.toolCall.name,
          arguments: item.toolCall.arguments,
        },
      }));
    } catch (error) {
      console.warn("Failed to parse LLM response:", error);
      return [];
    }
  }

  /**
   * Call Ollama API to generate completions
   */
  private async callOllama(prompt: string): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        prompt,
        stream: false,
        options: {
          temperature: 0.8, // Higher for diversity
          num_predict: 1024,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`
      );
    }

    const data = (await response.json()) as { response: string };
    return data.response;
  }

  /**
   * Check if Ollama is running and the model is available
   */
  async checkAvailability(): Promise<{
    available: boolean;
    models: string[];
    error?: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      
      if (!response.ok) {
        return { available: false, models: [], error: `HTTP ${response.status}` };
      }

      const data = (await response.json()) as { models: Array<{ name: string }> };
      const modelNames = data.models.map((m) => m.name);
      
      return {
        available: modelNames.some(
          (name) => name.startsWith(this.model) || name === this.model
        ),
        models: modelNames,
      };
    } catch (error) {
      return {
        available: false,
        models: [],
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Generate training examples for a single tool
   * 
   * @param tool - Tool schema to generate examples for
   * @param count - Number of examples to generate
   * @param existingQueries - Existing queries to avoid duplicates
   * @returns Array of training examples
   */
  async generateExamples(
    tool: ToolSchema,
    count: number,
    existingQueries: string[] = []
  ): Promise<TrainingExample[]> {
    const examples: TrainingExample[] = [];
    const allQueries = [...existingQueries];
    
    // Generate in batches of 5
    const batchSize = 5;
    const batches = Math.ceil(count / batchSize);

    for (let i = 0; i < batches && examples.length < count; i++) {
      const prompt = this.buildPrompt(tool, allQueries.slice(-10)); // Use last 10 as context
      
      try {
        const response = await this.callOllama(prompt);
        const parsed = this.parseResponse(response, tool);

        for (const item of parsed) {
          if (examples.length >= count) break;
          
          // Skip duplicates
          if (allQueries.includes(item.userQuery.toLowerCase())) continue;

          examples.push({
            userQuery: item.userQuery,
            expectedToolCalls: [item.toolCall],
          });
          
          allQueries.push(item.userQuery.toLowerCase());
        }
      } catch (error) {
        console.warn(`Batch ${i + 1} failed:`, error);
        // Continue with next batch
      }

      // Small delay between batches to avoid overwhelming Ollama
      if (i < batches - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    return examples;
  }

  /**
   * Generate examples for multiple tools
   */
  async generateForTools(
    tools: ToolSchema[],
    examplesPerTool: number,
    onProgress?: (tool: string, generated: number, total: number) => void
  ): Promise<Map<string, TrainingExample[]>> {
    const results = new Map<string, TrainingExample[]>();
    
    for (const tool of tools) {
      onProgress?.(tool.name, 0, examplesPerTool);
      
      const examples = await this.generateExamples(tool, examplesPerTool);
      results.set(tool.name, examples);
      
      onProgress?.(tool.name, examples.length, examplesPerTool);
    }

    return results;
  }
}
