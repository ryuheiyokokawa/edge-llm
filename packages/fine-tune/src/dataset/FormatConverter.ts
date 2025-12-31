/**
 * FormatConverter - Converts training examples to FunctionGemma format
 */

import {
  FUNCTION_GEMMA_TOKENS,
  type TrainingExample,
  type FormattedTrainingSample,
  type ToolSchema,
  type ExpectedToolCall,
} from "../types";

const {
  START_OF_TURN,
  END_OF_TURN,
  START_FUNCTION_DECLARATION,
  END_FUNCTION_DECLARATION,
  START_FUNCTION_CALL,
  END_FUNCTION_CALL,
  START_FUNCTION_RESPONSE,
  END_FUNCTION_RESPONSE,
} = FUNCTION_GEMMA_TOKENS;

/**
 * Converts training examples to FunctionGemma format
 */
export class FormatConverter {
  /**
   * Format tool definitions as FunctionGemma function declarations
   */
  static formatToolDeclarations(tools: ToolSchema[]): string {
    const declarations = tools
      .map((tool) => {
        const toolJson = JSON.stringify(
          {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
          },
          null,
          2
        );
        return `${START_FUNCTION_DECLARATION}\n${toolJson}\n${END_FUNCTION_DECLARATION}`;
      })
      .join("\n\n");

    return `${START_OF_TURN}developer\nYou have access to the following functions:\n\n${declarations}\n${END_OF_TURN}`;
  }

  /**
   * Format a user message
   */
  static formatUserMessage(query: string): string {
    return `${START_OF_TURN}user\n${query}${END_OF_TURN}`;
  }

  /**
   * Format model's function call response
   */
  static formatFunctionCall(toolCalls: ExpectedToolCall[]): string {
    const calls = toolCalls
      .map((call) => {
        const callJson = JSON.stringify({
          name: call.name,
          arguments: call.arguments,
        });
        return `${START_FUNCTION_CALL}\n${callJson}\n${END_FUNCTION_CALL}`;
      })
      .join("\n");

    return `${START_OF_TURN}model\n${calls}\n${END_OF_TURN}`;
  }

  /**
   * Format tool response
   */
  static formatToolResponse(response: unknown): string {
    const responseJson = JSON.stringify(response);
    return `${START_OF_TURN}tool\n${START_FUNCTION_RESPONSE}\n${responseJson}\n${END_FUNCTION_RESPONSE}\n${END_OF_TURN}`;
  }

  /**
   * Format model's final text response
   */
  static formatModelResponse(text: string): string {
    return `${START_OF_TURN}model\n${text}${END_OF_TURN}`;
  }

  /**
   * Convert a training example to FunctionGemma format
   * 
   * @param example - The training example
   * @param tools - Tool definitions available
   * @returns Formatted training sample
   */
  static convertExample(
    example: TrainingExample,
    tools: ToolSchema[]
  ): FormattedTrainingSample {
    const parts: string[] = [];

    // 1. Developer turn with tool declarations
    parts.push(this.formatToolDeclarations(tools));

    // 2. User query
    parts.push(this.formatUserMessage(example.userQuery));

    // 3. Model's function call
    parts.push(this.formatFunctionCall(example.expectedToolCalls));

    // 4. Tool response (if provided)
    if (example.toolResponses) {
      // For each tool call, add the corresponding response
      for (const call of example.expectedToolCalls) {
        const response = example.toolResponses[call.name];
        if (response !== undefined) {
          parts.push(this.formatToolResponse(response));
        }
      }

      // 5. Model's final response (if provided)
      if (example.expectedFinalResponse) {
        parts.push(this.formatModelResponse(example.expectedFinalResponse));
      }
    }

    return {
      text: parts.join("\n"),
    };
  }

  /**
   * Convert multiple examples to formatted samples
   */
  static convertExamples(
    examples: TrainingExample[],
    tools: ToolSchema[]
  ): FormattedTrainingSample[] {
    return examples.map((example) => this.convertExample(example, tools));
  }

  /**
   * Convert a ToolDefinition to ToolSchema (strips handler)
   */
  static toolDefinitionToSchema(tool: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required?: string[];
    };
  }): ToolSchema {
    return {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters as ToolSchema["parameters"],
    };
  }
}
