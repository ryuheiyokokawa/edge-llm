/**
 * User input as a tool pattern
 * Allows the model to request user input during tool execution
 */
import type { ToolDefinition } from "./types.js";

export interface UserInputRequest {
  prompt: string;
  options?: string[];
  tool_call_id: string;
}

export interface UserInputResponse {
  tool_call_id: string;
  userResponse: string;
}

/**
 * Create a user input tool definition
 * This tool allows the model to request additional information from the user
 */
export function createUserInputTool(): ToolDefinition {
  return {
    name: "requestUserInput",
    description:
      "Ask the user for additional information, clarification, or a choice between options. Use this when you need user input to continue.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "The question or prompt to show to the user",
        },
        options: {
          type: "array",
          items: { type: "string" },
          description:
            "Optional: Array of choices for the user to select from (multiple choice)",
        },
      },
      required: ["prompt"],
    },
    handler: async (args: Record<string, unknown>) => {
      // This handler should be replaced by the developer with their own UI implementation
      // The default implementation throws an error to indicate it needs to be implemented
      const prompt = args.prompt as string;
      const options = args.options as string[] | undefined;
      throw new Error(
        `requestUserInput tool handler not implemented. Prompt: ${prompt}, Options: ${options?.join(", ")}. Please provide a custom handler that shows a modal or input dialog to the user.`
      );
    },
  };
}

/**
 * Helper to create a user input tool with a custom handler
 */
export function createUserInputToolWithHandler(
  handler: (args: { prompt: string; options?: string[] }) => Promise<string>
): ToolDefinition {
  return {
    name: "requestUserInput",
    description:
      "Ask the user for additional information, clarification, or a choice between options. Use this when you need user input to continue.",
    parameters: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "The question or prompt to show to the user",
        },
        options: {
          type: "array",
          items: { type: "string" },
          description:
            "Optional: Array of choices for the user to select from (multiple choice)",
        },
      },
      required: ["prompt"],
    },
    handler: async (args: Record<string, unknown>) => {
      const prompt = args.prompt as string;
      const options = args.options as string[] | undefined;
      return await handler({ prompt, options });
    },
  };
}

