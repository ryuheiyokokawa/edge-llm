import { Message, ToolDefinition } from "../types";

/**
 * Format messages and tools for FunctionGemma
 * Uses specific control tokens: <start_of_turn>, <end_of_turn>, <start_function_declaration>, etc.
 */
export function formatFunctionGemmaPrompt(
  messages: Message[],
  tools: ToolDefinition[]
): string {
  let prompt = "";

  // 1. Developer turn with tool definitions (if tools exist)
  if (tools && tools.length > 0) {
    prompt += "<start_of_turn>developer\n";
    prompt +=
      "You are a model that can do function calling with the following functions\n";

    for (const tool of tools) {
      prompt += `<start_function_declaration>declaration:${tool.name}`;
      
      // We need to construct the function signature object
      // FunctionGemma expects a specific JSON-like structure inside the declaration
      const functionDef = {
        description: `<escape>${tool.description}<escape>`,
        parameters: {
          type: "object",
          properties: {},
          required: tool.parameters.required || [],
        },
      };

      // Map parameters
      if (tool.parameters.properties) {
        const props: Record<string, any> = {};
        for (const [key, value] of Object.entries(tool.parameters.properties)) {
          props[key] = {
            type: `<escape>${(value.type as string).toUpperCase()}<escape>`,
            description: value.description
              ? `<escape>${value.description}<escape>`
              : undefined,
          };
          if (value.enum) {
            props[key].enum = value.enum.map(
              (e) => `<escape>${String(e)}<escape>`
            );
          }
        }
        functionDef.parameters.properties = props;
      }

      prompt += JSON.stringify(functionDef);
      prompt += "<end_function_declaration>\n";
    }
    prompt += "<end_of_turn>\n";
  }

  // 2. Conversation history
  for (const msg of messages) {
    if (msg.role === "system") {
      // FunctionGemma doesn't have a dedicated 'system' role in the same way
      // We'll treat it as a developer turn or prepend to user turn
      // For now, let's use developer turn
      prompt += `<start_of_turn>developer\n${msg.content}<end_of_turn>\n`;
    } else if (msg.role === "user") {
      prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
    } else if (msg.role === "assistant") {
      prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
    } else if (msg.role === "tool") {
      // Tool results
      // FunctionGemma expects: <start_function_response>response:tool_name{...}<end_function_response>
      // We need to reconstruct the response format
      // Note: The message content should be the result
      prompt += `<start_function_response>response:${msg.name || "unknown"}`;
      // The content is usually a JSON string, but we need to ensure it's properly formatted for the model
      // FunctionGemma expects the result object directly
      prompt += msg.content; 
      prompt += "<end_function_response>\n";
    }
  }

  // 3. Prompt for model generation
  prompt += "<start_of_turn>model\n";

  return prompt;
}
