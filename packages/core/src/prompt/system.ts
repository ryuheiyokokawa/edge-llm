import { ToolDefinition } from "../types";

export function generateSystemPrompt(tools: ToolDefinition[]): string {
  if (!tools || tools.length === 0) {
    return "You are a helpful AI assistant.";
  }

  const toolDescriptions = tools.map(tool => {
    return `- ${tool.name}: ${tool.description}\n  Parameters: ${JSON.stringify(tool.parameters)}`;
  }).join("\n");

  return `You are a helpful AI assistant with access to the following tools:

${toolDescriptions}

To use a tool, you MUST output a JSON object in the following format:
{
  "tool": "tool_name",
  "arguments": {
    "arg_name": "value"
  }
}

If you do not need to use a tool, just respond with normal text.
IMPORTANT: When using a tool, output ONLY the JSON object. Do not add any other text.
`;
}
