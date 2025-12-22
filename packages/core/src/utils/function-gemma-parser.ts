import { ToolCall } from "../types";

/**
 * Parse FunctionGemma response to extract tool calls
 * Looks for <start_function_call>call:tool_name{args}<end_function_call>
 */
export function parseFunctionGemmaResponse(response: string): ToolCall[] {
  const toolCalls: ToolCall[] = [];
  
  // Regex to match function calls
  // Pattern: <start_function_call>call:tool_name{arguments}<end_function_call>
  // We need to be careful with nested braces in arguments
  const callPattern = /<start_function_call>call:([^{]+)({[\s\S]*?})<end_function_call>/g;
  
  let match;
  while ((match = callPattern.exec(response)) !== null) {
    const toolName = match[1].trim();
    let argsString = match[2];
    
    // FunctionGemma uses <escape> tags for strings in arguments sometimes
    // We need to clean them up before parsing JSON
    // e.g. {location:<escape>Tokyo<escape>} -> {"location":"Tokyo"}
    
    // 1. Replace <escape>...<escape> with "..."
    // This is a bit tricky because we need to handle the content inside
    argsString = argsString.replace(/<escape>(.*?)<escape>/g, '"$1"');
    
    // 2. Fix unquoted keys if necessary (FunctionGemma might output {key: value})
    // Simple regex to quote keys: {key: -> {"key":
    argsString = argsString.replace(/([{,])\s*([a-zA-Z0-9_]+)\s*:/g, '$1"$2":');
    
    try {
      const args = JSON.parse(argsString);
      toolCalls.push({
        id: `call_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: toolName,
        arguments: args,
      });
    } catch (e) {
      console.warn("Failed to parse FunctionGemma tool arguments:", argsString, e);
    }
  }
  
  return toolCalls;
}

/**
 * Clean response text by removing function call tokens
 */
export function cleanFunctionGemmaResponse(response: string): string {
  // Remove all function call blocks
  return response.replace(/<start_function_call>[\s\S]*?<end_function_call>/g, "").trim();
}
