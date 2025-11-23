/**
 * useLLM hook for interacting with the LLM
 */
import { useCallback, useRef } from "react";
import { useLLMContext } from "../context.js";
import type {
  ToolDefinition,
  Message,
  ModelResponse,
  ChatOptions,
} from "@edge-llm/core";

import type { RuntimeStatus } from "@edge-llm/core";

export interface UseLLMOptions {
  onStatusChange?: (status: RuntimeStatus) => void;
}

export interface UseLLMReturn {
  /**
   * Send a message to the LLM
   */
  send: (
    message:
      | string
      | Message
      | { toolResults: Array<{ tool_call_id: string; result: unknown }> },
    options?: ChatOptions
  ) => Promise<ModelResponse>;

  /**
   * Register a single tool
   */
  registerTool: (tool: ToolDefinition) => void;

  /**
   * Register multiple tools
   */
  registerTools: (tools: ToolDefinition[]) => void;

  /**
   * Unregister a tool
   */
  unregisterTool: (name: string) => void;

  /**
   * Get current status
   */
  status: RuntimeStatus;

  /**
   * Check if initialized
   */
  initialized: boolean;
}

export function useLLM(_options?: UseLLMOptions): UseLLMReturn {
  const { client, status, initialized } = useLLMContext();
  const toolsRef = useRef<ToolDefinition[]>([]);

  const send = useCallback(
    async (
      message:
        | string
        | Message
        | { toolResults: Array<{ tool_call_id: string; result: unknown }> },
      sendOptions?: ChatOptions
    ): Promise<ModelResponse> => {
      if (!client) {
        throw new Error("LLM client not initialized");
      }

      // Convert string to message format
      let messages: Message[];
      if (typeof message === "string") {
        messages = [{ role: "user", content: message }];
      } else if ("toolResults" in message) {
        // For tool results, we need to send them as tool messages
        // The conversation history should already include the assistant's tool_call message
        messages = message.toolResults.map(
          (tr: { tool_call_id: string; result: unknown }) => ({
            role: "tool" as const,
            content: JSON.stringify(tr.result),
            tool_call_id: tr.tool_call_id,
          })
        );
      } else {
        messages = [message];
      }

      return client.chat(messages, toolsRef.current, sendOptions);
    },
    [client]
  );

  const registerTool = useCallback((tool: ToolDefinition) => {
    toolsRef.current = toolsRef.current.filter((t) => t.name !== tool.name);
    toolsRef.current.push(tool);
  }, []);

  const registerTools = useCallback(
    (tools: ToolDefinition[]) => {
      for (const tool of tools) {
        registerTool(tool);
      }
    },
    [registerTool]
  );

  const unregisterTool = useCallback((name: string) => {
    toolsRef.current = toolsRef.current.filter((t) => t.name !== name);
  }, []);

  return {
    send,
    registerTool,
    registerTools,
    unregisterTool,
    status,
    initialized,
  };
}
