import { useEffect, useCallback } from "react";
import { useLLM } from "@edge-llm/react";
import type { Message } from "@edge-llm/core";
import { exampleTools } from "../utils/tools";
import { useChatState, useChatActions } from "../store";

/**
 * Chat hook that orchestrates LLM interactions
 * Uses hooks consistently - no direct store.getState() access
 */
export function useChat() {
  const { send, registerTools, status, initialized, clearCache } = useLLM();

  // State from hooks
  const { input, loading, conversationHistory } = useChatState();

  // Actions from hooks
  const actions = useChatActions();

  // Sync LLM status to store
  useEffect(() => {
    actions.syncLLMStatus(status, initialized);
  }, [status, initialized, actions]);

  // Register tools on mount
  useEffect(() => {
    if (initialized) {
      registerTools(exampleTools);
    }
  }, [initialized, registerTools]);

  /**
   * Handle sending a message
   * Captures state at call time, tracks history locally during async loop
   */
  const handleSend = useCallback(async () => {
    // Capture current state at call time
    const currentInput = input.trim();

    if (!currentInput || loading || !initialized) {
      return;
    }

    // Update state via actions
    actions.clearInput();
    actions.setLoading(true);
    actions.setError(null);
    actions.addUserMessage(currentInput);

    // Build local history for this request (starts from captured state)
    const userMsg: Message = { role: "user", content: currentInput };
    actions.appendToHistory(userMsg);
    let localHistory: Message[] = [...conversationHistory, userMsg];

    try {
      let keepGoing = true;
      let iterations = 0;
      const MAX_ITERATIONS = 5;

      while (keepGoing && iterations < MAX_ITERATIONS) {
        iterations++;

        const response = await send(localHistory as any);

        if (response.type === "content") {
          // Final answer
          const assistantMsg: Message = {
            role: "assistant",
            content: response.text || "",
          };
          actions.appendToHistory(assistantMsg);
          actions.addAssistantMessage(response.text || "");
          keepGoing = false;
        } else if (response.type === "tool_calls") {
          // Handle tool calls
          const toolCalls = response.calls;
          const assistantMsg: Message = {
            role: "assistant",
            content: response.text || "",
            tool_calls: toolCalls.map((tc) => ({
              id: tc.id,
              name: tc.name,
              arguments: tc.arguments,
            })),
          };

          // Update local history and store
          localHistory.push(assistantMsg);
          actions.appendToHistory(assistantMsg);

          // Show tool calls in UI
          for (const call of toolCalls) {
            actions.addToolCallMessage(call.name, call.arguments);
          }

          // Execute tools
          const toolResults: Message[] = [];
          for (const call of toolCalls) {
            try {
              const tool = exampleTools.find((t) => t.name === call.name);
              if (!tool) throw new Error(`Tool ${call.name} not found`);

              const result = await tool.handler(call.arguments);
              actions.addToolResultMessage(call.name, result, true);

              toolResults.push({
                role: "tool",
                content: JSON.stringify(result),
                tool_call_id: call.id,
                name: call.name,
              });
            } catch (e) {
              console.error(`[useChat] Tool execution failed:`, e);
              const errorResult = { error: String(e) };
              actions.addToolResultMessage(call.name, errorResult, false);

              toolResults.push({
                role: "tool",
                content: JSON.stringify(errorResult),
                tool_call_id: call.id,
                name: call.name,
              });
            }
          }

          // Update local history and store
          localHistory.push(...toolResults);
          actions.appendMultipleToHistory(toolResults);

          // Loop continues
        } else {
          keepGoing = false;
        }
      }
    } catch (err) {
      console.error("[useChat] Error in chat loop:", err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      actions.setError(errorMessage);
      actions.addAssistantMessage(`Error: ${errorMessage}`);
    } finally {
      actions.setLoading(false);
    }
  }, [input, loading, initialized, conversationHistory, send, actions]);

  return {
    handleSend,
    clearCache,
  };
}
