import { useState, useEffect, useCallback } from "react";
import { useLLM } from "@edge-llm/react";
import type { Message } from "@edge-llm/core";
import { exampleTools } from "../utils/tools";

export function useChat() {
  const { send, registerTools, status, initialized } = useLLM();
  const [messages, setMessages] = useState<
    Array<{ role: string; content: string }>
  >([]);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Register tools on mount
  useEffect(() => {
    if (initialized) {
      registerTools(exampleTools);
    }
  }, [initialized, registerTools]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading || !initialized) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);
    setError(null);

    // Add user message to history
    const userMsg: Message = { role: "user", content: userMessage };
    setConversationHistory((prev) => [...prev, userMsg]);
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      let currentHistory = [...conversationHistory, userMsg];
      let keepGoing = true;
      let iterations = 0;
      const MAX_ITERATIONS = 5;

      while (keepGoing && iterations < MAX_ITERATIONS) {
        iterations++;

        
        const response = await send(currentHistory as any);


        if (response.type === "content") {
          // Final answer
          const assistantMsg: Message = { 
            role: "assistant", 
            content: response.text || "" 
          };
          setConversationHistory((prev) => [...prev, assistantMsg]);
          setMessages((prev) => [...prev, { role: "assistant", content: response.text || "" }]);
          keepGoing = false;
        } else if (response.type === "tool_calls") {
          // Handle tool calls
          const toolCalls = response.calls;
          const assistantMsg: Message = {
            role: "assistant",
            content: "", // JSON mode usually puts content in the message, but for history we can leave empty or put the JSON
            tool_calls: toolCalls.map(tc => ({
              id: tc.id,
              name: tc.name,
              arguments: tc.arguments
            }))
          };
          
          // Add assistant message with tool calls to history
          currentHistory.push(assistantMsg);
          setConversationHistory((prev) => [...prev, assistantMsg]);

          // Show tool execution in UI
          const toolCallUIMessages = toolCalls.map((call) => ({
            role: "assistant",
            content: `🔧 Calling tool: ${call.name}(${JSON.stringify(call.arguments)})`,
          }));
          setMessages((prev) => [...prev, ...toolCallUIMessages as any]);

          // Execute tools
          const toolResults = [];
          for (const call of toolCalls) {

            try {
              // Find tool
              const tool = exampleTools.find(t => t.name === call.name);
              if (!tool) throw new Error(`Tool ${call.name} not found`);

              const result = await tool.handler(call.arguments);
              toolResults.push({
                tool_call_id: call.id,
                result: result
              });
              setMessages((prev) => [...prev, { role: "assistant", content: `✅ Tool ${call.name} result: ${JSON.stringify(result)}` }]);
            } catch (e) {
              console.error(`[App] Tool execution failed:`, e);
              const errorResult = { error: String(e) };
              toolResults.push({
                tool_call_id: call.id,
                result: errorResult
              });
              setMessages((prev) => [...prev, { role: "assistant", content: `❌ Tool ${call.name} failed: ${JSON.stringify(errorResult)}` }]);
            }
          }

          // Add tool results to history
          const toolResultMessages: Message[] = toolResults.map(tr => ({
            role: "tool",
            content: JSON.stringify(tr.result),
            tool_call_id: tr.tool_call_id,
            name: toolCalls.find(tc => tc.id === tr.tool_call_id)?.name // Add name for FunctionGemma prompt
          }));
          
          currentHistory.push(...toolResultMessages);
          setConversationHistory((prev) => [...prev, ...toolResultMessages]);
          
          // Loop continues to send tool results back to model
        } else {
          keepGoing = false;
        }
      }
    } catch (err) {
      console.error("[App] Error in chat loop:", err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(errorMessage);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMessage}` },
      ]);
      setConversationHistory((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, an error occurred." }
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, initialized, send, conversationHistory]);

  return {
    messages,
    input,
    setInput,
    handleSend,
    loading,
    error,
    status,
    initialized,
    clearCache: useLLM().clearCache,
  };
}
