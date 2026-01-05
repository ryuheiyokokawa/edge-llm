import type { Message } from "@edge-llm/core";

/**
 * UI message displayed in chat window
 */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Runtime type for LLM inference
 */
export type RuntimeType = "webllm" | "transformers" | "api";

/**
 * Chat state
 */
export interface ChatState {
  // Input
  input: string;

  // Loading/Error
  loading: boolean;
  error: string | null;

  // Message histories
  messages: ChatMessage[]; // UI display messages
  conversationHistory: Message[]; // Full context for LLM

  // Runtime selection
  runtime: RuntimeType;

  // LLM status (synced from @edge-llm/react)
  llmStatus: string;
  llmInitialized: boolean;
}

/**
 * Chat actions - pure state mutations, NO async/network calls
 */
export interface ChatActions {
  // Input
  setInput: (input: string) => void;
  clearInput: () => void;

  // Messages
  addUserMessage: (content: string) => void;
  addAssistantMessage: (content: string) => void;
  addToolCallMessage: (
    toolName: string,
    args: Record<string, unknown>
  ) => void;
  addToolResultMessage: (
    toolName: string,
    result: unknown,
    success: boolean
  ) => void;

  // Conversation history (for LLM context)
  appendToHistory: (message: Message) => void;
  appendMultipleToHistory: (messages: Message[]) => void;

  // Loading/Error
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Runtime
  setRuntime: (runtime: RuntimeType) => void;

  // LLM sync
  syncLLMStatus: (status: string, initialized: boolean) => void;

  // Reset
  clearMessages: () => void;
  reset: () => void;
}

/**
 * Combined store type
 */
export type ChatStore = ChatState & ChatActions;
