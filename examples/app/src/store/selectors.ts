import type { ChatStore } from "./types";

// ============================================================
// COMBINED STATE SELECTORS
// These return objects - use with useChatStoreShallow
// ============================================================

/**
 * Core chat state (messages, input, loading, error)
 */
export const selectChatState = (state: ChatStore) => ({
  messages: state.messages,
  input: state.input,
  loading: state.loading,
  error: state.error,
  conversationHistory: state.conversationHistory,
  canSend: state.llmInitialized && !state.loading && state.input.trim().length > 0,
});

/**
 * LLM status state
 */
export const selectLLMState = (state: ChatStore) => ({
  llmStatus: state.llmStatus,
  llmInitialized: state.llmInitialized,
});

/**
 * Runtime selection state
 */
export const selectRuntimeState = (state: ChatStore) => ({
  runtime: state.runtime,
});

// ============================================================
// COMBINED ACTION SELECTORS
// These return stable function refs - use with useChatStoreShallow
// ============================================================

/**
 * Input-related actions
 */
export const selectInputActions = (state: ChatStore) => ({
  setInput: state.setInput,
  clearInput: state.clearInput,
});

/**
 * Message-related actions (UI messages)
 */
export const selectMessageActions = (state: ChatStore) => ({
  addUserMessage: state.addUserMessage,
  addAssistantMessage: state.addAssistantMessage,
  addToolCallMessage: state.addToolCallMessage,
  addToolResultMessage: state.addToolResultMessage,
});

/**
 * Conversation history actions (LLM context)
 */
export const selectHistoryActions = (state: ChatStore) => ({
  appendToHistory: state.appendToHistory,
  appendMultipleToHistory: state.appendMultipleToHistory,
});

/**
 * Runtime selection actions
 */
export const selectRuntimeActions = (state: ChatStore) => ({
  setRuntime: state.setRuntime,
});

/**
 * Reset/clear actions
 */
export const selectResetActions = (state: ChatStore) => ({
  clearMessages: state.clearMessages,
  reset: state.reset,
});

/**
 * All chat-related actions combined
 */
export const selectChatActions = (state: ChatStore) => ({
  // Input
  setInput: state.setInput,
  clearInput: state.clearInput,
  // Messages
  addUserMessage: state.addUserMessage,
  addAssistantMessage: state.addAssistantMessage,
  addToolCallMessage: state.addToolCallMessage,
  addToolResultMessage: state.addToolResultMessage,
  // History
  appendToHistory: state.appendToHistory,
  appendMultipleToHistory: state.appendMultipleToHistory,
  // Loading/Error
  setLoading: state.setLoading,
  setError: state.setError,
  // LLM sync
  syncLLMStatus: state.syncLLMStatus,
  // Reset
  clearMessages: state.clearMessages,
  reset: state.reset,
});
