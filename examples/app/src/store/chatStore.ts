import { create } from "zustand";
import type { ChatStore, ChatState } from "./types";

/**
 * Initial state
 */
const initialState: ChatState = {
  input: "",
  loading: false,
  error: null,
  messages: [],
  conversationHistory: [],
  runtime: "transformers",
  llmStatus: "idle",
  llmInitialized: false,
};

/**
 * Create a chat store instance
 */
export const createChatStore = () =>
  create<ChatStore>((set) => ({
    ...initialState,

    // Input actions
    setInput: (input) => set({ input }),
    clearInput: () => set({ input: "" }),

    // Message actions
    addUserMessage: (content) =>
      set((state) => ({
        messages: [...state.messages, { role: "user", content }],
      })),

    addAssistantMessage: (content) =>
      set((state) => ({
        messages: [...state.messages, { role: "assistant", content }],
      })),

    addToolCallMessage: (toolName, args) =>
      set((state) => ({
        messages: [
          ...state.messages,
          {
            role: "assistant",
            content: `🔧 Calling: ${toolName}(${JSON.stringify(args)})`,
          },
        ],
      })),

    addToolResultMessage: (toolName, result, success) =>
      set((state) => ({
        messages: [
          ...state.messages,
          {
            role: "assistant",
            content: `${success ? "✅" : "❌"} ${toolName}: ${JSON.stringify(result)}`,
          },
        ],
      })),

    // History actions
    appendToHistory: (message) =>
      set((state) => ({
        conversationHistory: [...state.conversationHistory, message],
      })),

    appendMultipleToHistory: (messages) =>
      set((state) => ({
        conversationHistory: [...state.conversationHistory, ...messages],
      })),

    // Loading/Error
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),

    // Runtime
    setRuntime: (runtime) => set({ runtime }),

    // LLM sync
    syncLLMStatus: (llmStatus, llmInitialized) =>
      set({ llmStatus, llmInitialized }),

    // Reset
    clearMessages: () =>
      set({ messages: [], conversationHistory: [], error: null }),

    reset: () => set(initialState),
  }));

/**
 * Store API type (for context)
 */
export type ChatStoreApi = ReturnType<typeof createChatStore>;
