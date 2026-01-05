// Types
export type {
  ChatMessage,
  RuntimeType,
  ChatState,
  ChatActions,
  ChatStore,
} from "./types";

// Store
export { createChatStore, type ChatStoreApi } from "./chatStore";

// Provider and low-level hooks
export {
  ChatStoreProvider,
  useChatStore,
  useChatStoreShallow,
  useChatStoreApi,
} from "./StoreProvider";

// Composite hooks (preferred API for components)
export {
  useChatState,
  useLLMState,
  useRuntimeState,
  useInputActions,
  useMessageActions,
  useHistoryActions,
  useRuntimeActions,
  useResetActions,
  useChatActions,
} from "./hooks";

// Selectors (for advanced use cases)
export {
  selectChatState,
  selectLLMState,
  selectRuntimeState,
  selectInputActions,
  selectMessageActions,
  selectHistoryActions,
  selectRuntimeActions,
  selectResetActions,
  selectChatActions,
} from "./selectors";
