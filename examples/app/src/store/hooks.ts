/**
 * Thin wrapper hooks that apply the correct shallow comparison
 * These are the preferred API for components
 */

import { useChatStoreShallow } from "./StoreProvider";
import {
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

// ============================================================
// STATE HOOKS
// ============================================================

export const useChatState = () => useChatStoreShallow(selectChatState);
export const useLLMState = () => useChatStoreShallow(selectLLMState);
export const useRuntimeState = () => useChatStoreShallow(selectRuntimeState);

// ============================================================
// ACTION HOOKS
// ============================================================

export const useInputActions = () => useChatStoreShallow(selectInputActions);
export const useMessageActions = () => useChatStoreShallow(selectMessageActions);
export const useHistoryActions = () => useChatStoreShallow(selectHistoryActions);
export const useRuntimeActions = () => useChatStoreShallow(selectRuntimeActions);
export const useResetActions = () => useChatStoreShallow(selectResetActions);
export const useChatActions = () => useChatStoreShallow(selectChatActions);
