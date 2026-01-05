import {
  createContext,
  useContext,
  useRef,
  type ReactNode,
} from "react";
import { useStore } from "zustand";
import { useShallow } from "zustand/react/shallow";
import { createChatStore, type ChatStoreApi } from "./chatStore";
import type { ChatStore } from "./types";

/**
 * Context for the chat store
 */
const ChatStoreContext = createContext<ChatStoreApi | null>(null);

/**
 * Provider that creates a singleton store instance using useRef
 */
export function ChatStoreProvider({ children }: { children: ReactNode }) {
  const storeRef = useRef<ChatStoreApi | null>(null);

  // Create store only once (singleton pattern)
  if (!storeRef.current) {
    storeRef.current = createChatStore();
  }

  return (
    <ChatStoreContext.Provider value={storeRef.current}>
      {children}
    </ChatStoreContext.Provider>
  );
}

/**
 * Hook to access store with a selector (for primitive values)
 */
export function useChatStore<T>(selector: (state: ChatStore) => T): T {
  const store = useContext(ChatStoreContext);
  if (!store) {
    throw new Error("useChatStore must be used within ChatStoreProvider");
  }
  return useStore(store, selector);
}

/**
 * Hook to access store with shallow comparison (for object/action selectors)
 */
export function useChatStoreShallow<T>(selector: (state: ChatStore) => T): T {
  const store = useContext(ChatStoreContext);
  if (!store) {
    throw new Error("useChatStoreShallow must be used within ChatStoreProvider");
  }
  return useStore(store, useShallow(selector));
}

/**
 * Hook to access the store API directly
 * Use this for imperative access (e.g., in async handlers)
 */
export function useChatStoreApi(): ChatStoreApi {
  const store = useContext(ChatStoreContext);
  if (!store) {
    throw new Error("useChatStoreApi must be used within ChatStoreProvider");
  }
  return store;
}
