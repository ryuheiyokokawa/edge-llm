/**
 * React context for LLM provider
 */
import { createContext, useContext } from "react";
import type { LLMClient } from "@edge-llm/core";
import type { RuntimeConfig, RuntimeStatus, RuntimeType } from "@edge-llm/core";

export interface LLMContextValue {
  client: LLMClient | null;
  status: RuntimeStatus;
  activeRuntime: RuntimeType | null;
  config: RuntimeConfig;
  initialized: boolean;
  clearCache: () => Promise<void>;
}

export const LLMContext = createContext<LLMContextValue | null>(null);

export function useLLMContext(): LLMContextValue {
  const context = useContext(LLMContext);
  if (!context) {
    throw new Error("useLLM must be used within LLMProvider");
  }
  return context;
}
