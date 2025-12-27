/**
 * LLM Provider component
 */
import React, { useEffect, useState } from "react";
import { LLMClient } from "@edge-llm/core";
import type { RuntimeConfig, RuntimeStatus, RuntimeType } from "@edge-llm/core";
import { LLMContext, type LLMContextValue } from "./context.js";

export interface LLMProviderProps {
  config?: RuntimeConfig;
  serviceWorkerPath?: string;
  enableServiceWorker?: boolean; // Allow disabling service worker (useful for dev)
  children: React.ReactNode;
}

// Module-level singletons to prevent double initialization in Strict Mode
// and ensure state is synced across remounts.
let globalClient: LLMClient | null = null;
let isInitializing = false; // Synchronous lock to prevent races

/**
 * Reset global client (internal use for testing)
 */
export function resetGlobalClient() {
  globalClient = null;
  isInitializing = false;
}

export function LLMProvider({
  config = {},
  serviceWorkerPath = "/sw.js",
  enableServiceWorker = false, // Disabled by default for now (dev mode compatibility)
  children,
}: LLMProviderProps) {
  const [client, setClient] = useState<LLMClient | null>(globalClient);
  const [status, setStatus] = useState<RuntimeStatus>("idle");
  const [activeRuntime, setActiveRuntime] = useState<RuntimeType | null>(null);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function setup() {
      // If we already have a client instance, check if we need to re-initialize
      if (globalClient) {
        if (mounted) {
          setClient(globalClient);
          
          // Re-initialize with new config - LLMClient now handles disposal internally
          try {
            setStatus("initializing");
            await globalClient.initialize(config);
            const statusResponse = await globalClient.getStatusWithDetails();
            setStatus(statusResponse.status);
            setActiveRuntime(statusResponse.activeRuntime || null);
            if (statusResponse.status === "ready") {
              setInitialized(true);
            }
          } catch (error) {
            if (error instanceof Error && error.message === "Aborted") {
              console.log("[Edge LLM] Re-initialization aborted (new request started)");
              return;
            }
            console.error("[Edge LLM] Re-initialization error:", error);
            setStatus("error");
          }
        }
        return;
      }

      // Synchronous lock check
      if (isInitializing) return;
      isInitializing = true;

      try {
        // Register service worker (optional, disabled by default for dev mode)
        if (
          enableServiceWorker &&
          typeof navigator !== "undefined" &&
          "serviceWorker" in navigator
        ) {
          try {
            await navigator.serviceWorker.register(serviceWorkerPath);
            await navigator.serviceWorker.ready;
          } catch (error) {
            // Silently fail in dev mode
            if (process.env.NODE_ENV !== "development") {
              console.warn("Service worker registration failed:", error);
            }
          }
        }

        const newClient = new LLMClient();
        globalClient = newClient;

        // We set the client immediately so we have a reference
        if (mounted) {
          setClient(newClient);
          setStatus("initializing");
          if (config.debug) {
            console.log("[Edge LLM] Initializing client...");
          }
        }

        try {
          await newClient.initialize(config);
          const statusResponse = await newClient.getStatusWithDetails();
          if (config.debug) {
            console.log(
              "[Edge LLM] Initialization complete, status:",
              statusResponse.status
            );
          }

          if (mounted) {
            console.log("[Edge LLM] Setting initialized=true, status=", statusResponse.status);
            setStatus(statusResponse.status);
            setActiveRuntime(statusResponse.activeRuntime || null);
            setInitialized(true);
          }
        } catch (initError) {
          console.error("[Edge LLM] Initialization error:", initError);
          if (mounted) {
            setStatus("error");
          }
          // Reset on error so we can try again
          globalClient = null;
          isInitializing = false;
        }
      } catch (error) {
        console.error("Failed to setup LLM client:", error);
        if (mounted) {
          setStatus("error");
        }
        isInitializing = false;
      }
    }

    // Always call setup on mount (debounced)
    let initTimer: ReturnType<typeof setTimeout> | null = null;
    
    initTimer = setTimeout(() => {
      if (mounted) {
        setup();
      }
    }, 250);

    return () => {
      mounted = false;
      if (initTimer) clearTimeout(initTimer);
    };
  }, []); // Only run once on mount

  // Update status periodically (even during initialization)
  useEffect(() => {
    if (!client) return;

    const interval = setInterval(async () => {
      try {
        const statusResponse = await client.getStatusWithDetails();
        setStatus(statusResponse.status);
        setActiveRuntime(statusResponse.activeRuntime || null);
      } catch (error) {
        // Silently handle errors during initialization
        console.debug(
          "Status check failed (may be normal during init):",
          error
        );
      }
    }, 1000); // Check every second

    return () => clearInterval(interval);
  }, [client]);

  const clearCache = async () => {
    if (client) {
      await client.clearCache();
    }
  };

  const contextValue: LLMContextValue = {
    client,
    status,
    activeRuntime,
    config,
    initialized,
    clearCache,
  };

  return (
    <LLMContext.Provider value={contextValue}>{children}</LLMContext.Provider>
  );
}
