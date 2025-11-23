/**
 * LLM Provider component
 */
import React, { useEffect, useState } from "react";
import { LLMClient } from "@edge-llm/core";
import type { RuntimeConfig, RuntimeStatus } from "@edge-llm/core";
import { LLMContext, type LLMContextValue } from "./context.js";

export interface LLMProviderProps {
  config?: RuntimeConfig;
  serviceWorkerPath?: string;
  enableServiceWorker?: boolean; // Allow disabling service worker (useful for dev)
  children: React.ReactNode;
}

export function LLMProvider({
  config = {},
  serviceWorkerPath = "/sw.js",
  enableServiceWorker = false, // Disabled by default for now (dev mode compatibility)
  children,
}: LLMProviderProps) {
  const [client, setClient] = useState<LLMClient | null>(null);
  const [status, setStatus] = useState<RuntimeStatus>("idle");
  const [initialized, setInitialized] = useState(false);


  // Module-level singleton to prevent double initialization in Strict Mode
  // and ensure state is synced to the latest mounted component.
  useEffect(() => {
    let mounted = true;

    async function setup() {
      // If we already have a client instance in the state, we don't need to do anything
      if (client) return;

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

        // We need to create a new client for this mount if one doesn't exist
        // But we want to avoid double-creation in Strict Mode.
        // Since we can't easily use a global variable (it would persist across HMR),
        // we'll rely on the fact that the second effect run happens after the first cleanup.
        
        // Actually, the most robust way for this specific app structure (where we want a singleton client)
        // is to check if we can reuse an existing one or just be careful.
        
        // Let's try a simpler approach: just allow the client to be created, 
        // but ensure we update the state of the *current* mount.
        
        const newClient = new LLMClient();
        
        // We set the client immediately so we have a reference
        if (mounted) {
          setClient(newClient);
          setStatus("initializing");
          console.log("[Edge LLM] Initializing client...");
        }

        try {
          await newClient.initialize(config);
          const currentStatus = await newClient.getStatus();
          console.log("[Edge LLM] Initialization complete, status:", currentStatus);

          if (mounted) {
            setStatus(currentStatus);
            setInitialized(true);
          }
        } catch (initError) {
          console.error("[Edge LLM] Initialization error:", initError);
          if (mounted) {
            setStatus("error");
          }
        }
      } catch (error) {
        console.error("Failed to setup LLM client:", error);
        if (mounted) {
          setStatus("error");
        }
      }
    }

    // Only run setup if we haven't initialized yet
    if (!initialized && !client) {
        setup();
    }

    return () => {
      mounted = false;
    };
  }, []); // Only run once on mount

  // Update status periodically (even during initialization)
  useEffect(() => {
    if (!client) return;

    const interval = setInterval(async () => {
      try {
        const currentStatus = await client.getStatus();
        setStatus(currentStatus);
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

  const contextValue: LLMContextValue = {
    client,
    status,
    config,
    initialized,
  };

  return (
    <LLMContext.Provider value={contextValue}>{children}</LLMContext.Provider>
  );
}
