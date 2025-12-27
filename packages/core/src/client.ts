/**
 * Main thread client for communicating with service worker
 */
import type {
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
  RuntimeType,
  RuntimeStatus,
  ServiceWorkerMessage,
  ServiceWorkerResponse,
  StatusResponse,
} from "./types.js";
import { RuntimeManager } from "./runtime/manager.js";
import { ToolRegistry } from "./tool-registry.js";

export class LLMClient {
  private serviceWorkerRegistration: ServiceWorkerRegistration | null = null;
  private serviceWorkerReady: Promise<void>;
  private requestIdCounter = 0;
  private runtimeManager: RuntimeManager | null = null;
  private toolRegistry: ToolRegistry = new ToolRegistry();
  private debug: boolean = false;

  constructor() {
    this.serviceWorkerReady = this.ensureServiceWorker();
  }

  private log(...args: any[]) {
    if (this.debug) {
      console.log(...args);
    }
  }

  /**
   * Ensure service worker is registered and ready
   * Note: Service worker should be registered by the provider/application
   * For dev mode, service worker is optional
   */
  private async ensureServiceWorker(): Promise<void> {
    this.log("[LLMClient] Checking service worker support...");
    if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) {
      this.log(
        "[LLMClient] Service workers not supported, using direct execution"
      );
      // Service workers not supported - will work without them
      this.serviceWorkerRegistration = null;
      return;
    }

    try {
      // Check if there are any service workers registered
      const registrations = await navigator.serviceWorker.getRegistrations();
      this.log(
        "[LLMClient] Found",
        registrations.length,
        "service worker registrations"
      );

      if (registrations.length === 0) {
        this.log(
          "[LLMClient] No service workers registered, using direct execution"
        );
        this.serviceWorkerRegistration = null;
        return;
      }

      this.log("[LLMClient] Waiting for service worker to be ready...");
      // Wait for service worker to be ready if it exists
      // The service worker should already be registered by the provider
      this.serviceWorkerRegistration = await navigator.serviceWorker.ready;
      this.log("[LLMClient] Service worker is ready");
    } catch (error) {
      // Service worker not available - this is expected in dev mode
      this.log(
        "[LLMClient] Service worker not available, using direct execution:",
        error
      );
      this.serviceWorkerRegistration = null;
    }
  }

  /**
   * Send message to service worker and wait for response
   * Falls back to direct execution if service worker is not available
   */
  private async sendMessage(
    message: ServiceWorkerMessage
  ): Promise<ServiceWorkerResponse> {
    await this.serviceWorkerReady;

    // If no service worker, use direct execution
    if (!this.serviceWorkerRegistration?.active) {
      return this.handleDirectExecution(message);
    }

    return new Promise((resolve, reject) => {
      const channel = new MessageChannel();
      const timeout = setTimeout(() => {
        reject(new Error("Service worker message timeout"));
      }, 30000); // 30 second timeout

      channel.port1.onmessage = (event) => {
        clearTimeout(timeout);
        const response = event.data as ServiceWorkerResponse;
        resolve(response);
      };

      this.serviceWorkerRegistration!.active!.postMessage(message, [
        channel.port2,
      ]);
    });
  }

  /**
   * Handle direct execution without service worker (for dev mode)
   */
  private async handleDirectExecution(
    message: ServiceWorkerMessage
  ): Promise<ServiceWorkerResponse> {
    this.log("[LLMClient] Handling direct execution for:", message.type);
    switch (message.type) {
      case "INITIALIZE":
        try {
          // 1. Create the new manager first
          const nextManager = new RuntimeManager(message.config);
          
          // 2. Clear out the old one
          if (this.runtimeManager) {
            this.log("[LLMClient] Disposing existing RuntimeManager and clearing its cache to free quota...");
            try {
              // Proactively clear cache to ensure the new runtime has space
              await this.runtimeManager.clearCache();
            } catch (e) {
              console.warn("[LLMClient] Failed to clear cache during switch:", e);
            }
            await this.runtimeManager.dispose();
            this.runtimeManager = null;
          }

          // 3. Mark the new one as current
          this.runtimeManager = nextManager;
          
          this.log("[LLMClient] Initializing RuntimeManager...");
          await nextManager.initialize();
          
          // 4. Double check if we were replaced while initializing
          if (this.runtimeManager !== nextManager) {
            this.log("[LLMClient] Initialization complete but manager was already replaced. Disposing this one...");
            await nextManager.dispose();
            return { type: "INITIALIZE_RESPONSE", success: true };
          }
          
          this.log("[LLMClient] RuntimeManager initialized successfully");
          return { type: "INITIALIZE_RESPONSE", success: true };
        } catch (error) {
          const errorMsg =
            error instanceof Error ? error.message : String(error);
          
          // If it was an abort error, we don't necessarily want to log it as a failure
          if (errorMsg === "Aborted") {
            this.log("[LLMClient] Initialization aborted by a newer request");
            return { type: "INITIALIZE_RESPONSE", success: true }; // Still return success so caller knows it was intentional
          }

          console.error(
            "[LLMClient] Direct execution initialization failed:",
            errorMsg
          );
          return {
            type: "INITIALIZE_RESPONSE",
            success: false,
            error: errorMsg,
          };
        }

      case "CLEAR_CACHE":
        try {
          if (this.runtimeManager) {
            await this.runtimeManager.clearCache();
          } else {
            // Even if no manager, we can clear known caches
            const tempManager = new RuntimeManager({});
            await tempManager.clearCache();
          }
          return { type: "CLEAR_CACHE_RESPONSE", requestId: message.requestId, success: true };
        } catch (error) {
          return {
            type: "CLEAR_CACHE_RESPONSE",
            requestId: message.requestId,
            success: false,
            error: String(error),
          };
        }

      case "CHAT": {
        if (!this.runtimeManager) {
          return {
            type: "CHAT_RESPONSE",
            requestId: message.requestId,
            response: { type: "content", text: "" },
            error: "Runtime not initialized",
          };
        }

        try {
          const runtime = this.runtimeManager.getRuntime();
          const tools = this.toolRegistry.getAll();
          
          try {
            const response = await runtime.chat(
              message.messages,
              tools,
              message.options
            );

            return {
              type: "CHAT_RESPONSE",
              requestId: message.requestId,
              response,
            };
          } catch (runtimeError) {
            // SILENT FAILOVER: If local runtime fails, try falling back to API
            const runtimeType = runtime.getType();
            if (runtimeType !== "api") {
               console.warn(`[LLMClient] Runtime ${runtimeType} failed, attempting fallback to API:`, runtimeError);
               
               // Force re-initialization to pick up the next available runtime (usually API)
               // This will dispose of the failed runtime
               await this.runtimeManager.initialize();
               
               const newRuntime = this.runtimeManager.getRuntime();
               const response = await newRuntime.chat(
                 message.messages,
                 tools,
                 message.options
               );

               return {
                 type: "CHAT_RESPONSE",
                 requestId: message.requestId,
                 response,
               };
            }
            throw runtimeError;
          }
        } catch (error) {
          return {
            type: "CHAT_RESPONSE",
            requestId: message.requestId,
            response: { type: "content", text: "" },
            error: error instanceof Error ? error.message : String(error),
          };
        }
      }

      case "STATUS": {
        return {
          type: "STATUS_RESPONSE",
          requestId: message.requestId,
          status: this.runtimeManager?.getStatus() || "idle",
          activeRuntime: this.runtimeManager?.getActiveRuntimeType(),
        };
      }

      case "TOOL_REGISTRY_UPDATE":
        this.toolRegistry.clear();
        this.toolRegistry.registerMany(message.tools);
        return { type: "ERROR", error: "Tool registry updated" };

      default:
        return {
          type: "ERROR",
          error: `Unknown message type: ${
            (message as ServiceWorkerMessage).type
          }`,
        };
    }
  }

  /**
   * Initialize the service worker
   */
  async initialize(config: RuntimeConfig): Promise<void> {
    this.debug = config.debug || false;
    this.log("[LLMClient] Starting initialize...");
    this.log("[LLMClient] Waiting for service worker ready...");
    await this.serviceWorkerReady;
    this.log("[LLMClient] Service worker ready check complete");
    this.log(
      "[LLMClient] Service worker available:",
      !!this.serviceWorkerRegistration?.active
    );

    this.log("[LLMClient] Sending INITIALIZE message...");
    const response = await this.sendMessage({
      type: "INITIALIZE",
      config,
    });
    this.log("[LLMClient] Received response:", response.type);

    if (response.type === "INITIALIZE_RESPONSE") {
      if (!response.success) {
        const errorMsg = response.error || "Initialization failed";
        console.error("[LLMClient] Initialization failed:", errorMsg);
        throw new Error(errorMsg);
      }
      this.log("[LLMClient] Initialization successful");
    } else {
      console.error("[LLMClient] Unexpected response type:", response);
      throw new Error("Unexpected response type");
    }
  }

  /**
   * Send chat message
   */
  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse> {
    await this.serviceWorkerReady;
    const requestId = `chat-${++this.requestIdCounter}`;

    // First update tool registry
    await this.sendMessage({
      type: "TOOL_REGISTRY_UPDATE",
      tools,
    });

    // Then send chat message
    const response = await this.sendMessage({
      type: "CHAT",
      messages,
      tools,
      options,
      requestId,
    });

    if (response.type === "CHAT_RESPONSE") {
      if (response.error) {
        throw new Error(response.error);
      }
      return response.response;
    } else {
      throw new Error("Unexpected response type");
    }
  }

  /**
   * Get current status
   */
  async getStatus(): Promise<RuntimeStatus> {
    const response = await this.getStatusWithDetails();
    return response.status;
  }

  /**
   * Get complete status including active runtime type
   */
  async getStatusWithDetails(): Promise<StatusResponse> {
    await this.serviceWorkerReady;

    // If using direct execution, get status from runtime manager
    if (!this.serviceWorkerRegistration?.active && this.runtimeManager) {
      return {
        type: "STATUS_RESPONSE",
        requestId: Math.random().toString(36).substring(7), // Generate a dummy requestId
        status: this.runtimeManager.getStatus(),
        activeRuntime: this.runtimeManager.getActiveRuntimeType(),
      };
    }

    const response = await this.sendMessage({
      type: "STATUS",
      requestId: Math.random().toString(36).substring(7),
    });

    if (response.type === "STATUS_RESPONSE") {
      return response;
    }

    throw new Error("Invalid status response");
  }

  /**
   * Clear all runtime caches
   */
  async clearCache(): Promise<void> {
    await this.serviceWorkerReady;
    const requestId = `clear-${++this.requestIdCounter}`;

    const response = await this.sendMessage({
      type: "CLEAR_CACHE",
      requestId,
    });

    if (response.type === "CLEAR_CACHE_RESPONSE") {
      if (!response.success) {
        throw new Error(response.error || "Failed to clear cache");
      }
    } else {
      throw new Error("Unexpected response type");
    }
  }


  /**
   * Dispose client resources
   */
  async dispose(): Promise<void> {
    if (this.runtimeManager) {
      await this.runtimeManager.dispose();
      this.runtimeManager = null;
    }
  }
}
