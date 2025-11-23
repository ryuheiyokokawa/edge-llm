/**
 * Main thread client for communicating with service worker
 */
import type {
  RuntimeConfig,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
  RuntimeStatus,
  ServiceWorkerMessage,
  ServiceWorkerResponse,
} from "./types.js";
import { RuntimeManager } from "./runtime/manager.js";
import { ToolRegistry } from "./tool-registry.js";

export class LLMClient {
  private serviceWorkerRegistration: ServiceWorkerRegistration | null = null;
  private serviceWorkerReady: Promise<void>;
  private requestIdCounter = 0;
  private runtimeManager: RuntimeManager | null = null;
  private toolRegistry: ToolRegistry = new ToolRegistry();

  constructor() {
    this.serviceWorkerReady = this.ensureServiceWorker();
  }

  /**
   * Ensure service worker is registered and ready
   * Note: Service worker should be registered by the provider/application
   * For dev mode, service worker is optional
   */
  private async ensureServiceWorker(): Promise<void> {
    console.log("[LLMClient] Checking service worker support...");
    if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) {
      console.log(
        "[LLMClient] Service workers not supported, using direct execution"
      );
      // Service workers not supported - will work without them
      this.serviceWorkerRegistration = null;
      return;
    }

    try {
      // Check if there are any service workers registered
      const registrations = await navigator.serviceWorker.getRegistrations();
      console.log(
        "[LLMClient] Found",
        registrations.length,
        "service worker registrations"
      );

      if (registrations.length === 0) {
        console.log(
          "[LLMClient] No service workers registered, using direct execution"
        );
        this.serviceWorkerRegistration = null;
        return;
      }

      console.log("[LLMClient] Waiting for service worker to be ready...");
      // Wait for service worker to be ready if it exists
      // The service worker should already be registered by the provider
      this.serviceWorkerRegistration = await navigator.serviceWorker.ready;
      console.log("[LLMClient] Service worker is ready");
    } catch (error) {
      // Service worker not available - this is expected in dev mode
      console.log(
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
    console.log("[LLMClient] Handling direct execution for:", message.type);
    switch (message.type) {
      case "INITIALIZE":
        try {
          if (!this.runtimeManager) {
            console.log("[LLMClient] Creating RuntimeManager...");
            this.runtimeManager = new RuntimeManager(message.config);
            console.log("[LLMClient] Initializing RuntimeManager...");
            await this.runtimeManager.initialize();
            console.log("[LLMClient] RuntimeManager initialized successfully");
          }
          return { type: "INITIALIZE_RESPONSE", success: true };
        } catch (error) {
          const errorMsg =
            error instanceof Error ? error.message : String(error);
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
        const status = this.runtimeManager?.getStatus() || "idle";
        return {
          type: "STATUS_RESPONSE",
          requestId: message.requestId,
          status,
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
    console.log("[LLMClient] Starting initialize...");
    console.log("[LLMClient] Waiting for service worker ready...");
    await this.serviceWorkerReady;
    console.log("[LLMClient] Service worker ready check complete");
    console.log(
      "[LLMClient] Service worker available:",
      !!this.serviceWorkerRegistration?.active
    );

    console.log("[LLMClient] Sending INITIALIZE message...");
    const response = await this.sendMessage({
      type: "INITIALIZE",
      config,
    });
    console.log("[LLMClient] Received response:", response.type);

    if (response.type === "INITIALIZE_RESPONSE") {
      if (!response.success) {
        const errorMsg = response.error || "Initialization failed";
        console.error("[LLMClient] Initialization failed:", errorMsg);
        throw new Error(errorMsg);
      }
      console.log("[LLMClient] Initialization successful");
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
    await this.serviceWorkerReady;

    // If using direct execution, get status from runtime manager
    if (!this.serviceWorkerRegistration?.active && this.runtimeManager) {
      return this.runtimeManager.getStatus();
    }

    const requestId = `status-${++this.requestIdCounter}`;

    const response = await this.sendMessage({
      type: "STATUS",
      requestId,
    });

    if (response.type === "STATUS_RESPONSE") {
      return response.status;
    } else {
      throw new Error("Unexpected response type");
    }
  }
}
