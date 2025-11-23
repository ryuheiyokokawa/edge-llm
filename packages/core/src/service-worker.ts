/**
 * Service worker controller for managing LLM inference
 * This will be used by the service worker script
 */
import { RuntimeManager } from "./runtime/manager.js";
import { ToolRegistry } from "./tool-registry.js";
import type {
  RuntimeConfig,
  ServiceWorkerMessage,
  ServiceWorkerResponse,
} from "./types.js";

export class ServiceWorkerController {
  private runtimeManager: RuntimeManager | null = null;
  private toolRegistry: ToolRegistry = new ToolRegistry();

  /**
   * Initialize the service worker controller
   */
  async initialize(config: RuntimeConfig): Promise<void> {
    this.runtimeManager = new RuntimeManager(config);
    await this.runtimeManager.initialize();
  }

  /**
   * Handle messages from main thread
   */
  async handleMessage(
    message: ServiceWorkerMessage
  ): Promise<ServiceWorkerResponse> {
    switch (message.type) {
      case "INITIALIZE":
        try {
          await this.initialize(message.config);
          return { type: "INITIALIZE_RESPONSE", success: true };
        } catch (error) {
          return {
            type: "INITIALIZE_RESPONSE",
            success: false,
            error: error instanceof Error ? error.message : String(error),
          };
        }

      case "CHAT":
        return this.handleChat(message);

      case "STATUS":
        return this.handleStatus(message);

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

  private async handleChat(
    message: Extract<ServiceWorkerMessage, { type: "CHAT" }>
  ): Promise<ServiceWorkerResponse> {
    if (!this.runtimeManager) {
      return {
        type: "CHAT_RESPONSE",
        requestId: message.requestId,
        response: {
          type: "content",
          text: "",
        },
        error: "Service worker not initialized",
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
        response: {
          type: "content",
          text: "",
        },
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  private handleStatus(
    message: Extract<ServiceWorkerMessage, { type: "STATUS" }>
  ): ServiceWorkerResponse {
    const status = this.runtimeManager?.getStatus() || "idle";
    return {
      type: "STATUS_RESPONSE",
      requestId: message.requestId,
      status,
    };
  }

  /**
   * Get tool registry
   */
  getToolRegistry(): ToolRegistry {
    return this.toolRegistry;
  }

  /**
   * Dispose resources
   */
  async dispose(): Promise<void> {
    if (this.runtimeManager) {
      await this.runtimeManager.dispose();
      this.runtimeManager = null;
    }
    this.toolRegistry.clear();
  }
}
