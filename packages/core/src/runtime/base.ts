/**
 * Base runtime implementation
 */
import type {
  Runtime,
  RuntimeConfig,
  RuntimeStatus,
  Message,
  ToolDefinition,
  ModelResponse,
  ChatOptions,
} from "../types.js";

export abstract class BaseRuntime implements Runtime {
  protected status: RuntimeStatus = "idle";
  protected config: RuntimeConfig | null = null;

  abstract initialize(config: RuntimeConfig): Promise<void>;
  abstract chat(
    messages: Message[],
    tools: ToolDefinition[],
    options?: ChatOptions
  ): Promise<ModelResponse>;

  getStatus(): RuntimeStatus {
    return this.status;
  }

  protected setStatus(status: RuntimeStatus): void {
    this.status = status;
  }

  protected log(...args: any[]): void {
    if (this.config?.debug) {
      console.log(...args);
    }
  }

  async dispose(): Promise<void> {
    this.status = "idle";
    this.config = null;
  }

  async clearCache(): Promise<void> {
    // Default implementation does nothing
    this.log("[BaseRuntime] clearCache not implemented for this runtime");
  }
}
