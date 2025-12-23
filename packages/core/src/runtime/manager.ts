/**
 * Runtime manager for selecting and managing runtimes
 */
import { WebLLMRuntime } from "./webllm.js";
import { TransformersRuntime } from "./transformers.js";
import { APIRuntime } from "./api.js";
import type {
  Runtime,
  RuntimeConfig,
  RuntimeType,
  RuntimeStatus,
} from "../types.js";

export class RuntimeManager {
  private currentRuntime: Runtime | null = null;
  private initializingRuntime: Runtime | null = null;
  private config: RuntimeConfig;
  private fallbackChain: RuntimeType[] = [];
  private abortController: AbortController = new AbortController();

  constructor(config: RuntimeConfig) {
    this.config = config;
    this.buildFallbackChain();
  }

  private log(...args: any[]) {
    if (this.config.debug) {
      console.log(...args);
    }
  }

  private buildFallbackChain(): void {
    const preferred = this.config.preferredRuntime || "auto";

    if (preferred === "auto") {
      // Auto-detect: WebGPU -> WASM -> API
      this.fallbackChain = ["webllm", "transformers", "api"];
    } else {
      // Use preferred, then fallback based on strategy
      this.fallbackChain = [preferred];

      if (preferred === "webllm") {
        this.fallbackChain.push("api", "transformers");
      } else if (preferred === "transformers") {
        this.fallbackChain.push("api");
      }
    }
  }

  /**
   * Initialize the best available runtime
   */
  async initialize(): Promise<void> {
    let lastError: Error | null = null;

    if (this.config.debug) {
      this.log("[RuntimeManager] Fallback chain:", this.fallbackChain);
    }

    // Check WebGPU support for debugging
    const hasWebGPU = await RuntimeManager.checkWebGPUSupport();
    this.log("[RuntimeManager] WebGPU available:", hasWebGPU);

    for (const runtimeType of this.fallbackChain) {
      if (this.abortController.signal.aborted) {
        this.log("[RuntimeManager] Initialization aborted");
        return;
      }

      try {
        this.log(`[RuntimeManager] Attempting to initialize ${runtimeType}...`);

        if (runtimeType === "webllm" && !hasWebGPU) {
          this.log(
            "[RuntimeManager] Skipping WebLLM because WebGPU is not available"
          );
          continue;
        }

        const runtime = this.createRuntime(runtimeType);
        this.initializingRuntime = runtime;
        
        // Extract model ID from models map if available
        const runtimeModelId = this.config.models?.[runtimeType as keyof NonNullable<typeof this.config.models>];

        // Pass parent abort signal to runtime
        await runtime.initialize({
          ...this.config,
          modelId: runtimeModelId || this.config.modelId,
          signal: this.abortController.signal,
        });

        if (this.abortController.signal.aborted) {
          this.log("[RuntimeManager] Successfully initialized but already aborted, disposing...");
          await runtime.dispose();
          this.initializingRuntime = null;
          return;
        }

        this.currentRuntime = runtime;
        this.initializingRuntime = null;
        this.log(`[RuntimeManager] Successfully initialized ${runtimeType}`);
        return;
      } catch (error) {
        if (this.abortController.signal.aborted) {
          this.log("[RuntimeManager] Aborted during error handling");
          return;
        }
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.warn(
          `[RuntimeManager] Failed to initialize ${runtimeType}:`,
          errorMsg
        );
        lastError = error instanceof Error ? error : new Error(String(error));
        // Try next runtime in fallback chain
        continue;
      }
    }

    const errorMsg = `Failed to initialize any runtime: ${
      lastError?.message || "Unknown error"
    }`;
    console.error("[RuntimeManager]", errorMsg);
    throw new Error(errorMsg);
  }

  /**
   * Create a runtime instance
   */
  private createRuntime(type: RuntimeType): Runtime {
    switch (type) {
      case "webllm":
        return new WebLLMRuntime();
      case "transformers":
        return new TransformersRuntime();
      case "api":
        return new APIRuntime();
      default:
        throw new Error(`Unknown runtime type: ${type}`);
    }
  }

  /**
   * Get current runtime
   */
  getRuntime(): Runtime {
    if (!this.currentRuntime) {
      throw new Error("Runtime not initialized");
    }
    return this.currentRuntime;
  }

  /**
   * Get current status
   */
  getStatus(): RuntimeStatus {
    return this.currentRuntime?.getStatus() || "idle";
  }

  /**
   * Dispose current runtime
   */
  async dispose(): Promise<void> {
    this.abortController.abort();
    if (this.currentRuntime) {
      await this.currentRuntime.dispose();
      this.currentRuntime = null;
    }
    if (this.initializingRuntime) {
      await this.initializingRuntime.dispose();
      this.initializingRuntime = null;
    }
  }

  /**
   * Clear all runtime caches
   */
  async clearCache(): Promise<void> {
    this.log("[RuntimeManager] Clearing all caches...");
    
    // 1. Clear current runtime if active
    if (this.currentRuntime) {
      await this.currentRuntime.clearCache();
    }
    
    // 2. Proactively clear others known caches
    // For WebLLM
    const webllm = new WebLLMRuntime();
    await webllm.clearCache();
    
    // For Transformers.js
    const transformers = new TransformersRuntime();
    await transformers.clearCache();
    
    this.log("[RuntimeManager] All caches cleared");
  }

  /**
   * Check if WebGPU is available
   */
  static async checkWebGPUSupport(): Promise<boolean> {
    if (typeof navigator === "undefined") {
      return false;
    }
    const gpu = (navigator as any).gpu;
    if (!gpu) {
      return false;
    }
    try {
      const adapter = await gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  /**
   * Check if WASM is available
   */
  static checkWASMSupport(): boolean {
    return typeof WebAssembly !== "undefined";
  }
}
