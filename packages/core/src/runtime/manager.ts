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
    const hasWebGPU = await RuntimeManager.checkWebGPUSupport();
    this.log("[RuntimeManager] WebGPU available:", hasWebGPU);

    // Initial pass: Fast-path API if available to get "ready" instantly
    if (this.fallbackChain.includes("api")) {
      try {
        this.log("[RuntimeManager] Fast-pathing API runtime for instant availability...");
        const apiRuntime = await this.tryInitializeRuntime("api");
        this.currentRuntime = apiRuntime;
      } catch (e) {
        this.log("[RuntimeManager] API fast-path failed, will follow standard chain");
      }
    }

    // Secondary pass: Background load the preferred local runtime if not already active
    const preferredLocal = this.fallbackChain.find(t => t === "webllm" || t === "transformers");
    if (preferredLocal && this.currentRuntime?.getType() !== preferredLocal) {
        // Start background loading
        this.backgroundInitialize(preferredLocal).catch(err => {
            console.warn(`[RuntimeManager] Background loading of ${preferredLocal} failed:`, err);
        });
    }

    // If we have NO runtime yet, we must wait for the first successful one in the chain
    if (!this.currentRuntime) {
        await this.sequentialInitialize();
    }
  }

  private async backgroundInitialize(type: RuntimeType): Promise<void> {
    try {
        this.log(`[RuntimeManager] Background initializing ${type}...`);
        const runtime = await this.tryInitializeRuntime(type);
        
        // If successful, swap it in as the primary
        if (runtime && !this.abortController.signal.aborted) {
            this.log(`[RuntimeManager] Hot-swapping to ${type}`);
            const oldRuntime = this.currentRuntime;
            this.currentRuntime = runtime;
            
            // Only dispose if it's NOT the same runtime instance (sanity check)
            if (oldRuntime && oldRuntime !== runtime) {
                this.log(`[RuntimeManager] Disposing old runtime: ${oldRuntime.getType()}`);
                await oldRuntime.dispose();
            }
        }
    } catch (e) {
        this.log(`[RuntimeManager] Background initialization of ${type} failed, staying on current.`);
    }
  }

  private async sequentialInitialize(): Promise<void> {
    let lastError: Error | null = null;

    for (const runtimeType of this.fallbackChain) {
      if (this.currentRuntime?.getType() === runtimeType) return;
      if (this.abortController.signal.aborted) return;

      try {
        const runtime = await this.tryInitializeRuntime(runtimeType as RuntimeType);
        this.currentRuntime = runtime;
        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        continue;
      }
    }

    throw new Error(`Failed to initialize any runtime: ${lastError?.message || "Unknown error"}`);
  }

  private async tryInitializeRuntime(type: RuntimeType): Promise<Runtime> {
    const hasWebGPU = await RuntimeManager.checkWebGPUSupport();
    if (type === "webllm" && !hasWebGPU) {
        throw new Error("WebGPU not supported");
    }

    const runtime = this.createRuntime(type);
    this.initializingRuntime = runtime;

    const runtimeModelId = this.config.models?.[type as keyof NonNullable<typeof this.config.models>];

    await runtime.initialize({
      ...this.config,
      modelId: runtimeModelId || this.config.modelId,
      signal: this.abortController.signal,
    });

    if (this.abortController.signal.aborted) {
      await runtime.dispose();
      throw new Error("Aborted");
    }

    // Pure: Do NOT set `currentRuntime` here. Let the caller decide.
    this.initializingRuntime = null;
    return runtime;
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
   * Get the type of the active runtime
   */
  getActiveRuntimeType(): RuntimeType | undefined {
    return this.currentRuntime?.getType();
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
