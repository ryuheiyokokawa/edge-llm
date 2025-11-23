/**
 * Tool executor with timeout, retry, and safety features
 */
import type {
  ToolDefinition,
  ToolCall,
  ToolExecutionConfig,
  ToolExecutionResult,
} from "./types.js";

export class ToolExecutor {
  private config: Required<ToolExecutionConfig>;
  private executionStartTime: number = 0;

  constructor(config: ToolExecutionConfig = {}) {
    this.config = {
      maxIterations: config.maxIterations ?? 5,
      executionTimeout: config.executionTimeout ?? 30000, // 30 seconds
      totalTimeout: config.totalTimeout ?? 120000, // 2 minutes
      retryStrategy: config.retryStrategy ?? "exponential",
      maxRetries: config.maxRetries ?? 2,
    };
  }

  /**
   * Execute a single tool call with timeout and retry logic
   */
  async executeTool(
    call: ToolCall,
    tool: ToolDefinition,
    _iteration: number
  ): Promise<ToolExecutionResult> {
    const startTime = Date.now();
    const toolCallId = call.id;

    // Check total timeout
    if (this.executionStartTime > 0) {
      const elapsed = Date.now() - this.executionStartTime;
      if (elapsed > this.config.totalTimeout) {
        throw new Error(
          `Total execution timeout exceeded (${this.config.totalTimeout}ms)`
        );
      }
    }

    let lastError: Error | null = null;
    let attempt = 0;

    while (attempt <= this.config.maxRetries) {
      try {
        // Create timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(
            () => reject(new Error(`Tool execution timeout: ${tool.name}`)),
            this.config.executionTimeout
          );
        });

        // Execute tool with timeout
        const result = await Promise.race([
          tool.handler(call.arguments),
          timeoutPromise,
        ]);

        const executionTime = Date.now() - startTime;

        return {
          tool_call_id: toolCallId,
          result,
          executionTime,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Check if error is retryable
        const isRetryable = this.isRetryableError(error);

        if (!isRetryable || attempt >= this.config.maxRetries) {
          // Don't retry or max retries reached
          const executionTime = Date.now() - startTime;
          return {
            tool_call_id: toolCallId,
            result: null,
            error: lastError.message,
            executionTime,
          };
        }

        // Calculate retry delay
        const delay = this.calculateRetryDelay(attempt);
        await this.sleep(delay);
        attempt++;
      }
    }

    // Should never reach here, but TypeScript needs it
    const executionTime = Date.now() - startTime;
    return {
      tool_call_id: toolCallId,
      result: null,
      error: lastError?.message || "Unknown error",
      executionTime,
    };
  }

  /**
   * Execute multiple tool calls in parallel
   */
  async executeTools(
    calls: ToolCall[],
    tools: Map<string, ToolDefinition>,
    iteration: number
  ): Promise<ToolExecutionResult[]> {
    return Promise.all(
      calls.map(async (call) => {
        const tool = tools.get(call.name);
        if (!tool) {
          return {
            tool_call_id: call.id,
            result: null,
            error: `Tool ${call.name} not found`,
          };
        }
        return this.executeTool(call, tool, iteration);
      })
    );
  }

  /**
   * Check if error is retryable
   */
  private isRetryableError(error: unknown): boolean {
    if (!(error instanceof Error)) {
      return false;
    }

    // Network errors are retryable
    if (
      error.message.includes("timeout") ||
      error.message.includes("network") ||
      error.message.includes("fetch")
    ) {
      return true;
    }

    // Rate limit errors are retryable
    if (error.message.includes("rate limit") || error.message.includes("429")) {
      return true;
    }

    // Other errors are not retryable by default
    return false;
  }

  /**
   * Calculate retry delay based on strategy
   */
  private calculateRetryDelay(attempt: number): number {
    switch (this.config.retryStrategy) {
      case "exponential":
        return Math.min(1000 * Math.pow(2, attempt), 10000); // Max 10 seconds
      case "linear":
        return 1000 * (attempt + 1); // 1s, 2s, 3s...
      case "none":
        return 0;
      default:
        return 1000;
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Start execution timer
   */
  startTimer(): void {
    this.executionStartTime = Date.now();
  }

  /**
   * Reset execution timer
   */
  resetTimer(): void {
    this.executionStartTime = 0;
  }

  /**
   * Check if max iterations exceeded
   */
  checkMaxIterationsExceeded(iteration: number): boolean {
    return iteration >= this.config.maxIterations;
  }

  /**
   * Get elapsed time since start
   */
  getElapsedTime(): number {
    if (this.executionStartTime === 0) {
      return 0;
    }
    return Date.now() - this.executionStartTime;
  }
}

