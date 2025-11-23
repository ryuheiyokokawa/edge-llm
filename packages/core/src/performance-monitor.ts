/**
 * Performance monitoring for tool execution and inference
 */
export interface PerformanceMetrics {
  inferenceTime: number; // milliseconds
  toolExecutionTime: number; // milliseconds
  totalTime: number; // milliseconds
  iterations: number;
  toolCalls: number;
  tokensGenerated?: number;
  tokensInput?: number;
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    inferenceTime: 0,
    toolExecutionTime: 0,
    totalTime: 0,
    iterations: 0,
    toolCalls: 0,
  };

  private startTime: number = 0;
  private inferenceStartTime: number = 0;
  private toolExecutionStartTime: number = 0;

  /**
   * Start monitoring a complete operation
   */
  start(): void {
    this.startTime = Date.now();
    this.metrics = {
      inferenceTime: 0,
      toolExecutionTime: 0,
      totalTime: 0,
      iterations: 0,
      toolCalls: 0,
    };
  }

  /**
   * Start inference timing
   */
  startInference(): void {
    this.inferenceStartTime = Date.now();
  }

  /**
   * End inference timing
   */
  endInference(): void {
    if (this.inferenceStartTime > 0) {
      this.metrics.inferenceTime += Date.now() - this.inferenceStartTime;
      this.inferenceStartTime = 0;
    }
  }

  /**
   * Start tool execution timing
   */
  startToolExecution(): void {
    this.toolExecutionStartTime = Date.now();
  }

  /**
   * End tool execution timing
   */
  endToolExecution(): void {
    if (this.toolExecutionStartTime > 0) {
      this.metrics.toolExecutionTime += Date.now() - this.toolExecutionStartTime;
      this.toolExecutionStartTime = 0;
    }
  }

  /**
   * Record an iteration
   */
  recordIteration(): void {
    this.metrics.iterations++;
  }

  /**
   * Record tool calls
   */
  recordToolCalls(count: number): void {
    this.metrics.toolCalls += count;
  }

  /**
   * Record token usage
   */
  recordTokens(input?: number, generated?: number): void {
    if (input !== undefined) {
      this.metrics.tokensInput = (this.metrics.tokensInput || 0) + input;
    }
    if (generated !== undefined) {
      this.metrics.tokensGenerated =
        (this.metrics.tokensGenerated || 0) + generated;
    }
  }

  /**
   * Get current metrics
   */
  getMetrics(): PerformanceMetrics {
    if (this.startTime > 0) {
      this.metrics.totalTime = Date.now() - this.startTime;
    }
    return { ...this.metrics };
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.startTime = 0;
    this.inferenceStartTime = 0;
    this.toolExecutionStartTime = 0;
    this.metrics = {
      inferenceTime: 0,
      toolExecutionTime: 0,
      totalTime: 0,
      iterations: 0,
      toolCalls: 0,
    };
  }
}

