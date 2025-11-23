/**
 * Multi-turn tool calling loop manager
 */
import type {
  Message,
  ToolDefinition,
  ModelResponse,
  ContentResponse,
  ToolExecutionResult,
  ToolExecutionConfig,
  ChatOptions,
  ToolCall,
} from "./types.js";
import { ToolExecutor } from "./tool-executor.js";
import { ToolRegistry } from "./tool-registry.js";

export interface ToolLoopOptions extends ChatOptions {
  toolConfig?: ToolExecutionConfig;
  onIteration?: (iteration: number, calls: ToolCall[]) => void;
  onToolResult?: (result: ToolExecutionResult) => void;
}

/**
 * Manages multi-turn tool calling loops
 */
export class ToolLoop {
  private executor: ToolExecutor;
  private toolRegistry: ToolRegistry;
  private conversationHistory: Message[] = [];

  constructor(
    toolRegistry: ToolRegistry,
    toolConfig?: ToolExecutionConfig
  ) {
    this.toolRegistry = toolRegistry;
    this.executor = new ToolExecutor(toolConfig);
  }

  /**
   * Execute a complete tool calling loop
   * This handles the multi-turn conversation until the model stops calling tools
   */
  async execute(
    initialMessage: string | Message,
    chatFn: (
      messages: Message[],
      tools: ToolDefinition[],
      options?: ChatOptions
    ) => Promise<ModelResponse>,
    options?: ToolLoopOptions
  ): Promise<ContentResponse> {
    // Initialize conversation
    const userMessage: Message =
      typeof initialMessage === "string"
        ? { role: "user", content: initialMessage }
        : initialMessage;

    this.conversationHistory = [userMessage];
    this.executor.startTimer();

    let iteration = 0;
    let response: ModelResponse;

    try {
      while (true) {
        // Check max iterations
        if (this.executor.checkMaxIterationsExceeded(iteration)) {
          throw new Error(
            `Max iterations exceeded (${iteration} >= ${this.executor.checkMaxIterationsExceeded(iteration)})`
          );
        }

        // Get tools for this iteration
        const tools = this.toolRegistry.getAll();

        // Call model
        response = await chatFn(
          this.conversationHistory,
          tools,
          options
        );

        // If content response, we're done
        if (response.type === "content") {
          // Add assistant response to history
          this.conversationHistory.push({
            role: "assistant",
            content: response.text || "",
          });
          return response;
        }

        // Handle tool calls
        if (response.type === "tool_calls") {
          iteration++;

          // Notify iteration callback
          if (options?.onIteration) {
            options.onIteration(iteration, response.calls);
          }

          // Execute all tools in parallel
          const toolMap = new Map(
            tools.map((t) => [t.name, t])
          );
          const results = await this.executor.executeTools(
            response.calls,
            toolMap,
            iteration
          );

          // Notify tool result callbacks
          if (options?.onToolResult) {
            for (const result of results) {
              options.onToolResult(result);
            }
          }

          // Add assistant message with tool calls to history
          this.conversationHistory.push({
            role: "assistant",
            content: "",
            tool_calls: response.calls,
          });

          // Add tool results to history
          for (const result of results) {
            const toolCall = response.calls.find(
              (tc) => tc.id === result.tool_call_id
            );
            this.conversationHistory.push({
              role: "tool",
              content:
                result.error || JSON.stringify(result.result),
              tool_call_id: result.tool_call_id,
              name: toolCall?.name,
            });
          }

          // Continue loop to get next response
          continue;
        }
      }
    } catch (error) {
      // Return error as content response
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        type: "content",
        text: `Error: ${errorMessage}`,
      };
    } finally {
      this.executor.resetTimer();
    }
  }

  /**
   * Get conversation history
   */
  getHistory(): Message[] {
    return [...this.conversationHistory];
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /**
   * Add message to history
   */
  addMessage(message: Message): void {
    this.conversationHistory.push(message);
  }
}

