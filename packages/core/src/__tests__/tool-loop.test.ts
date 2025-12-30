import { jest, describe, it, expect, beforeEach } from "@jest/globals";
import { ToolLoop } from "../tool-loop";
import { ToolRegistry } from "../tool-registry";
import type { ModelResponse } from "../types";

// Mock ToolExecutor
const mockExecuteTools = jest.fn<any>();
const mockCheckMaxIterationsExceeded = jest.fn<any>();
const mockStartTimer = jest.fn();
const mockResetTimer = jest.fn();

jest.mock("../tool-executor", () => {
  return {
    ToolExecutor: jest.fn().mockImplementation(() => ({
      executeTools: mockExecuteTools,
      checkMaxIterationsExceeded: mockCheckMaxIterationsExceeded,
      startTimer: mockStartTimer,
      resetTimer: mockResetTimer,
    })),
  };
});

describe("ToolLoop", () => {
  let toolLoop: ToolLoop;
  let registry: ToolRegistry;

  beforeEach(() => {
    jest.clearAllMocks();
    mockExecuteTools.mockResolvedValue([]);
    mockCheckMaxIterationsExceeded.mockReturnValue(false);
    
    registry = new ToolRegistry();
    toolLoop = new ToolLoop(registry);
  });

  it("should handle single-turn content response", async () => {
    const chatFn = jest.fn<() => Promise<ModelResponse>>().mockResolvedValue({
      type: "content",
      text: "Hello world",
    });

    // @ts-ignore
    const result = await toolLoop.execute("Hi", chatFn);

    expect(result).toEqual({ type: "content", text: "Hello world" });
    expect(chatFn).toHaveBeenCalledTimes(1);
    expect(toolLoop.getHistory()).toHaveLength(2); // User + Assistant
    expect(toolLoop.getHistory()[0]).toEqual({ role: "user", content: "Hi" });
    expect(toolLoop.getHistory()[1]).toEqual({
      role: "assistant",
      content: "Hello world",
    });
  });

  it("should handle multi-turn tool calling", async () => {
    // 1. First call returns tool calls
    // 2. Second call returns final content
    const chatFn = jest.fn<() => Promise<ModelResponse>>()
      .mockResolvedValueOnce({
        type: "tool_calls",
        calls: [{ id: "call_1", name: "test_tool", arguments: {} }],
      })
      .mockResolvedValueOnce({
        type: "content",
        text: "Result is 42",
      });

    // Mock tool execution result
    mockExecuteTools.mockResolvedValueOnce([
      { tool_call_id: "call_1", result: "42" },
    ]);

    // Register a dummy tool so it's available
    registry.register({
      name: "test_tool",
      description: "Test tool",
      parameters: { type: "object", properties: {} },
      handler: async () => "42",
    });

    // @ts-ignore
    const result = await toolLoop.execute("Calculate this", chatFn);

    expect(result).toEqual({ type: "content", text: "Result is 42" });
    expect(chatFn).toHaveBeenCalledTimes(2);
    expect(mockExecuteTools).toHaveBeenCalledTimes(1);
    
    // Check history accumulation
    const history = toolLoop.getHistory();
    expect(history).toHaveLength(4);
    // 0: User "Calculate this"
    // 1: Assistant tool_calls
    // 2: Tool result
    // 3: Assistant "Result is 42"
    expect(history[1]!.role).toBe("assistant");
    expect(history[1]!.tool_calls).toBeDefined();
    expect(history[2]!.role).toBe("tool");
    expect(history[2]!.content).toBe(JSON.stringify("42"));
  });

  it("should enforce max iterations", async () => {
    // Mock executor to say max iterations exceeded
    mockCheckMaxIterationsExceeded.mockReturnValue(5);

    const chatFn = jest.fn<() => Promise<ModelResponse>>();

    // @ts-ignore
    await expect(toolLoop.execute("Start", chatFn)).resolves.toEqual({
      type: "content",
      text: "Error: Max iterations exceeded (0 >= 5)",
    });
  });

  it("should handle errors gracefully", async () => {
    // Chat function throws error
    const chatFn = jest.fn<() => Promise<ModelResponse>>().mockRejectedValue(new Error("API Error"));

    // @ts-ignore
    const result = await toolLoop.execute("Start", chatFn);

    expect(result).toEqual({
      type: "content",
      text: "Error: API Error",
    });
  });

  it("should clear history", () => {
    toolLoop.addMessage({ role: "user", content: "test" });
    expect(toolLoop.getHistory()).toHaveLength(1);
    
    toolLoop.clearHistory();
    expect(toolLoop.getHistory()).toHaveLength(0);
  });
});
