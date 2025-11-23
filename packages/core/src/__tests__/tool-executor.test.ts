/**
 * Unit tests for ToolExecutor
 */
import { ToolExecutor } from "../tool-executor.js";
import type { ToolCall, ToolDefinition, ToolExecutionConfig } from "../types.js";

describe("ToolExecutor", () => {
  let executor: ToolExecutor;

  beforeEach(() => {
    executor = new ToolExecutor();
  });

  describe("executeTool", () => {
    it("should execute a tool successfully", async () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "Test tool",
        parameters: { type: "object", properties: {} },
        handler: async () => ({ result: "success" }),
      };

      const call: ToolCall = {
        id: "call1",
        name: "testTool",
        arguments: {},
      };

      const result = await executor.executeTool(call, tool, 0);
      expect(result.tool_call_id).toBe("call1");
      expect(result.result).toEqual({ result: "success" });
      expect(result.error).toBeUndefined();
    });

    it("should handle tool execution errors", async () => {
      const tool: ToolDefinition = {
        name: "failingTool",
        description: "Failing tool",
        parameters: { type: "object", properties: {} },
        handler: async () => {
          throw new Error("Tool execution failed");
        },
      };

      const call: ToolCall = {
        id: "call1",
        name: "failingTool",
        arguments: {},
      };

      const result = await executor.executeTool(call, tool, 0);
      expect(result.tool_call_id).toBe("call1");
      expect(result.error).toBeDefined();
      expect(result.error).toContain("Tool execution failed");
    });

    it("should timeout on slow tools", async () => {
      const config: ToolExecutionConfig = {
        executionTimeout: 100, // 100ms timeout
      };
      executor = new ToolExecutor(config);

      const tool: ToolDefinition = {
        name: "slowTool",
        description: "Slow tool",
        parameters: { type: "object", properties: {} },
        handler: async () => {
          await new Promise((resolve) => setTimeout(resolve, 500));
          return "too slow";
        },
      };

      const call: ToolCall = {
        id: "call1",
        name: "slowTool",
        arguments: {},
      };

      const result = await executor.executeTool(call, tool, 0);
      expect(result.error).toBeDefined();
      expect(result.error).toContain("timeout");
    });
  });

  describe("checkMaxIterationsExceeded", () => {
    it("should detect when max iterations exceeded", () => {
      const config: ToolExecutionConfig = {
        maxIterations: 3,
      };
      executor = new ToolExecutor(config);

      expect(executor.checkMaxIterationsExceeded(2)).toBe(false);
      expect(executor.checkMaxIterationsExceeded(3)).toBe(true);
      expect(executor.checkMaxIterationsExceeded(4)).toBe(true);
    });
  });

  describe("timer management", () => {
    it("should track elapsed time", async () => {
      executor.startTimer();
      await new Promise((resolve) => setTimeout(resolve, 10));
      const elapsed = executor.getElapsedTime();
      expect(elapsed).toBeGreaterThan(0);
      expect(elapsed).toBeLessThan(100); // Should be around 10ms
    });

    it("should reset timer", () => {
      executor.startTimer();
      executor.resetTimer();
      expect(executor.getElapsedTime()).toBe(0);
    });
  });
});

