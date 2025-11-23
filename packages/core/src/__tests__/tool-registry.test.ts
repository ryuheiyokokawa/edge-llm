/**
 * Unit tests for ToolRegistry
 */
import { ToolRegistry } from "../tool-registry.js";
import type { ToolDefinition } from "../types.js";

describe("ToolRegistry", () => {
  let registry: ToolRegistry;

  beforeEach(() => {
    registry = new ToolRegistry();
  });

  describe("register", () => {
    it("should register a single tool", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "A test tool",
        parameters: {
          type: "object",
          properties: {},
        },
        handler: async () => "result",
      };

      registry.register(tool);

      expect(registry.has("testTool")).toBe(true);
      expect(registry.get("testTool")).toEqual(tool);
    });

    it("should replace existing tool with same name", () => {
      const tool1: ToolDefinition = {
        name: "testTool",
        description: "First version",
        parameters: { type: "object", properties: {} },
        handler: async () => "result1",
      };

      const tool2: ToolDefinition = {
        name: "testTool",
        description: "Second version",
        parameters: { type: "object", properties: {} },
        handler: async () => "result2",
      };

      registry.register(tool1);
      registry.register(tool2);

      expect(registry.get("testTool")).toEqual(tool2);
    });
  });

  describe("registerMany", () => {
    it("should register multiple tools", () => {
      const tools: ToolDefinition[] = [
        {
          name: "tool1",
          description: "Tool 1",
          parameters: { type: "object", properties: {} },
          handler: async () => "result1",
        },
        {
          name: "tool2",
          description: "Tool 2",
          parameters: { type: "object", properties: {} },
          handler: async () => "result2",
        },
      ];

      registry.registerMany(tools);

      expect(registry.size()).toBe(2);
      expect(registry.has("tool1")).toBe(true);
      expect(registry.has("tool2")).toBe(true);
    });
  });

  describe("unregister", () => {
    it("should remove a tool", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "A test tool",
        parameters: { type: "object", properties: {} },
        handler: async () => "result",
      };

      registry.register(tool);
      expect(registry.has("testTool")).toBe(true);

      const removed = registry.unregister("testTool");
      expect(removed).toBe(true);
      expect(registry.has("testTool")).toBe(false);
    });

    it("should return false if tool doesn't exist", () => {
      const removed = registry.unregister("nonexistent");
      expect(removed).toBe(false);
    });
  });

  describe("getAll", () => {
    it("should return all registered tools", () => {
      const tools: ToolDefinition[] = [
        {
          name: "tool1",
          description: "Tool 1",
          parameters: { type: "object", properties: {} },
          handler: async () => "result1",
        },
        {
          name: "tool2",
          description: "Tool 2",
          parameters: { type: "object", properties: {} },
          handler: async () => "result2",
        },
      ];

      registry.registerMany(tools);
      const allTools = registry.getAll();

      expect(allTools).toHaveLength(2);
      expect(allTools).toContainEqual(tools[0]);
      expect(allTools).toContainEqual(tools[1]);
    });
  });

  describe("toOpenAIFormat", () => {
    it("should convert tools to OpenAI format", () => {
      const tool: ToolDefinition = {
        name: "getWeather",
        description: "Get weather for a city",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
          required: ["city"],
        },
        handler: async () => ({}),
      };

      registry.register(tool);
      const openAIFormat = registry.toOpenAIFormat();

      expect(openAIFormat).toHaveLength(1);
      expect(openAIFormat[0]).toEqual({
        type: "function",
        function: {
          name: "getWeather",
          description: "Get weather for a city",
          parameters: tool.parameters,
        },
      });
    });
  });

  describe("clear", () => {
    it("should remove all tools", () => {
      registry.registerMany([
        {
          name: "tool1",
          description: "Tool 1",
          parameters: { type: "object", properties: {} },
          handler: async () => "result1",
        },
        {
          name: "tool2",
          description: "Tool 2",
          parameters: { type: "object", properties: {} },
          handler: async () => "result2",
        },
      ]);

      expect(registry.size()).toBe(2);
      registry.clear();
      expect(registry.size()).toBe(0);
    });
  });
});

