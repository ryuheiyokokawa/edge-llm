/**
 * Unit tests for ToolValidator
 */
import { ToolValidator } from "../tool-validator.js";
import type { ToolCall, ToolDefinition } from "../types.js";

describe("ToolValidator", () => {
  describe("validateArguments", () => {
    it("should validate correct arguments", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "Test tool",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string" },
            age: { type: "number" },
          },
          required: ["name"],
        },
        handler: async () => ({}),
      };

      const call: ToolCall = {
        id: "call1",
        name: "testTool",
        arguments: { name: "John", age: 30 },
      };

      const result = ToolValidator.validateArguments(call, tool);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it("should detect missing required parameters", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "Test tool",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
          required: ["name"],
        },
        handler: async () => ({}),
      };

      const call: ToolCall = {
        id: "call1",
        name: "testTool",
        arguments: {},
      };

      const result = ToolValidator.validateArguments(call, tool);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain("Missing required parameter: name");
    });

    it("should detect type mismatches", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "Test tool",
        parameters: {
          type: "object",
          properties: {
            age: { type: "number" },
          },
        },
        handler: async () => ({}),
      };

      const call: ToolCall = {
        id: "call1",
        name: "testTool",
        arguments: { age: "not a number" },
      };

      const result = ToolValidator.validateArguments(call, tool);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it("should validate enum values", () => {
      const tool: ToolDefinition = {
        name: "testTool",
        description: "Test tool",
        parameters: {
          type: "object",
          properties: {
            status: { type: "string", enum: ["active", "inactive"] },
          },
        },
        handler: async () => ({}),
      };

      const validCall: ToolCall = {
        id: "call1",
        name: "testTool",
        arguments: { status: "active" },
      };

      const invalidCall: ToolCall = {
        id: "call2",
        name: "testTool",
        arguments: { status: "unknown" },
      };

      expect(ToolValidator.validateArguments(validCall, tool).valid).toBe(true);
      expect(ToolValidator.validateArguments(invalidCall, tool).valid).toBe(false);
    });
  });

  describe("formatResult", () => {
    it("should format objects as JSON", () => {
      const result = { key: "value", number: 42 };
      const formatted = ToolValidator.formatResult(result);
      expect(formatted).toBe(JSON.stringify(result));
    });

    it("should return strings as-is", () => {
      const result = "simple string";
      const formatted = ToolValidator.formatResult(result);
      expect(formatted).toBe("simple string");
    });

    it("should handle null and undefined", () => {
      expect(ToolValidator.formatResult(null)).toBe("null");
      expect(ToolValidator.formatResult(undefined)).toBe("null");
    });
  });
});

