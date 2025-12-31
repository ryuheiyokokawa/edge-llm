/**
 * Tests for FormatConverter
 */

import { FormatConverter } from "../dataset/FormatConverter";
import { FUNCTION_GEMMA_TOKENS, type ToolSchema, type TrainingExample } from "../types";

describe("FormatConverter", () => {
  const sampleTools: ToolSchema[] = [
    {
      name: "get_weather",
      description: "Get the current weather for a location",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "City name" },
        },
        required: ["location"],
      },
    },
    {
      name: "calculate",
      description: "Evaluate a math expression",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression" },
        },
        required: ["expression"],
      },
    },
  ];

  describe("formatToolDeclarations", () => {
    it("should format tool declarations with correct tokens", () => {
      const result = FormatConverter.formatToolDeclarations(sampleTools);

      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_OF_TURN);
      expect(result).toContain("developer");
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_FUNCTION_DECLARATION);
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_FUNCTION_DECLARATION);
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_OF_TURN);
      expect(result).toContain("get_weather");
      expect(result).toContain("calculate");
    });

    it("should include tool descriptions", () => {
      const result = FormatConverter.formatToolDeclarations(sampleTools);

      expect(result).toContain("Get the current weather for a location");
      expect(result).toContain("Evaluate a math expression");
    });
  });

  describe("formatUserMessage", () => {
    it("should format user message with correct tokens", () => {
      const result = FormatConverter.formatUserMessage("What's the weather?");

      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_OF_TURN);
      expect(result).toContain("user");
      expect(result).toContain("What's the weather?");
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_OF_TURN);
    });
  });

  describe("formatFunctionCall", () => {
    it("should format function call with correct tokens", () => {
      const result = FormatConverter.formatFunctionCall([
        { name: "get_weather", arguments: { location: "Tokyo" } },
      ]);

      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_OF_TURN);
      expect(result).toContain("model");
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_FUNCTION_CALL);
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_FUNCTION_CALL);
      expect(result).toContain("get_weather");
      expect(result).toContain("Tokyo");
    });

    it("should handle multiple tool calls", () => {
      const result = FormatConverter.formatFunctionCall([
        { name: "get_weather", arguments: { location: "Tokyo" } },
        { name: "calculate", arguments: { expression: "2 + 2" } },
      ]);

      expect(result).toContain("get_weather");
      expect(result).toContain("calculate");
      expect(result).toContain("Tokyo");
      expect(result).toContain("2 + 2");
    });
  });

  describe("formatToolResponse", () => {
    it("should format tool response with correct tokens", () => {
      const result = FormatConverter.formatToolResponse({ temperature: 22 });

      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_OF_TURN);
      expect(result).toContain("tool");
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_FUNCTION_RESPONSE);
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_FUNCTION_RESPONSE);
      expect(result).toContain("temperature");
      expect(result).toContain("22");
    });
  });

  describe("formatModelResponse", () => {
    it("should format model response with correct tokens", () => {
      const result = FormatConverter.formatModelResponse("The weather is sunny.");

      expect(result).toContain(FUNCTION_GEMMA_TOKENS.START_OF_TURN);
      expect(result).toContain("model");
      expect(result).toContain("The weather is sunny.");
      expect(result).toContain(FUNCTION_GEMMA_TOKENS.END_OF_TURN);
    });
  });

  describe("convertExample", () => {
    it("should convert a simple example", () => {
      const example: TrainingExample = {
        userQuery: "What's the weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
      };

      const result = FormatConverter.convertExample(example, sampleTools);

      expect(result.text).toBeDefined();
      expect(result.text).toContain("developer");
      expect(result.text).toContain("user");
      expect(result.text).toContain("model");
      expect(result.text).toContain("What's the weather in Tokyo?");
      expect(result.text).toContain("get_weather");
    });

    it("should include tool response and final response when provided", () => {
      const example: TrainingExample = {
        userQuery: "What's the weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
        toolResponses: { get_weather: { temperature: 22 } },
        expectedFinalResponse: "It's 22 degrees in Tokyo.",
      };

      const result = FormatConverter.convertExample(example, sampleTools);

      expect(result.text).toContain("temperature");
      expect(result.text).toContain("22");
      expect(result.text).toContain("It's 22 degrees in Tokyo.");
    });
  });

  describe("convertExamples", () => {
    it("should convert multiple examples", () => {
      const examples: TrainingExample[] = [
        {
          userQuery: "Weather in Tokyo?",
          expectedToolCalls: [{ name: "get_weather", arguments: { location: "Tokyo" } }],
        },
        {
          userQuery: "Calculate 2 + 2",
          expectedToolCalls: [{ name: "calculate", arguments: { expression: "2 + 2" } }],
        },
      ];

      const results = FormatConverter.convertExamples(examples, sampleTools);

      expect(results).toHaveLength(2);
      expect(results[0]!.text).toContain("Tokyo");
      expect(results[1]!.text).toContain("2 + 2");
    });
  });

  describe("toolDefinitionToSchema", () => {
    it("should strip handler from tool definition", () => {
      const toolDef = {
        name: "test_tool",
        description: "A test tool",
        parameters: {
          type: "object" as const,
          properties: {
            arg1: { type: "string" as const },
          },
          required: ["arg1"],
        },
        handler: async () => ({ result: "test" }),
      };

      const schema = FormatConverter.toolDefinitionToSchema(toolDef);

      expect(schema.name).toBe("test_tool");
      expect(schema.description).toBe("A test tool");
      expect(schema.parameters).toBeDefined();
      expect((schema as any).handler).toBeUndefined();
    });
  });
});
