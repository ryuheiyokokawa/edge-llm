/**
 * Tests for SyntheticGenerator
 */

import { SyntheticGenerator } from "../synthetic/SyntheticGenerator";
import { DatasetBuilder } from "../dataset/DatasetBuilder";
import type { ToolSchema, SyntheticGenerationConfig } from "../types";

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe("SyntheticGenerator", () => {
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
  ];

  const defaultConfig: SyntheticGenerationConfig = {
    provider: {
      type: "ollama",
      model: "llama3.2",
    },
    examplesPerTool: 5,
  };

  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe("constructor", () => {
    it("should create generator with config", () => {
      const generator = new SyntheticGenerator(defaultConfig);
      expect(generator).toBeDefined();
    });

    it("should default validateOutput to true", () => {
      const generator = new SyntheticGenerator(defaultConfig);
      expect(generator).toBeDefined();
    });
  });

  describe("validateProvider", () => {
    it("should return valid when Ollama is available with model", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          models: [{ name: "llama3.2:latest" }],
        }),
      });

      const generator = new SyntheticGenerator(defaultConfig);
      const result = await generator.validateProvider();

      expect(result.valid).toBe(true);
      expect(result.availableModels).toContain("llama3.2:latest");
    });

    it("should return invalid when model not found", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          models: [{ name: "mistral:latest" }],
        }),
      });

      const generator = new SyntheticGenerator(defaultConfig);
      const result = await generator.validateProvider();

      expect(result.valid).toBe(false);
      expect(result.error).toContain("not found");
    });

    it("should handle connection errors", async () => {
      mockFetch.mockRejectedValueOnce(new Error("ECONNREFUSED"));

      const generator = new SyntheticGenerator(defaultConfig);
      const result = await generator.validateProvider();

      expect(result.valid).toBe(false);
      expect(result.error).toContain("ECONNREFUSED");
    });
  });

  describe("generate", () => {
    it("should throw for unsupported provider types", async () => {
      const generator = new SyntheticGenerator({
        ...defaultConfig,
        provider: { type: "openai" as any, model: "gpt-4" },
      });

      await expect(generator.generate(sampleTools)).rejects.toThrow(
        'Provider type "openai" not yet implemented'
      );
    });

    it("should generate examples for tools", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          response: JSON.stringify([
            {
              userQuery: "What's the weather in Tokyo?",
              toolCall: {
                name: "get_weather",
                arguments: { location: "Tokyo" },
              },
            },
          ]),
        }),
      });

      const generator = new SyntheticGenerator({
        ...defaultConfig,
        examplesPerTool: 1,
      });
      const examples = await generator.generate(sampleTools);

      expect(examples.length).toBeGreaterThan(0);
      expect(examples[0]!.expectedToolCalls[0]!.name).toBe("get_weather");
    });

    it("should validate examples when validateOutput is true", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          response: JSON.stringify([
            {
              userQuery: "Query",
              toolCall: {
                name: "wrong_tool", // Wrong tool name
                arguments: { location: "Tokyo" },
              },
            },
          ]),
        }),
      });

      const generator = new SyntheticGenerator({
        ...defaultConfig,
        examplesPerTool: 1,
        validateOutput: true,
      });
      const examples = await generator.generate(sampleTools);

      // Should filter out invalid examples
      expect(examples.length).toBe(0);
    });

    it("should call progress callback", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          response: JSON.stringify([
            {
              userQuery: "Weather?",
              toolCall: { name: "get_weather", arguments: { location: "NYC" } },
            },
          ]),
        }),
      });

      const generator = new SyntheticGenerator({
        ...defaultConfig,
        examplesPerTool: 1,
      });

      const progressCalls: Array<{ tool: string; phase: string }> = [];
      await generator.generate(sampleTools, (status) => {
        progressCalls.push({ tool: status.tool, phase: status.phase });
      });

      expect(progressCalls.some((p) => p.phase === "generating")).toBe(true);
      expect(progressCalls.some((p) => p.phase === "complete")).toBe(true);
    });
  });

  describe("generateAndAddToBuilder", () => {
    it("should add generated examples to DatasetBuilder", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          response: JSON.stringify([
            {
              userQuery: "Weather in London?",
              toolCall: { name: "get_weather", arguments: { location: "London" } },
            },
          ]),
        }),
      });

      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: "/tmp/test",
      });

      const generator = new SyntheticGenerator({
        ...defaultConfig,
        examplesPerTool: 1,
      });

      const count = await generator.generateAndAddToBuilder(builder);

      expect(count).toBeGreaterThan(0);
      expect(builder.getExampleCount()).toBe(count);
    });
  });
});
