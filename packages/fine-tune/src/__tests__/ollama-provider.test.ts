/**
 * Tests for OllamaProvider
 */

import { OllamaProvider } from "../synthetic/OllamaProvider";
import type { ToolSchema } from "../types";

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe("OllamaProvider", () => {
  const sampleTool: ToolSchema = {
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name" },
      },
      required: ["location"],
    },
  };

  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe("constructor", () => {
    it("should use default config when not provided", () => {
      const provider = new OllamaProvider();
      expect(provider).toBeDefined();
    });

    it("should accept custom config", () => {
      const provider = new OllamaProvider({
        type: "ollama",
        model: "custom-model",
        baseUrl: "http://custom:11434",
      });
      expect(provider).toBeDefined();
    });
  });

  describe("checkAvailability", () => {
    it("should return available when model exists", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          models: [{ name: "llama3.2:latest" }, { name: "mistral:latest" }],
        }),
      });

      const provider = new OllamaProvider({ type: "ollama", model: "llama3.2" });
      const result = await provider.checkAvailability();

      expect(result.available).toBe(true);
      expect(result.models).toContain("llama3.2:latest");
    });

    it("should return unavailable when model not found", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          models: [{ name: "mistral:latest" }],
        }),
      });

      const provider = new OllamaProvider({ type: "ollama", model: "llama3.2" });
      const result = await provider.checkAvailability();

      expect(result.available).toBe(false);
    });

    it("should handle connection errors", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

      const provider = new OllamaProvider();
      const result = await provider.checkAvailability();

      expect(result.available).toBe(false);
      expect(result.error).toContain("Connection refused");
    });
  });

  describe("generateExamples", () => {
    it("should generate examples from Ollama response", async () => {
      // Mock tags endpoint for implicit check
      mockFetch.mockResolvedValueOnce({
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
            {
              userQuery: "Tell me the weather for New York",
              toolCall: {
                name: "get_weather",
                arguments: { location: "New York" },
              },
            },
          ]),
        }),
      });

      const provider = new OllamaProvider();
      const examples = await provider.generateExamples(sampleTool, 2);

      expect(examples.length).toBeGreaterThanOrEqual(1);
      expect(examples[0]!.userQuery).toBeDefined();
      expect(examples[0]!.expectedToolCalls[0]!.name).toBe("get_weather");
    });

    it("should handle malformed responses gracefully", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          response: "This is not valid JSON",
        }),
      });

      const provider = new OllamaProvider();
      const examples = await provider.generateExamples(sampleTool, 5);

      // Should return empty array, not throw
      expect(Array.isArray(examples)).toBe(true);
    });

    it("should handle markdown-wrapped JSON", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          response: `\`\`\`json
[
  {
    "userQuery": "Weather in Paris?",
    "toolCall": {"name": "get_weather", "arguments": {"location": "Paris"}}
  }
]
\`\`\``,
        }),
      });

      const provider = new OllamaProvider();
      const examples = await provider.generateExamples(sampleTool, 1);

      expect(examples.length).toBe(1);
      expect(examples[0]!.userQuery).toContain("Paris");
    });

    it("should skip duplicate queries", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          response: JSON.stringify([
            {
              userQuery: "existing query",
              toolCall: { name: "get_weather", arguments: { location: "Tokyo" } },
            },
          ]),
        }),
      });

      const provider = new OllamaProvider();
      const examples = await provider.generateExamples(sampleTool, 1, [
        "existing query",
      ]);

      // Should skip the duplicate
      expect(examples.length).toBe(0);
    });
  });

  describe("generateForTools", () => {
    it("should generate examples for multiple tools", async () => {
      const tools: ToolSchema[] = [
        sampleTool,
        {
          name: "calculate",
          description: "Calculate expression",
          parameters: {
            type: "object",
            properties: {
              expression: { type: "string" },
            },
            required: ["expression"],
          },
        },
      ];

      // Mock responses for each tool
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            response: JSON.stringify([
              {
                userQuery: "Weather?",
                toolCall: { name: "get_weather", arguments: { location: "NYC" } },
              },
            ]),
          }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            response: JSON.stringify([
              {
                userQuery: "Calculate 1+1",
                toolCall: { name: "calculate", arguments: { expression: "1+1" } },
              },
            ]),
          }),
        });

      const provider = new OllamaProvider();
      const results = await provider.generateForTools(tools, 1);

      expect(results.size).toBe(2);
      expect(results.has("get_weather")).toBe(true);
      expect(results.has("calculate")).toBe(true);
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

      const provider = new OllamaProvider();
      const progressCalls: Array<{ tool: string; generated: number }> = [];

      await provider.generateForTools([sampleTool], 1, (tool, generated) => {
        progressCalls.push({ tool, generated });
      });

      expect(progressCalls.length).toBeGreaterThan(0);
      expect(progressCalls.some((p) => p.tool === "get_weather")).toBe(true);
    });
  });
});
