import type { ToolDefinition } from "@edge-llm/core";

export const exampleTools: ToolDefinition[] = [
  {
    name: "calculate",
    description: "Evaluate a mathematical expression",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description:
            "Mathematical expression to evaluate (e.g., '2+2', '10*5')",
        },
      },
      required: ["expression"],
    },
    handler: async (args: Record<string, unknown>) => {
      const expression = args.expression as string;
      try {
        // Safe evaluation - in production, use a proper math parser
        const result = Function(`"use strict"; return (${expression})`)();
        return { result, expression };
      } catch (error) {
        return { error: `Invalid expression: ${expression}` };
      }
    },
  },
  {
    name: "getCurrentTime",
    description: "Get the current date and time",
    parameters: {
      type: "object",
      properties: {},
      required: [],
    },
    handler: async () => {
      return {
        time: new Date().toISOString(),
        timestamp: Date.now(),
      };
    },
  },
  {
    name: "searchWeb",
    description: "Search the web (mock implementation)",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
    },
    handler: async (args: Record<string, unknown>) => {
      const query = args.query as string;
      // Mock search results
      return {
        query,
        results: [
          { title: `Result 1 for "${query}"`, url: "https://example.com/1" },
          { title: `Result 2 for "${query}"`, url: "https://example.com/2" },
        ],
      };
    },
  },
];
