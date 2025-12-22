/**
 * Unit tests for tool calling formats (JSON and XML)
 */
import { TransformersRuntime } from "../runtime/transformers.js";
import { WebLLMRuntime } from "../runtime/webllm.js";
import type { ToolDefinition, Message } from "../types.js";

// Mock @huggingface/transformers
const mockPipeline = jest.fn();
const mockAutoTokenizer = {
  from_pretrained: jest.fn(),
};

jest.mock("@huggingface/transformers", () => ({
  pipeline: (...args: any[]) => mockPipeline(...args),
  AutoTokenizer: {
    from_pretrained: (...args: any[]) => mockAutoTokenizer.from_pretrained(...args),
  },
  env: { allowLocalModels: true, useBrowserCache: false },
}));

// Mock @mlc-ai/web-llm
jest.mock("@mlc-ai/web-llm", () => ({
  CreateMLCEngine: jest.fn(),
}));

import * as webllm from "@mlc-ai/web-llm";

// Mock RuntimeManager
jest.mock("../runtime/manager.js", () => ({
  RuntimeManager: {
    checkWASMSupport: jest.fn().mockReturnValue(true),
    checkWebGPUSupport: jest.fn().mockResolvedValue(true),
  },
}));

describe("Tool Calling Formats", () => {
  const tools: ToolDefinition[] = [{
    name: "calculate",
    description: "Calculate math",
    parameters: { type: "object", properties: { expression: { type: "string" } } },
    handler: async () => ({}),
  }];

  const messages: Message[] = [{ role: "user", content: "What is 2+2?" }];

  describe("TransformersRuntime", () => {
    let runtime: TransformersRuntime;

    beforeEach(async () => {
      jest.clearAllMocks();
      runtime = new TransformersRuntime();
      
      mockAutoTokenizer.from_pretrained.mockResolvedValue({
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn(),
      });
      
      mockPipeline.mockResolvedValue({
        model: {
          generate: jest.fn().mockResolvedValue({ slice: jest.fn().mockReturnValue([]) })
        }
      });
    });

    it("should parse XML format when configured", async () => {
      await runtime.initialize({ toolCallFormat: "xml" });
      
      const tokenizer = await mockAutoTokenizer.from_pretrained();
      tokenizer.decode.mockReturnValue('<start_function_call>call:calculate{expression:<escape>2+2<escape>}<end_function_call>');
      
      const response = await runtime.chat(messages, tools);
      
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls") {
        expect(response.calls[0]!.name).toBe("calculate");
        expect(response.calls[0]!.arguments).toEqual({ expression: "2+2" });
      }
    });

    it("should parse JSON format when configured", async () => {
      await runtime.initialize({ toolCallFormat: "json" });
      
      const tokenizer = await mockAutoTokenizer.from_pretrained();
      tokenizer.decode.mockReturnValue('{"tool": "calculate", "arguments": {"expression": "2+2"}}');
      
      const response = await runtime.chat(messages, tools);
      
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls") {
        expect(response.calls[0]!.name).toBe("calculate");
        expect(response.calls[0]!.arguments).toEqual({ expression: "2+2" });
      }
    });
  });

  describe("WebLLMRuntime", () => {
    let runtime: WebLLMRuntime;

    beforeEach(() => {
      jest.clearAllMocks();
      runtime = new WebLLMRuntime();
    });

    it("should parse XML format when configured", async () => {
      const mockEngine = {
        chat: {
          completions: {
            create: jest.fn().mockResolvedValue({
              choices: [{ message: { content: '<start_function_call>call:calculate{expression:<escape>2+2<escape>}<end_function_call>' } }]
            })
          }
        },
        unload: jest.fn(),
      };
      (webllm.CreateMLCEngine as jest.Mock).mockResolvedValue(mockEngine);
      
      await runtime.initialize({ toolCallFormat: "xml" });
      const response = await runtime.chat(messages, tools);
      
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls") {
        expect(response.calls[0]!.name).toBe("calculate");
        expect(response.calls[0]!.arguments).toEqual({ expression: "2+2" });
      }
    });

    it("should parse JSON format when configured", async () => {
      const mockEngine = {
        chat: {
          completions: {
            create: jest.fn().mockResolvedValue({
              choices: [{ message: { content: '{"tool": "calculate", "arguments": {"expression": "2+2"}}' } }]
            })
          }
        },
        unload: jest.fn(),
      };
      (webllm.CreateMLCEngine as jest.Mock).mockResolvedValue(mockEngine);
      
      await runtime.initialize({ toolCallFormat: "json" });
      const response = await runtime.chat(messages, tools);
      
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls") {
        expect(response.calls[0]!.name).toBe("calculate");
        expect(response.calls[0]!.arguments).toEqual({ expression: "2+2" });
      }
    });
  });
});
