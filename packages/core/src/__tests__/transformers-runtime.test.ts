/**
 * Unit tests for TransformersRuntime
 * Tests initialization, prompt formatting, and response parsing
 */
import { TransformersRuntime } from "../runtime/transformers.js";
import type { Message, ToolDefinition } from "../types.js";

// Mock @huggingface/transformers
const mockPipeline = jest.fn();
const mockAutoTokenizer = {
  from_pretrained: jest.fn(),
};
const mockEnv = {
  allowLocalModels: true,
  useBrowserCache: false,
};

jest.mock("@huggingface/transformers", () => ({
  pipeline: (...args: any[]) => mockPipeline(...args),
  AutoTokenizer: {
    from_pretrained: (...args: any[]) => mockAutoTokenizer.from_pretrained(...args),
  },
  env: mockEnv,
}));

// Mock RuntimeManager for WASM check
jest.mock("../runtime/manager.js", () => ({
  RuntimeManager: {
    checkWASMSupport: jest.fn().mockReturnValue(true),
    checkWebGPUSupport: jest.fn().mockResolvedValue(false),
  },
}));

import { RuntimeManager } from "../runtime/manager.js";

describe("TransformersRuntime", () => {
  let runtime: TransformersRuntime;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset WASM mock to return true by default
    (RuntimeManager.checkWASMSupport as jest.Mock).mockReturnValue(true);
    
    runtime = new TransformersRuntime();
    
    // Setup default successful mocks
    mockAutoTokenizer.from_pretrained.mockResolvedValue({
      apply_chat_template: jest.fn().mockReturnValue("formatted prompt"),
      decode: jest.fn().mockReturnValue("Hello, I am an AI assistant."),
    });
    
    // The pipeline function returns a mock structure with model.generate
    const mockPipelineInstance = {
      model: {
        generate: jest.fn().mockResolvedValue({
          slice: jest.fn().mockReturnValue([1, 2, 3]) // mock tokens
        })
      }
    };
    mockPipeline.mockResolvedValue(mockPipelineInstance);
  });

  describe("constructor", () => {
    it("should set default model name", () => {
      const runtime = new TransformersRuntime();
      expect((runtime as any).modelName).toBe("onnx-community/functiongemma-270m-it-ONNX");
    });
  });

  describe("initialize", () => {
    // Note: The WASM availability test was removed since TransformersRuntime now uses
    // inline `typeof WebAssembly !== "undefined"` check which cannot be mocked in Node.js
    // where WebAssembly is always available.

    it("should use model from config if provided", async () => {
      await runtime.initialize({
        models: { transformers: "custom/model-name" },
      });
      
      expect(mockAutoTokenizer.from_pretrained).toHaveBeenCalledWith(
        "custom/model-name",
        expect.any(Object)
      );
    });

    it("should use default model if not specified in config", async () => {
      await runtime.initialize({});
      
      expect(mockAutoTokenizer.from_pretrained).toHaveBeenCalledWith(
        "onnx-community/functiongemma-270m-it-ONNX",
        expect.any(Object)
      );
    });

    it("should initialize tokenizer and pipeline", async () => {
      await runtime.initialize({});
      
      expect(mockAutoTokenizer.from_pretrained).toHaveBeenCalled();
      expect(mockPipeline).toHaveBeenCalledWith(
        "text-generation",
        expect.any(String),
        expect.any(Object)
      );
    });

    it("should set status to ready on success", async () => {
      await runtime.initialize({});
      
      expect(runtime.getStatus()).toBe("ready");
    });

    it("should set status to error on failure", async () => {
      mockAutoTokenizer.from_pretrained.mockRejectedValue(new Error("Load failed"));
      
      await expect(runtime.initialize({})).rejects.toThrow();
      expect(runtime.getStatus()).toBe("error");
    });

    it("should configure environment settings", async () => {
      await runtime.initialize({});
      
      expect(mockEnv.allowLocalModels).toBe(false);
      expect(mockEnv.useBrowserCache).toBe(false);
    });
  });

  describe("chat", () => {
    // Note: Tests that need custom pipeline behavior must set up the mock
    // BEFORE calling initialize(), since the pipeline is captured at init time.

    it("should throw if not initialized", async () => {
      const uninitRuntime = new TransformersRuntime();
      
      await expect(
        uninitRuntime.chat([{ role: "user", content: "Hello" }], [])
      ).rejects.toThrow("not initialized");
    });

    it("should format messages and generate response", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue("Hello!"),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      await runtime.initialize({});
      
      const messages: Message[] = [{ role: "user", content: "Hi" }];
      const response = await runtime.chat(messages, []);
      
      expect(response.type).toBe("content");
      if (response.type === "content") {
        expect(response.text).toBe("Hello!");
      }
    });

    it("should detect and parse tool calls from FunctionGemma format", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue('<start_function_call>call:calculate{expression:<escape>2+2<escape>}<end_function_call>'),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      await runtime.initialize({});
 
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate math",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      }];
 
      const response = await runtime.chat(
        [{ role: "user", content: "What is 2+2?" }],
        tools
      );
 
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls" && response.calls.length > 0) {
        expect(response.calls[0]?.name).toBe("calculate");
        expect(response.calls[0]?.arguments).toEqual({ expression: "2+2" });
      }
    });

    it("should return content if no tool call detected", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue("I cannot help with that."),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      await runtime.initialize({});
 
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate math",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      }];
 
      const response = await runtime.chat(
        [{ role: "user", content: "Tell me a joke" }],
        tools
      );
 
      expect(response.type).toBe("content");
    });

    it("should include tool descriptions in prompt when tools provided", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue("formatted"),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);

      // Re-initialize to get the new tokenizer
      await runtime.initialize({});

      const tools: ToolDefinition[] = [{
        name: "getWeather",
        description: "Get weather for a city",
        parameters: { 
          type: "object", 
          properties: { city: { type: "string" } },
          required: ["city"],
        },
        handler: async () => ({}),
      }];

      await runtime.chat([{ role: "user", content: "What's the weather?" }], tools);

      // Check that apply_chat_template was called with a developer message
      expect(mockTokenizer.apply_chat_template).toHaveBeenCalled();
      const callArgs = mockTokenizer.apply_chat_template.mock.calls[0][0];
      const templateTools = mockTokenizer.apply_chat_template.mock.calls[0][1].tools;
      
      const devMsg = callArgs.find((m: any) => m.role === "developer");
      expect(devMsg).toBeDefined();
      expect(devMsg.content).toContain("function calling");
      
      // Verification that tools are passed to template
      expect(templateTools).toBeDefined();
      expect(templateTools[0].function.name).toBe("getWeather");
    });
  });

  describe("tool call parsing", () => {
    // Note: These tests need to set up the mock BEFORE initialize,
    // so we don't use a beforeEach with initialize here

    it("should parse tool call embedded in text", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue('Sure, let me calculate that. <start_function_call>call:calculate{expression:<escape>5*5<escape>}<end_function_call>'),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      // Initialize after setting up the mock
      await runtime.initialize({});
 
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      }];
 
      const response = await runtime.chat([{ role: "user", content: "5*5?" }], tools);
 
      expect(response.type).toBe("tool_calls");
      if (response.type === "tool_calls" && response.calls.length > 0) {
        expect(response.calls[0]?.name).toBe("calculate");
      }
    });

    it("should ignore unknown tools", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue('<start_function_call>call:unknownTool{arg:1}<end_function_call>'),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      // Initialize after setting up the mock
      await runtime.initialize({});
 
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      }];
 
      const response = await runtime.chat([{ role: "user", content: "Test" }], tools);
 
      // Should return as content since the tool wasn't found
      expect(response.type).toBe("content");
    });
 
    it("should handle malformed JSON gracefully", async () => {
      const mockTokenizer = {
        apply_chat_template: jest.fn().mockReturnValue({ input_ids: { dims: [1, 1] } }),
        decode: jest.fn().mockReturnValue('<start_function_call>call:calculate{expression:{invalid}}<end_function_call>'),
      };
      mockAutoTokenizer.from_pretrained.mockResolvedValue(mockTokenizer);
      
      // Initialize after setting up the mock
      await runtime.initialize({});
 
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      }];
 
      const response = await runtime.chat([{ role: "user", content: "Test" }], tools);
 
      // Should return as content since JSON is invalid
      expect(response.type).toBe("content");
    });
  });

  describe("dispose", () => {
    it("should clear pipeline and tokenizer", async () => {
      await runtime.initialize({});
      await runtime.dispose();
      
      expect((runtime as any).pipeline).toBeNull();
      expect((runtime as any).tokenizer).toBeNull();
    });
  });
});
