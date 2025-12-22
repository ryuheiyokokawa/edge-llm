/**
 * Unit tests for TransformersRuntime
 * Tests initialization, prompt formatting, and response parsing
 */
import { TransformersRuntime } from "../runtime/transformers.js";
import type { Message, ToolDefinition } from "../types.js";

// Mock @xenova/transformers
const mockPipeline = jest.fn();
const mockAutoTokenizer = {
  from_pretrained: jest.fn(),
};
const mockEnv = {
  allowLocalModels: true,
  useBrowserCache: false,
};

jest.mock("@xenova/transformers", () => ({
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
    });
    
    // The pipeline function returns a callable pipeline object
    // When called with text, it returns the generation result
    const mockPipelineInstance = jest.fn().mockResolvedValue([{
      generated_text: "Hello, I am an AI assistant.",
    }]);
    mockPipeline.mockResolvedValue(mockPipelineInstance);
  });

  describe("constructor", () => {
    it("should set default model name", () => {
      const runtime = new TransformersRuntime();
      expect((runtime as any).modelName).toBe("Xenova/Qwen2.5-0.5B-Instruct");
    });
  });

  describe("initialize", () => {
    it("should throw if WASM is not available", async () => {
      (RuntimeManager.checkWASMSupport as jest.Mock).mockReturnValue(false);
      
      await expect(runtime.initialize({})).rejects.toThrow(
        "WASM not available"
      );
    });

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
        "Xenova/Qwen2.5-0.5B-Instruct",
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
      expect(mockEnv.useBrowserCache).toBe(true);
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
      // Set up custom mock before initialize
      const mockPipelineInstance = jest.fn().mockResolvedValue([{ generated_text: "Hello!" }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
      await runtime.initialize({});
      
      const messages: Message[] = [{ role: "user", content: "Hi" }];
      const response = await runtime.chat(messages, []);
      
      expect(response.type).toBe("content");
      if (response.type === "content") {
        expect(response.text).toBe("Hello!");
      }
    });

    it("should detect and parse tool calls from JSON response", async () => {
      // Set up custom mock before initialize
      const mockPipelineInstance = jest.fn().mockResolvedValue([{
        generated_text: '{"tool": "calculate", "arguments": {"expression": "2+2"}}',
      }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
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
      // Set up custom mock before initialize
      const mockPipelineInstance = jest.fn().mockResolvedValue([{
        generated_text: "I cannot help with that.",
      }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
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
      const mockApplyTemplate = jest.fn().mockReturnValue("formatted");
      mockAutoTokenizer.from_pretrained.mockResolvedValue({
        apply_chat_template: mockApplyTemplate,
      });

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

      // Check that apply_chat_template was called with a system message
      expect(mockApplyTemplate).toHaveBeenCalled();
      const callArgs = mockApplyTemplate.mock.calls[0][0];
      const systemMsg = callArgs.find((m: any) => m.role === "system");
      expect(systemMsg).toBeDefined();
      expect(systemMsg.content).toContain("getWeather");
      expect(systemMsg.content).toContain("Get weather for a city");
    });
  });

  describe("tool call parsing", () => {
    // Note: These tests need to set up the mock BEFORE initialize,
    // so we don't use a beforeEach with initialize here

    it("should parse tool call embedded in text", async () => {
      const mockPipelineInstance = jest.fn().mockResolvedValue([{
        generated_text: 'Sure, let me calculate that. {"tool": "calculate", "arguments": {"expression": "5*5"}}',
      }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
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
      const mockPipelineInstance = jest.fn().mockResolvedValue([{
        generated_text: '{"tool": "unknownTool", "arguments": {}}',
      }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
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
      const mockPipelineInstance = jest.fn().mockResolvedValue([{
        generated_text: '{"tool": "calculate", "arguments": {invalid json}',
      }]);
      mockPipeline.mockResolvedValue(mockPipelineInstance);
      
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
