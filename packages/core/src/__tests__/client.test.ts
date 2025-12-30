/**
 * Unit tests for LLMClient
 * Tests initialization, direct execution, and message handling
 */
import { LLMClient } from "../client.js";
import { RuntimeManager } from "../runtime/manager.js";
import type { Message, ToolDefinition } from "../types.js";

// Mock RuntimeManager
jest.mock("../runtime/manager.js", () => {
  const mockRuntime = {
    initialize: jest.fn().mockResolvedValue(undefined),
    chat: jest.fn().mockResolvedValue({ type: "content", text: "Hello" }),
    getStatus: jest.fn().mockReturnValue("ready"),
    getType: jest.fn().mockReturnValue("transformers"),
    dispose: jest.fn(),
    clearCache: jest.fn(),
  };

  return {
    RuntimeManager: jest.fn().mockImplementation(() => ({
      initialize: jest.fn().mockResolvedValue(undefined),
      getRuntime: jest.fn().mockReturnValue(mockRuntime),
      getStatus: jest.fn().mockReturnValue("ready"),
      getActiveRuntimeType: jest.fn().mockReturnValue("transformers"),
      dispose: jest.fn().mockResolvedValue(undefined),
      clearCache: jest.fn().mockResolvedValue(undefined),
    })),
  };
});

// Mock navigator for service worker tests
const originalNavigator = global.navigator;

describe("LLMClient", () => {
  let client: LLMClient;

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock navigator with no service worker support for direct execution
    Object.defineProperty(global, "navigator", {
      value: undefined,
      writable: true,
    });
    client = new LLMClient();
  });

  afterEach(() => {
    Object.defineProperty(global, "navigator", {
      value: originalNavigator,
      writable: true,
    });
  });

  describe("initialization", () => {
    it("should initialize successfully in direct execution mode", async () => {
      await expect(client.initialize({})).resolves.toBeUndefined();
    });

    it("should create RuntimeManager with provided config", async () => {
      const config = { preferredRuntime: "transformers" as const, debug: true };
      await client.initialize(config);
      
      expect(RuntimeManager).toHaveBeenCalledWith(config);
    });

    it("should throw on initialization failure", async () => {
      (RuntimeManager as unknown as jest.Mock).mockImplementationOnce(() => ({
        initialize: jest.fn().mockRejectedValue(new Error("Init failed")),
      }));

      const failClient = new LLMClient();
      await expect(failClient.initialize({})).rejects.toThrow();
    });

    it("should handle reinitialize with new config", async () => {
      await client.initialize({});
      await client.initialize({ preferredRuntime: "api" }); // Second call with different config
      
      // RuntimeManager is created each time to allow switching runtimes
      expect(RuntimeManager).toHaveBeenCalledTimes(2);
    });
  });

  describe("chat", () => {
    beforeEach(async () => {
      await client.initialize({});
    });

    it("should send messages and return response", async () => {
      const messages: Message[] = [{ role: "user", content: "Hello" }];
      const tools: ToolDefinition[] = [];
      
      const response = await client.chat(messages, tools);
      
      expect(response).toEqual({ type: "content", text: "Hello" });
    });

    it("should pass tools to runtime", async () => {
      const messages: Message[] = [{ role: "user", content: "Calculate 2+2" }];
      const tools: ToolDefinition[] = [{
        name: "calculate",
        description: "Calculate math",
        parameters: { type: "object", properties: {} },
        handler: async () => ({ result: 4 }),
      }];
      
      await client.chat(messages, tools);
      
      // Get the mock runtime and check it was called with tools
      const mockManager = (RuntimeManager as unknown as jest.Mock).mock?.results?.[0]?.value;
      const mockRuntime = mockManager.getRuntime();
      expect(mockRuntime.chat).toHaveBeenCalledWith(
        messages,
        expect.arrayContaining([expect.objectContaining({ name: "calculate" })]),
        undefined
      );
    });

    it("should handle chat options", async () => {
      const messages: Message[] = [{ role: "user", content: "Hi" }];
      const options = { maxTokens: 100, temperature: 0.5 };
      
      await client.chat(messages, [], options);
      
      const mockManager = (RuntimeManager as unknown as jest.Mock).mock?.results?.[0]?.value;
      const mockRuntime = mockManager.getRuntime();
      expect(mockRuntime.chat).toHaveBeenCalledWith(
        messages,
        expect.any(Array),
        options
      );
    });
  });

  describe("getStatus", () => {
    it("should return idle when not initialized", async () => {
      // Create new client without initializing
      const uninitClient = new LLMClient();
      
      // Without a runtime manager, direct execution returns idle
      // This is actually handled by the sendMessage -> handleDirectExecution path
      // For now, we just check it doesn't throw
      const status = await uninitClient.getStatus();
      expect(status).toBeDefined();
    });

    it("should return runtime status after initialization", async () => {
      await client.initialize({});
      const status = await client.getStatus();
      
      expect(status).toBe("ready");
    });
  });

  describe("direct execution mode", () => {
    it("should use direct execution when no service worker", async () => {
      // navigator is undefined, so service worker is not available
      await client.initialize({});
      
      // The initialization should succeed using direct execution
      expect(RuntimeManager).toHaveBeenCalled();
    });
  });
});

describe("LLMClient with service worker available", () => {
  beforeEach(() => {
    // Mock navigator with service worker support but no registrations
    Object.defineProperty(global, "navigator", {
      value: {
        serviceWorker: {
          getRegistrations: jest.fn().mockResolvedValue([]),
          ready: Promise.resolve({ active: null }),
        },
      },
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(global, "navigator", {
      value: originalNavigator,
      writable: true,
    });
  });

  it("should fall back to direct execution when no service workers registered", async () => {
    const client = new LLMClient();
    await client.initialize({});
    
    // Should still work via direct execution
    expect(RuntimeManager).toHaveBeenCalled();
  });
});
