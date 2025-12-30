/**
 * Unit tests for RuntimeManager
 */
import { RuntimeManager } from "../runtime/manager.js";
import type { RuntimeConfig } from "../types.js";

// Mock the runtime implementations
jest.mock("../runtime/webllm.js", () => ({
  WebLLMRuntime: jest.fn().mockImplementation(() => ({
    initialize: jest.fn(),
    chat: jest.fn(),
    getStatus: jest.fn().mockReturnValue("ready"),
    getType: jest.fn().mockReturnValue("webllm"),
    dispose: jest.fn(),
    clearCache: jest.fn(),
  })),
}));

jest.mock("../runtime/transformers.js", () => ({
  TransformersRuntime: jest.fn().mockImplementation(() => ({
    initialize: jest.fn(),
    chat: jest.fn(),
    getStatus: jest.fn().mockReturnValue("ready"),
    getType: jest.fn().mockReturnValue("transformers"),
    dispose: jest.fn(),
    clearCache: jest.fn(),
  })),
}));

jest.mock("../runtime/api.js", () => ({
  APIRuntime: jest.fn().mockImplementation(() => ({
    initialize: jest.fn(),
    chat: jest.fn(),
    getStatus: jest.fn().mockReturnValue("ready"),
    getType: jest.fn().mockReturnValue("api"),
    dispose: jest.fn(),
    clearCache: jest.fn(),
  })),
}));

// Get mocked modules
import { WebLLMRuntime } from "../runtime/webllm.js";
import { TransformersRuntime } from "../runtime/transformers.js";
import { APIRuntime } from "../runtime/api.js";

const MockWebLLMRuntime = WebLLMRuntime as jest.MockedClass<typeof WebLLMRuntime>;
const MockTransformersRuntime = TransformersRuntime as jest.MockedClass<typeof TransformersRuntime>;
const MockAPIRuntime = APIRuntime as jest.MockedClass<typeof APIRuntime>;

describe("RuntimeManager", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset static method mocks
    jest.spyOn(RuntimeManager, "checkWebGPUSupport").mockResolvedValue(false);
    jest.spyOn(RuntimeManager, "checkWASMSupport").mockReturnValue(true);
    
    // By default, make API fail so it doesn't fast-path - tests can override this
    MockAPIRuntime.mockImplementation(() => ({
      initialize: jest.fn().mockRejectedValue(new Error("No API URL configured")),
      chat: jest.fn(),
      getStatus: jest.fn().mockReturnValue("error"),
      getType: jest.fn().mockReturnValue("api"),
      dispose: jest.fn(),
      clearCache: jest.fn(),
    }) as any);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("fallback chain building", () => {
    it("should build auto fallback chain: webllm -> transformers -> api", () => {
      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      // Access private field for testing (normally not recommended, but useful here)
      expect((manager as any).fallbackChain).toEqual(["webllm", "transformers", "api"]);
    });

    it("should build chain for preferred runtime with api fallback", () => {
      const config: RuntimeConfig = { preferredRuntime: "transformers" };
      const manager = new RuntimeManager(config);
      
      expect((manager as any).fallbackChain).toEqual(["transformers", "api"]);
    });

    it("should build chain for webllm with transformers and api fallback", () => {
      const config: RuntimeConfig = { preferredRuntime: "webllm" };
      const manager = new RuntimeManager(config);
      
      expect((manager as any).fallbackChain).toEqual(["webllm", "api", "transformers"]);
    });

    it("should only have api in chain when api is preferred", () => {
      const config: RuntimeConfig = { preferredRuntime: "api" };
      const manager = new RuntimeManager(config);
      
      expect((manager as any).fallbackChain).toEqual(["api"]);
    });

    it("should default to auto when no preference specified", () => {
      const config: RuntimeConfig = {};
      const manager = new RuntimeManager(config);
      
      expect((manager as any).fallbackChain).toEqual(["webllm", "transformers", "api"]);
    });
  });

  describe("initialize", () => {
    it("should skip webllm when WebGPU is not available", async () => {
      jest.spyOn(RuntimeManager, "checkWebGPUSupport").mockResolvedValue(false);
      
      // Make transformers succeed
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      await manager.initialize();
      
      // WebLLM should not have been instantiated since WebGPU is not available
      expect(MockWebLLMRuntime).not.toHaveBeenCalled();
      // API is tried first (fast-path), then transformers in sequential fallback
      expect(MockAPIRuntime).toHaveBeenCalled();
      expect(MockTransformersRuntime).toHaveBeenCalled();
    });

    it("should use webllm when WebGPU is available", async () => {
      jest.spyOn(RuntimeManager, "checkWebGPUSupport").mockResolvedValue(true);
      
      MockWebLLMRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("webllm"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      await manager.initialize();
      
      // API fast-path is attempted, then webllm is background loaded
      expect(MockAPIRuntime).toHaveBeenCalled();
      expect(MockWebLLMRuntime).toHaveBeenCalled();
    });

    it("should fall back to next runtime when current fails", async () => {
      jest.spyOn(RuntimeManager, "checkWebGPUSupport").mockResolvedValue(true);
      
      // Make WebLLM fail
      MockWebLLMRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockRejectedValue(new Error("WebLLM init failed")),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("error"),
        getType: jest.fn().mockReturnValue("webllm"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      // Make Transformers succeed
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      await manager.initialize();
      
      // API tried first (fast-path fails), then webllm tried (fails), then transformers
      expect(MockAPIRuntime).toHaveBeenCalled();
      expect(MockWebLLMRuntime).toHaveBeenCalled();
      expect(MockTransformersRuntime).toHaveBeenCalled();
    });

    it("should throw when all runtimes fail", async () => {
      jest.spyOn(RuntimeManager, "checkWebGPUSupport").mockResolvedValue(false);
      
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockRejectedValue(new Error("Transformers init failed")),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("error"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      // API already set to fail in beforeEach

      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      await expect(manager.initialize()).rejects.toThrow("Failed to initialize any runtime");
    });

    it("should initialize preferred runtime directly when specified", async () => {
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "transformers" };
      const manager = new RuntimeManager(config);
      
      await manager.initialize();
      
      expect(MockTransformersRuntime).toHaveBeenCalled();
      expect(MockWebLLMRuntime).not.toHaveBeenCalled();
    });
    
    it("should fast-path API when available for instant readiness", async () => {
      // Make API succeed
      MockAPIRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("api"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "auto" };
      const manager = new RuntimeManager(config);
      
      await manager.initialize();
      
      // API should be fast-pathed and become the initial runtime
      expect(MockAPIRuntime).toHaveBeenCalled();
      expect(manager.getStatus()).toBe("ready");
    });
  });

  describe("getRuntime", () => {
    it("should throw if not initialized", () => {
      const config: RuntimeConfig = {};
      const manager = new RuntimeManager(config);
      
      expect(() => manager.getRuntime()).toThrow("Runtime not initialized");
    });

    it("should return the current runtime after initialization", async () => {
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "transformers" };
      const manager = new RuntimeManager(config);
      await manager.initialize();
      
      const runtime = manager.getRuntime();
      expect(runtime).toBeDefined();
      expect(runtime.getStatus()).toBe("ready");
    });
  });

  describe("getStatus", () => {
    it("should return idle when not initialized", () => {
      const config: RuntimeConfig = {};
      const manager = new RuntimeManager(config);
      
      expect(manager.getStatus()).toBe("idle");
    });

    it("should return runtime status after initialization", async () => {
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: jest.fn(),
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "transformers" };
      const manager = new RuntimeManager(config);
      await manager.initialize();
      
      expect(manager.getStatus()).toBe("ready");
    });
  });

  describe("dispose", () => {
    it("should dispose the current runtime", async () => {
      const mockDispose = jest.fn().mockResolvedValue(undefined);
      MockTransformersRuntime.mockImplementation(() => ({
        initialize: jest.fn().mockResolvedValue(undefined),
        chat: jest.fn(),
        getStatus: jest.fn().mockReturnValue("ready"),
        getType: jest.fn().mockReturnValue("transformers"),
        dispose: mockDispose,
        clearCache: jest.fn(),
      }) as any);

      const config: RuntimeConfig = { preferredRuntime: "transformers" };
      const manager = new RuntimeManager(config);
      await manager.initialize();
      await manager.dispose();
      
      expect(mockDispose).toHaveBeenCalled();
    });

    it("should handle dispose when not initialized", async () => {
      const config: RuntimeConfig = {};
      const manager = new RuntimeManager(config);
      
      // Should not throw
      await expect(manager.dispose()).resolves.toBeUndefined();
    });
  });

  describe("static methods", () => {
    beforeEach(() => {
      jest.restoreAllMocks();
    });

    it("checkWASMSupport should return true when WebAssembly is defined", () => {
      // WebAssembly is always defined in Node.js
      expect(RuntimeManager.checkWASMSupport()).toBe(true);
    });

    it("checkWebGPUSupport should return false when navigator is undefined", async () => {
      // In Node.js, navigator is undefined
      const result = await RuntimeManager.checkWebGPUSupport();
      expect(result).toBe(false);
    });
  });
});
