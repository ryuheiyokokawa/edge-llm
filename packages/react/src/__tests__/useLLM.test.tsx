/**
 * Unit tests for useLLM hook
 */
import React from "react";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLLM } from "../hooks/useLLM";
import { LLMContext, type LLMContextValue } from "../context";

// Create a mock client
function createMockClient() {
  return {
    initialize: jest.fn().mockResolvedValue(undefined),
    getStatus: jest.fn().mockResolvedValue("ready"),
    chat: jest.fn().mockResolvedValue({ type: "content", text: "Hello!" }),
  };
}

// Helper to create complete context value with all required properties
function createContextValue(overrides: Partial<LLMContextValue> = {}): LLMContextValue {
  return {
    client: null,
    status: "idle",
    activeRuntime: null,
    config: {},
    initialized: false,
    clearCache: async () => {},
    ...overrides,
  };
}

// Test wrapper component
function createWrapper(contextValue: LLMContextValue) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <LLMContext.Provider value={contextValue}>
        {children}
      </LLMContext.Provider>
    );
  };
}

// Test component that uses useLLM
function TestComponent({ onSend }: { onSend?: (result: any) => void }) {
  const { send, registerTool, registerTools, unregisterTool, status, initialized } = useLLM();

  const handleSend = async () => {
    const result = await send("Hello");
    onSend?.(result);
  };

  const handleRegisterTool = () => {
    registerTool({
      name: "testTool",
      description: "A test tool",
      parameters: { type: "object", properties: {} },
      handler: async () => ({ result: "test" }),
    });
  };

  const handleRegisterMultiple = () => {
    registerTools([
      {
        name: "tool1",
        description: "Tool 1",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      },
      {
        name: "tool2",
        description: "Tool 2",
        parameters: { type: "object", properties: {} },
        handler: async () => ({}),
      },
    ]);
  };

  const handleUnregister = () => {
    unregisterTool("testTool");
  };

  return (
    <div>
      <span data-testid="status">{status}</span>
      <span data-testid="initialized">{initialized ? "yes" : "no"}</span>
      <button data-testid="send" onClick={handleSend}>Send</button>
      <button data-testid="register" onClick={handleRegisterTool}>Register</button>
      <button data-testid="registerMultiple" onClick={handleRegisterMultiple}>Register Multiple</button>
      <button data-testid="unregister" onClick={handleUnregister}>Unregister</button>
    </div>
  );
}

describe("useLLM", () => {
  describe("status and initialized", () => {
    it("should return status from context", () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      render(<TestComponent />, { wrapper: createWrapper(contextValue) });

      expect(screen.getByTestId("status")).toHaveTextContent("ready");
      expect(screen.getByTestId("initialized")).toHaveTextContent("yes");
    });

    it("should reflect not initialized state", () => {
      const contextValue = createContextValue({
        status: "idle",
        initialized: false,
      });

      render(<TestComponent />, { wrapper: createWrapper(contextValue) });

      expect(screen.getByTestId("status")).toHaveTextContent("idle");
      expect(screen.getByTestId("initialized")).toHaveTextContent("no");
    });
  });

  describe("send", () => {
    it("should throw when client is not initialized", async () => {
      const contextValue = createContextValue({
        client: null,
        status: "idle",
        initialized: false,
      });

      // We need to catch the error in the component itself
      let caughtError: Error | null = null;
      function ErrorCatchingComponent() {
        const { send } = useLLM();

        const handleSend = async () => {
          try {
            await send("Hello");
          } catch (e) {
            caughtError = e as Error;
          }
        };

        return <button data-testid="send" onClick={handleSend}>Send</button>;
      }

      render(<ErrorCatchingComponent />, { wrapper: createWrapper(contextValue) });

      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      expect(caughtError).not.toBeNull();
      expect((caughtError as any)?.message).toBe("LLM client not initialized");
    });

    it("should call client.chat with message", async () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      const onSend = jest.fn();
      render(<TestComponent onSend={onSend} />, { wrapper: createWrapper(contextValue) });

      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      expect(mockClient.chat).toHaveBeenCalledWith(
        [{ role: "user", content: "Hello" }],
        expect.any(Array),
        expect.objectContaining({ stream: false })
      );

      expect(onSend).toHaveBeenCalledWith({ type: "content", text: "Hello!" });
    });
  });

  describe("tool registration", () => {
    it("should register a tool", async () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      render(<TestComponent />, { wrapper: createWrapper(contextValue) });

      await act(async () => {
        await userEvent.click(screen.getByTestId("register"));
      });

      // Now send a message - the tool should be included
      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      expect(mockClient.chat).toHaveBeenCalledWith(
        expect.any(Array),
        expect.arrayContaining([
          expect.objectContaining({ name: "testTool" }),
        ]),
        expect.any(Object)
      );
    });

    it("should register multiple tools", async () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      render(<TestComponent />, { wrapper: createWrapper(contextValue) });

      await act(async () => {
        await userEvent.click(screen.getByTestId("registerMultiple"));
      });

      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      const chatCall = mockClient.chat.mock.calls[0];
      const tools = chatCall[1];
      expect(tools).toHaveLength(2);
      expect(tools[0].name).toBe("tool1");
      expect(tools[1].name).toBe("tool2");
    });

    it("should unregister a tool", async () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      render(<TestComponent />, { wrapper: createWrapper(contextValue) });

      // Register then unregister
      await act(async () => {
        await userEvent.click(screen.getByTestId("register"));
      });

      await act(async () => {
        await userEvent.click(screen.getByTestId("unregister"));
      });

      // Now send - tool should not be included
      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      const chatCall = mockClient.chat.mock.calls[0];
      const tools = chatCall[1];
      expect(tools).not.toContainEqual(
        expect.objectContaining({ name: "testTool" })
      );
    });

    it("should replace tool with same name on re-register", async () => {
      const mockClient = createMockClient();
      const contextValue = createContextValue({
        client: mockClient as any,
        status: "ready",
        activeRuntime: "transformers",
        initialized: true,
      });

      // Custom test component for this test
      function DuplicateRegisterTest() {
        const { send, registerTool } = useLLM();

        const handleRegisterFirst = () => {
          registerTool({
            name: "myTool",
            description: "First version",
            parameters: { type: "object", properties: {} },
            handler: async () => ({}),
          });
        };

        const handleRegisterSecond = () => {
          registerTool({
            name: "myTool",
            description: "Second version",
            parameters: { type: "object", properties: {} },
            handler: async () => ({}),
          });
        };

        const handleSend = async () => {
          await send("test");
        };

        return (
          <div>
            <button data-testid="first" onClick={handleRegisterFirst}>First</button>
            <button data-testid="second" onClick={handleRegisterSecond}>Second</button>
            <button data-testid="send" onClick={handleSend}>Send</button>
          </div>
        );
      }

      render(<DuplicateRegisterTest />, { wrapper: createWrapper(contextValue) });

      // Register first, then second (replacement)
      await act(async () => {
        await userEvent.click(screen.getByTestId("first"));
      });

      await act(async () => {
        await userEvent.click(screen.getByTestId("second"));
      });

      await act(async () => {
        await userEvent.click(screen.getByTestId("send"));
      });

      const chatCall = mockClient.chat.mock.calls[0];
      const tools = chatCall[1];
      
      // Should only have one tool
      const myTools = tools.filter((t: any) => t.name === "myTool");
      expect(myTools).toHaveLength(1);
      expect(myTools[0].description).toBe("Second version");
    });
  });
});
