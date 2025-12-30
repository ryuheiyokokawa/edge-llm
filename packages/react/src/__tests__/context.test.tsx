/**
 * Unit tests for LLMContext and useLLMContext
 */
import { render, screen } from "@testing-library/react";
import { LLMContext, useLLMContext, type LLMContextValue } from "../context";

// Test component that uses the context
function TestConsumer() {
  const { status, initialized } = useLLMContext();
  return (
    <div>
      <span data-testid="status">{status}</span>
      <span data-testid="initialized">{initialized ? "true" : "false"}</span>
    </div>
  );
}

describe("LLMContext", () => {
  describe("useLLMContext", () => {
    it("should throw when used outside of LLMProvider", () => {
      // Suppress console.error for the expected error
      const consoleSpy = jest.spyOn(console, "error").mockImplementation(() => {});
      
      expect(() => {
        render(<TestConsumer />);
      }).toThrow("useLLM must be used within LLMProvider");
      
      consoleSpy.mockRestore();
    });

    it("should return context value when inside provider", () => {
      const mockContextValue: LLMContextValue = {
        client: null,
        status: "ready",
        activeRuntime: "transformers",
        config: {},
        initialized: true,
        clearCache: async () => {},
      };

      render(
        <LLMContext.Provider value={mockContextValue}>
          <TestConsumer />
        </LLMContext.Provider>
      );

      expect(screen.getByTestId("status")).toHaveTextContent("ready");
      expect(screen.getByTestId("initialized")).toHaveTextContent("true");
    });

    it("should provide idle status when not initialized", () => {
      const mockContextValue: LLMContextValue = {
        client: null,
        status: "idle",
        activeRuntime: null,
        config: {},
        initialized: false,
        clearCache: async () => {},
      };

      render(
        <LLMContext.Provider value={mockContextValue}>
          <TestConsumer />
        </LLMContext.Provider>
      );

      expect(screen.getByTestId("status")).toHaveTextContent("idle");
      expect(screen.getByTestId("initialized")).toHaveTextContent("false");
    });

    it("should provide error status on failure", () => {
      const mockContextValue: LLMContextValue = {
        client: null,
        status: "error",
        activeRuntime: null,
        config: {},
        initialized: false,
        clearCache: async () => {},
      };

      render(
        <LLMContext.Provider value={mockContextValue}>
          <TestConsumer />
        </LLMContext.Provider>
      );

      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });
  });
});
