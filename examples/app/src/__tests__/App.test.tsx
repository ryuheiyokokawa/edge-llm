/**
 * Unit tests for Example App
 */
import { render, screen, act } from "@testing-library/react";
import App from "../App";

// Mock the LLMProvider from @edge-llm/react
jest.mock("@edge-llm/react", () => ({
  LLMProvider: jest.fn(({ children }) => (
    <div data-testid="llm-provider">{children}</div>
  )),
  useLLM: () => ({
    send: jest.fn(),
    registerTool: jest.fn(),
    registerTools: jest.fn(),
    unregisterTool: jest.fn(),
    status: "ready",
    initialized: true,
  }),
}));

import { LLMProvider } from "@edge-llm/react";

// Mock ChatInterface to simplify App tests
jest.mock("../components/ChatInterface", () => ({
  ChatInterface: () => <div data-testid="chat-interface">Chat Interface</div>,
}));

describe("App", () => {
  it("should render without crashing", async () => {
    await act(async () => {
      render(<App />);
    });
    
    expect(screen.getByTestId("llm-provider")).toBeInTheDocument();
    expect(screen.getByText("Edge LLM Playground")).toBeInTheDocument();
  });

  it("should render ChatInterface", async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByTestId("chat-interface")).toBeInTheDocument();
  });

  it("should switch runtime and update config", async () => {
    render(<App />);
    
    // Default should be transformers
    expect(LLMProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({ preferredRuntime: "transformers" })
      }),
      expect.anything()
    );

    // Switch to webllm
    const select = screen.getByRole("combobox");
    await act(async () => {
      render(<App />); // Re-render to trigger state change? No, fire event.
    });

    // Actually use fireEvent or userEvent
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.change(select, { target: { value: "webllm" } });

    expect(LLMProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({ preferredRuntime: "webllm" })
      }),
      expect.anything()
    );
  });
});
