/**
 * Unit tests for Example App
 */
import { render, screen, act } from "@testing-library/react";
import App from "../App";

// Mock the LLMProvider from @edge-llm/react
jest.mock("@edge-llm/react", () => ({
  LLMProvider: ({ children }: { children: any }) => (
    <div data-testid="llm-provider">{children}</div>
  ),
  useLLM: () => ({
    send: jest.fn(),
    registerTool: jest.fn(),
    registerTools: jest.fn(),
    unregisterTool: jest.fn(),
    status: "ready",
    initialized: true,
  }),
}));

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
});
