/**
 * Unit tests for ChatInterface component
 */
import { render, screen, act, fireEvent } from "@testing-library/react";
import { ChatInterface } from "../ChatInterface";
import { useChat } from "../../hooks/useChat";

// Mock the useChat hook
jest.mock("../../hooks/useChat");
const mockUseChat = useChat as jest.MockedFunction<typeof useChat>;

describe("ChatInterface", () => {
  const defaultMockValue = {
    messages: [],
    input: "",
    setInput: jest.fn(),
    handleSend: jest.fn(),
    loading: false,
    error: null,
    status: "ready" as const,
    initialized: true,
    clearCache: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseChat.mockReturnValue(defaultMockValue);
  });

  it("should render welcome message when no messages exist", () => {
    render(<ChatInterface />);
    expect(screen.getByText("Start a conversation! Try asking:")).toBeInTheDocument();
  });

  it("should render messages", () => {
    mockUseChat.mockReturnValue({
      ...defaultMockValue,
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there!" },
      ],
    });

    render(<ChatInterface />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("should handle input change", () => {
    render(<ChatInterface />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "test message" } });
    expect(defaultMockValue.setInput).toHaveBeenCalledWith("test message");
  });

  it("should call handleSend on button click", () => {
    mockUseChat.mockReturnValue({
      ...defaultMockValue,
      input: "test",
    });

    render(<ChatInterface />);
    const sendButton = screen.getByText("Send");
    fireEvent.click(sendButton);
    expect(defaultMockValue.handleSend).toHaveBeenCalled();
  });

  it("should show loading state", () => {
    mockUseChat.mockReturnValue({
      ...defaultMockValue,
      loading: true,
    });

    render(<ChatInterface />);
    expect(screen.getByText("Thinking...")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Type your message...")).toBeDisabled();
  });

  it("should show error message", () => {
    mockUseChat.mockReturnValue({
      ...defaultMockValue,
      error: "Something went wrong",
    });

    render(<ChatInterface />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("should call clearCache with confirmation", async () => {
    const confirmSpy = jest.spyOn(window, "confirm").mockReturnValue(true);
    
    render(<ChatInterface />);
    const clearButton = screen.getByText("Clear Models Cache");
    
    await act(async () => {
      fireEvent.click(clearButton);
    });

    expect(confirmSpy).toHaveBeenCalled();
    expect(defaultMockValue.clearCache).toHaveBeenCalled();
    
    confirmSpy.mockRestore();
  });

  it("should not call clearCache if confirmation is cancelled", async () => {
    const confirmSpy = jest.spyOn(window, "confirm").mockReturnValue(false);
    
    render(<ChatInterface />);
    const clearButton = screen.getByText("Clear Models Cache");
    
    await act(async () => {
      fireEvent.click(clearButton);
    });

    expect(defaultMockValue.clearCache).not.toHaveBeenCalled();
    
    confirmSpy.mockRestore();
  });
});
