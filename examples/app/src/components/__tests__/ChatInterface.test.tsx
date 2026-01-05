/**
 * Unit tests for ChatInterface component
 */
import { render, screen, act, fireEvent } from "@testing-library/react";
import { ChatInterface } from "../ChatInterface";

// Mock the store hooks
jest.mock("../../store", () => ({
  useChatState: jest.fn(),
  useLLMState: jest.fn(),
  useInputActions: jest.fn(),
  useResetActions: jest.fn(),
}));

// Mock useChat hook
jest.mock("../../hooks/useChat", () => ({
  useChat: jest.fn(),
}));

import { useChatState, useLLMState, useInputActions, useResetActions } from "../../store";
import { useChat } from "../../hooks/useChat";

const mockUseChatState = useChatState as jest.MockedFunction<typeof useChatState>;
const mockUseLLMState = useLLMState as jest.MockedFunction<typeof useLLMState>;
const mockUseInputActions = useInputActions as jest.MockedFunction<typeof useInputActions>;
const mockUseResetActions = useResetActions as jest.MockedFunction<typeof useResetActions>;
const mockUseChat = useChat as jest.MockedFunction<typeof useChat>;

describe("ChatInterface", () => {
  const defaultChatState = {
    messages: [],
    input: "",
    loading: false,
    error: null,
    conversationHistory: [],
    canSend: false,
  };

  const defaultLLMState = {
    llmStatus: "ready",
    llmInitialized: true,
  };

  const mockSetInput = jest.fn();
  const mockClearMessages = jest.fn();
  const mockHandleSend = jest.fn();
  const mockClearCache = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockUseChatState.mockReturnValue(defaultChatState);
    mockUseLLMState.mockReturnValue(defaultLLMState);
    mockUseInputActions.mockReturnValue({ setInput: mockSetInput, clearInput: jest.fn() });
    mockUseResetActions.mockReturnValue({ clearMessages: mockClearMessages, reset: jest.fn() });
    mockUseChat.mockReturnValue({ handleSend: mockHandleSend, clearCache: mockClearCache });
  });

  it("should render welcome message when no messages exist", () => {
    render(<ChatInterface />);
    expect(screen.getByText("Start a conversation! Try asking:")).toBeInTheDocument();
  });

  it("should render messages", () => {
    mockUseChatState.mockReturnValue({
      ...defaultChatState,
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
    expect(mockSetInput).toHaveBeenCalledWith("test message");
  });

  it("should call handleSend on button click when canSend is true", () => {
    mockUseChatState.mockReturnValue({
      ...defaultChatState,
      input: "test",
      canSend: true,
    });

    render(<ChatInterface />);
    const sendButton = screen.getByText("Send");
    fireEvent.click(sendButton);
    expect(mockHandleSend).toHaveBeenCalled();
  });

  it("should show loading state", () => {
    mockUseChatState.mockReturnValue({
      ...defaultChatState,
      loading: true,
    });

    render(<ChatInterface />);
    expect(screen.getByText("Thinking...")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Type your message...")).toBeDisabled();
  });

  it("should show error message", () => {
    mockUseChatState.mockReturnValue({
      ...defaultChatState,
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
    expect(mockClearCache).toHaveBeenCalled();

    confirmSpy.mockRestore();
  });

  it("should not call clearCache if confirmation is cancelled", async () => {
    const confirmSpy = jest.spyOn(window, "confirm").mockReturnValue(false);

    render(<ChatInterface />);
    const clearButton = screen.getByText("Clear Models Cache");

    await act(async () => {
      fireEvent.click(clearButton);
    });

    expect(mockClearCache).not.toHaveBeenCalled();

    confirmSpy.mockRestore();
  });
});
