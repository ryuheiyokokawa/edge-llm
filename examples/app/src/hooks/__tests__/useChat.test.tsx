import { renderHook, act } from "@testing-library/react";
import { useChat } from "../useChat";

// Mock useLLM
const mockSend = jest.fn();
const mockRegisterTools = jest.fn();
const mockClearCache = jest.fn();

jest.mock("@edge-llm/react", () => ({
  useLLM: () => ({
    send: mockSend,
    registerTools: mockRegisterTools,
    clearCache: mockClearCache,
    status: "ready",
    initialized: true,
  }),
}));

// Mock state values
const mockState = {
  input: "",
  loading: false,
  conversationHistory: [] as any[],
  canSend: false,
  messages: [],
  error: null,
};

// Mock action functions
const mockActions = {
  clearInput: jest.fn(),
  setLoading: jest.fn(),
  setError: jest.fn(),
  setInput: jest.fn(),
  addUserMessage: jest.fn(),
  addAssistantMessage: jest.fn(),
  addToolCallMessage: jest.fn(),
  addToolResultMessage: jest.fn(),
  appendToHistory: jest.fn(),
  appendMultipleToHistory: jest.fn(),
  syncLLMStatus: jest.fn(),
  clearMessages: jest.fn(),
  reset: jest.fn(),
};

jest.mock("../../store", () => ({
  useChatState: () => mockState,
  useChatActions: () => mockActions,
}));

describe("useChat", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset mock state
    mockState.input = "";
    mockState.loading = false;
    mockState.conversationHistory = [];
  });

  it("should return handleSend and clearCache", () => {
    const { result } = renderHook(() => useChat());

    expect(result.current.handleSend).toBeDefined();
    expect(result.current.clearCache).toBeDefined();
    expect(typeof result.current.handleSend).toBe("function");
  });

  it("should register tools on mount", () => {
    renderHook(() => useChat());
    expect(mockRegisterTools).toHaveBeenCalled();
  });

  it("should not send if input is empty", async () => {
    mockState.input = "";

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockSend).not.toHaveBeenCalled();
  });

  it("should not send if already loading", async () => {
    mockState.input = "test";
    mockState.loading = true;

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockSend).not.toHaveBeenCalled();
  });

  it("should send message when input is valid", async () => {
    mockState.input = "Hello";
    mockSend.mockResolvedValueOnce({ type: "content", text: "Hi there!" });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockActions.clearInput).toHaveBeenCalled();
    expect(mockActions.setLoading).toHaveBeenCalledWith(true);
    expect(mockActions.addUserMessage).toHaveBeenCalledWith("Hello");
    expect(mockSend).toHaveBeenCalled();
    expect(mockActions.addAssistantMessage).toHaveBeenCalledWith("Hi there!");
    expect(mockActions.setLoading).toHaveBeenCalledWith(false);
  });

  it("should handle errors during send", async () => {
    mockState.input = "Hello";
    mockSend.mockRejectedValueOnce(new Error("Network error"));
    const consoleSpy = jest.spyOn(console, "error").mockImplementation(() => {});

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockActions.setError).toHaveBeenCalledWith("Network error");
    expect(mockActions.setLoading).toHaveBeenCalledWith(false);

    consoleSpy.mockRestore();
  });

  it("should handle tool calls", async () => {
    mockState.input = "Calculate 2+2";

    // First call returns tool_calls
    mockSend.mockResolvedValueOnce({
      type: "tool_calls",
      calls: [{ id: "call_1", name: "calculate", arguments: { expression: "2+2" } }],
    });
    // Second call returns content
    mockSend.mockResolvedValueOnce({ type: "content", text: "The result is 4." });

    const { result } = renderHook(() => useChat());

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockActions.addToolCallMessage).toHaveBeenCalled();
    expect(mockActions.addToolResultMessage).toHaveBeenCalled();
    expect(mockSend).toHaveBeenCalledTimes(2);
    expect(mockActions.addAssistantMessage).toHaveBeenCalledWith("The result is 4.");
  });
});
