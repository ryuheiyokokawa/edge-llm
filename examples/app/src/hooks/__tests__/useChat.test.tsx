import { renderHook, act } from "@testing-library/react";
import { useChat } from "../useChat";

// Mock useLLM
const mockSend = jest.fn();
const mockRegisterTools = jest.fn();

jest.mock("@edge-llm/react", () => ({
  useLLM: () => ({
    send: mockSend,
    registerTools: mockRegisterTools,
    status: "ready",
    initialized: true,
  }),
}));

describe("useChat", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should initialize with default state", () => {
    const { result } = renderHook(() => useChat());

    expect(result.current.messages).toEqual([]);
    expect(result.current.input).toBe("");
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("should register tools on mount", () => {
    renderHook(() => useChat());
    expect(mockRegisterTools).toHaveBeenCalled();
  });

  it("should handle sending a message", async () => {
    mockSend.mockResolvedValueOnce({ type: "content", text: "Hello there!" });
    
    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.setInput("Hello");
    });

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockSend).toHaveBeenCalled();
    expect(result.current.messages).toHaveLength(2); // User + Assistant
    expect(result.current.messages[0]).toEqual({ role: "user", content: "Hello" });
    expect(result.current.messages[1]).toEqual({ role: "assistant", content: "Hello there!" });
    expect(result.current.input).toBe("");
  });

  it("should handle errors during send", async () => {
    mockSend.mockRejectedValueOnce(new Error("Network error"));
    const consoleSpy = jest.spyOn(console, "error").mockImplementation(() => {});
    
    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.setInput("Hello");
    });

    await act(async () => {
      await result.current.handleSend();
    });

    expect(result.current.error).toBe("Network error");
    expect(result.current.loading).toBe(false);
    
    consoleSpy.mockRestore();
  });

  it("should handle a tool call loop", async () => {
    // 1st call returns tool call
    mockSend.mockResolvedValueOnce({
      type: "tool_calls",
      calls: [{ id: "call_1", name: "calculate", arguments: { expression: "2+2" } }],
    });
    // 2nd call returns final content
    mockSend.mockResolvedValueOnce({ type: "content", text: "The result is 4." });

    const { result } = renderHook(() => useChat());

    act(() => {
      result.current.setInput("Calculate 2+2");
    });

    await act(async () => {
      await result.current.handleSend();
    });

    // Check messages sequence
    // 1. User: Calculate 2+2
    // 2. Assistant: 🔧 Calling tool...
    // 3. Assistant: ✅ Tool calculate result...
    // 4. Assistant: The result is 4.
    expect(result.current.messages).toHaveLength(4);
    expect(result.current.messages[1]!.content).toContain("🔧 Calling tool: calculate");
    expect(result.current.messages[2]!.content).toContain("✅ Tool calculate result");
    expect(result.current.messages[3]!.content).toBe("The result is 4.");

    expect(mockSend).toHaveBeenCalledTimes(2);
  });
});
