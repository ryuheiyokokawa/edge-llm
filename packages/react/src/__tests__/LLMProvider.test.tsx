/**
 * Unit tests for LLMProvider component
 */
import { render, screen, waitFor, act } from "@testing-library/react";
import { LLMProvider, resetGlobalClient } from "../LLMProvider";
import { useLLMContext } from "../context";

// Mock @edge-llm/core
const mockInitialize = jest.fn();
const mockGetStatus = jest.fn();
const mockGetStatusWithDetails = jest.fn();
const mockChat = jest.fn();
const mockClearCache = jest.fn();

jest.mock("@edge-llm/core", () => ({
  LLMClient: jest.fn().mockImplementation(() => ({
    initialize: mockInitialize,
    getStatus: mockGetStatus,
    getStatusWithDetails: mockGetStatusWithDetails,
    chat: mockChat,
    clearCache: mockClearCache,
  })),
}));

// Test component that exposes context values
function TestConsumer() {
  const { status, initialized, client } = useLLMContext();
  return (
    <div>
      <span data-testid="status">{status}</span>
      <span data-testid="initialized">{initialized ? "yes" : "no"}</span>
      <span data-testid="hasClient">{client ? "yes" : "no"}</span>
    </div>
  );
}

describe("LLMProvider", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    resetGlobalClient();
    
    // Default successful mock behavior
    mockInitialize.mockResolvedValue(undefined);
    mockGetStatus.mockResolvedValue("ready");
    mockGetStatusWithDetails.mockResolvedValue({ 
      type: "STATUS_RESPONSE", 
      status: "ready", 
      activeRuntime: "transformers" 
    });
    mockClearCache.mockResolvedValue(undefined);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("should render children", () => {
    render(
      <LLMProvider>
        <div data-testid="child">Hello</div>
      </LLMProvider>
    );

    expect(screen.getByTestId("child")).toHaveTextContent("Hello");
  });

  it("should transition status during initialization", () => {
    render(
      <LLMProvider>
        <TestConsumer />
      </LLMProvider>
    );

    // Status will be idle or initializing depending on when the effect runs
    const status = screen.getByTestId("status").textContent;
    expect(["idle", "initializing"]).toContain(status);
  });

  it("should transition to ready on successful init", async () => {
    render(
      <LLMProvider>
        <TestConsumer />
      </LLMProvider>
    );

    // Run the debounce timer
    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    // Wait for initialization to complete
    await waitFor(() => {
      expect(mockInitialize).toHaveBeenCalled();
    });

    // After init completes
    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("ready");
    });

    await waitFor(() => {
      expect(screen.getByTestId("initialized")).toHaveTextContent("yes");
    });
  });

  it("should set error status on initialization failure", async () => {
    mockInitialize.mockRejectedValue(new Error("Init failed"));

    // Suppress console.error for expected error
    const consoleSpy = jest.spyOn(console, "error").mockImplementation(() => {});

    render(
      <LLMProvider>
        <TestConsumer />
      </LLMProvider>
    );

    // Run the debounce timer
    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(screen.getByTestId("status")).toHaveTextContent("error");
    });

    expect(screen.getByTestId("initialized")).toHaveTextContent("no");

    consoleSpy.mockRestore();
  });

  it("should pass config to client.initialize", async () => {
    const config = { preferredRuntime: "transformers" as const, debug: true };

    render(
      <LLMProvider config={config}>
        <TestConsumer />
      </LLMProvider>
    );

    // Run the debounce timer
    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    await waitFor(() => {
      expect(mockInitialize).toHaveBeenCalledWith(config);
    });
  });

  it("should provide client to children after setup", async () => {
    render(
      <LLMProvider>
        <TestConsumer />
      </LLMProvider>
    );

    // Wait for async setup
    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    // After useEffect runs, client should be set
    await waitFor(() => {
      expect(screen.getByTestId("hasClient")).toHaveTextContent("yes");
    });
  });

  it("should poll status after initialization", async () => {
    render(
      <LLMProvider>
        <TestConsumer />
      </LLMProvider>
    );

    await act(async () => {
      jest.advanceTimersByTime(300);
    });

    // Wait for client to be created
    await waitFor(() => {
      expect(screen.getByTestId("hasClient")).toHaveTextContent("yes");
    });

    // Clear the initial getStatusWithDetails calls
    mockGetStatusWithDetails.mockClear();

    // Advance timers to trigger status polling
    await act(async () => {
      jest.advanceTimersByTime(1000);
    });

    expect(mockGetStatusWithDetails).toHaveBeenCalled();
  });
});
