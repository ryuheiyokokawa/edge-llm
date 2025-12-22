// Test setup for React testing
import "@testing-library/jest-dom";

// Mock matchMedia for components that might use it
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock alert and location
window.alert = jest.fn();
Object.defineProperty(window, 'location', {
  value: {
    reload: jest.fn(),
  },
  writable: true,
});

// Mock @edge-llm/core
jest.mock("@edge-llm/core", () => ({
  LLMClient: jest.fn().mockImplementation(() => ({
    initialize: jest.fn().mockResolvedValue(undefined),
    getStatus: jest.fn().mockResolvedValue("ready"),
    chat: jest.fn().mockResolvedValue({ type: "content", text: "Mock response" }),
    clearCache: jest.fn().mockResolvedValue(undefined),
  })),
}));

// Mock @edge-llm/react
jest.mock("@edge-llm/react", () => ({
  LLMProvider: ({ children }: any) => children,
  useLLM: () => ({
    send: jest.fn().mockResolvedValue({ type: "content", text: "Mock response" }),
    registerTool: jest.fn(),
    registerTools: jest.fn(),
    unregisterTool: jest.fn(),
    status: "ready",
    initialized: true,
    clearCache: jest.fn().mockResolvedValue(undefined),
  }),
  useLLMContext: () => ({
    client: {},
    status: "ready",
    config: {},
    initialized: true,
    clearCache: jest.fn().mockResolvedValue(undefined),
  }),
}));
