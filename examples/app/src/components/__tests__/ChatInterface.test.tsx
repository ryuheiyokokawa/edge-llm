import { render, screen, fireEvent } from "@testing-library/react";
import { ChatInterface } from "../ChatInterface";
import * as useChatHook from "../../hooks/useChat";

// Mock useChat
const mockHandleSend = jest.fn();
const mockSetInput = jest.fn();

jest.mock("../../hooks/useChat", () => ({
  useChat: jest.fn(),
}));

describe("ChatInterface", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (useChatHook.useChat as jest.Mock).mockReturnValue({
      messages: [],
      input: "",
      setInput: mockSetInput,
      handleSend: mockHandleSend,
      loading: false,
      error: null,
      status: "ready",
      initialized: true,
    });
  });

  it("should render status and empty state", () => {
    render(<ChatInterface />);
    expect(screen.getByText(/Status:/)).toBeInTheDocument();
    expect(screen.getByText(/Start a conversation!/)).toBeInTheDocument();
  });

  it("should render messages", () => {
    (useChatHook.useChat as jest.Mock).mockReturnValue({
      ...useChatHook.useChat(),
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ],
    });

    render(<ChatInterface />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("should handle input change", () => {
    render(<ChatInterface />);
    const input = screen.getByPlaceholderText(/Type your message/i);
    fireEvent.change(input, { target: { value: "New message" } });
    expect(mockSetInput).toHaveBeenCalledWith("New message");
  });

  it("should call handleSend on button click", () => {
    (useChatHook.useChat as jest.Mock).mockReturnValue({
      ...useChatHook.useChat(),
      input: "Hello",
    });

    render(<ChatInterface />);
    const button = screen.getByRole("button", { name: /Send/i });
    fireEvent.click(button);
    expect(mockHandleSend).toHaveBeenCalled();
  });
});
