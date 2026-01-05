import { useChat } from "../hooks/useChat";
import { exampleTools } from "../utils/tools";
import {
  useChatState,
  useLLMState,
  useInputActions,
  useResetActions,
} from "../store";

export function ChatInterface() {
  // State from composite hooks
  const { messages, input, loading, error, canSend } = useChatState();
  const { llmStatus, llmInitialized } = useLLMState();

  // Actions from composite hooks
  const { setInput } = useInputActions();
  const { clearMessages } = useResetActions();

  // Async handlers from useChat hook
  const { handleSend, clearCache } = useChat();

  return (
    <div style={styles.container}>
      <h1>Edge LLM Example App</h1>

      <div style={styles.status}>
        <strong>Status:</strong> {llmStatus} {llmInitialized ? "✓" : "..."}
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button onClick={clearMessages} style={styles.secondaryButton}>
            Clear Chat
          </button>
          <button
            onClick={async () => {
              if (
                confirm(
                  "Clear all model caches? This will require re-downloading model files."
                )
              ) {
                try {
                  await clearCache();
                  alert("Caches cleared. The page will now reload.");
                  window.location.reload();
                } catch (e) {
                  alert("Failed to clear cache: " + e);
                }
              }
            }}
            style={styles.clearButton}
          >
            Clear Models Cache
          </button>
        </div>
      </div>

      {error && (
        <div style={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <div style={styles.chatWindow}>
        {messages.length === 0 ? (
          <div style={styles.emptyState}>
            Start a conversation! Try asking:
            <ul>
              <li>"What is 15 * 23?"</li>
              <li>"What time is it?"</li>
              <li>"Search for React hooks"</li>
            </ul>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                ...styles.message,
                background: msg.role === "user" ? "#e3f2fd" : "#fff",
                borderLeft: `3px solid ${
                  msg.role === "user" ? "#2196f3" : "#4caf50"
                }`,
              }}
            >
              <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong>{" "}
              {msg.content}
            </div>
          ))
        )}
        {loading && <div style={styles.emptyState}>Thinking...</div>}
      </div>

      <div style={styles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your message..."
          disabled={!llmInitialized || loading}
          style={styles.input}
        />
        <button
          onClick={handleSend}
          disabled={!canSend}
          style={{
            ...styles.button,
            background: canSend ? "#2196f3" : "#ccc",
            cursor: canSend ? "pointer" : "not-allowed",
          }}
        >
          Send
        </button>
      </div>

      <div style={styles.toolsContainer}>
        <h3>Available Tools:</h3>
        <ul>
          {exampleTools.map((tool) => (
            <li key={tool.name}>
              <strong>{tool.name}</strong>: {tool.description}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    padding: "2rem",
  },
  status: {
    marginBottom: "1rem",
    padding: "0.5rem",
    background: "#f0f0f0",
    borderRadius: "4px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  secondaryButton: {
    padding: "0.25rem 0.5rem",
    fontSize: "0.8rem",
    background: "#666",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  },
  clearButton: {
    padding: "0.25rem 0.5rem",
    fontSize: "0.8rem",
    background: "#ff5252",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  },
  error: {
    marginBottom: "1rem",
    padding: "0.5rem",
    background: "#ffebee",
    color: "#c62828",
    borderRadius: "4px",
  },
  chatWindow: {
    border: "1px solid #ddd",
    borderRadius: "4px",
    padding: "1rem",
    marginBottom: "1rem",
    minHeight: "400px",
    maxHeight: "600px",
    overflowY: "auto",
    background: "#fafafa",
  },
  emptyState: {
    color: "#666",
    fontStyle: "italic",
  },
  message: {
    marginBottom: "1rem",
    padding: "0.5rem",
    borderRadius: "4px",
  },
  inputContainer: {
    display: "flex",
    gap: "0.5rem",
  },
  input: {
    flex: 1,
    padding: "0.5rem",
    border: "1px solid #ddd",
    borderRadius: "4px",
    fontSize: "1rem",
  },
  button: {
    padding: "0.5rem 1rem",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontSize: "1rem",
  },
  toolsContainer: {
    marginTop: "2rem",
    padding: "1rem",
    background: "#f5f5f5",
    borderRadius: "4px",
  },
} as const;
