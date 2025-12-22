import { useChat } from "../hooks/useChat";
import { exampleTools } from "../utils/tools";

export function ChatInterface() {
  const {
    messages,
    input,
    setInput,
    handleSend,
    loading,
    error,
    status,
    initialized,
    clearCache,
  } = useChat();

  return (
    <div style={styles.container}>
      <h1>Edge LLM Example App</h1>

      <div style={styles.status}>
        <strong>Status:</strong> {status} {initialized ? "✓" : "..."}
        <button 
          onClick={async () => {
            if (confirm("Clear all model caches? This will require re-downloading model files.")) {
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
        {loading && (
          <div style={styles.emptyState}>Thinking...</div>
        )}
      </div>

      <div style={styles.inputContainer}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type your message..."
          disabled={!initialized || loading}
          style={styles.input}
        />
        <button
          onClick={handleSend}
          disabled={!initialized || loading || !input.trim()}
          style={{
            ...styles.button,
            background:
              initialized && !loading && input.trim() ? "#2196f3" : "#ccc",
            cursor:
              initialized && !loading && input.trim()
                ? "pointer"
                : "not-allowed",
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
