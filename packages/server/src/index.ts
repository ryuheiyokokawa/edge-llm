import express from "express";
import cors from "cors";
import ollama from "ollama";

const app = express();
const port = process.env.PORT || 3001;
const DEFAULT_MODEL = "llama3.2";

app.use(cors());
app.use(express.json());

// Main chat completions endpoint
app.post("/v1/chat/completions", async (req: express.Request, res: express.Response) => {
  const { messages, tools, stream, model } = req.body;

  try {
    const modelToUse = model || DEFAULT_MODEL;
    console.log(`[Bridge] Request for model: ${modelToUse}`);
    
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const response = await ollama.chat({
        model: modelToUse,
        messages,
        tools,
        stream: true,
      });

      for await (const part of response) {
        const chunk = {
          choices: [
            {
              delta: {
                content: part.message.content,
              },
            },
          ],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const response = await ollama.chat({
        model: modelToUse,
        messages,
        tools,
        stream: false,
      });

      // Format to OpenAI-compatible structure
      const formattedResponse = {
        choices: [
          {
            message: {
              role: "assistant",
              content: response.message.content,
              tool_calls: response.message.tool_calls?.map((tc: any, index: number) => ({
                id: `call_${Date.now()}_${index}`,
                type: "function",
                function: {
                  name: tc.function.name,
                  arguments: JSON.stringify(tc.function.arguments),
                },
              })),
            },
          },
        ],
      };

      res.json(formattedResponse);
    }
  } catch (error: any) {
    console.error("[Bridge] Error:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`[Bridge] Server running at http://localhost:${port}`);
  console.log(`[Bridge] API Endpoint: http://localhost:${port}/v1/chat/completions`);
});
