import { exampleTools } from "../tools";

describe("exampleTools", () => {
  const calculateTool = exampleTools.find((t) => t.name === "calculate")!;
  const timeTool = exampleTools.find((t) => t.name === "getCurrentTime")!;
  const searchTool = exampleTools.find((t) => t.name === "searchWeb")!;

  describe("calculate", () => {
    it("should evaluate valid expressions", async () => {
      const result = await calculateTool.handler({ expression: "2 + 2" });
      expect(result).toEqual({ result: 4, expression: "2 + 2" });
    });

    it("should handle invalid expressions", async () => {
      const result = await calculateTool.handler({ expression: "invalid code" });
      expect(result).toHaveProperty("error");
    });
  });

  describe("getCurrentTime", () => {
    it("should return time and timestamp", async () => {
      const result = await timeTool.handler({});
      expect(result).toHaveProperty("time");
      expect(result).toHaveProperty("timestamp");
    });
  });

  describe("searchWeb", () => {
    it("should return mock results", async () => {
      const result = await searchTool.handler({ query: "react" });
      expect(result).toEqual({
        query: "react",
        results: expect.arrayContaining([
          expect.objectContaining({ title: expect.stringContaining("react") }),
        ]),
      });
    });
  });
});
