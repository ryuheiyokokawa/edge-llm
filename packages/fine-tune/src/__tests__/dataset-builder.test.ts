/**
 * Tests for DatasetBuilder
 */

import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import { DatasetBuilder } from "../dataset/DatasetBuilder";
import type { ToolSchema, TrainingExample } from "../types";

describe("DatasetBuilder", () => {
  const sampleTools: ToolSchema[] = [
    {
      name: "get_weather",
      description: "Get the current weather for a location",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "City name" },
        },
        required: ["location"],
      },
    },
    {
      name: "calculate",
      description: "Evaluate a math expression",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression" },
        },
        required: ["expression"],
      },
    },
  ];

  let tempDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "dataset-builder-test-"));
  });

  afterEach(async () => {
    try {
      await fs.rm(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe("constructor", () => {
    it("should create a DatasetBuilder with tools", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      expect(builder.getTools()).toHaveLength(2);
      expect(builder.getExampleCount()).toBe(0);
    });

    it("should use default split ratio", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      // Default is 80/10/10
      expect(builder).toBeDefined();
    });

    it("should validate split ratio sums to 1.0", () => {
      expect(() => {
        new DatasetBuilder({
          tools: sampleTools as any,
          outputDir: tempDir,
          splitRatio: { train: 0.5, valid: 0.3, test: 0.1 }, // Sums to 0.9
        });
      }).toThrow("Split ratio must sum to 1.0");
    });

    it("should accept valid custom split ratio", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
        splitRatio: { train: 0.7, valid: 0.2, test: 0.1 },
      });

      expect(builder).toBeDefined();
    });
  });

  describe("addExample", () => {
    it("should add a valid example", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      const example: TrainingExample = {
        userQuery: "What's the weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
      };

      builder.addExample(example);

      expect(builder.getExampleCount()).toBe(1);
    });

    it("should throw for unknown tool reference", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      const example: TrainingExample = {
        userQuery: "Send an email",
        expectedToolCalls: [
          { name: "send_email", arguments: { to: "test@example.com" } },
        ],
      };

      expect(() => builder.addExample(example)).toThrow(
        'Training example references unknown tool: "send_email"'
      );
    });

    it("should support chaining", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      builder
        .addExample({
          userQuery: "Weather in Tokyo?",
          expectedToolCalls: [
            { name: "get_weather", arguments: { location: "Tokyo" } },
          ],
        })
        .addExample({
          userQuery: "Calculate 2+2",
          expectedToolCalls: [
            { name: "calculate", arguments: { expression: "2+2" } },
          ],
        });

      expect(builder.getExampleCount()).toBe(2);
    });
  });

  describe("addExamples", () => {
    it("should add multiple examples", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      const examples: TrainingExample[] = [
        {
          userQuery: "Weather in Tokyo?",
          expectedToolCalls: [
            { name: "get_weather", arguments: { location: "Tokyo" } },
          ],
        },
        {
          userQuery: "Weather in NYC?",
          expectedToolCalls: [
            { name: "get_weather", arguments: { location: "New York" } },
          ],
        },
      ];

      builder.addExamples(examples);

      expect(builder.getExampleCount()).toBe(2);
    });
  });

  describe("preview", () => {
    it("should return formatted preview of examples", () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      builder.addExample({
        userQuery: "Weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
      });

      const preview = builder.preview(1);

      expect(preview).toHaveLength(1);
      expect(preview[0]!.text).toContain("Tokyo");
      expect(preview[0]!.text).toContain("get_weather");
    });
  });

  describe("build", () => {
    it("should throw if no examples added", async () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      await expect(builder.build()).rejects.toThrow(
        "No training examples added"
      );
    });

    it("should create train/valid/test JSONL files", async () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      // Add 10 examples to get a meaningful split
      for (let i = 0; i < 10; i++) {
        builder.addExample({
          userQuery: `Weather in City${i}?`,
          expectedToolCalls: [
            { name: "get_weather", arguments: { location: `City${i}` } },
          ],
        });
      }

      const result = await builder.build();

      // Check files exist
      expect(result.trainPath).toContain("train.jsonl");
      expect(result.validPath).toContain("valid.jsonl");
      expect(result.testPath).toContain("test.jsonl");

      // Check files are created
      const trainExists = await fs
        .access(result.trainPath)
        .then(() => true)
        .catch(() => false);
      expect(trainExists).toBe(true);

      // Check counts (with default 80/10/10 split)
      expect(result.counts.train).toBe(8);
      expect(result.counts.valid).toBe(1);
      expect(result.counts.test).toBe(1);
    });

    it("should create directory if it doesn't exist", async () => {
      const nestedDir = path.join(tempDir, "nested", "output");
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: nestedDir,
      });

      builder.addExample({
        userQuery: "Weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
      });

      await builder.build();

      const dirExists = await fs
        .access(nestedDir)
        .then(() => true)
        .catch(() => false);
      expect(dirExists).toBe(true);
    });

    it("should write valid JSONL format", async () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
        splitRatio: { train: 1.0, valid: 0, test: 0 },
      });

      builder.addExample({
        userQuery: "Weather in Tokyo?",
        expectedToolCalls: [
          { name: "get_weather", arguments: { location: "Tokyo" } },
        ],
      });

      const result = await builder.build();
      const content = await fs.readFile(result.trainPath, "utf-8");
      const lines = content.trim().split("\n");

      expect(lines).toHaveLength(1);

      // Each line should be valid JSON
      const parsed = JSON.parse(lines[0]!);
      expect(parsed.text).toBeDefined();
      expect(parsed.text).toContain("Tokyo");
    });
  });

  describe("saveToolSchemas", () => {
    it("should save tool schemas to JSON file", async () => {
      const builder = new DatasetBuilder({
        tools: sampleTools as any,
        outputDir: tempDir,
      });

      const outputPath = await builder.saveToolSchemas();

      const content = await fs.readFile(outputPath, "utf-8");
      const tools = JSON.parse(content);

      expect(tools).toHaveLength(2);
      expect(tools[0].name).toBe("get_weather");
      expect(tools[1].name).toBe("calculate");
    });
  });
});
