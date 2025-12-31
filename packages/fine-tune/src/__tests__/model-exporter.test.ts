/**
 * Tests for ModelExporter
 */

import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import { ModelExporter } from "../export/ModelExporter";

// Mock child_process
jest.mock("child_process", () => ({
  spawn: jest.fn(),
}));

import { spawn } from "child_process";
import { EventEmitter } from "events";

const mockSpawn = spawn as jest.MockedFunction<typeof spawn>;

describe("ModelExporter", () => {
  let tempDir: string;
  let exporter: ModelExporter;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "model-exporter-test-"));
    exporter = new ModelExporter();
    mockSpawn.mockReset();
  });

  afterEach(async () => {
    try {
      await fs.rm(tempDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  });

  describe("constructor", () => {
    it("should create an exporter instance", () => {
      const exp = new ModelExporter();
      expect(exp).toBeDefined();
    });
  });

  describe("checkGGUFAvailability", () => {
    it("should return available when llama-cpp-python is installed", async () => {
      const mockProcess = new EventEmitter() as any;
      mockProcess.stdout = new EventEmitter();
      mockProcess.stderr = new EventEmitter();

      mockSpawn.mockReturnValue(mockProcess);

      const resultPromise = exporter.checkGGUFAvailability();

      setImmediate(() => {
        mockProcess.stdout.emit("data", "available");
        mockProcess.emit("close", 0);
      });

      const result = await resultPromise;

      expect(result.available).toBe(true);
    });

    it("should return unavailable with error message when not installed", async () => {
      const mockProcess = new EventEmitter() as any;
      mockProcess.stdout = new EventEmitter();
      mockProcess.stderr = new EventEmitter();

      mockSpawn.mockReturnValue(mockProcess);

      const resultPromise = exporter.checkGGUFAvailability();

      setImmediate(() => {
        mockProcess.stderr.emit("data", "ModuleNotFoundError");
        mockProcess.emit("close", 1);
      });

      const result = await resultPromise;

      expect(result.available).toBe(false);
      expect(result.error).toContain("llama-cpp-python");
    });
  });

  describe("exportSafetensors", () => {
    it("should copy safetensors files to output directory", async () => {
      // Create mock adapter directory with safetensors files
      const adapterDir = path.join(tempDir, "adapters");
      await fs.mkdir(adapterDir, { recursive: true });
      await fs.writeFile(
        path.join(adapterDir, "adapter.safetensors"),
        "mock safetensors content"
      );
      await fs.writeFile(
        path.join(adapterDir, "config.json"),
        '{"test": true}'
      );

      const outputDir = path.join(tempDir, "output");

      const result = await exporter.exportSafetensors(adapterDir, outputDir);

      expect(result.success).toBe(true);
      expect(result.format).toBe("safetensors");
      expect(result.outputPath).toBe(outputDir);

      // Verify files were copied
      const files = await fs.readdir(outputDir);
      expect(files).toContain("adapter.safetensors");
      expect(files).toContain("config.json");
    });

    it("should fail when no safetensors files exist", async () => {
      const adapterDir = path.join(tempDir, "empty-adapters");
      await fs.mkdir(adapterDir, { recursive: true });
      await fs.writeFile(path.join(adapterDir, "config.json"), "{}");

      const outputDir = path.join(tempDir, "output");

      const result = await exporter.exportSafetensors(adapterDir, outputDir);

      expect(result.success).toBe(false);
      expect(result.error).toContain("No safetensors files found");
    });

    it("should call progress callback", async () => {
      const adapterDir = path.join(tempDir, "adapters");
      await fs.mkdir(adapterDir, { recursive: true });
      await fs.writeFile(
        path.join(adapterDir, "model.safetensors"),
        "content"
      );

      const outputDir = path.join(tempDir, "output");
      const progressCalls: any[] = [];

      await exporter.exportSafetensors(adapterDir, outputDir, (status) => {
        progressCalls.push(status);
      });

      expect(progressCalls.some((p) => p.phase === "preparing")).toBe(true);
      expect(progressCalls.some((p) => p.phase === "complete")).toBe(true);
    });
  });

  describe("exportGGUF", () => {
    it("should fail when GGUF tools are not available", async () => {
      // Mock llama-cpp-python not installed
      const mockProcess = new EventEmitter() as any;
      mockProcess.stdout = new EventEmitter();
      mockProcess.stderr = new EventEmitter();

      mockSpawn.mockReturnValue(mockProcess);

      const resultPromise = exporter.exportGGUF(
        path.join(tempDir, "model"),
        path.join(tempDir, "output.gguf")
      );

      setImmediate(() => {
        mockProcess.stderr.emit("data", "ModuleNotFoundError");
        mockProcess.emit("close", 1);
      });

      const result = await resultPromise;

      expect(result.success).toBe(false);
      expect(result.format).toBe("gguf");
      expect(result.error).toContain("llama-cpp-python");
    });
  });

  describe("getModelInfo", () => {
    it("should return info for a file", async () => {
      const modelFile = path.join(tempDir, "model.gguf");
      await fs.writeFile(modelFile, "mock model content");

      const info = await exporter.getModelInfo(modelFile);

      expect(info.exists).toBe(true);
      expect(info.format).toBe("gguf");
      expect(info.size).toBeGreaterThan(0);
    });

    it("should return info for a directory with safetensors", async () => {
      const modelDir = path.join(tempDir, "model");
      await fs.mkdir(modelDir, { recursive: true });
      await fs.writeFile(
        path.join(modelDir, "model.safetensors"),
        "content"
      );
      await fs.writeFile(
        path.join(modelDir, "config.json"),
        "{}"
      );

      const info = await exporter.getModelInfo(modelDir);

      expect(info.exists).toBe(true);
      expect(info.format).toBe("safetensors");
      expect(info.files).toContain("model.safetensors");
      expect(info.files).toContain("config.json");
    });

    it("should return exists: false for non-existent path", async () => {
      const info = await exporter.getModelInfo(
        path.join(tempDir, "nonexistent")
      );

      expect(info.exists).toBe(false);
    });
  });

  describe("export", () => {
    it("should export to multiple formats", async () => {
      // Create mock adapter directory
      const adapterDir = path.join(tempDir, "adapters");
      await fs.mkdir(adapterDir, { recursive: true });
      await fs.writeFile(
        path.join(adapterDir, "adapter.safetensors"),
        "content"
      );

      const outputDir = path.join(tempDir, "output");

      const results = await exporter.export({
        adapterPath: adapterDir,
        baseModel: "test-model",
        outputFormats: ["safetensors"],
        outputDir,
      });

      expect(results.size).toBe(1);
      expect(results.get("safetensors")?.success).toBe(true);
    });

    it("should handle ONNX format as not implemented", async () => {
      const adapterDir = path.join(tempDir, "adapters");
      await fs.mkdir(adapterDir, { recursive: true });

      const results = await exporter.export({
        adapterPath: adapterDir,
        baseModel: "test-model",
        outputFormats: ["onnx"],
        outputDir: path.join(tempDir, "output"),
      });

      expect(results.get("onnx")?.success).toBe(false);
      expect(results.get("onnx")?.error).toContain("not yet implemented");
    });
  });
});
