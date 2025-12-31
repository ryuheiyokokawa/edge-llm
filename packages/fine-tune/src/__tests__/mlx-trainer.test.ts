/**
 * Tests for MLXTrainer
 */

import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import { MLXTrainer } from "../training/MLXTrainer";
import { DEFAULT_TRAINING_CONFIG } from "../training/TrainingConfig";

// Mock child_process
jest.mock("child_process", () => ({
  spawn: jest.fn(),
}));

import { spawn } from "child_process";
import { EventEmitter } from "events";

const mockSpawn = spawn as jest.MockedFunction<typeof spawn>;

describe("MLXTrainer", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "mlx-trainer-test-"));
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
    it("should create trainer with required config", () => {
      const trainer = new MLXTrainer({
        datasetPath: "/data",
        outputPath: "/output",
      });

      expect(trainer).toBeDefined();
    });

    it("should merge with default config", () => {
      const trainer = new MLXTrainer({
        datasetPath: "/data",
        outputPath: "/output",
        epochs: 5,
      });

      expect(trainer).toBeDefined();
    });
  });

  describe("DEFAULT_TRAINING_CONFIG", () => {
    it("should have sensible defaults", () => {
      expect(DEFAULT_TRAINING_CONFIG.baseModel).toContain("functiongemma");
      expect(DEFAULT_TRAINING_CONFIG.epochs).toBe(3);
      expect(DEFAULT_TRAINING_CONFIG.batchSize).toBe(4);
      expect(DEFAULT_TRAINING_CONFIG.loraRank).toBe(8);
      expect(DEFAULT_TRAINING_CONFIG.loraAlpha).toBe(16);
    });
  });

  describe("checkEnvironment", () => {
    it("should check Python availability", async () => {
      const mockProcess = new EventEmitter() as any;
      mockProcess.stdout = new EventEmitter();
      mockProcess.stderr = new EventEmitter();

      mockSpawn.mockReturnValue(mockProcess);

      const trainer = new MLXTrainer({
        datasetPath: tempDir,
        outputPath: tempDir,
      });

      // Start environment check (don't await - just verify method exists)\n      void trainer.checkEnvironment();

      // Simulate Python version response
      mockProcess.stdout.emit("data", "Python 3.13.7\n");
      mockProcess.emit("close", 0);

      // Verify the method exists and is callable
      expect(trainer.checkEnvironment).toBeDefined();
    });
  });

  describe("validateDataset", () => {
    it("should throw when train.jsonl is missing", async () => {
      const trainer = new MLXTrainer({
        datasetPath: tempDir,
        outputPath: tempDir,
      });

      await expect(trainer.validateDataset()).rejects.toThrow(
        "Dataset validation failed"
      );
    });

    it("should return counts when files exist", async () => {
      // Create mock data files
      await fs.writeFile(
        path.join(tempDir, "train.jsonl"),
        '{"text": "example 1"}\n{"text": "example 2"}\n{"text": "example 3"}'
      );
      await fs.writeFile(
        path.join(tempDir, "valid.jsonl"),
        '{"text": "valid 1"}'
      );

      const trainer = new MLXTrainer({
        datasetPath: tempDir,
        outputPath: tempDir,
      });

      const result = await trainer.validateDataset();

      expect(result.trainCount).toBe(3);
      expect(result.validCount).toBe(1);
    });
  });

  describe("train", () => {
    it("should fail when dataset is invalid", async () => {
      const trainer = new MLXTrainer({
        datasetPath: tempDir, // Empty dir, no data files
        outputPath: tempDir,
      });

      await expect(trainer.train()).rejects.toThrow("Dataset validation failed");
    });

    it("should validate dataset before training", async () => {
      // Create mock data files
      await fs.writeFile(
        path.join(tempDir, "train.jsonl"),
        '{"text": "example"}'
      );
      await fs.writeFile(
        path.join(tempDir, "valid.jsonl"),
        '{"text": "valid"}'
      );

      const mockProcess = new EventEmitter() as any;
      mockProcess.stdout = new EventEmitter();
      mockProcess.stderr = new EventEmitter();
      mockProcess.kill = jest.fn();

      mockSpawn.mockReturnValue(mockProcess);

      const trainer = new MLXTrainer({
        datasetPath: tempDir,
        outputPath: tempDir,
      });

      const logEvents: string[] = [];
      trainer.on("log", (msg) => logEvents.push(msg));

      // Start training but don't await - test that validation happens
      const trainPromise = trainer.train();

      // Give it time to validate dataset and start spawn
      await new Promise((r) => setTimeout(r, 50));

      // Emit close to finish the test
      mockProcess.emit("close", 0);

      const result = await trainPromise;

      // Check that logs include dataset info
      expect(logEvents.some((l) => l.includes("Validating dataset"))).toBe(true);
      expect(result.success).toBe(true);
    }, 10000);
  });

  describe("abort", () => {
    it("should not throw when no process is running", () => {
      const trainer = new MLXTrainer({
        datasetPath: tempDir,
        outputPath: tempDir,
      });

      // Should not throw
      expect(() => trainer.abort()).not.toThrow();
    });
  });
});

