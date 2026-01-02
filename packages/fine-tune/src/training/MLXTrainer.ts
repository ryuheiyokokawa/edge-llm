/**
 * MLXTrainer - TypeScript orchestrator for MLX LoRA training
 */

import { spawn, type ChildProcess } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import { fileURLToPath } from "url";
import { EventEmitter } from "events";
import {
  DEFAULT_TRAINING_CONFIG,
  type MLXTrainingConfig,
  type MLXTrainingProgress,
  type TrainingResult,
} from "./TrainingConfig";

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Events emitted by MLXTrainer
 */
export interface MLXTrainerEvents {
  progress: (progress: MLXTrainingProgress) => void;
  log: (message: string) => void;
  error: (error: Error) => void;
  complete: (result: TrainingResult) => void;
}

/**
 * Python environment check result
 */
export interface PythonEnvCheck {
  pythonAvailable: boolean;
  pythonVersion?: string;
  mlxAvailable: boolean;
  mlxVersion?: string;
  mlxLmAvailable: boolean;
  mlxLmVersion?: string;
  error?: string;
}

/**
 * MLXTrainer orchestrates Python MLX training from TypeScript
 */
export class MLXTrainer extends EventEmitter {
  private config: MLXTrainingConfig;
  private process: ChildProcess | null = null;
  private pythonPath: string;
  private scriptDir: string;

  constructor(config: Partial<MLXTrainingConfig> & Pick<MLXTrainingConfig, "datasetPath" | "outputPath">) {
    super();
    
    this.config = {
      ...DEFAULT_TRAINING_CONFIG,
      ...config,
    };
    
    // Default python path
    this.pythonPath = process.env.PYTHON_PATH ?? "python3";
    
    // Script directory - tsup bundles everything into dist/
    // From dist/ go up one level to package root, then into python/
    this.scriptDir = path.join(__dirname, "..", "python");
  }

  /**
   * Check Python environment for required packages
   */
  async checkEnvironment(): Promise<PythonEnvCheck> {
    const result: PythonEnvCheck = {
      pythonAvailable: false,
      mlxAvailable: false,
      mlxLmAvailable: false,
    };

    try {
      // Check Python version
      const pythonVersion = await this.runCommand(this.pythonPath, ["--version"]);
      result.pythonAvailable = true;
      result.pythonVersion = pythonVersion.trim().replace("Python ", "");

      // Check MLX
      try {
        const mlxCheck = await this.runCommand(this.pythonPath, [
          "-c",
          "import mlx.core as mx; print('available')",
        ]);
        result.mlxAvailable = mlxCheck.includes("available");
        result.mlxVersion = "available";
      } catch {
        result.mlxAvailable = false;
      }

      // Check MLX-LM
      try {
        const mlxLmCheck = await this.runCommand(this.pythonPath, [
          "-c",
          "import mlx_lm; print('available')",
        ]);
        result.mlxLmAvailable = mlxLmCheck.includes("available");
        result.mlxLmVersion = "available";
      } catch {
        result.mlxLmAvailable = false;
      }
    } catch (error) {
      result.error = error instanceof Error ? error.message : "Unknown error";
    }

    return result;
  }

  /**
   * Run a command and return stdout
   */
  private runCommand(command: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const proc = spawn(command, args, {
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      proc.stdout?.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr?.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(stderr || `Process exited with code ${code}`));
        }
      });

      proc.on("error", reject);
    });
  }

  /**
   * Install required Python packages
   */
  async installDependencies(): Promise<void> {
    const requirementsPath = path.join(this.scriptDir, "requirements.txt");
    
    this.emit("log", "Installing Python dependencies...");
    
    try {
      await this.runCommand(this.pythonPath, [
        "-m",
        "pip",
        "install",
        "-r",
        requirementsPath,
        "--quiet",
      ]);
      this.emit("log", "Dependencies installed successfully");
    } catch (error) {
      throw new Error(`Failed to install dependencies: ${error}`);
    }
  }

  /**
   * Validate training data directory
   */
  async validateDataset(): Promise<{ trainCount: number; validCount: number }> {
    const trainPath = path.join(this.config.datasetPath, "train.jsonl");
    const validPath = path.join(this.config.datasetPath, "valid.jsonl");

    try {
      await fs.access(trainPath);
      await fs.access(validPath);
    } catch {
      throw new Error(
        `Dataset validation failed. Ensure train.jsonl and valid.jsonl exist in ${this.config.datasetPath}`
      );
    }

    const trainContent = await fs.readFile(trainPath, "utf-8");
    const validContent = await fs.readFile(validPath, "utf-8");

    const trainCount = trainContent.trim().split("\n").length;
    const validCount = validContent.trim().split("\n").length;

    return { trainCount, validCount };
  }

  /**
   * Build command line arguments for training script
   */
  private buildArgs(): string[] {
    const args = [
      path.join(this.scriptDir, "train_lora.py"),
      "--model", this.config.baseModel,
      "--data", this.config.datasetPath,
      "--output", this.config.outputPath,
      "--epochs", String(this.config.epochs),
      "--batch-size", String(this.config.batchSize),
      "--learning-rate", String(this.config.learningRate),
      "--lora-rank", String(this.config.loraRank),
      "--lora-alpha", String(this.config.loraAlpha),
    ];

    if (this.config.maxTokens) {
      args.push("--max-tokens", String(this.config.maxTokens));
    }

    if (this.config.gradientCheckpointing) {
      args.push("--grad-checkpoint");
    }

    if (this.config.seed) {
      args.push("--seed", String(this.config.seed));
    }

    if (this.config.loraLayers) {
      args.push("--lora-layers", String(this.config.loraLayers));
    }

    args.push("--verbose");

    return args;
  }

  /**
   * Parse training output for progress information
   */
  private parseOutput(line: string): MLXTrainingProgress | null {
    // Parse progress patterns from MLX output
    // Example: "Epoch 1/3, Step 100/500, Loss: 0.5432"
    
    const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/i);
    const stepMatch = line.match(/Step\s+(\d+)\/(\d+)/i);
    const lossMatch = line.match(/Loss:\s*([\d.]+)/i);
    
    if (epochMatch || stepMatch || lossMatch) {
      return {
        epoch: epochMatch ? parseInt(epochMatch[1]!) : 0,
        totalEpochs: epochMatch ? parseInt(epochMatch[2]!) : this.config.epochs!,
        step: stepMatch ? parseInt(stepMatch[1]!) : 0,
        totalSteps: stepMatch ? parseInt(stepMatch[2]!) : 0,
        loss: lossMatch ? parseFloat(lossMatch[1]!) : 0,
        learningRate: this.config.learningRate!,
        status: "training",
      };
    }

    // Check for status messages
    if (line.includes("Loading model")) {
      return {
        epoch: 0,
        totalEpochs: this.config.epochs!,
        step: 0,
        totalSteps: 0,
        loss: 0,
        learningRate: this.config.learningRate!,
        status: "loading",
        message: "Loading model...",
      };
    }

    if (line.includes("Training complete")) {
      return {
        epoch: this.config.epochs!,
        totalEpochs: this.config.epochs!,
        step: 0,
        totalSteps: 0,
        loss: 0,
        learningRate: this.config.learningRate!,
        status: "complete",
        message: "Training complete",
      };
    }

    return null;
  }

  /**
   * Start LoRA training
   */
  async train(): Promise<TrainingResult> {
    const startTime = Date.now();

    // Validate dataset first
    this.emit("log", "Validating dataset...");
    const { trainCount, validCount } = await this.validateDataset();
    this.emit("log", `Dataset: ${trainCount} train, ${validCount} valid examples`);

    // Create output directory
    await fs.mkdir(this.config.outputPath, { recursive: true });

    // Build and run training command
    const args = this.buildArgs();
    this.emit("log", `Starting training with command: ${this.pythonPath} ${args.join(" ")}`);

    return new Promise((resolve) => {
      this.process = spawn(this.pythonPath, args, {
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1", // Ensure real-time output
        },
      });

      let lastProgress: MLXTrainingProgress | null = null;

      this.process.stdout?.on("data", (data) => {
        const lines = data.toString().split("\n");
        for (const line of lines) {
          if (line.trim()) {
            this.emit("log", line);
            
            const progress = this.parseOutput(line);
            if (progress) {
              lastProgress = progress;
              this.emit("progress", progress);
            }
          }
        }
      });

      this.process.stderr?.on("data", (data) => {
        const message = data.toString();
        this.emit("log", `[stderr] ${message}`);
      });

      this.process.on("close", (code) => {
        const trainingTime = Date.now() - startTime;

        if (code === 0) {
          const result: TrainingResult = {
            success: true,
            adapterPath: path.join(this.config.outputPath, "adapters"),
            configPath: path.join(this.config.outputPath, "training_config.json"),
            trainingTime,
            finalLoss: lastProgress?.loss,
          };
          
          this.emit("complete", result);
          resolve(result);
        } else {
          const result: TrainingResult = {
            success: false,
            error: `Training process exited with code ${code}`,
            trainingTime,
          };
          
          this.emit("complete", result);
          resolve(result);
        }

        this.process = null;
      });

      this.process.on("error", (error) => {
        const result: TrainingResult = {
          success: false,
          error: error.message,
          trainingTime: Date.now() - startTime,
        };
        
        this.emit("error", error);
        this.emit("complete", result);
        resolve(result);
        
        this.process = null;
      });
    });
  }

  /**
   * Abort training
   */
  abort(): void {
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
      this.emit("log", "Training aborted");
    }
  }

  /**
   * Merge trained adapters into base model
   */
  async mergeAdapters(adapterPath?: string): Promise<{ success: boolean; outputPath?: string; error?: string }> {
    const adapters = adapterPath ?? path.join(this.config.outputPath, "adapters");
    const mergedOutput = path.join(this.config.outputPath, "merged");

    this.emit("log", "Merging adapters into base model...");

    try {
      await this.runCommand(this.pythonPath, [
        path.join(this.scriptDir, "merge_adapters.py"),
        "--model", this.config.baseModel,
        "--adapters", adapters,
        "--output", mergedOutput,
      ]);

      this.emit("log", `Merged model saved to: ${mergedOutput}`);
      
      return { success: true, outputPath: mergedOutput };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      return { success: false, error: errorMessage };
    }
  }
}
