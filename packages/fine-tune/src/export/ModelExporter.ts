/**
 * ModelExporter - Export fine-tuned models to various formats
 */

import { spawn } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import type { ExportConfig, ExportFormat } from "../types";

/**
 * Result of an export operation
 */
export interface ExportResult {
  success: boolean;
  format: ExportFormat;
  outputPath?: string;
  fileSize?: number;
  error?: string;
}

/**
 * Progress callback for export operations
 */
export type ExportProgressCallback = (status: {
  format: ExportFormat;
  phase: "preparing" | "converting" | "saving" | "complete" | "error";
  message?: string;
}) => void;

/**
 * ModelExporter handles exporting fine-tuned models to deployment formats
 */
export class ModelExporter {
  private pythonPath: string;
  private scriptDir: string;

  constructor() {
    this.pythonPath = process.env.PYTHON_PATH ?? "python3";
    this.scriptDir = path.join(__dirname, "..", "..", "python");
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
   * Check if llama.cpp's convert script is available
   */
  async checkGGUFAvailability(): Promise<{
    available: boolean;
    error?: string;
  }> {
    try {
      // Check for llama-cpp-python which includes conversion tools
      await this.runCommand(this.pythonPath, [
        "-c",
        "import llama_cpp; print('available')",
      ]);
      return { available: true };
    } catch {
      return {
        available: false,
        error:
          "llama-cpp-python not installed. Install with: pip install llama-cpp-python",
      };
    }
  }

  /**
   * Export model to safetensors format
   * This is the native MLX format, so it's mostly a copy operation
   */
  async exportSafetensors(
    adapterPath: string,
    outputDir: string,
    onProgress?: ExportProgressCallback
  ): Promise<ExportResult> {
    onProgress?.({
      format: "safetensors",
      phase: "preparing",
      message: "Preparing safetensors export...",
    });

    try {
      // Create output directory
      await fs.mkdir(outputDir, { recursive: true });

      // Find and copy safetensors files
      const files = await fs.readdir(adapterPath);
      const safetensorsFiles = files.filter((f) => f.endsWith(".safetensors"));

      if (safetensorsFiles.length === 0) {
        return {
          success: false,
          format: "safetensors",
          error: "No safetensors files found in adapter path",
        };
      }

      onProgress?.({
        format: "safetensors",
        phase: "converting",
        message: `Copying ${safetensorsFiles.length} files...`,
      });

      let totalSize = 0;

      for (const file of safetensorsFiles) {
        const srcPath = path.join(adapterPath, file);
        const dstPath = path.join(outputDir, file);
        await fs.copyFile(srcPath, dstPath);
        
        const stats = await fs.stat(dstPath);
        totalSize += stats.size;
      }

      // Also copy config files
      const configFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"];
      for (const configFile of configFiles) {
        const srcPath = path.join(adapterPath, configFile);
        try {
          await fs.access(srcPath);
          await fs.copyFile(srcPath, path.join(outputDir, configFile));
        } catch {
          // Config file doesn't exist, skip
        }
      }

      onProgress?.({
        format: "safetensors",
        phase: "complete",
        message: "Safetensors export complete",
      });

      return {
        success: true,
        format: "safetensors",
        outputPath: outputDir,
        fileSize: totalSize,
      };
    } catch (error) {
      onProgress?.({
        format: "safetensors",
        phase: "error",
        message: error instanceof Error ? error.message : "Unknown error",
      });

      return {
        success: false,
        format: "safetensors",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Export model to GGUF format for llama.cpp compatibility
   */
  async exportGGUF(
    modelPath: string,
    outputPath: string,
    quantization: "4bit" | "8bit" | "none" = "4bit",
    onProgress?: ExportProgressCallback
  ): Promise<ExportResult> {
    onProgress?.({
      format: "gguf",
      phase: "preparing",
      message: "Checking GGUF conversion tools...",
    });

    // Check availability first
    const availability = await this.checkGGUFAvailability();
    if (!availability.available) {
      return {
        success: false,
        format: "gguf",
        error: availability.error,
      };
    }

    onProgress?.({
      format: "gguf",
      phase: "converting",
      message: `Converting to GGUF with ${quantization} quantization...`,
    });

    try {
      // Use the Python export script
      const exportScript = path.join(this.scriptDir, "export_gguf.py");
      
      // Check if script exists
      try {
        await fs.access(exportScript);
      } catch {
        // Script doesn't exist - provide helpful error
        return {
          success: false,
          format: "gguf",
          error: "GGUF export script not found. GGUF export requires additional setup.",
        };
      }

      await this.runCommand(this.pythonPath, [
        exportScript,
        "--model", modelPath,
        "--output", outputPath,
        "--quantization", quantization,
      ]);

      const stats = await fs.stat(outputPath);

      onProgress?.({
        format: "gguf",
        phase: "complete",
        message: "GGUF export complete",
      });

      return {
        success: true,
        format: "gguf",
        outputPath,
        fileSize: stats.size,
      };
    } catch (error) {
      onProgress?.({
        format: "gguf",
        phase: "error",
        message: error instanceof Error ? error.message : "Unknown error",
      });

      return {
        success: false,
        format: "gguf",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Export model to ONNX format for browser/edge deployment
   * This fuses MLX LoRA adapters with the base model and converts to ONNX
   */
  async exportONNX(
    adapterPath: string,
    outputDir: string,
    baseModel: string = "mlx-community/functiongemma-270m-it-4bit",
    quantization: "int8" | "int4" | "fp16" | "none" = "none",
    onProgress?: ExportProgressCallback
  ): Promise<ExportResult> {
    onProgress?.({
      format: "onnx",
      phase: "preparing",
      message: "Preparing ONNX export...",
    });

    try {
      // Create output directory
      await fs.mkdir(outputDir, { recursive: true });

      // Use the Python fusion and ONNX export script
      const exportScript = path.join(this.scriptDir, "fuse_and_export_onnx.py");
      
      // Check if script exists
      try {
        await fs.access(exportScript);
      } catch {
        return {
          success: false,
          format: "onnx",
          error: "ONNX export script not found at: " + exportScript,
        };
      }

      onProgress?.({
        format: "onnx",
        phase: "converting",
        message: "Fusing adapters and converting to ONNX (this may take a few minutes)...",
      });

      const fusedDir = path.join(outputDir, "fused");
      const onnxDir = path.join(outputDir, "onnx");

      const args =  [
        exportScript,
        "--adapter", adapterPath,
        "--base-model", baseModel,
        "--output", fusedDir,
        "--onnx-output", onnxDir,
      ];

      if (quantization !== "none") {
        args.push("--quantize", quantization);
      }

      await this.runCommand(this.pythonPath, args);

      // Get size of ONNX directory
      let totalSize = 0;
      const files = await fs.readdir(onnxDir);
      for (const file of files) {
        const fileStat = await fs.stat(path.join(onnxDir, file));
        if (fileStat.isFile()) {
          totalSize += fileStat.size;
        }
      }

      onProgress?.({
        format: "onnx",
        phase: "complete",
        message: "ONNX export complete",
      });

      return {
        success: true,
        format: "onnx",
        outputPath: onnxDir,
        fileSize: totalSize,
      };
    } catch (error) {
      onProgress?.({
        format: "onnx",
        phase: "error",
        message: error instanceof Error ? error.message : "Unknown error",
      });

      return {
        success: false,
        format: "onnx",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Export model to multiple formats
   */
  async export(
    config: ExportConfig,
    onProgress?: ExportProgressCallback
  ): Promise<Map<ExportFormat, ExportResult>> {
    const results = new Map<ExportFormat, ExportResult>();

    await fs.mkdir(config.outputDir, { recursive: true });

    for (const format of config.outputFormats) {
      switch (format) {
        case "safetensors":
          const safetensorsDir = path.join(config.outputDir, "safetensors");
          const safetensorsResult = await this.exportSafetensors(
            config.adapterPath,
            safetensorsDir,
            onProgress
          );
          results.set("safetensors", safetensorsResult);
          break;

        case "gguf":
          const ggufPath = path.join(
            config.outputDir,
            `model-${config.quantization ?? "4bit"}.gguf`
          );
          const ggufResult = await this.exportGGUF(
            config.adapterPath,
            ggufPath,
            config.quantization ?? "4bit",
            onProgress
          );
          results.set("gguf", ggufResult);
          break;

        case "onnx":
          const onnxDir = path.join(config.outputDir, "onnx");
          const onnxResult = await this.exportONNX(
            config.adapterPath,
            onnxDir,
            config.baseModel,
            config.quantization as "int8" | "int4" | "fp16" | "none",
            onProgress
          );
          results.set("onnx", onnxResult);
          break;

        default:
          results.set(format, {
            success: false,
            format,
            error: `Unknown format: ${format}`,
          });
      }
    }

    return results;
  }

  /**
   * Get information about an exported model
   */
  async getModelInfo(modelPath: string): Promise<{
    exists: boolean;
    format?: string;
    size?: number;
    files?: string[];
  }> {
    try {
      const stats = await fs.stat(modelPath);
      
      if (stats.isFile()) {
        return {
          exists: true,
          format: path.extname(modelPath).replace(".", ""),
          size: stats.size,
          files: [path.basename(modelPath)],
        };
      }

      if (stats.isDirectory()) {
        const files = await fs.readdir(modelPath);
        let totalSize = 0;
        
        for (const file of files) {
          const fileStat = await fs.stat(path.join(modelPath, file));
          if (fileStat.isFile()) {
            totalSize += fileStat.size;
          }
        }

        const hasSafetensors = files.some((f) => f.endsWith(".safetensors"));
        
        return {
          exists: true,
          format: hasSafetensors ? "safetensors" : "unknown",
          size: totalSize,
          files,
        };
      }

      return { exists: false };
    } catch {
      return { exists: false };
    }
  }
}
