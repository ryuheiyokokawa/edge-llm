/**
 * Export command - Export trained model to deployment formats
 */

import { parseArgs } from "node:util";
import * as path from "path";
import { ModelExporter } from "../../export/ModelExporter";
import type { ExportFormat } from "../../types";

const HELP = `
fine-tune export - Export trained model to deployment formats

USAGE:
  fine-tune export [options]

OPTIONS:
  -m, --model <path>     Path to trained adapters or model
  -o, --output <dir>     Output directory for exported model
  -f, --format <fmt>     Output format: safetensors, gguf (default: safetensors)
  -q, --quantization     Quantization level: 4bit, 8bit, none (default: 4bit)
  --info                 Show model information and exit
  -h, --help             Show this help message

FORMATS:
  safetensors  MLX native format (for inference with mlx-lm)
  gguf         llama.cpp format (requires llama-cpp-python)

EXAMPLES:
  fine-tune export -m ./output/adapters -o ./models -f safetensors
  fine-tune export --model ./output/adapters --output ./models --format gguf
  fine-tune export -m ./output/adapters --info
`;

export async function exportCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      model: { type: "string", short: "m" },
      output: { type: "string", short: "o" },
      format: { type: "string", short: "f" },
      quantization: { type: "string", short: "q" },
      info: { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    console.log(HELP);
    return;
  }

  // Validate required options
  if (!values.model) {
    throw new Error("Missing required option: --model");
  }

  const exporter = new ModelExporter();
  const modelPath = path.resolve(values.model);

  // Info mode
  if (values.info) {
    console.log("📊 Model Information\n");
    
    const info = await exporter.getModelInfo(modelPath);
    
    if (!info.exists) {
      console.log(`❌ Model not found: ${modelPath}`);
      process.exit(1);
    }

    console.log(`  Path:   ${modelPath}`);
    console.log(`  Format: ${info.format}`);
    console.log(`  Size:   ${formatBytes(info.size ?? 0)}`);
    
    if (info.files) {
      console.log(`  Files:  ${info.files.length}`);
      for (const file of info.files.slice(0, 10)) {
        console.log(`          - ${file}`);
      }
      if (info.files.length > 10) {
        console.log(`          ... and ${info.files.length - 10} more`);
      }
    }

    return;
  }

  // Export mode
  if (!values.output) {
    throw new Error("Missing required option: --output");
  }

  const format = (values.format ?? "safetensors") as ExportFormat;
  const quantization = (values.quantization ?? "4bit") as "4bit" | "8bit" | "none";
  const outputDir = path.resolve(values.output);

  console.log("📦 Exporting model...\n");
  console.log(`  Model:        ${modelPath}`);
  console.log(`  Output:       ${outputDir}`);
  console.log(`  Format:       ${format}`);
  console.log(`  Quantization: ${quantization}`);

  // Check GGUF availability if needed
  if (format === "gguf") {
    const check = await exporter.checkGGUFAvailability();
    if (!check.available) {
      console.log("\n⚠️  GGUF export not available:");
      console.log(`   ${check.error}`);
      process.exit(1);
    }
  }

  // Export
  console.log("\n");
  const results = await exporter.export(
    {
      adapterPath: modelPath,
      baseModel: "functiongemma", // Not used for safetensors
      outputFormats: [format],
      outputDir,
      quantization,
    },
    (status) => {
      const icon = status.phase === "complete" ? "✅" : 
                   status.phase === "error" ? "❌" : "⏳";
      console.log(`  ${icon} ${status.format}: ${status.message ?? status.phase}`);
    }
  );

  // Report results
  console.log("\n" + "=".repeat(50));
  
  let hasErrors = false;
  for (const [fmt, result] of results) {
    if (result.success) {
      console.log(`\n✅ ${fmt} export successful`);
      console.log(`   Path: ${result.outputPath}`);
      if (result.fileSize) {
        console.log(`   Size: ${formatBytes(result.fileSize)}`);
      }
    } else {
      hasErrors = true;
      console.log(`\n❌ ${fmt} export failed`);
      console.log(`   Error: ${result.error}`);
    }
  }

  if (hasErrors) {
    process.exit(1);
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
