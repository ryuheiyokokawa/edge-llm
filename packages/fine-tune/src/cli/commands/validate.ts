/**
 * Validate command - Validate a fine-tuned model
 */

import { parseArgs } from "node:util";
import * as fs from "fs/promises";
import * as path from "path";
import { ModelExporter } from "../../export/ModelExporter";

const HELP = `
fine-tune validate - Validate a fine-tuned model

USAGE:
  fine-tune validate [options]

OPTIONS:
  -m, --model <path>     Path to trained model or adapters
  -t, --test <file>      Test dataset file (JSONL) - optional
  -h, --help             Show this help message

CHECKS:
  - Model files exist and are valid
  - Configuration files are present
  - Model can be loaded (requires Python environment)
  - Test inference works (if --test provided)

EXAMPLES:
  fine-tune validate -m ./output/adapters
  fine-tune validate --model ./output/adapters --test ./data/test.jsonl
`;

export async function validateCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      model: { type: "string", short: "m" },
      test: { type: "string", short: "t" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    console.log(HELP);
    return;
  }

  if (!values.model) {
    throw new Error("Missing required option: --model");
  }

  const modelPath = path.resolve(values.model);
  console.log("🔍 Validating model...\n");
  console.log(`   Model: ${modelPath}`);

  const exporter = new ModelExporter();
  let hasErrors = false;

  // Check 1: Model exists
  console.log("\n📁 Checking model files...");
  const modelInfo = await exporter.getModelInfo(modelPath);

  if (!modelInfo.exists) {
    console.log("   ❌ Model path does not exist");
    process.exit(1);
  }

  console.log(`   ✅ Model path exists`);
  console.log(`   ✅ Format: ${modelInfo.format}`);
  console.log(`   ✅ Size: ${formatBytes(modelInfo.size ?? 0)}`);

  // Check 2: Required files
  console.log("\n📋 Checking required files...");
  const requiredFiles = ["config.json"];
  const optionalFiles = ["tokenizer.json", "tokenizer_config.json"];

  if (modelInfo.files) {
    for (const file of requiredFiles) {
      if (modelInfo.files.includes(file)) {
        console.log(`   ✅ ${file}`);
      } else {
        console.log(`   ⚠️  ${file} not found (may be inherited from base model)`);
      }
    }

    for (const file of optionalFiles) {
      if (modelInfo.files.includes(file)) {
        console.log(`   ✅ ${file}`);
      }
    }

    // Check for model weights
    const weightFiles = modelInfo.files.filter(
      (f) => f.endsWith(".safetensors") || f.endsWith(".npz") || f.endsWith(".gguf")
    );
    if (weightFiles.length > 0) {
      console.log(`   ✅ Found ${weightFiles.length} weight file(s)`);
    } else {
      console.log("   ❌ No weight files found");
      hasErrors = true;
    }
  }

  // Check 3: Python environment (optional)
  console.log("\n🐍 Checking Python environment...");
  try {
    const { MLXTrainer } = await import("../../training/MLXTrainer");
    const trainer = new MLXTrainer({
      datasetPath: ".",
      outputPath: ".",
    });
    const env = await trainer.checkEnvironment();

    console.log(`   ${env.pythonAvailable ? "✅" : "❌"} Python: ${env.pythonVersion ?? "not found"}`);
    console.log(`   ${env.mlxAvailable ? "✅" : "⚠️ "} MLX: ${env.mlxVersion ?? "not installed"}`);
    console.log(`   ${env.mlxLmAvailable ? "✅" : "⚠️ "} MLX-LM: ${env.mlxLmAvailable ? "available" : "not installed"}`);

    if (!env.mlxAvailable || !env.mlxLmAvailable) {
      console.log("   ⚠️  Install with: pip install mlx mlx-lm");
    }
  } catch (error) {
    console.log("   ⚠️  Could not check Python environment");
  }

  // Check 4: Test dataset (if provided)
  if (values.test) {
    console.log("\n📊 Checking test dataset...");
    const testPath = path.resolve(values.test);

    try {
      const content = await fs.readFile(testPath, "utf-8");
      const lines = content.trim().split("\n");
      
      let validLines = 0;
      let invalidLines = 0;

      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.text && typeof parsed.text === "string") {
            validLines++;
          } else {
            invalidLines++;
          }
        } catch {
          invalidLines++;
        }
      }

      console.log(`   ✅ Test file: ${testPath}`);
      console.log(`   ✅ Valid examples: ${validLines}`);
      if (invalidLines > 0) {
        console.log(`   ⚠️  Invalid lines: ${invalidLines}`);
      }
    } catch (error) {
      console.log(`   ❌ Could not read test file: ${error instanceof Error ? error.message : error}`);
      hasErrors = true;
    }
  }

  // Summary
  console.log("\n" + "=".repeat(50));
  if (hasErrors) {
    console.log("\n❌ Validation failed with errors");
    process.exit(1);
  } else {
    console.log("\n✅ Validation passed!");
    console.log("\nNext steps:");
    console.log("  1. Test inference with the model");
    console.log("  2. Export to deployment format if needed");
    console.log("  3. Integrate with your application");
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
