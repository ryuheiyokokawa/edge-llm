/**
 * Train command - Train a LoRA adapter using MLX
 */

import { parseArgs } from "node:util";
import * as path from "path";
import { MLXTrainer } from "../../training/MLXTrainer";
import { DEFAULT_TRAINING_CONFIG } from "../../training/TrainingConfig";

const HELP = `
fine-tune train - Train a LoRA adapter using MLX

USAGE:
  fine-tune train [options]

OPTIONS:
  -d, --data <dir>       Training data directory (contains train.jsonl, valid.jsonl)
  -o, --output <dir>     Output directory for trained adapters
  -m, --model <name>     Base model (default: mlx-community/functiongemma-270m-it-4bit)
  --epochs <n>           Number of training epochs (default: 3)
  --batch-size <n>       Batch size (default: 4)
  --lr <rate>            Learning rate (default: 0.0002)
  --lora-rank <n>        LoRA rank (default: 8)
  --lora-alpha <n>       LoRA alpha (default: 16)
  --check-env            Check Python environment and exit
  -h, --help             Show this help message

EXAMPLES:
  fine-tune train -d ./data -o ./output
  fine-tune train --data ./data --output ./output --epochs 5
  fine-tune train --check-env
`;

export async function trainCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      data: { type: "string", short: "d" },
      output: { type: "string", short: "o" },
      model: { type: "string", short: "m" },
      epochs: { type: "string" },
      "batch-size": { type: "string" },
      lr: { type: "string" },
      "lora-rank": { type: "string" },
      "lora-alpha": { type: "string" },
      "check-env": { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    console.log(HELP);
    return;
  }

  // Check environment mode
  if (values["check-env"]) {
    console.log("🔍 Checking Python environment...\n");
    
    const trainer = new MLXTrainer({
      datasetPath: ".",
      outputPath: ".",
    });

    const env = await trainer.checkEnvironment();

    console.log("Python Environment Check:");
    console.log(`  Python:  ${env.pythonAvailable ? "✅ " + env.pythonVersion : "❌ Not found"}`);
    console.log(`  MLX:     ${env.mlxAvailable ? "✅ " + env.mlxVersion : "❌ Not installed"}`);
    console.log(`  MLX-LM:  ${env.mlxLmAvailable ? "✅ Available" : "❌ Not installed"}`);

    if (!env.mlxAvailable || !env.mlxLmAvailable) {
      console.log("\n⚠️  Install missing packages with:");
      console.log("   pip install mlx mlx-lm");
    } else {
      console.log("\n✅ Environment is ready for training!");
    }

    return;
  }

  // Validate required options
  if (!values.data) {
    throw new Error("Missing required option: --data");
  }
  if (!values.output) {
    throw new Error("Missing required option: --output");
  }

  // Build config
  const config = {
    datasetPath: path.resolve(values.data),
    outputPath: path.resolve(values.output),
    baseModel: values.model ?? DEFAULT_TRAINING_CONFIG.baseModel,
    epochs: values.epochs ? parseInt(values.epochs) : DEFAULT_TRAINING_CONFIG.epochs,
    batchSize: values["batch-size"] ? parseInt(values["batch-size"]) : DEFAULT_TRAINING_CONFIG.batchSize,
    learningRate: values.lr ? parseFloat(values.lr) : DEFAULT_TRAINING_CONFIG.learningRate,
    loraRank: values["lora-rank"] ? parseInt(values["lora-rank"]) : DEFAULT_TRAINING_CONFIG.loraRank,
    loraAlpha: values["lora-alpha"] ? parseInt(values["lora-alpha"]) : DEFAULT_TRAINING_CONFIG.loraAlpha,
  };

  console.log("🚀 Starting LoRA fine-tuning...\n");
  console.log("Configuration:");
  console.log(`  Dataset:      ${config.datasetPath}`);
  console.log(`  Output:       ${config.outputPath}`);
  console.log(`  Base Model:   ${config.baseModel}`);
  console.log(`  Epochs:       ${config.epochs}`);
  console.log(`  Batch Size:   ${config.batchSize}`);
  console.log(`  Learning Rate: ${config.learningRate}`);
  console.log(`  LoRA Rank:    ${config.loraRank}`);
  console.log(`  LoRA Alpha:   ${config.loraAlpha}`);

  // Create trainer
  const trainer = new MLXTrainer(config);

  // Set up event handlers
  trainer.on("log", (message) => {
    console.log(message);
  });

  trainer.on("progress", (progress) => {
    if (progress.status === "training") {
      const pct = progress.totalSteps > 0 
        ? Math.round((progress.step / progress.totalSteps) * 100)
        : 0;
      process.stdout.write(`\r  Epoch ${progress.epoch}/${progress.totalEpochs} | Step ${progress.step} | Loss: ${progress.loss.toFixed(4)} | ${pct}%  `);
    }
  });

  // Start training
  console.log("\n" + "=".repeat(60));
  const result = await trainer.train();
  console.log("\n" + "=".repeat(60));

  if (result.success) {
    const timeSeconds = Math.round((result.trainingTime ?? 0) / 1000);
    console.log("\n✅ Training complete!");
    console.log(`   Time: ${timeSeconds}s`);
    console.log(`   Adapters: ${result.adapterPath}`);
    if (result.finalLoss) {
      console.log(`   Final Loss: ${result.finalLoss.toFixed(4)}`);
    }
  } else {
    console.log("\n❌ Training failed!");
    console.log(`   Error: ${result.error}`);
    process.exit(1);
  }
}
