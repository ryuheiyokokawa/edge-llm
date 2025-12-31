#!/usr/bin/env node
/**
 * @edge-llm/fine-tune CLI
 * Command-line interface for fine-tuning FunctionGemma models
 */

// Import commands
import { datasetCommand } from "./commands/dataset";
import { trainCommand } from "./commands/train";
import { exportCommand } from "./commands/export";
import { validateCommand } from "./commands/validate";

const VERSION = "0.1.0";

const HELP = `
@edge-llm/fine-tune v${VERSION}
Fine-tune FunctionGemma for custom tool calling

USAGE:
  fine-tune <command> [options]

COMMANDS:
  dataset   Prepare training dataset from examples
  train     Train a LoRA adapter using MLX
  export    Export trained model to deployment formats
  validate  Validate a fine-tuned model

OPTIONS:
  -h, --help     Show this help message
  -v, --version  Show version number

EXAMPLES:
  # Prepare a dataset from examples
  fine-tune dataset --input examples.json --output ./data --tools tools.json

  # Train a LoRA adapter
  fine-tune train --data ./data --output ./output

  # Export to GGUF format
  fine-tune export --model ./output/adapters --output ./models --format gguf

  # Validate the trained model
  fine-tune validate --model ./output/adapters

For more information, run:
  fine-tune <command> --help
`;

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === "-h" || args[0] === "--help") {
    console.log(HELP);
    process.exit(0);
  }

  if (args[0] === "-v" || args[0] === "--version") {
    console.log(VERSION);
    process.exit(0);
  }

  const command = args[0];
  const commandArgs = args.slice(1);

  try {
    switch (command) {
      case "dataset":
        await datasetCommand(commandArgs);
        break;

      case "train":
        await trainCommand(commandArgs);
        break;

      case "export":
        await exportCommand(commandArgs);
        break;

      case "validate":
        await validateCommand(commandArgs);
        break;

      default:
        console.error(`Unknown command: ${command}`);
        console.log(HELP);
        process.exit(1);
    }
  } catch (error) {
    console.error(
      `\n❌ Error: ${error instanceof Error ? error.message : error}`
    );
    process.exit(1);
  }
}

main();
