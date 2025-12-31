/**
 * Dataset command - Prepare training dataset from examples
 */

import { parseArgs } from "node:util";
import * as fs from "fs/promises";
import * as path from "path";
import { DatasetBuilder } from "../../dataset/DatasetBuilder";
import type { ToolSchema, TrainingExample } from "../../types";

const HELP = `
fine-tune dataset - Prepare training dataset from examples

USAGE:
  fine-tune dataset [options]

OPTIONS:
  -i, --input <file>     Input file with training examples (JSON)
  -o, --output <dir>     Output directory for dataset files
  -t, --tools <file>     Tool definitions file (JSON)
  --split <ratio>        Train/valid/test split ratio (default: 0.8,0.1,0.1)
  -h, --help             Show this help message

EXAMPLES:
  fine-tune dataset -i examples.json -o ./data -t tools.json
  fine-tune dataset --input examples.json --output ./data --tools tools.json --split 0.7,0.2,0.1
`;

export async function datasetCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      input: { type: "string", short: "i" },
      output: { type: "string", short: "o" },
      tools: { type: "string", short: "t" },
      split: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    console.log(HELP);
    return;
  }

  // Validate required options
  if (!values.input) {
    throw new Error("Missing required option: --input");
  }
  if (!values.output) {
    throw new Error("Missing required option: --output");
  }
  if (!values.tools) {
    throw new Error("Missing required option: --tools");
  }

  console.log("📂 Preparing dataset...");
  console.log(`   Input: ${values.input}`);
  console.log(`   Output: ${values.output}`);
  console.log(`   Tools: ${values.tools}`);

  // Load tools
  console.log("\n🔧 Loading tool definitions...");
  const toolsContent = await fs.readFile(values.tools, "utf-8");
  const tools: ToolSchema[] = JSON.parse(toolsContent);
  console.log(`   Found ${tools.length} tool(s): ${tools.map(t => t.name).join(", ")}`);

  // Load examples
  console.log("\n📝 Loading training examples...");
  const examplesContent = await fs.readFile(values.input, "utf-8");
  const examples: TrainingExample[] = JSON.parse(examplesContent);
  console.log(`   Found ${examples.length} example(s)`);

  // Parse split ratio
  let splitRatio = { train: 0.8, valid: 0.1, test: 0.1 };
  if (values.split) {
    const parts = values.split.split(",").map(Number);
    if (parts.length !== 3 || parts.some(isNaN)) {
      throw new Error("Invalid split ratio. Use format: 0.8,0.1,0.1");
    }
    const [train, valid, test] = parts;
    splitRatio = { train: train!, valid: valid!, test: test! };
  }

  console.log(`   Split ratio: train=${splitRatio.train}, valid=${splitRatio.valid}, test=${splitRatio.test}`);

  // Create dataset builder
  const builder = new DatasetBuilder({
    tools: tools as any,
    outputDir: values.output,
    splitRatio,
  });

  // Add examples
  builder.addExamples(examples);

  // Preview first example
  console.log("\n👀 Preview of formatted example:");
  const preview = builder.preview(1);
  if (preview[0]) {
    const previewText = preview[0].text.substring(0, 500);
    console.log("   " + previewText.split("\n").join("\n   ") + "...");
  }

  // Build dataset
  console.log("\n🔨 Building dataset...");
  const result = await builder.build();

  console.log("\n✅ Dataset created successfully!");
  console.log(`   Train: ${result.trainPath} (${result.counts.train} examples)`);
  console.log(`   Valid: ${result.validPath} (${result.counts.valid} examples)`);
  console.log(`   Test:  ${result.testPath} (${result.counts.test} examples)`);

  // Save tool schemas
  const toolsPath = await builder.saveToolSchemas();
  console.log(`   Tools: ${toolsPath}`);
}
