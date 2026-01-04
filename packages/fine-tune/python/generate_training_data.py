#!/usr/bin/env python3
"""
Generate training data in FunctionGemma format.

This script converts tool definitions and training examples into the
FunctionGemma format required for fine-tuning.

Input formats:
- Tool definitions: JSON file with tool schemas
- Training examples: JSON file with user queries and expected tool calls

Output format:
- JSONL files with <bos><start_of_turn>... conversations in FunctionGemma format

Usage:
  # Use default examples (from examples/ folder)
  python generate_training_data.py

  # Use custom data
  python generate_training_data.py \
    --tools-file my-tools.json \
    --examples-file my-examples.json \
    --output-dir my-training-data

Input File Formats:
-------------------

tools-file (JSON array):
[
  {
    "name": "calculate",
    "description": "Evaluate a mathematical expression",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "Math expression"}
      },
      "required": ["expression"]
    }
  }
]

examples-file (JSON array):
[
  {
    "userQuery": "What is 5 * 12?",
    "expectedToolCalls": [{"name": "calculate", "arguments": {"expression": "5 * 12"}}]
  }
]
"""

import json
import random
import argparse
from pathlib import Path


def format_tool_declaration(tool: dict) -> str:
    """Format a tool definition in FunctionGemma format."""
    name = tool["name"]
    desc = tool["description"]
    params = tool.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    
    # Format parameters
    param_items = []
    for key, val in properties.items():
        param_type = val.get("type", "STRING").upper()
        param_desc = val.get("description", "")
        param_str = f'{key}:{{description:<escape>{param_desc}<escape>,type:<escape>{param_type}<escape>'
        if "enum" in val:
            enum_str = ",".join(f"<escape>{e}<escape>" for e in val["enum"])
            param_str += f",enum:[{enum_str}]"
        param_str += "}"
        param_items.append(param_str)
    
    params_str = ",".join(param_items)
    required_str = ",".join(f"<escape>{r}<escape>" for r in required)
    
    return f'<start_function_declaration>declaration:{name}{{description:<escape>{desc}<escape>,parameters:{{{params_str}}},required:[{required_str}],type:<escape>OBJECT<escape>}}<end_function_declaration>'


def format_function_call(name: str, args: dict) -> str:
    """Format a function call in FunctionGemma format."""
    def escape_value(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)
    
    args_str = ",".join(f"{k}:<escape>{escape_value(v)}<escape>" for k, v in args.items())
    return f'<start_function_call>call:{name}{{{args_str}}}<end_function_call>'


def load_tools(tools_path: Path) -> list[dict]:
    """Load tool definitions from a JSON file or directory."""
    tools = []
    
    if tools_path.is_dir():
        # Load from directory (multiple files)
        for tool_file in tools_path.glob("*.json"):
            with open(tool_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tools.extend(data)
                else:
                    tools.append(data)
    else:
        # Load from single file
        with open(tools_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                tools.extend(data)
            else:
                tools.append(data)
    
    return tools


def load_examples(examples_path: Path) -> list[dict]:
    """Load training examples from JSON file."""
    with open(examples_path) as f:
        return json.load(f)


def generate_training_sample(example: dict, tools: list[dict]) -> dict | None:
    """Generate a single training sample in FunctionGemma format."""
    # Build the system prompt with tool declarations
    tool_declarations = "\n".join(format_tool_declaration(tool) for tool in tools)
    
    system_prompt = f"""You are a model that can do function calling with the following functions.
Must use the EXACT format: <start_function_call>call:name{{arg:<escape>val<escape>}}<end_function_call>
Example: <start_function_call>call:calculate{{expression:<escape>5*12<escape>}}<end_function_call>
{tool_declarations}"""

    # Get the expected tool call
    tool_calls = example.get("expectedToolCalls", [])
    if not tool_calls:
        return None
    
    # Only use first tool call for simplicity
    tc = tool_calls[0]
    function_call = format_function_call(tc["name"], tc["arguments"])
    
    # Full conversation format
    text = f"""<bos><start_of_turn>developer
{system_prompt}<end_of_turn>
<start_of_turn>user
{example["userQuery"]}<end_of_turn>
<start_of_turn>model
{function_call}<end_of_turn>"""
    
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(
        description="Generate FunctionGemma training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (with defaults to examples/ folder)
    parser.add_argument(
        "--tools-file", type=Path,
        default=Path("examples/tool-definitions"),
        help="Path to tool definitions JSON file or directory (default: examples/tool-definitions/)"
    )
    parser.add_argument(
        "--examples-file", type=Path,
        default=Path("examples/training-examples.json"),
        help="Path to training examples JSON file (default: examples/training-examples.json)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=Path, default=Path("training-data"),
        help="Output directory for JSONL files (default: training-data/)"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.85,
        help="Fraction of data for training (default: 0.85)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.tools_file.exists():
        print(f"Error: Tool definitions not found: {args.tools_file}")
        print("\nCreate a tools JSON file with the following format:")
        print('''[
  {"name": "my_tool", "description": "...", "parameters": {...}}
]''')
        return 1
    
    if not args.examples_file.exists():
        print(f"Error: Training examples not found: {args.examples_file}")
        print("\nCreate an examples JSON file with the following format:")
        print('''[
  {"userQuery": "...", "expectedToolCalls": [{"name": "...", "arguments": {...}}]}
]''')
        return 1
    
    # Load data
    tools = load_tools(args.tools_file)
    examples = load_examples(args.examples_file)
    
    print(f"Loaded {len(tools)} tools from {args.tools_file}")
    print(f"Loaded {len(examples)} examples from {args.examples_file}")
    
    # Generate training data
    training_data = []
    for ex in examples:
        sample = generate_training_sample(ex, tools)
        if sample:
            training_data.append(sample)
    
    print(f"Generated {len(training_data)} training samples")
    
    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(training_data)
    split_idx = int(len(training_data) * args.train_split)
    train_data = training_data[:split_idx]
    valid_data = training_data[split_idx:]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write output files
    train_file = args.output_dir / "train.jsonl"
    valid_file = args.output_dir / "valid.jsonl"
    
    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(valid_file, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n✅ Written {len(train_data)} training samples to {train_file}")
    print(f"✅ Written {len(valid_data)} validation samples to {valid_file}")
    
    # Print a sample
    print("\n" + "=" * 60)
    print("Sample output (truncated):")
    print("=" * 60)
    sample_text = training_data[0]["text"]
    print(sample_text[:400] + "..." if len(sample_text) > 400 else sample_text)
    
    return 0


if __name__ == "__main__":
    exit(main())
