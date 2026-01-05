# FunctionGemma Fine-Tuning System Plan

**Version:** 0.1.0  
**Status:** ✅ Complete (January 2025)  
**Target Platform:** macOS (M1 Pro 16GB) as first-class citizen, with RTX 4070 WSL fallback

---

## Executive Summary

This document outlines the implementation plan for a fine-tuning system within the `@edge-llm/fine-tune` package. The goal is to enable developers to create custom tool-calling models tailored to their specific tool definitions, improving accuracy and reducing inference latency for edge deployment.

**Core Value:** Take a user's registered tools → generate training data → fine-tune FunctionGemma → export optimized model for edge inference.

---

## Hardware Analysis

### Primary: M1 Pro 16GB (macOS)

| Aspect | Assessment |
|--------|------------|
| **Memory** | ✅ **Excellent fit** - FunctionGemma is 270M params (~540MB fp16). With QLoRA (4-bit), base model fits in ~135MB. Peak training memory estimated at 4-6GB, well under 16GB limit |
| **Compute** | ✅ 8-core GPU, Metal acceleration via MLX framework |
| **Framework** | ✅ Apple MLX + `mlx-lm` - native support for LoRA/QLoRA on Apple Silicon |
| **Comparison** | Mistral 7B (7B params) trains successfully on 16GB M1 with ~13-14GB peak memory. FunctionGemma at 270M is ~26x smaller |

> [!TIP]
> Your M1 Pro 16GB is **significantly overpowered** for FunctionGemma fine-tuning. This is the ideal scenario - you have headroom for larger batch sizes and faster training.

### Fallback: RTX 4070 WSL

| Aspect | Assessment |
|--------|------------|
| **Memory** | ✅ 12GB VRAM - also more than sufficient |
| **Compute** | ✅ ~5,888 CUDA cores, faster than M1 for raw throughput |
| **Framework** | ✅ Unsloth or standard PyTorch + Transformers + PEFT |
| **Trade-off** | Faster training, but requires cross-platform code paths |

### Recommendation

**macOS-first is the right call.** FunctionGemma's small size means your M1 Pro handles it easily. We'll use Apple MLX as the primary training backend, with optional CUDA support for the RTX 4070 as a secondary path.

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    @edge-llm/fine-tune                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐│
│  │  DatasetBuilder │───▶│  TrainingPipeline│───▶│ ModelExporter││
│  │                 │    │                 │    │              ││
│  │ - Tool registry │    │ - MLX backend   │    │ - GGUF export││
│  │ - Example gen   │    │ - CUDA backend  │    │ - ONNX export││
│  │ - JSONL output  │    │ - LoRA/QLoRA    │    │ - HF upload  ││
│  └─────────────────┘    └─────────────────┘    └──────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Edge Deployment                             │
├─────────────────────────────────────────────────────────────────┤
│  WebLLM (WebGPU)  │  Transformers.js (WASM)  │  Ollama (Local) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset Format

### FunctionGemma Chat Format

FunctionGemma uses a structured format with special control tokens:

```text
<start_of_turn>developer
You have access to the following functions:

<start_function_declaration>
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"}
    },
    "required": ["location"]
  }
}
<end_function_declaration>
<end_of_turn>
<start_of_turn>user
What's the weather in Tokyo?<end_of_turn>
<start_of_turn>model
<start_function_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
<end_function_call>
<end_of_turn>
<start_of_turn>tool
<start_function_response>
{"temperature": 22, "condition": "sunny"}
<end_function_response>
<end_of_turn>
<start_of_turn>model
The weather in Tokyo is sunny with a temperature of 22°C.<end_of_turn>
```

### Training Data Structure (JSONL)

Each line in training JSONL files:

```jsonl
{"text": "<start_of_turn>developer\nYou have access to...<end_of_turn><start_of_turn>user\nWhat's the weather?<end_of_turn><start_of_turn>model\n<start_function_call>\n{\"name\": \"get_weather\", ...}\n<end_function_call>\n<end_of_turn>"}
```

### Dataset Requirements

| File | Purpose | Required |
|------|---------|----------|
| `train.jsonl` | Training examples | ✅ Yes |
| `valid.jsonl` | Validation during training | ✅ Yes |
| `test.jsonl` | Post-training evaluation | Optional |

---

## Implementation Plan

### Phase 1: Dataset Preparation Tools

Create TypeScript utilities that run in Node.js to prepare training data.

#### 1.1 DatasetBuilder Class

```typescript
interface DatasetBuilderConfig {
  tools: ToolDefinition[];           // From @edge-llm/core
  outputDir: string;
  splitRatio?: { train: number; valid: number; test: number };
}

interface TrainingExample {
  userQuery: string;
  expectedToolCalls: ToolCall[];     // What the model should call
  toolResponses?: Record<string, any>; // Mock responses for multi-turn
  expectedFinalResponse?: string;    // Final text response
}
```

#### 1.2 Example Generation Strategies

| Strategy | Description |
|----------|-------------|
| **Manual** | Developer provides explicit examples |
| **Template** | Generate variations from template patterns |
| **Synthetic** | Use existing LLM to generate diverse examples (optional) |

#### Files to Create

- `src/dataset/DatasetBuilder.ts` - Main dataset preparation class
- `src/dataset/FormatConverter.ts` - Convert to FunctionGemma format
- `src/dataset/ExampleGenerator.ts` - Template-based example generation
- `src/types.ts` - Extended type definitions

---

### Phase 2: Training Infrastructure

Two backends for different platforms:

#### 2.1 MLX Backend (macOS - Primary)

The training will be orchestrated from Node.js but executed via Python subprocess using MLX.

```text
packages/fine-tune/
├── src/                          # TypeScript code
│   ├── training/
│   │   ├── MLXTrainer.ts        # Orchestrates Python MLX scripts
│   │   └── TrainingConfig.ts    # Configuration types
│   └── ...
└── python/                       # Python MLX scripts
    ├── train_lora.py            # LoRA training script
    ├── train_qlora.py           # QLoRA training (4-bit)
    ├── merge_adapters.py        # Merge LoRA weights into base
    └── requirements.txt         # mlx, mlx-lm, etc.
```

#### 2.2 Training Parameters (Defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| **LoRA Rank** | 8 | Good balance for small models |
| **LoRA Alpha** | 16 | Scaling factor |
| **Target Modules** | q_proj, v_proj | Standard attention layers |
| **Batch Size** | 4 | Safe for 16GB, can increase |
| **Learning Rate** | 2e-4 | Standard for LoRA |
| **Epochs** | 3 | Often sufficient for tool-calling |
| **Quantization** | 4-bit (QLoRA) | Optional, saves memory |

#### Files to Create

- `src/training/MLXTrainer.ts` - TypeScript orchestrator
- `src/training/CUDATrainer.ts` - Alternative CUDA backend (Phase 2+)
- `src/training/TrainingConfig.ts` - Unified config
- `python/train_lora.py` - MLX training script
- `python/requirements.txt` - Python dependencies

---

### Phase 3: Model Export & Deployment

Convert fine-tuned models to formats compatible with edge runtimes.

#### 3.1 Export Formats

| Format | Target Runtime | Priority |
|--------|----------------|----------|
| **GGUF** | Ollama, llama.cpp, WebLLM | 🔴 High |
| **Safetensors** | Hugging Face, Transformers.js | 🔴 High |
| **ONNX** | ONNX Runtime Web | 🟡 Medium |

#### 3.2 ModelExporter Class

```typescript
interface ExportConfig {
  adapterPath: string;      // Path to LoRA adapters
  baseModel: string;        // Base model identifier
  outputFormats: ('gguf' | 'safetensors' | 'onnx')[];
  quantization?: '4bit' | '8bit' | 'none';
}
```

#### Files to Create

- `src/export/ModelExporter.ts` - Export orchestration
- `src/export/GGUFConverter.ts` - GGUF conversion
- `python/export_gguf.py` - Python export script
- `python/merge_and_export.py` - Merge + export pipeline

---

### Phase 4: CLI & Developer Experience

Command-line interface for the entire fine-tuning workflow.

```bash
# Generate dataset from tool definitions
npx @edge-llm/fine-tune dataset \
  --tools ./my-tools.json \
  --examples ./examples.json \
  --output ./training-data

# Train the model
npx @edge-llm/fine-tune train \
  --dataset ./training-data \
  --output ./my-model \
  --epochs 3 \
  --backend mlx

# Export for deployment
npx @edge-llm/fine-tune export \
  --model ./my-model \
  --format gguf,safetensors \
  --output ./dist
```

#### Files to Create

- `src/cli/index.ts` - CLI entry point
- `src/cli/commands/dataset.ts` - Dataset command
- `src/cli/commands/train.ts` - Train command
- `src/cli/commands/export.ts` - Export command

---

## Verification Plan

### Automated Tests

1. **Dataset Generation Tests**
   - Verify JSONL format correctness
   - Validate FunctionGemma token structure
   - Test train/valid/test splits

2. **Training Pipeline Tests**
   - Mock MLX subprocess calls
   - Verify config file generation
   - Test error handling

3. **Export Tests**
   - Verify output file creation
   - Validate GGUF metadata

### Manual Verification

1. **End-to-End Training Test**
   ```bash
   # Small dataset, 1 epoch
   npx @edge-llm/fine-tune train --dataset ./test-data --epochs 1
   ```

2. **Model Quality Check**
   - Load fine-tuned model in Ollama
   - Test with registered tools
   - Compare accuracy to base model

3. **Edge Deployment Test**
   - Export to GGUF
   - Load in WebLLM (example app)
   - Verify tool calling works

---

## Dependencies

### Node.js / TypeScript

```json
{
  "dependencies": {
    "@edge-llm/core": "*"
  },
  "devDependencies": {
    "commander": "^12.0.0",
    "execa": "^8.0.0"
  }
}
```

### Python (MLX Backend)

```text
mlx>=0.20.0
mlx-lm>=0.19.0
huggingface_hub>=0.20.0
safetensors>=0.4.0
```

### Optional (CUDA Backend)

```text
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0  # For QLoRA
```

---

## File Structure After Implementation

```
packages/fine-tune/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                    # Public exports
│   ├── types.ts                    # Type definitions
│   ├── dataset/
│   │   ├── DatasetBuilder.ts       # Main dataset class
│   │   ├── FormatConverter.ts      # FunctionGemma format
│   │   └── ExampleGenerator.ts     # Template generation
│   ├── training/
│   │   ├── MLXTrainer.ts           # macOS MLX backend
│   │   ├── CUDATrainer.ts          # Windows/Linux CUDA
│   │   └── TrainingConfig.ts       # Config types
│   ├── export/
│   │   ├── ModelExporter.ts        # Export orchestration
│   │   └── GGUFConverter.ts        # GGUF conversion
│   └── cli/
│       ├── index.ts                # CLI entry
│       └── commands/
│           ├── dataset.ts
│           ├── train.ts
│           └── export.ts
├── python/
│   ├── requirements.txt
│   ├── train_lora.py               # MLX LoRA training
│   ├── train_qlora.py              # MLX QLoRA training
│   ├── merge_adapters.py           # Merge LoRA into base
│   └── export_gguf.py              # GGUF export
└── __tests__/
    ├── dataset.test.ts
    ├── training.test.ts
    └── export.test.ts
```

---

## Decisions (Resolved)

| Question | Decision |
|----------|----------|
| **Synthetic Data Generation** | ✅ Yes - Support **local Ollama** + **OpenAI/Claude APIs** for synthetic training data generation |
| **Pre-built Tool Templates** | ✅ Yes - Ship example training data for common tools (calculator, weather, search, calendar) |
| **Model Size Options** | Start with **270M only**. Larger models can be added later if needed |
| **Continuous Fine-Tuning** | ✅ Explore - Worth investigating adapter merging for incremental updates |
| **Base Model Source** | ✅ Use **MLX-optimized FunctionGemma** from `mlx-community` on Hugging Face |

---

## Gap Analysis

### 🔴 Critical Gaps (Must Address Before Implementation)

#### 1. MLX FunctionGemma Availability ✅ VERIFIED

**Status:** ✅ **Available** - No conversion needed!

**Verified Models on HuggingFace:**
- `mlx-community/functiongemma-270m-it-4bit` (recommended - smallest)
- `mlx-community/functiongemma-270m-it-6bit`
- `mlx-community/functiongemma-270m-it-bf16` (full precision)

**Usage:**
```python
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/functiongemma-270m-it-4bit")
```

**Risk Level:** ✅ Resolved - Model ready to use

---

#### 2. Synthetic Data Generation Architecture

**Issue:** Plan mentions synthetic data but doesn't detail the implementation.

**What's Needed:**

```typescript
interface SyntheticDataProvider {
  provider: 'ollama' | 'openai' | 'claude';
  model?: string;           // e.g., 'llama3.2', 'gpt-4o-mini', 'claude-3-haiku'
  baseUrl?: string;         // For Ollama: http://localhost:11434
  apiKey?: string;          // For OpenAI/Claude
}

interface SyntheticGenerationConfig {
  providers: SyntheticDataProvider[];
  examplesPerTool: number;    // How many variations to generate
  diversityPrompt?: string;   // Custom prompt for diversity
  validateOutput: boolean;    // Validate generated JSON is valid
}
```

**Files to Add:**
- `src/dataset/SyntheticGenerator.ts` - Core generation logic
- `src/dataset/providers/OllamaProvider.ts` - Local Ollama integration
- `src/dataset/providers/OpenAIProvider.ts` - OpenAI API
- `src/dataset/providers/ClaudeProvider.ts` - Anthropic API

**CLI Extension:**
```bash
npx @edge-llm/fine-tune dataset \
  --tools ./my-tools.json \
  --synthetic --provider ollama --model llama3.2 \
  --examples-per-tool 50 \
  --output ./training-data
```

**Effort Impact:** +2-3 days for synthetic data providers

---

#### 3. Continuous Fine-Tuning Strategy (P3 - Nice to Have)

**Priority:** 🟢 **P3 - Future Enhancement** (moved from critical)

**Summary:** LoRA adapter merging is feasible but adds complexity. For initial release, users will retrain from scratch. Can be added in a future phase.

**Future Implementation (if needed):**
- Train new LoRA adapter for new tools
- Merge with existing adapter using weighted average or TIES/DARE
- Export single merged model

**Files to Add:**
- `src/training/AdapterMerger.ts` - Merge multiple LoRA adapters
- `python/merge_lora_adapters.py` - Python merging logic

**Effort Impact:** +1-2 days

**Risk:** Alpha parameter drift with repeated merges - need to test empirically

---

#### 4. Example Training Data Content

**Issue:** Need to define what example data ships with the package.

**Proposed Example Tools:**

| Tool | Examples | Complexity |
|------|----------|------------|
| `calculate` | 30 | Simple - single arg |
| `get_weather` | 30 | Simple - location lookup |
| `search_web` | 30 | Medium - query parsing |
| `get_calendar_events` | 30 | Medium - date ranges |
| `send_email` | 20 | Complex - multiple args |
| `create_reminder` | 20 | Medium - time parsing |

**Total:** ~160 examples in starter dataset

**File Structure:**
```
packages/fine-tune/
├── examples/
│   ├── starter-dataset/
│   │   ├── train.jsonl      # 128 examples (80%)
│   │   ├── valid.jsonl      # 16 examples (10%)
│   │   └── test.jsonl       # 16 examples (10%)
│   └── tool-definitions/
│       └── common-tools.json # Tool schema definitions
```

**Effort Impact:** +1 day

---

### 🟡 Medium Gaps (Should Address)

#### 5. Python Environment Management

**Issue:** Users need Python + MLX installed. How do we handle this?

**Options:**

| Approach | User Experience | Complexity |
|----------|-----------------|------------|
| **Require manual install** | User installs Python/MLX themselves | Low |
| **Bundled Python** | Ship Python with package | High |
| **Docker container** | `docker run edge-llm/fine-tune` | Medium |
| **Check + guide** | Detect Python, provide install instructions | Low-Medium |

**Recommended:** **Check + guide** approach
- On first run, check for Python 3.10+ and MLX
- If missing, print clear installation instructions
- Optionally offer to run `pip install` commands

**Files to Add:**
- `src/utils/PythonEnvChecker.ts` - Check Python environment
- `src/utils/SetupGuide.ts` - Print setup instructions

---

#### 6. Training Progress Reporting

**Issue:** MLX training runs as subprocess. How do we report progress to user?

**Solution:**
- Parse MLX stdout for loss values, epoch progress
- Stream to user via CLI progress bar
- Emit events for programmatic use

```typescript
interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  step: number;
  loss: number;
  learningRate: number;
  eta: string;
}

// Event-based API
trainer.on('progress', (progress: TrainingProgress) => {
  console.log(`Epoch ${progress.epoch}/${progress.totalEpochs} - Loss: ${progress.loss}`);
});
```

---

#### 7. Model Validation After Training

**Issue:** How do we verify the fine-tuned model actually works?

**Solution:** Add validation step that:
1. Loads the fine-tuned model
2. Runs test examples
3. Measures tool-calling accuracy
4. Reports pass/fail with metrics

```bash
npx @edge-llm/fine-tune validate \
  --model ./my-model \
  --test-data ./test.jsonl \
  --threshold 0.9   # Fail if accuracy < 90%
```

---

### 🟢 Nice-to-Have (Future Enhancements)

#### 8. Web UI for Dataset Curation

Instead of CLI-only, a local web UI could help users:
- View and edit training examples
- Run synthetic generation
- Monitor training progress
- Test the model interactively

**Effort:** +3-5 days (future phase)

---

#### 9. Hugging Face Hub Integration

Allow users to:
- Push fine-tuned models to HF Hub
- Pull community-shared LoRA adapters
- Share training datasets

---

#### 10. A/B Testing Infrastructure

Compare base model vs fine-tuned model performance on same test set.

---

## Updated Phased Delivery

| Phase | Scope | Priority | Effort |
|-------|-------|----------|--------|
| **Phase 1** | Dataset preparation (DatasetBuilder, FormatConverter, Example data) | P1 | 2-3 days |
| **Phase 2** | Synthetic data generation (Ollama provider - llama3.2) | P1 | 1-2 days |
| **Phase 3** | MLX training backend (MLXTrainer, Python scripts) | P1 | 3-4 days |
| **Phase 4** | Export pipeline (GGUF, Safetensors) | P2 | 2-3 days |
| **Phase 5** | CLI interface + validation command | P2 | 2-3 days |
| **Phase 6** | CUDA backend (optional) | P3 | 2-3 days |
| **Phase 7** | Continuous fine-tuning (LoRA adapter merging) | P3 | 1-2 days |

**Total Estimated Effort:** 14-21 days for full implementation (including gap analysis additions)

---

## Gap Analysis Summary

| Category | Items | Impact |
|----------|-------|--------|
| 🔴 **Critical** | MLX model availability, Synthetic data architecture, Continuous training, Example data | +5-8 days |
| 🟡 **Medium** | Python env management, Progress reporting, Model validation | +2-3 days |
| 🟢 **Nice-to-have** | Web UI, HF Hub integration, A/B testing | Future phases |

**Key Risks:**
1. FunctionGemma 270M may need manual conversion to MLX format
2. Repeated LoRA merging may cause alpha drift (needs empirical testing)
3. Python dependency management adds friction for Node.js users

---

## Summary

Your M1 Pro 16GB is **more than capable** of fine-tuning FunctionGemma locally. The MLX framework provides native Apple Silicon support with LoRA/QLoRA, and the small model size (270M params) means you'll have excellent training performance with memory to spare.

The implementation prioritizes:
1. **macOS as first-class citizen** via MLX
2. **Simple developer experience** via TypeScript + CLI
3. **Edge-ready exports** via GGUF and Safetensors
4. **Synthetic data generation** via Ollama (local) and cloud APIs
5. **Continuous fine-tuning** via LoRA adapter merging
6. **Optional CUDA fallback** for Windows/Linux users

---

_Last updated: December 2025_

