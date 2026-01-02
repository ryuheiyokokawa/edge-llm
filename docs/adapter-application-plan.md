# LoRA Adapter Integration Plan

## Overview

This plan outlines how to integrate LoRA adapter support into the `@edge-llm/core` package and example app, enabling users to load fine-tuned FunctionGemma adapters for custom tool-calling capabilities.

## Background

**Current State:**
- Fine-tuning system (`@edge-llm/fine-tune`) creates LoRA adapters (`.safetensors` files)
- Core package uses base FunctionGemma model: `onnx-community/functiongemma-270m-it-ONNX`
- TransformersRuntime loads models via `@huggingface/transformers` pipeline
- No mechanism to load or merge adapters

**Goal:**
Enable loading fine-tuned LoRA adapters on top of the base model in the TransformersRuntime, allowing custom tool calling without modifying the base model.

---

## Part 1: Core Package Modifications

### 1.1 Type Definitions

**File:** `packages/core/src/types.ts`

**Changes:**
Add adapter configuration to `RuntimeConfig`:

```typescript
export interface RuntimeConfig {
  // ... existing fields ...
  
  /**
   * LoRA adapter configuration
   */
  adapter?: AdapterConfig;
}

/**
 * LoRA adapter configuration
 */
export interface AdapterConfig {
  /**
   * Path or URL to adapter weights (safetensors file or directory)
   */
  path: string;
  
  /**
   * Type of adapter loading
   * - 'local': Load from local file system (Node.js only)
   * - 'url': Load from URL (browser-compatible)
   * - 'huggingface': Load from Hugging Face Hub
   */
  source: 'local' | 'url' | 'huggingface';
  
  /**
   * Whether to merge adapter immediately or apply dynamically
   * - 'merge': Merge adapter weights into model (slower init, faster inference)
   * - 'dynamic': Apply adapter dynamically (faster init, slightly slower inference)
   */
  mode?: 'merge' | 'dynamic';
  
  /**
   * Optional: adapter scale factor (LoRA alpha / rank)
   * If not provided, will be read from adapter_config.json
   */
  scale?: number;
}
```

**Rationale:** Browser environment cannot access file system, so we need URL-based loading. Hugging Face loading allows easy sharing of adapters.

---

### 1.2 TransformersRuntime Modifications

**File:** `packages/core/src/runtime/transformers.ts`

#### Changes Overview:
1. Add adapter loading logic to `initialize()` method
2. Implement adapter merging or dynamic application
3. Handle adapter-specific errors and fallbacks

#### 1.2.1 Add Adapter Loading After Pipeline Initialization

**Location:** After line 211 (after pipeline is loaded)

```typescript
// After pipeline initialization
if (this.config.adapter) {
  await this.loadAdapter(this.config.adapter);
}
```

#### 1.2.2 Implement `loadAdapter()` Method

```typescript
/**
 * Load and apply LoRA adapter
 */
private async loadAdapter(adapterConfig: AdapterConfig): Promise<void> {
  this.log('[Transformers.js] Loading LoRA adapter...');
  
  try {
    // Step 1: Fetch adapter files
    const adapterData = await this.fetchAdapter(adapterConfig);
    
    // Step 2: Load adapter config
    const config = adapterData.config || {
      rank: 8,
      alpha: 16,
      layers: 16,
    };
    
    this.log('[Transformers.js] Adapter config:', config);
    
    // Step 3: Apply adapter
    if (adapterConfig.mode === 'merge') {
      await this.mergeAdapter(adapterData, config);
    } else {
      await this.applyAdapterDynamic(adapterData, config);
    }
    
    this.log('[Transformers.js] Adapter loaded successfully');
  } catch (error) {
    console.error('[Transformers.js] Failed to load adapter:', error);
    // Don't throw - fall back to base model
    this.log('[Transformers.js] Continuing with base model');
  }
}

/**
 * Fetch adapter files from source
 */
private async fetchAdapter(config: AdapterConfig): Promise<AdapterData> {
  switch (config.source) {
    case 'url':
      return await this.fetchAdapterFromURL(config.path);
    case 'huggingface':
      return await this.fetchAdapterFromHF(config.path);
    case 'local':
      throw new Error('Local adapter loading not supported in browser');
    default:
      throw new Error(`Unknown adapter source: ${config.source}`);
  }
}

/**
 * Fetch adapter from URL
 */
private async fetchAdapterFromURL(url: string): Promise<AdapterData> {
  // Fetch adapters.safetensors
  const weightsUrl = url.endsWith('.safetensors') 
    ? url 
    : `${url}/adapters.safetensors`;
    
  const weightsResponse = await fetch(weightsUrl);
  if (!weightsResponse.ok) {
    throw new Error(`Failed to fetch adapter weights: ${weightsResponse.statusText}`);
  }
  
  const weightsBlob = await weightsResponse.blob();
  const weightsBuffer = await weightsBlob.arrayBuffer();
  
  // Fetch adapter_config.json (optional)
  let config = null;
  try {
    const configUrl = url.endsWith('.safetensors')
      ? url.replace('adapters.safetensors', 'adapter_config.json')
      : `${url}/adapter_config.json`;
      
    const configResponse = await fetch(configUrl);
    if (configResponse.ok) {
      config = await configResponse.json();
    }
  } catch (e) {
    // Config is optional
    this.log('[Transformers.js] adapter_config.json not found, using defaults');
  }
  
  return {
    weights: weightsBuffer,
    config,
  };
}

/**
 * Fetch adapter from Hugging Face Hub
 */
private async fetchAdapterFromHF(repoId: string): Promise<AdapterData> {
  const baseUrl = `https://huggingface.co/${repoId}/resolve/main`;
  return await this.fetchAdapterFromURL(baseUrl);
}
```

**Challenge:** Transformers.js doesn't natively support LoRA adapter loading in the browser. We have two options:

**Option A (Recommended): Server-Side Merging**
- Merge adapters during fine-tuning or as a separate step
- Export merged model to ONNX
- Load merged model directly

**Option B: Client-Side LoRA Application**
- Implement lightweight LoRA weight application in JavaScript
- Requires manual tensor operations
- More complex but more flexible

**Initial Implementation:** We'll focus on **Option A** since it's simpler and leverages existing infrastructure.

---

### 1.3 Model Export Modifications

Since client-side LoRA merging is complex, we'll modify the export pipeline to create merged ONNX models:

**File:** `packages/fine-tune/src/export/ModelExporter.ts`

**Add Method:**
```typescript
/**
 * Export merged model to ONNX format
 */
async exportONNX(
  adapterPath: string,
  outputPath: string,
  baseModel: string,
  onProgress?: ExportProgressCallback
): Promise<ExportResult> {
  // Implementation will call Python script for ONNX export
  // This requires additional Python dependencies
}
```

**Python Script:** `packages/fine-tune/python/export_onnx.py`
```python
# Merge MLX adapters with base model
# Export to ONNX using optimum or onnxruntime
```

---

## Part 2: React Package Modifications

### 2.1 LLMProvider Props

**File:** `packages/react/src/LLMProvider.tsx`

**No changes required** - `RuntimeConfig` is already passed through, which will include the new `adapter` field.

---

## Part 3: Example App Integration

### 3.1 Adapter Selection UI

**File:** `examples/app/src/App.tsx`

**Changes:**

```typescript
function App() {
  const [runtime, setRuntime] = useState<"webllm" | "transformers" | "api">("transformers");
  const [adapter, setAdapter] = useState<string | null>(null);

  return (
    <div>
      {/* ... existing runtime selector ... */}
      
      {runtime === "transformers" && (
        <label>
          Adapter:
          <select
            value={adapter || "none"}
            onChange={(e) => setAdapter(e.target.value === "none" ? null : e.target.value)}
          >
            <option value="none">Base Model (No Adapter)</option>
            <option value="example">Example Adapter (Local)</option>
            <option value="custom">Custom URL...</option>
          </select>
        </label>
      )}
      
      <LLMProvider
        config={{
          preferredRuntime: runtime,
          models: {
            transformers: "onnx-community/functiongemma-270m-it-ONNX",
          },
          adapter: adapter ? {
            path: adapter === "example" 
              ? "/adapters/example" 
              : promptForURL(),
            source: "url",
            mode: "merge",
          } : undefined,
        }}
      />
    </div>
  );
}
```

### 3.2 Serve Adapters

**Add:** `examples/app/public/adapters/example/`
- `adapters.safetensors` (or merged ONNX model)
- `adapter_config.json`

**File:** `examples/app/vite.config.ts`
```typescript
export default defineConfig({
  // ... existing config ...
  publicDir: 'public', // Ensure public directory is served
});
```

---

## Part 4: Alternative Approach (Simpler MVP)

Given the complexity of client-side LoRA merging, here's a **simpler alternative**:

### 4.1 Use Merged Models Directly

**Approach:**
1. Export merged model from fine-tuning (Python)
2. Convert to ONNX using `optimum-cli`
3. Push to Hugging Face Hub
4. Load in TransformersRuntime like any other model

**Changes:**

**`RuntimeConfig`:**
```typescript
models?: {
  transformers?: string; // Can point to merged model
}
```

**Example App:**
```typescript
<LLMProvider
  config={{
    preferredRuntime: "transformers",
    models: {
      // Base model
      transformers: "onnx-community/functiongemma-270m-it-ONNX",
      
      // OR fine-tuned/merged model
      transformers: "your-org/functiongemma-270m-custom-tools-ONNX",
    },
  }}
/>
```

**Benefits:**
- ✅ No runtime modifications needed
- ✅ Uses existing infrastructure
- ✅ Works today with zero code changes
- ✅ Better performance (no dynamic adapter application)

**Drawbacks:**
- ❌ Requires model re-export for each adapter
- ❌ Larger model files (can't share base model)
- ❌ Less flexible

---

## Gap Analysis

### What We Have:
✅ Fine-tuning system that creates LoRA adapters  
✅ TransformersRuntime that loads ONNX models  
✅ Export pipeline for safetensors  

### What We Need:

#### For Full LoRA Support (Advanced):
1. ❌ Client-side LoRA weight application library
2. ❌ Safetensors parser for JavaScript
3. ❌ Tensor manipulation utilities
4. ❌ Adapter caching system
5. ⚠️  Testing infrastructure for adapters

#### For Merged Model Approach (MVP):
1. ⚠️  Python script to merge adapters with base model
2. ⚠️  ONNX export from merged MLX model
3. ⚠️  Documentation for export workflow
4. ✅ Everything else exists

---

## Recommended Approach

**Phase 1: MVP (Merged Models)** ← START HERE
1. Add Python export script for merged ONNX models
2. Document workflow: fine-tune → merge → export → upload → use
3. Test with example adapter in demo app

**Phase 2: Direct Adapter Support (Future)**
1. Research/implement JS LoRA library
2. Add adapter loading to TransformersRuntime
3. Implement caching and optimization

---

## Verification Plan

### 1. Unit Tests

**New Test File:** `packages/core/src/runtime/__tests__/adapter-loading.test.ts`

```typescript
describe('Adapter Loading', () => {
  it('should load adapter from URL', async () => {
    const runtime = new TransformersRuntime();
    await runtime.initialize({
      models: { transformers: 'base-model' },
      adapter: {
        path: 'https://example.com/adapters',
        source: 'url',
      },
    });
    // Verify adapter loaded
  });
});
```

**Run:** `cd packages/core && npm test -- adapter-loading.test.ts`

### 2. Integration Test

**File:** `examples/app/src/__tests__/adapter-integration.test.tsx`

Test that fine-tuned model can be loaded and used.

### 3. Manual Verification

**Steps:**
1. Fine-tune model: `cd packages/fine-tune && npx fine-tune train ...`
2. Export to ONNX: `python export_onnx.py --adapter ./output/adapters --output ./merged`
3. Upload to HF Hub or local server
4. Update example app to use merged model
5. Test tool calling with custom tools
6. Verify model uses fine-tuned behavior

**Expected:** Model should correctly call custom tools that were in training data.

---

## Open Questions  

1. **ONNX Export:** Does MLX → ONNX export preserve LoRA-merged weights correctly?
2. **Model Size:** What's the size difference between base (270MB) and merged models?
3. **HF Hub:** Should we provide a reference merged model for testing?
4. **Fallback:** What happens if adapter fails to load? (Continue with base model)
5. **Caching:** Should we cache merged models in IndexedDB?

---

## Implementation Tasks

### MVP (Merged Model Approach):

- [ ] **1.1** Add `export_onnx.py` Python script
  - Merge MLX adapters with base model
  - Export to ONNX using optimum
  
- [ ] **1.2** Add ONNX export to `ModelExporter`
  - Add `exportONNX()` method
  - Wire to CLI `export` command
  
- [ ] **1.3** Test ONNX export
  - Export example adapter
  - Verify model loads in Transformers.js
  - Test tool calling
  
- [ ] **1.4** Update example app
  - Add model selector dropdown
  - Support custom model URLs/IDs
  
- [ ] **1.5** Documentation
  - Add export workflow to fine-tune README
  - Document how to use merged models
  - Create example/tutorial

### Future (Direct Adapter Support):

- [ ] **2.1** Research LoRA.js libraries
- [ ] **2.2** Add `AdapterConfig` types
- [ ] **2.3** Implement adapter fetch logic
- [ ] **2.4** Implement LoRA weight application
- [ ] **2.5** Add adapter caching
- [ ] **2.6** Comprehensive testing

---

## Timeline Estimate

**MVP:** 2-3 hours
- Python ONNX export script: 1 hour
- Testing and debugging: 1 hour
- Example app integration: 30 min
- Documentation: 30 min

**Full LoRA Support:** 1-2 days
- Research: 2-4 hours
- Implementation: 4-6 hours  
- Testing: 2-3 hours
- Documentation: 1 hour
