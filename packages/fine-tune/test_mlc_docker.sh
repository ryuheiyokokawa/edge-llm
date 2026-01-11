#!/bin/bash
# Test MLC model inference using Docker container
# This mimics what WebLLM does in the browser, but runs on CPU via Docker
#
# Usage: ./test_mlc_docker.sh [model_path] [prompt]

MODEL_PATH="${1:-working/mlc-model-x86}"
PROMPT="${2:-What is 5*12?}"

echo "============================================================"
echo "🧪 MLC-LLM Docker Test Fixture"
echo "============================================================"
echo "Model Path: $MODEL_PATH"
echo "Test Prompt: $PROMPT"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH/mlc-chat-config.json" ]; then
    echo "❌ mlc-chat-config.json not found at $MODEL_PATH"
    exit 1
fi

# Check for WASM file
if [ -f "$MODEL_PATH/model-webgpu.wasm" ]; then
    WASM_SIZE=$(ls -lh "$MODEL_PATH/model-webgpu.wasm" | awk '{print $5}')
    echo "✅ Found model-webgpu.wasm: $WASM_SIZE"
else
    echo "⚠️  No model-webgpu.wasm found (weights-only export)"
fi

# Parse model config
echo ""
echo "📋 Model Config:"
python3 -c "
import json
with open('$MODEL_PATH/mlc-chat-config.json') as f:
    c = json.load(f)
print(f\"   - model_type: {c.get('model_type')}\")
print(f\"   - quantization: {c.get('quantization')}\")
print(f\"   - vocab_size: {c.get('vocab_size')}\")
print(f\"   - context_window: {c.get('context_window_size')}\")
" 2>/dev/null || echo "   (Could not parse config)"

echo ""
echo "📁 Model Files:"
ls -lh "$MODEL_PATH" | tail -n +2 | while read line; do
    echo "   $line"
done

echo ""
echo "🔄 Running MLC inference in Docker (CPU mode)..."
echo "   This uses the mlc_llm Python API for testing."
echo ""

# Create a temp Python test script
TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Test MLC model inference on CPU."""
import sys
import os

# Add paths
sys.path.insert(0, "/opt/mlc-llm/python")

model_path = sys.argv[1] if len(sys.argv) > 1 else "working/mlc-model-x86"
prompt = sys.argv[2] if len(sys.argv) > 2 else "What is 5*12?"

print(f"Loading model from: {model_path}")

try:
    from mlc_llm import MLCEngine
    
    # Create engine with CPU device
    engine = MLCEngine(model_path, device="cpu")
    
    print("✅ Model loaded successfully!")
    print("")
    print("🎯 Running inference...")
    
    # Simple chat completion
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = engine.chat.completions.create(
        messages=messages,
        max_tokens=128,
        temperature=0.0,
    )
    
    print("")
    print("📝 Output:")
    print("-" * 40)
    print(response.choices[0].message.content)
    print("-" * 40)
    print("")
    print("✅ Inference complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

# Run in Docker
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "$TEST_SCRIPT:/tmp/test_mlc.py" \
    mlc-compiler \
    python3 /tmp/test_mlc.py "/workspace/$MODEL_PATH" "$PROMPT"

EXIT_CODE=$?
rm -f "$TEST_SCRIPT"

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test complete!"
else
    echo "❌ Test failed with exit code $EXIT_CODE"
fi
echo "============================================================"

exit $EXIT_CODE
