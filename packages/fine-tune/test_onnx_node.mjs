#!/usr/bin/env node
/**
 * Node.js test fixture for validating ONNX models with Transformers.js
 * This mimics what the browser runtime does, but in a more debuggable environment.
 * 
 * Usage: node test_onnx_node.mjs <model_path> [prompt]
 */

import { pipeline, env } from '@huggingface/transformers';
import path from 'path';
import fs from 'fs';

// Disable remote models for local testing
env.allowLocalModels = true;
env.allowRemoteModels = false;

const modelPath = process.argv[2];
const testPrompt = process.argv[3] || "What is 5*12?";

if (!modelPath) {
    console.error("Usage: node test_onnx_node.mjs <model_path> [prompt]");
    console.error("Example: node test_onnx_node.mjs ./working/onnx-model-q4-int8");
    process.exit(1);
}

const absolutePath = path.resolve(modelPath);

console.log("=".repeat(60));
console.log("🧪 Transformers.js Node.js Test Fixture");
console.log("=".repeat(60));
console.log(`Model Path: ${absolutePath}`);
console.log(`Test Prompt: ${testPrompt}`);
console.log("");

// Check model files exist
const configPath = path.join(absolutePath, 'config.json');
if (!fs.existsSync(configPath)) {
    console.error(`❌ config.json not found at ${configPath}`);
    process.exit(1);
}

const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
console.log("📋 Model Config:");
console.log(`   - model_type: ${config.model_type}`);
console.log(`   - architectures: ${config.architectures?.join(', ')}`);
console.log(`   - use_cache: ${config.use_cache}`);
console.log(`   - transformers_version: ${config.transformers_version}`);
console.log(`   - has transformers.js_config: ${!!config['transformers.js_config']}`);
console.log("");

// Check ONNX files
const onnxDir = path.join(absolutePath, 'onnx');
const hasOnnxSubdir = fs.existsSync(onnxDir);
console.log("📁 ONNX Structure:");
console.log(`   - Has onnx/ subdirectory: ${hasOnnxSubdir}`);

if (hasOnnxSubdir) {
    const onnxFiles = fs.readdirSync(onnxDir);
    onnxFiles.forEach(f => {
        const stats = fs.statSync(path.join(onnxDir, f));
        const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
        console.log(`   - ${f}: ${sizeMB} MB`);
    });
} else {
    // Check for model.onnx in root
    const modelFile = path.join(absolutePath, 'model.onnx');
    if (fs.existsSync(modelFile)) {
        const stats = fs.statSync(modelFile);
        const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
        console.log(`   - model.onnx (root): ${sizeMB} MB`);
    }
}
console.log("");

// Try to load the pipeline
console.log("🔄 Loading pipeline...");
try {
    const generator = await pipeline('text-generation', absolutePath, {
        local_files_only: true,
        progress_callback: (progress) => {
            if (progress.status === 'progress') {
                process.stdout.write(`\r   Loading: ${progress.file} - ${Math.round(progress.progress)}%`);
            } else if (progress.status === 'done') {
                console.log(`\n   ✅ Loaded: ${progress.file}`);
            }
        }
    });
    
    console.log("\n✅ Pipeline loaded successfully!");
    console.log("");
    
    // Test generation
    console.log("🎯 Running inference...");
    const messages = [
        { role: "system", content: "You are a helpful assistant that can do function calling." },
        { role: "user", content: testPrompt }
    ];
    
    const result = await generator(messages, {
        max_new_tokens: 128,
        do_sample: false,
        return_full_text: false
    });
    
    console.log("✅ Inference complete!");
    console.log("");
    console.log("📝 Output:");
    console.log("-".repeat(40));
    console.log(result[0]?.generated_text || result);
    console.log("-".repeat(40));
    
} catch (error) {
    console.error("\n❌ Pipeline failed:");
    console.error(error.message);
    console.error("");
    console.error("Stack trace:");
    console.error(error.stack);
    process.exit(1);
}

console.log("");
console.log("=".repeat(60));
console.log("✅ Test complete!");
