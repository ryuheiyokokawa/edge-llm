/** @type {import('jest').Config} */
export default {
  preset: "ts-jest/presets/default-esm",
  testEnvironment: "jsdom",
  extensionsToTreatAsEsm: [".ts", ".tsx"],
  moduleNameMapper: {
    "^(\\.{1,2}/.*)\\.js$": "$1",
    "^@edge-llm/core$": "<rootDir>/../../packages/core/src/index.ts",
    "^@edge-llm/react$": "<rootDir>/../../packages/react/src/index.ts",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        useESM: true,
        tsconfig: {
          jsx: "react-jsx",
        },
      },
    ],
  },
  transformIgnorePatterns: [
    "node_modules/(?!(@mlc-ai/web-llm|@huggingface/transformers)/)"
  ],
  setupFilesAfterEnv: ["<rootDir>/src/test/setup.ts"],
  testMatch: [
    "**/__tests__/**/*.test.{ts,tsx}",
    "**/?(*.)+(spec|test).{ts,tsx}",
  ],
};
