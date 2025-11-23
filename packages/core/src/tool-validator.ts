/**
 * Tool result formatting and validation
 */
import type { ToolDefinition, ToolCall, JSONSchema } from "./types.js";

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  formatted: unknown;
}

export class ToolValidator {
  /**
   * Validate tool call arguments against schema
   */
  static validateArguments(
    call: ToolCall,
    tool: ToolDefinition
  ): ValidationResult {
    const errors: string[] = [];
    const args = call.arguments;

    // Check required parameters
    if (tool.parameters.required) {
      for (const required of tool.parameters.required) {
        if (!(required in args) || args[required] === undefined) {
          errors.push(`Missing required parameter: ${required}`);
        }
      }
    }

    // Validate parameter types
    if (tool.parameters.properties) {
      for (const [key, value] of Object.entries(args)) {
        const schema = tool.parameters.properties[key];
        if (!schema) {
          // Check if additionalProperties is allowed
          if (tool.parameters.additionalProperties === false) {
            errors.push(`Unknown parameter: ${key}`);
          }
          continue;
        }

        const validationError = this.validateValue(value, schema, key);
        if (validationError) {
          errors.push(validationError);
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      formatted: args,
    };
  }

  /**
   * Validate a single value against a schema property
   */
  private static validateValue(
    value: unknown,
    schema: JSONSchema["properties"][string],
    key: string
  ): string | null {
    // Type validation
    if (schema.type === "string" && typeof value !== "string") {
      return `Parameter ${key} must be a string, got ${typeof value}`;
    }
    if (schema.type === "number" && typeof value !== "number") {
      return `Parameter ${key} must be a number, got ${typeof value}`;
    }
    if (schema.type === "boolean" && typeof value !== "boolean") {
      return `Parameter ${key} must be a boolean, got ${typeof value}`;
    }
    if (schema.type === "array" && !Array.isArray(value)) {
      return `Parameter ${key} must be an array, got ${typeof value}`;
    }
    if (schema.type === "object" && (typeof value !== "object" || value === null || Array.isArray(value))) {
      return `Parameter ${key} must be an object, got ${typeof value}`;
    }

    // Enum validation
    if (schema.enum && !schema.enum.includes(value as string | number)) {
      return `Parameter ${key} must be one of: ${schema.enum.join(", ")}`;
    }

    // Nested object validation
    if (schema.type === "object" && schema.properties && typeof value === "object" && value !== null) {
      const obj = value as Record<string, unknown>;
      for (const [nestedKey, nestedValue] of Object.entries(obj)) {
        const nestedSchema = schema.properties?.[nestedKey];
        if (nestedSchema) {
          const nestedError = this.validateValue(nestedValue, nestedSchema, `${key}.${nestedKey}`);
          if (nestedError) {
            return nestedError;
          }
        }
      }
    }

    // Array item validation
    if (schema.type === "array" && schema.items && Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        const itemError = this.validateValue(value[i], schema.items, `${key}[${i}]`);
        if (itemError) {
          return itemError;
        }
      }
    }

    return null;
  }

  /**
   * Format tool result for model consumption
   */
  static formatResult(result: unknown): string {
    if (result === null || result === undefined) {
      return "null";
    }

    // If already a string, return as-is (might be JSON)
    if (typeof result === "string") {
      return result;
    }

    // Try to stringify, with error handling
    try {
      return JSON.stringify(result);
    } catch (error) {
      return String(result);
    }
  }

  /**
   * Validate and format tool result
   */
  static validateAndFormat(
    call: ToolCall,
    tool: ToolDefinition,
    result: unknown
  ): { valid: boolean; formatted: string; errors: string[] } {
    // Validate arguments first
    const argValidation = this.validateArguments(call, tool);
    
    // Format result
    const formatted = this.formatResult(result);

    return {
      valid: argValidation.valid,
      formatted,
      errors: argValidation.errors,
    };
  }
}

