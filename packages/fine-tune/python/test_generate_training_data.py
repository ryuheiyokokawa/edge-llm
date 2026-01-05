#!/usr/bin/env python3
"""
Tests for generate_training_data.py formatting functions.

These tests ensure the FunctionGemma format is correctly generated.
Run with: python -m pytest python/test_generate_training_data.py -v
"""

import json
import pytest
import sys
from pathlib import Path

# Add python dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_training_data import (
    format_tool_declaration,
    format_function_call,
    generate_training_sample,
)


class TestFormatToolDeclaration:
    """Tests for format_tool_declaration()."""

    def test_basic_tool(self):
        """Test basic tool with one required parameter."""
        tool = {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression"
                    }
                },
                "required": ["expression"]
            }
        }
        
        result = format_tool_declaration(tool)
        
        # Check structure
        assert result.startswith("<start_function_declaration>declaration:calculate{")
        assert result.endswith("}<end_function_declaration>")
        
        # Check description
        assert "description:<escape>Evaluate a mathematical expression<escape>" in result
        
        # Check parameter
        assert "expression:{description:<escape>Math expression<escape>" in result
        assert "type:<escape>STRING<escape>" in result
        
        # Check required
        assert "required:[<escape>expression<escape>]" in result

    def test_tool_with_multiple_params(self):
        """Test tool with multiple parameters."""
        tool = {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"}
                },
                "required": ["query"]
            }
        }
        
        result = format_tool_declaration(tool)
        
        assert "declaration:search{" in result
        assert "query:{description:<escape>Search query<escape>" in result
        assert "limit:{description:<escape>Max results<escape>" in result
        assert "required:[<escape>query<escape>]" in result

    def test_tool_with_enum(self):
        """Test tool with enum parameter."""
        tool = {
            "name": "set_unit",
            "description": "Set temperature unit",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["unit"]
            }
        }
        
        result = format_tool_declaration(tool)
        
        assert "enum:[<escape>celsius<escape>,<escape>fahrenheit<escape>]" in result

    def test_tool_with_no_parameters(self):
        """Test tool with no parameters."""
        tool = {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {}
        }
        
        result = format_tool_declaration(tool)
        
        assert "declaration:get_time{" in result
        assert "parameters:{}" in result
        assert "required:[]" in result


class TestFormatFunctionCall:
    """Tests for format_function_call()."""

    def test_basic_call(self):
        """Test basic function call with string argument."""
        result = format_function_call("calculate", {"expression": "5 * 12"})
        
        assert result == "<start_function_call>call:calculate{expression:<escape>5 * 12<escape>}<end_function_call>"

    def test_call_with_multiple_args(self):
        """Test function call with multiple arguments."""
        result = format_function_call("search", {"query": "weather", "limit": 5})
        
        assert result.startswith("<start_function_call>call:search{")
        assert result.endswith("}<end_function_call>")
        assert "query:<escape>weather<escape>" in result
        assert "limit:<escape>5<escape>" in result

    def test_call_with_dict_arg(self):
        """Test function call with dict argument (should be JSON serialized)."""
        result = format_function_call("update", {"data": {"key": "value"}})
        
        # Dict should be JSON-serialized
        assert 'data:<escape>{"key": "value"}<escape>' in result

    def test_call_with_list_arg(self):
        """Test function call with list argument (should be JSON serialized)."""
        result = format_function_call("process", {"items": [1, 2, 3]})
        
        # List should be JSON-serialized  
        assert "items:<escape>[1, 2, 3]<escape>" in result

    def test_call_with_empty_args(self):
        """Test function call with no arguments."""
        result = format_function_call("get_time", {})
        
        assert result == "<start_function_call>call:get_time{}<end_function_call>"


class TestGenerateTrainingSample:
    """Tests for generate_training_sample()."""

    def test_basic_sample(self):
        """Test generating a complete training sample."""
        tools = [{
            "name": "calculate",
            "description": "Math",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression"}
                },
                "required": ["expression"]
            }
        }]
        
        example = {
            "userQuery": "What is 5 * 12?",
            "expectedToolCalls": [{"name": "calculate", "arguments": {"expression": "5 * 12"}}]
        }
        
        result = generate_training_sample(example, tools)
        
        assert result is not None
        assert "text" in result
        
        text = result["text"]
        
        # Check conversation structure
        assert text.startswith("<bos><start_of_turn>developer")
        assert "<end_of_turn>" in text
        assert "<start_of_turn>user" in text
        assert "What is 5 * 12?" in text
        assert "<start_of_turn>model" in text
        
        # Check tool call in model response
        assert "<start_function_call>call:calculate{expression:<escape>5 * 12<escape>}<end_function_call>" in text

    def test_sample_with_no_tool_calls(self):
        """Test that example with no tool calls returns None."""
        tools = [{"name": "test", "description": "Test", "parameters": {}}]
        
        example = {
            "userQuery": "Hello",
            "expectedToolCalls": []
        }
        
        result = generate_training_sample(example, tools)
        
        assert result is None

    def test_sample_includes_all_tools(self):
        """Test that all tool declarations are included in system prompt."""
        tools = [
            {"name": "tool1", "description": "First tool", "parameters": {}},
            {"name": "tool2", "description": "Second tool", "parameters": {}},
        ]
        
        example = {
            "userQuery": "Do something",
            "expectedToolCalls": [{"name": "tool1", "arguments": {}}]
        }
        
        result = generate_training_sample(example, tools)
        text = result["text"]
        
        # Both tools should be declared
        assert "declaration:tool1{" in text
        assert "declaration:tool2{" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
