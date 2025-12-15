"""
Instruction and chat formatting abstraction.

Provides pluggable prompt formats, role-based message handling,
and format abstraction for fine-tuning once, deploying many personas.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PromptFormat:
    """Base class for prompt formats."""
    
    def format_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Format messages into prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError
    
    def parse_response(
        self,
        generated_text: str,
    ) -> str:
        """
        Parse response from generated text.
        
        Args:
            generated_text: Generated text
            
        Returns:
            Extracted response
        """
        raise NotImplementedError


class ChatMLFormat(PromptFormat):
    """ChatML format (OpenAI-style)."""
    
    def format_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format messages in ChatML format."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        return "\n".join(formatted)
    
    def parse_response(self, generated_text: str) -> str:
        """Parse response from ChatML format."""
        # Extract content after last assistant tag
        if "<|im_start|>assistant" in generated_text:
            parts = generated_text.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1].split("<|im_end|>")[0].strip()
                return response
        return generated_text


class AlpacaFormat(PromptFormat):
    """Alpaca instruction format."""
    
    def format_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format messages in Alpaca format."""
        # Alpaca format: instruction + response
        instruction = ""
        response = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user" or role == "system":
                instruction += content + "\n"
            elif role == "assistant":
                response = content
        
        formatted = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n{response}"
        return formatted
    
    def parse_response(self, generated_text: str) -> str:
        """Parse response from Alpaca format."""
        if "### Response:" in generated_text:
            parts = generated_text.split("### Response:")
            if len(parts) > 1:
                return parts[-1].strip()
        return generated_text


class InstructionFormatter:
    """
    Pluggable instruction formatter.
    
    Supports multiple prompt formats and role-based handling.
    """
    
    def __init__(
        self,
        format_type: str = "chatml",
        custom_format: Optional[PromptFormat] = None,
    ):
        """
        Initialize instruction formatter.
        
        Args:
            format_type: Format type (chatml, alpaca, custom)
            custom_format: Custom format implementation
        """
        if custom_format:
            self.format = custom_format
        elif format_type == "chatml":
            self.format = ChatMLFormat()
        elif format_type == "alpaca":
            self.format = AlpacaFormat()
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def format_instruction(
        self,
        instruction: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Format instruction with optional context and examples.
        
        Args:
            instruction: Instruction text
            context: Optional context
            examples: Optional few-shot examples
            
        Returns:
            Formatted prompt
        """
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": context,
            })
        
        if examples:
            for example in examples:
                messages.append({
                    "role": "user",
                    "content": example.get("input", ""),
                })
                messages.append({
                    "role": "assistant",
                    "content": example.get("output", ""),
                })
        
        messages.append({
            "role": "user",
            "content": instruction,
        })
        
        return self.format.format_messages(messages)
    
    def format_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Format conversation history.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Formatted conversation
        """
        return self.format.format_messages(messages)
    
    def parse_response(self, generated_text: str) -> str:
        """Parse response from generated text."""
        return self.format.parse_response(generated_text)
    
    def create_instruction_mask(
        self,
        formatted_text: str,
        tokenizer_manager,
    ) -> List[int]:
        """
        Create mask for instruction vs response tokens.
        
        Args:
            formatted_text: Formatted text
            tokenizer_manager: Tokenizer manager
            
        Returns:
            List of 0 (instruction) or 1 (response) for each token
        """
        # Split instruction and response
        if "### Response:" in formatted_text:
            parts = formatted_text.split("### Response:")
            instruction_text = parts[0] + "### Response:"
        elif "<|im_start|>assistant" in formatted_text:
            parts = formatted_text.split("<|im_start|>assistant")
            instruction_text = parts[0] + "<|im_start|>assistant"
        else:
            # Assume all instruction
            instruction_text = formatted_text
        
        # Tokenize
        instruction_tokens = tokenizer_manager.encode(instruction_text, add_bos=True, add_eos=False)
        all_tokens = tokenizer_manager.encode(formatted_text, add_bos=True, add_eos=True)
        
        # Create mask
        mask = [0] * len(instruction_tokens) + [1] * (len(all_tokens) - len(instruction_tokens))
        
        return mask[:len(all_tokens)]

