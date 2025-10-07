# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .utils.router_context import conversation_context
from .utils.router_utils import MODEL_SPECS, LLMRouter, acall_model


class RouterTool(BaseTool):
    """A tool for routing requests to multiple tools for intelligent processing."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        # Default configuration for GPT-4o calls
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1024)
        self.api_key = config.get("api_key")  # Will use env var if not provided

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    def _tool_name_to_model_id(self, tool_name: str) -> str:
        """Convert tool name back to model ID."""
        # Extract model_id from tool_name (convert back from call_xxx format)
        model_id = tool_name.replace("call_", "")
        
        # Handle special cases for dots in model names
        # Convert back by checking against available models
        for available_model_id in MODEL_SPECS.keys():
            # Create the expected function name for this model
            expected_func_name = available_model_id.replace('-', '_').replace('.', '_')
            if model_id == expected_func_name:
                return available_model_id
        
        # Fallback to simple underscore to hyphen conversion
        model_id = model_id.replace("_", "-")
        return model_id

    async def create(self, instance_id: Optional[str] = None, messages: List[Dict[str, Any]] = [], **kwargs) -> str:
        """Create a tool instance for a conversation."""
        if instance_id is None:
            instance_id = str(uuid4())
        conversation_context.clear_context(instance_id)
        for msg in messages:
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to=msg.role, # user or system
                arguments=msg.content
            )
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute a GPT call with the provided parameters."""
        try:
            # Extract parameters
            optimized_system_prompt = parameters.get("optimized_system_prompt", "")
            max_tokens = parameters.get("max_tokens", self.default_max_tokens)
            temperature = parameters.get("temperature", self.default_temperature)
            
            if not optimized_system_prompt:
                raise ValueError("optimized_system_prompt is required when calling the tool")

            # Get the conversation context, might be for future use
            request_conversation_context = conversation_context.get_context(instance_id)
            
            # TODO: concatenate all the user messages
            assert request_conversation_context[0]["route_to"] == "system", "The first message should be the system prompt"
            initial_user_message = request_conversation_context[1]["arguments"]
            if "USER SYSTEM:" in initial_user_message:
                system_prompt = initial_user_message.split("USER SYSTEM:")[1].split("USER QUERY:")[0].strip()
                user_prompt = initial_user_message.split("USER QUERY:")[1].strip()
                messages = [{"role": "system", "content": optimized_system_prompt}, {"role": "user", "content": user_prompt}]
            else:
                user_prompt = initial_user_message.split("USER QUERY:")[1].strip()
                messages = [{"role": "system", "content": optimized_system_prompt}, {"role": "user", "content": user_prompt}]

            # Get the actual model ID from the tool name
            model_id = self._tool_name_to_model_id(self.name)
            
            # Prepare sampling parameters
            sampling_params = {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            router = LLMRouter()    
            response, metadata = await acall_model(router, model_id, messages, sampling_params)
            
            # Append interaction to conversation context
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to=model_id,
                arguments=parameters,
                feedback=response,
                metadata=metadata,
                tool_id=self.name,
            )
            
            # Calculate reward
            reward = 0.0
            metadata["success"] = True
            
            return response, reward, model_id, parameters, metadata
            
        except Exception as e:
            # Record failed attempt
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to=model_id,
                arguments=parameters,
                feedback=f"Error: {str(e)}",
                metadata={"success": False, "error": str(e)},
                tool_id=self.name,
            )
            
            return f"Error calling {self.name}: {str(e)}", 0.0, model_id, parameters, {"success": False, "error": str(e)}

    async def release(self, instance_id: str, **kwargs) -> None:
        # Release the context
        conversation_context.clear_context(instance_id)


class SelectTool(BaseTool):
    """A tool for selecting a previous used tool's feedback."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    async def create(self, instance_id: Optional[str] = None, messages: List[Dict[str, Any]] = [], **kwargs) -> str:
        return instance_id
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            # Extract parameters
            selected_model_call_name = parameters.get("model_call_name", "")

            if not selected_model_call_name:
                raise ValueError("model_call_name is required when calling the tool")
            
            request_conversation_context = conversation_context.get_context(instance_id)
            
            found = False
            for context in request_conversation_context[::-1]:
                if "tool_id" in context and context["tool_id"] == selected_model_call_name:
                    found = True
                    response = context["feedback"]
                    break

            if not found:
                raise ValueError(f"{selected_model_call_name} not found in the conversation context.")

            conversation_context.append_interaction(
                request_id=instance_id,
                route_to="user",
                arguments=parameters,
                feedback=response,
                metadata={"success": True},
                tool_id=self.name,
            )
            return response, 0.0, "user", parameters, {"success": True}
        
        except Exception as e:
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to="user",
                arguments=parameters,
                feedback=f"Error: {str(e)}",
                metadata={"success": False, "error": str(e)},
                tool_id=self.name,
            )
            return f"Error calling select_response: {str(e)}", 0.0, "user", parameters, {"success": False, "error": str(e)}

    
    async def release(self, instance_id: str, **kwargs) -> None:
        # Release the context
        conversation_context.clear_context(instance_id)