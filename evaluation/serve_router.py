#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-Style Router Server

This module implements an OpenAI-compatible API server that acts as an intelligent
LLM router. It analyzes incoming requests and routes them to the most appropriate
model from a curated set of available models.

Features:
- OpenAI-compatible API endpoints (/v1/chat/completions)
- Intelligent routing using gpt-5-nano by default
- Support for FIXED_MODEL_SET_2 models as available tools
- Cost-optimal model selection
- Agentic prompt engineering for each target model

Usage:
    python serve_router.py --port 8000
    
    # Then call via OpenAI client:
    import openai
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    response = client.chat.completions.create(
        model="router",
        messages=[{"role": "user", "content": "What is 2+2?"}]
    )
"""

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from data_preprocess.router_data_preprocess import (FIXED_MODEL_SET_1,
                                                    FIXED_MODEL_SET_2,
                                                    FIXED_MODEL_SET_3,
                                                    SYSTEM_PROMPT,
                                                    create_openai_tools)
# Import router utilities and model specifications
from verl.tools.utils.router_utils import MODEL_SPECS, LLMRouter, acall_model

app = FastAPI(title="Router API", version="1.0.0")

# Default router configuration
# DEFAULT_ROUTER_MODEL = "xRouter-7b"  # Qwen 2.5 7B: $0.04/M input, $0.10/M output
DEFAULT_ROUTER_MODEL = "gpt-5-nano"
AVAILABLE_MODELS = FIXED_MODEL_SET_1

# Router model cost specifications
ROUTER_MODEL_COSTS = {
    "xRouter-7b": {
        "input_price_per_million": 0.04,
        "output_price_per_million": 0.10
    },
    "xRouter-3b": {
        "input_price_per_million": 0.0012,
        "output_price_per_million": 0.0024
    },
    "gpt-5-nano": {
        "input_price_per_million": 0.05,  # From MODEL_SPECS
        "output_price_per_million": 0.40
    }
}

# Filter available models to only include those in MODEL_SPECS
FILTERED_AVAILABLE_MODELS = [
    model_id for model_id in AVAILABLE_MODELS 
    if model_id in MODEL_SPECS
]

print(f"Router initialized with {len(FILTERED_AVAILABLE_MODELS)} available models:")
for model_id in FILTERED_AVAILABLE_MODELS:
    spec = MODEL_SPECS[model_id]
    print(f"  - {model_id}: {spec.name} ({spec.quality_tier})")


def calculate_router_cost(router_model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for router model based on token usage.
    
    Args:
        router_model: Router model identifier
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        
    Returns:
        Cost in USD
    """
    # Check if router model is in MODEL_SPECS (e.g., gpt-5-nano)
    if router_model in MODEL_SPECS:
        spec = MODEL_SPECS[router_model]
        input_cost = (input_tokens / 1_000_000) * spec.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * spec.output_price_per_million
        return input_cost + output_cost
    
    # Check if router model is in ROUTER_MODEL_COSTS (e.g., qwen2.5-7b)
    elif router_model in ROUTER_MODEL_COSTS:
        costs = ROUTER_MODEL_COSTS[router_model]
        input_cost = (input_tokens / 1_000_000) * costs["input_price_per_million"]
        output_cost = (output_tokens / 1_000_000) * costs["output_price_per_million"]
        return input_cost + output_cost
    
    else:
        # Fallback: use qwen2.5-7b costs for unknown models
        print(f"Warning: Unknown router model '{router_model}', using xRouter-7b costs as fallback")
        costs = ROUTER_MODEL_COSTS["xRouter-7b"]
        input_cost = (input_tokens / 1_000_000) * costs["input_price_per_million"]
        output_cost = (output_tokens / 1_000_000) * costs["output_price_per_million"]
        return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count"""
    return len(text) // 4  # Rough approximation: 4 chars per token


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


class RouterAgent:
    """
    Intelligent router agent that analyzes queries and selects optimal models.
    """
    
    def __init__(self, router_model: str = DEFAULT_ROUTER_MODEL, max_turns: int = 3, hosted_port: Optional[int] = None):
        self.router_model = router_model
        self.max_turns = max_turns
        self.hosted_port = hosted_port
        self.available_models = FILTERED_AVAILABLE_MODELS
        self.tools = create_openai_tools(self.available_models)
        self.router_system = LLMRouter()
        self.router_costs = 0.0  # Initialize router costs at instance level
        
        # Configure base URL for self-hosted models
        if hosted_port is not None:
            self.base_url = f"http://localhost:{hosted_port}/v1"
            print(f"Router model {router_model} will be accessed via self-hosted endpoint: {self.base_url}")
        else:
            self.base_url = None
            print(f"Router model {router_model} will be accessed via default API")
        
        # Add select_response tool to the tools list
        select_tool = {
            "type": "function",
            "function": {
                "name": "select_response",
                "description": (
                    "Select the best response from previously executed model calls to provide as the final answer.\n\n"
                    "Workflow:\n"
                    "1. First, call one or more model tools (e.g., call_gpt_4o, call_deepseek_r1) to generate responses\n"
                    "2. Compare and evaluate the quality of different model responses\n"
                    "3. Use this tool to select the most appropriate response as your final answer\n\n"
                    "Requirements:\n"
                    "- Only call this tool AFTER executing other model tools\n"
                    "- The model_call_name must match exactly with a previously called function name\n"
                    "- Function names always start with 'call_'.\n\n"
                    "The selected response will be returned to the user as the definitive answer to their question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_call_name": {
                            "type": "string",
                            "description": "Exact name of the previously called model function whose response you want to select. Must start with 'call_' and match a function that was already executed (e.g., 'call_gpt_4o', 'call_deepseek_r1')."
                        }
                    },
                    "required": ["model_call_name"]
                }
            }
        }
        self.tools.append(select_tool)
        
        print(f"Router agent initialized with router model: {router_model}")
        print(f"Max turns: {max_turns}")
        print(f"Available tools: {len(self.tools)}")
    
    async def route_request(self, messages: List[Dict[str, str]], 
                          max_tokens: Optional[int] = None, 
                          temperature: float = 0.0) -> Dict[str, Any]:
        """
        Route a request through the intelligent routing system with agentic loop.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing final response and metadata
        """
        start_time = time.time()
        call_history = []
        self.router_costs = 0.0  # Reset router model costs for this request
        

        # Helper function to handle fallback responses (timeout/max turns/failure)
        def create_fallback_response(strategy: str, turns_used: int, last_model_response=None, last_call=None):
            if last_model_response and last_call:
                # Use last successful response
                content = last_model_response
                model_used = last_call["model_id"]
                if strategy == "max_turns_reached":
                    print(f"🔚 Max turns reached, returning last model response.")
                elif strategy.startswith("timeout"):
                    print(f"🔚 Timeout reached, returning last model response.")
                strategy_suffix = "last_response" if not strategy.endswith("_reached") else strategy
            else:
                # Fallback to error message
                if strategy == "max_turns_reached":
                    content = "No valid solution found."
                    print(f"❌ No valid solution returned after {self.max_turns} turns.")
                elif strategy.startswith("timeout"):
                    content = "The solution is not found or not valid. Request timed out after 540 seconds."
                    print(f"❌ Timeout after 540 seconds, no valid solution found.")
                else:
                    content = "No valid solution found."
                    print(f"❌ No valid solution found.")
                model_used = None
                strategy_suffix = "fallback" if not strategy.endswith("_reached") else strategy
            
            return {
                "content": content,
                "model_used": model_used,
                "total_cost": sum(call.get("cost", 0) for call in call_history) + self.router_costs,
                "router_cost": self.router_costs,
                "target_models_cost": sum(call.get("cost", 0) for call in call_history),
                "total_time": time.time() - start_time,
                "routing_strategy": strategy_suffix,
                "call_history": call_history,
                "turns_used": turns_used
            }

        # TODO: add user system prompt to user query
        # TODO: handle max tokens
        # TODO: Add logic from data_preprocess/router_data_preprocess.py to handle system prompt and user query formatting

        # Choose system prompt (no simple_mode here, so always use SYSTEM_PROMPT)
        router_system_prompt = SYSTEM_PROMPT

        # Copy messages to avoid mutating input
        user_messages = [dict(m) for m in messages]
        user_system = ""
        user_query = ""
        for msg in user_messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                # If the system prompt is too long, we consider it as a user query
                if len(system_content) > 4096:
                    user_query += system_content + "\n"
                else:
                    user_system += system_content + "\n"
            elif msg["role"] == "user":
                user_query += msg["content"] + "\n"

        user_system = "USER SYSTEM: \n" + user_system.strip() if user_system.strip() else ""
        user_query = "USER QUERY: \n" + user_query.strip() if user_query.strip() else ""
        router_user_prompt = (user_system + "\n" + user_query).strip()

        router_messages = [
            {"role": "system", "content": router_system_prompt},
            {"role": "user", "content": router_user_prompt}
        ]

        print(f"📝 Initial user message converted for router:")
        print(router_user_prompt)
        
        current_turn = 0
        last_model_response = None
        last_call = None
        
        try:
            # Agentic loop with timeout wrapper - execute with 500s timeout
            async def agentic_routing():
                nonlocal current_turn, last_model_response, last_call
                
                while current_turn < self.max_turns:
                    print(f"🔄 Router Turn {current_turn + 1}/{self.max_turns}")
                    
                    # Step 1: Get routing decision from router model
                    print(f"🤖 Calling router model: {self.router_model}")

                    # Prepare tools for router model call
                    router_params = {
                        "model": "openai/" + self.router_model,
                        "messages": router_messages,
                        "tools": self.tools,
                        "tool_choice": "auto",
                        "drop_params": True  # Let litellm handle parameter compatibility
                    }

                    # Handle temperature and max_tokens for different models
                    if self.router_model.startswith("gpt-5"):
                        router_params["temperature"] = 1.0  # GPT-5 only supports temperature=1.0
                        # outer_params["max_completion_tokens"] = 16384
                    else:
                        router_params["temperature"] = 0.0
                        # router_params["max_tokens"] = 16384
                    
                    # Add custom base URL for self-hosted models
                    if self.base_url is not None:
                        router_params["base_url"] = self.base_url
                        router_params["api_key"] = "dummy"  # Self-hosted models usually don't require real API keys
                    
                    router_response = litellm.completion(**router_params)
                    
                    # Track router model costs
                    if hasattr(router_response, 'usage') and router_response.usage:
                        router_input_tokens = getattr(router_response.usage, 'prompt_tokens', 0)
                        router_output_tokens = getattr(router_response.usage, 'completion_tokens', 0)
                    else:
                        # Fallback estimation if usage not available
                        router_input_tokens = estimate_tokens(" ".join([msg["content"] for msg in router_messages]))
                        router_output_tokens = estimate_tokens(router_response.choices[0].message.content or "")
                    
                    router_call_cost = calculate_router_cost(self.router_model, router_input_tokens, router_output_tokens)
                    self.router_costs += router_call_cost
                    
                    print(f"🤖 Router model {self.router_model} used {router_input_tokens} input + {router_output_tokens} output tokens, cost: ${router_call_cost:.6f}")
                    
                    # Parse router response
                    if not router_response.choices or not router_response.choices[0].message:
                        raise HTTPException(
                            status_code=500,
                            detail="Router model returned empty response"
                        )
                    
                    choice = router_response.choices[0]
                    message = choice.message
                    
                    # Check if router made tool calls
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls = message.tool_calls
                        
                        # Add assistant message with tool calls to conversation
                        router_messages.append({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                } for tc in tool_calls
                            ]
                        })
                        
                        # Process each tool call
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            arguments = json.loads(tool_call.function.arguments)
                            call_id = tool_call.id
                            
                            print(f"🔧 Router selected tool: {function_name}")
                            print(f"📝 Arguments: {arguments}")
                            
                            # Handle select_response tool
                            if function_name == "select_response":
                                # Find the call with the specified ID, the name starts with call_
                                selected_model_call_name = arguments.get("model_call_name", None)
                                selected_call = None
                                
                                for call in call_history[::-1]:
                                    if selected_model_call_name and call.get("function_name") == selected_model_call_name:
                                        selected_call = call
                                        break
                                
                                if selected_call:
                                    # if a call is selected, directly return it
                                    call_history.append({
                                        "call_id": call_id,
                                        "model_id": None,
                                        "function_name": "select_response",
                                        "arguments": arguments,
                                        "response": None,
                                        "cost": 0,
                                        "tokens_used": 0,
                                        "latency": 0
                                    })
                                    return {
                                        "content": selected_call["response"],
                                        "model_used": selected_call["model_id"],
                                        "total_cost": sum(call.get("cost", 0) for call in call_history) + self.router_costs,
                                        "router_cost": self.router_costs,
                                        "target_models_cost": sum(call.get("cost", 0) for call in call_history),
                                        "total_time": time.time() - start_time,
                                        "routing_strategy": "select_response",
                                        "call_history": call_history,
                                        "turns_used": current_turn + 1
                                    }
                                else:
                                    # Add error tool result and continue
                                    tool_result = f"Error calling select_response: {selected_model_call_name} not found in the conversation context."
                                    router_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": tool_result
                                    })
                                    continue
                            
                            # Handle model call tools
                            elif function_name.startswith("call_"):
                                # Extract model ID from function name
                                model_id = self._extract_model_id(function_name)
                                
                                if model_id not in self.available_models:
                                    # Add error tool result and continue
                                    tool_result = f"Error calling {function_name}: Model {model_id} is not available."
                                    router_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": tool_result
                                    })
                                    continue
                                
                                # Prepare optimized system prompt and sampling params
                                optimized_system_prompt = arguments.get("optimized_system_prompt", "")
                                if not optimized_system_prompt:
                                    tool_result = f"Error calling {function_name}: optimized_system_prompt is required when calling the tool."
                                    router_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": tool_result
                                    })
                                    continue
                                
                                sampling_params = {
                                    "temperature": arguments.get("temperature", temperature or 0.7),
                                }
                                
                                initial_user_message = router_messages[1]["content"]
                                if "USER SYSTEM:" in initial_user_message:
                                    system_prompt = initial_user_message.split("USER SYSTEM:")[1].split("USER QUERY:")[0].strip()
                                    # we put the user's system prompt in the user prompt
                                    user_prompt = system_prompt + "\n" + initial_user_message.split("USER QUERY:")[1].strip()
                                    target_messages = [{"role": "system", "content": optimized_system_prompt}, {"role": "user", "content": user_prompt}]
                                else:
                                    user_prompt = initial_user_message.split("USER QUERY:")[1].strip()
                                    target_messages = [{"role": "system", "content": optimized_system_prompt}, {"role": "user", "content": user_prompt}]
                                
                                # Call the target model
                                print(f"🎯 Calling target model: {model_id}")
                                try:
                                    try:
                                        response_content, metadata = await asyncio.wait_for(
                                            acall_model(
                                                self.router_system, 
                                                model_id, 
                                                target_messages,
                                                sampling_params
                                            ),
                                            timeout=600.0
                                        )
                                    except asyncio.TimeoutError:
                                        # Simulate timeout response
                                        response_content = "Timeout when trying to call this model. Please change to call another model to get the solution in your next turn."
                                        metadata = {
                                            "model_id": model_id,
                                            "model_name": model_id,
                                            "input_tokens": sum(len(msg.get("content", "")) for msg in target_messages) // 4,
                                            "output_tokens": len(response_content) // 4,
                                            "cost": 0.0,
                                            "latency": 600.0,
                                            "provider": "timeout"
                                        }
                                    
                                    # Record the call
                                    call_record = {
                                        "call_id": call_id,
                                        "model_id": model_id,
                                        "function_name": function_name,
                                        "arguments": arguments,
                                        "response": response_content,
                                        "cost": metadata.get("cost", 0),
                                        "tokens_used": metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0),
                                        "latency": metadata.get("latency", 0)
                                    }
                                    call_history.append(call_record)
                                    last_model_response = response_content
                                    last_call = call_record
                                    
                                    # Add tool result to conversation for next turn
                                    router_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": response_content
                                    })
                                    
                                except Exception as e:
                                    print(f"❌ Error calling {function_name}: {str(e)}")
                                    # Add error tool result and continue
                                    tool_result = f"Error calling {function_name}: {str(e)}"
                                    router_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": tool_result
                                    })
                    
                    # If no tool calls, router provided a direct response - exit loop
                    elif hasattr(message, 'content') and message.content:
                        print(f"💬 Router provided direct response, no tool calls found.")
                        return {
                            "content": message.content,
                            "model_used": None,
                            "total_cost": sum(call.get("cost", 0) for call in call_history) + self.router_costs,
                            "router_cost": self.router_costs,
                            "target_models_cost": sum(call.get("cost", 0) for call in call_history),
                            "total_time": time.time() - start_time,
                            "routing_strategy": "direct_response",
                            "call_history": call_history,
                            "turns_used": current_turn + 1
                        }
                    
                    current_turn += 1
                
                # Max turns reached - use fallback helper
                return create_fallback_response("max_turns_reached", self.max_turns, last_model_response, last_call)
                    
            # Execute routing with timeout
            try:
                return await asyncio.wait_for(agentic_routing(), timeout=540.0) # 540s timeout
            except asyncio.TimeoutError:
                # Timeout handling - use fallback helper
                return create_fallback_response("timeout", current_turn, last_model_response, last_call)
                
        except Exception as e:
            print(f"❌ Router error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Router error: {str(e)}"
            )
    
    def _extract_model_id(self, function_name: str) -> str:
        """Extract model ID from function name (e.g., call_gpt_4o -> gpt-4o)"""
        if not function_name.startswith("call_"):
            return function_name
        
        model_name = function_name[5:]  # Remove 'call_' prefix
        
        # Convert underscores back to hyphens and dots
        for available_model in self.available_models:
            expected_func_name = available_model.replace('-', '_').replace('.', '_')
            if model_name == expected_func_name:
                return available_model
        
        # Fallback: simple underscore to hyphen conversion
        return model_name.replace('_', '-')


# Initialize router agent
router_agent = RouterAgent()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Router API",
        "available_models": FILTERED_AVAILABLE_MODELS,
        "router_model": DEFAULT_ROUTER_MODEL
    }

@app.get("/v1")
async def v1_root():
    return {"status":"ok","endpoints":["/v1/models","/v1/chat/completions"]}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = []
    
    # Add the router model
    models.append({
        "id": "router",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "router-api"
    })
    
    # Add available models
    for model_id in FILTERED_AVAILABLE_MODELS:
        models.append({
            "id": model_id,
            "object": "model", 
            "created": int(time.time()),
            "owned_by": "router-api"
        })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main chat completions endpoint (OpenAI-compatible)"""
    
    try:
        # Convert pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Route the request with timeout
        try:
            result = await asyncio.wait_for(
                router_agent.route_request(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature or 0.0
                ),
                timeout=670.0  # 670s timeout
            )
        except asyncio.TimeoutError:
            print("⏰ Request timeout after 670 seconds")
            # Return a timeout response
            result = {
                "content": "The solution is not found or not valid. Request timed out after 570 seconds.",
                "model_used": None,
                "total_cost": 0,
                "router_cost": 0.0,
                "target_models_cost": 0.0,
                "total_time": 670.0,
                "routing_strategy": "timeout",
                "call_history": [],
                "turns_used": 0
            }
        
        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(msg["content"]) // 4 for msg in messages),
                "completion_tokens": len(result["content"]) // 4,
                "total_tokens": sum(len(msg["content"]) // 4 for msg in messages) + len(result["content"]) // 4
            },
            "router_metadata": {
                "model_used": result["model_used"],
                "total_cost": result["total_cost"],
                "router_cost": result.get("router_cost", 0.0),
                "target_models_cost": result.get("target_models_cost", 0.0),
                "total_time": result["total_time"],
                "routing_strategy": result["routing_strategy"],
                "call_history": result["call_history"],
                "turns_used": result.get("turns_used", 0)
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/debug/tools")
async def debug_tools():
    """Debug endpoint to view available tools"""
    return {
        "available_models": FILTERED_AVAILABLE_MODELS,
        "tools_count": len(router_agent.tools),
        "tools": router_agent.tools
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Router API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8800, help="Port to bind to")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL, 
                       help="Model to use for routing decisions")
    parser.add_argument("--max-turns", type=int, default=3,
                       help="Maximum number of turns for agentic routing")
    parser.add_argument("--hosted-port", type=int, default=None,
                       help="Port where the self-hosted router model is running (for OpenAI-compatible API)")
    parser.add_argument("--log-level", default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="Log level")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Router API Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Router Model: {args.router_model}")
    print(f"   Max Turns: {args.max_turns}")
    print(f"   Hosted Port: {args.hosted_port}")
    print(f"   Available Models: {len(FILTERED_AVAILABLE_MODELS)}")
    
    # Update router model if specified
    global router_agent
    if args.router_model != DEFAULT_ROUTER_MODEL or args.max_turns != 3 or args.hosted_port is not None:
        router_agent = RouterAgent(args.router_model, args.max_turns, args.hosted_port)
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()