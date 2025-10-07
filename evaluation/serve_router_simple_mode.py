#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-Style Router Server - Simple Mode

This module implements a simplified router that makes single-turn decisions:
1. Analyze user query 
2. Either answer directly OR call exactly one model
3. Return response to user (no multi-turn interactions)

Simple mode follows the SYSTEM_PROMPT_SIMPLE logic from data_preprocess/router_data_preprocess.py.

Features:
- Single-turn routing decisions only
- Router can answer directly for simple queries  
- Router can delegate to exactly one model for complex queries
- No select_response tool or multi-turn conversations
- Cost-optimal model selection
- OpenAI-compatible API endpoints

Usage:
    python serve_router_simple_mode.py --port 8800 --hosted-port 8000
"""

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from data_preprocess.router_data_preprocess import (FIXED_MODEL_SET_1,
                                                    SYSTEM_PROMPT_SIMPLE,
                                                    create_openai_tools)
# Import router utilities and model specifications
from verl.tools.utils.router_utils import MODEL_SPECS, LLMRouter, acall_model

app = FastAPI(title="Router API - Simple Mode", version="1.0.0")

# Default router configuration
DEFAULT_ROUTER_MODEL = "xRouter-7b"  # Qwen 2.5 7B: $0.04/M input, $0.10/M output
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

print(f"Simple Mode Router initialized with {len(FILTERED_AVAILABLE_MODELS)} available models:")
for model_id in FILTERED_AVAILABLE_MODELS:
    spec = MODEL_SPECS[model_id]
    print(f"  - {model_id}: {spec.name} ({spec.quality_tier})")


def calculate_router_cost(router_model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for router model based on token usage."""
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


class SimpleRouterAgent:
    """
    Simple mode router agent - single turn decisions only.
    
    Logic:
    1. Analyze user query with router model
    2. Router either answers directly OR calls exactly one model
    3. Return response to user (no multi-turn conversations)
    """
    
    def __init__(self, router_model: str = DEFAULT_ROUTER_MODEL, hosted_port: Optional[int] = None):
        self.router_model = router_model
        self.hosted_port = hosted_port
        self.available_models = FILTERED_AVAILABLE_MODELS
        self.router_system = LLMRouter()
        
        # Configure base URL for self-hosted models
        if hosted_port is not None:
            self.base_url = f"http://localhost:{hosted_port}/v1"
            print(f"Simple Mode Router model {router_model} will be accessed via self-hosted endpoint: {self.base_url}")
        else:
            self.base_url = None
            print(f"Simple Mode Router model {router_model} will be accessed via default API")
        
        # Create tools for simple mode (no select_response tool)
        self.tools = create_openai_tools(self.available_models)
        
        print(f"Simple Mode Router agent initialized with router model: {router_model}")
        print(f"Available tools: {len(self.tools)} (no select_response in simple mode)")
    
    async def route_request(self, messages: List[Dict[str, str]], 
                          max_tokens: Optional[int] = None, 
                          temperature: float = 0.0) -> Dict[str, Any]:
        """
        Route a request through the simple mode routing system.
        
        Simple mode logic:
        1. Single router call to analyze query
        2. Router either responds directly OR calls exactly one model
        3. Return final response (no multi-turn interactions)
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing final response and metadata
        """
        start_time = time.time()
        router_costs = 0.0
        
        # Process user messages according to simple mode format
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

        # Create router messages with simple mode system prompt
        router_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SIMPLE},
            {"role": "user", "content": router_user_prompt}
        ]

        print(f"📝 Simple Mode - User message converted for router:")
        print(router_user_prompt)
        
        try:
            # Single router call to make decision
            print(f"🤖 Calling router model: {self.router_model}")
            
            # Prepare router call parameters
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
                router_params["max_completion_tokens"] = 32768
            else:
                router_params["temperature"] = 0.0
                router_params["max_tokens"] = 32768
            
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
            router_costs += router_call_cost
            
            print(f"🤖 Router model {self.router_model} used {router_input_tokens} input + {router_output_tokens} output tokens, cost: ${router_call_cost:.6f}")
            
            # Parse router response
            if not router_response.choices or not router_response.choices[0].message:
                raise HTTPException(
                    status_code=500,
                    detail="Router model returned empty response"
                )
            
            choice = router_response.choices[0]
            message = choice.message
            
            # Check if router made a tool call (delegate to model)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = message.tool_calls
                
                if len(tool_calls) > 1:
                    print(f"⚠️  Simple mode: Router made {len(tool_calls)} tool calls, but only first one will be used")
                
                # In simple mode, only use the first tool call
                tool_call = tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                call_id = tool_call.id
                
                print(f"🔧 Simple Mode - Router selected tool: {function_name}")
                print(f"📝 Arguments: {arguments}")
                
                # Handle model call tools only (no select_response in simple mode)
                if function_name.startswith("call_"):
                    # Extract model ID from function name
                    model_id = self._extract_model_id(function_name)
                    
                    if model_id not in self.available_models:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Model {model_id} not available"
                        )
                    
                    # Prepare optimized system prompt and sampling params
                    optimized_system_prompt = arguments.get("optimized_system_prompt", "")
                    if not optimized_system_prompt:
                        raise HTTPException(
                            status_code=400,
                            detail=f"optimized_system_prompt is required when calling {function_name}"
                        )
                    
                    sampling_params = {
                        "temperature": arguments.get("temperature", temperature or 0.7)
                    }
                    
                    # Prepare target model messages
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
                    print(f"🎯 Simple Mode - Calling target model: {model_id}")
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
                        
                        # Calculate total costs
                        target_model_cost = metadata.get("cost", 0)
                        total_cost = router_costs + target_model_cost
                        
                        print(f"✅ Simple Mode - Target model call successful")
                        print(f"💰 Target model cost: ${target_model_cost:.6f}")
                        print(f"💰 Total cost: ${total_cost:.6f}")
                        
                        return {
                            "content": response_content,
                            "model_used": model_id,
                            "total_cost": total_cost,
                            "router_cost": router_costs,
                            "target_models_cost": target_model_cost,
                            "total_time": time.time() - start_time,
                            "routing_strategy": "delegate_to_model",
                            "call_history": [{
                                "call_id": call_id,
                                "model_id": model_id,
                                "function_name": function_name,
                                "arguments": arguments,
                                "response": response_content,
                                "cost": target_model_cost,
                                "tokens_used": metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0),
                                "latency": metadata.get("latency", 0)
                            }],
                            "turns_used": 1
                        }
                        
                    except asyncio.TimeoutError:
                        print(f"❌ Simple Mode - Target model call timed out after 600s")
                        return {
                            "content": f"The model {model_id} timed out. Please try a different approach or model.",
                            "model_used": None,
                            "total_cost": router_costs,
                            "router_cost": router_costs,
                            "target_models_cost": 0.0,
                            "total_time": time.time() - start_time,
                            "routing_strategy": "timeout_fallback",
                            "call_history": [],
                            "turns_used": 1
                        }
                    except Exception as e:
                        print(f"❌ Simple Mode - Error calling {model_id}: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error calling model {model_id}: {str(e)}"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid tool call in simple mode: {function_name}"
                    )
            
            # Router provided a direct response (no tool calls)
            elif hasattr(message, 'content') and message.content:
                print(f"💬 Simple Mode - Router provided direct response")
                
                # Parse the router's response to extract actual content
                router_content = message.content.strip()
                
                # Remove <think>...</think> tags if present
                if "<think>" in router_content and "</think>" in router_content:
                    # Extract content after </think>
                    think_end = router_content.find("</think>")
                    if think_end != -1:
                        actual_content = router_content[think_end + 8:].strip()
                        if actual_content:
                            router_content = actual_content
                
                return {
                    "content": router_content,
                    "model_used": None,  # Router answered directly
                    "total_cost": router_costs,
                    "router_cost": router_costs,
                    "target_models_cost": 0.0,
                    "total_time": time.time() - start_time,
                    "routing_strategy": "direct_response",
                    "call_history": [],
                    "turns_used": 1
                }
            
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Router model provided no valid response or tool calls"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Simple Mode Router error: {str(e)}")
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


# Initialize simple mode router agent
router_agent = SimpleRouterAgent()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Router API - Simple Mode",
        "available_models": FILTERED_AVAILABLE_MODELS,
        "router_model": DEFAULT_ROUTER_MODEL,
        "mode": "simple"
    }


@app.get("/v1")
async def v1_root():
    return {"status": "ok", "endpoints": ["/v1/models", "/v1/chat/completions"], "mode": "simple"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = []
    
    # Add the router model
    models.append({
        "id": "router-simple",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "router-api-simple"
    })
    
    # Add available models
    for model_id in FILTERED_AVAILABLE_MODELS:
        models.append({
            "id": model_id,
            "object": "model", 
            "created": int(time.time()),
            "owned_by": "router-api-simple"
        })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main chat completions endpoint (OpenAI-compatible) - Simple Mode"""
    
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
                timeout=670.0  # 670s timeout (router + model call + buffer)
            )
        except asyncio.TimeoutError:
            print("⏰ Simple Mode - Request timeout after 670 seconds")
            # Return a timeout response
            result = {
                "content": "Request timed out. Please try again with a simpler query or different model.",
                "model_used": None,
                "total_cost": 0,
                "router_cost": 0.0,
                "target_models_cost": 0.0,
                "total_time": 670.0,
                "routing_strategy": "timeout",
                "call_history": [],
                "turns_used": 1
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
                "turns_used": result.get("turns_used", 1),
                "mode": "simple"
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Simple Mode - Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time(), "mode": "simple"}


@app.get("/debug/tools")
async def debug_tools():
    """Debug endpoint to view available tools"""
    return {
        "available_models": FILTERED_AVAILABLE_MODELS,
        "tools_count": len(router_agent.tools),
        "tools": router_agent.tools,
        "mode": "simple",
        "note": "Simple mode: no select_response tool, single-turn decisions only"
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Router API Server - Simple Mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8800, help="Port to bind to")
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL, 
                       help="Model to use for routing decisions")
    parser.add_argument("--hosted-port", type=int, default=None,
                       help="Port where the self-hosted router model is running (for OpenAI-compatible API)")
    parser.add_argument("--log-level", default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="Log level")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Router API Server - Simple Mode")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Router Model: {args.router_model}")
    print(f"   Hosted Port: {args.hosted_port}")
    print(f"   Available Models: {len(FILTERED_AVAILABLE_MODELS)}")
    print(f"   Mode: Simple (single-turn decisions only)")
    
    # Update router model if specified
    global router_agent
    if args.router_model != DEFAULT_ROUTER_MODEL or args.hosted_port is not None:
        router_agent = SimpleRouterAgent(args.router_model, args.hosted_port)
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()