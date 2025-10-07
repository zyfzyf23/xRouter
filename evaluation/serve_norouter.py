#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-Style Direct Model Server

This module implements an OpenAI-compatible API server that directly routes
requests to specified models without intermediate routing.

Features:
- OpenAI-compatible API endpoints (/v1/chat/completions)
- Direct model routing based on model parameter
- Support for FIXED_MODEL_SET_2 models as available tools
- Cost tracking for target models only

Usage:
    python serve_norouter.py --port 8000
    
    # Then call via OpenAI client:
    import openai
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    response = client.chat.completions.create(
        model="gpt-4o",  # Direct model specification
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
                                                    CHEMEVAL_MODEL_SET,
                                                    SYSTEM_PROMPT,
                                                    create_openai_tools)
# Import router utilities and model specifications
from verl.tools.utils.router_utils import MODEL_SPECS, LLMRouter, acall_model

app = FastAPI(title="Direct Model API", version="1.0.0")

# Available models
AVAILABLE_MODELS = CHEMEVAL_MODEL_SET

# Filter available models to only include those in MODEL_SPECS
FILTERED_AVAILABLE_MODELS = [
    model_id for model_id in AVAILABLE_MODELS 
    if model_id in MODEL_SPECS
]

print(f"Server initialized with {len(FILTERED_AVAILABLE_MODELS)} available models:")
for model_id in FILTERED_AVAILABLE_MODELS:
    spec = MODEL_SPECS[model_id]
    print(f"  - {model_id}: {spec.name} ({spec.quality_tier})")


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


class DirectModelHandler:
    """
    Direct model handler that routes requests to specified models.
    """
    
    def __init__(self):
        self.available_models = FILTERED_AVAILABLE_MODELS
        self.router_system = LLMRouter()
        
        print(f"Direct model handler initialized")
        print(f"Available models: {len(self.available_models)}")
    
    async def handle_request(self, model_id: str, messages: List[Dict[str, str]], 
                           max_tokens: Optional[int] = None, 
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Handle a request by directly calling the specified model.
        
        Args:
            model_id: Model identifier to route to
            messages: List of chat messages
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Validate model
        if model_id not in self.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is not available. Available models: {self.available_models}"
            )
        
        # Prepare messages for target model
        target_messages = [dict(m) for m in messages]
        
        # Prepare sampling parameters
        sampling_params = {
            "temperature": temperature,
        }
        
        print(f"🎯 Calling target model: {model_id}")
        
        try:
            # Call the target model directly
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
                response_content = "Timeout when trying to call this model."
                metadata = {
                    "model_id": model_id,
                    "model_name": model_id,
                    "input_tokens": sum(len(msg.get("content", "")) for msg in target_messages) // 4,
                    "output_tokens": len(response_content) // 4,
                    "cost": 0.0,
                    "latency": 600.0,
                    "provider": "timeout"
                }
            
            return {
                "content": response_content,
                "model_used": model_id,
                "total_cost": metadata.get("cost", 0),
                "router_cost": 0.0,  # No router costs
                "target_models_cost": metadata.get("cost", 0),
                "total_time": time.time() - start_time,
                "routing_strategy": "direct",
                "call_history": [{
                    "model_id": model_id,
                    "response": response_content,
                    "cost": metadata.get("cost", 0),
                    "tokens_used": metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0),
                    "latency": metadata.get("latency", 0)
                }],
                "turns_used": 1
            }
            
        except Exception as e:
            print(f"❌ Error calling {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling model {model_id}: {str(e)}"
            )


# Initialize direct model handler
model_handler = DirectModelHandler()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Direct Model API",
        "available_models": FILTERED_AVAILABLE_MODELS
    }

@app.get("/v1")
async def v1_root():
    return {"status":"ok","endpoints":["/v1/models","/v1/chat/completions"]}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = []
    
    # Add available models
    for model_id in FILTERED_AVAILABLE_MODELS:
        models.append({
            "id": model_id,
            "object": "model", 
            "created": int(time.time()),
            "owned_by": "direct-model-api"
        })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main chat completions endpoint (OpenAI-compatible)"""
    
    try:
        # Convert pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Route the request directly to the specified model
        try:
            result = await asyncio.wait_for(
                model_handler.handle_request(
                    model_id=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature or 0.7
                ),
                timeout=670.0  # 670s timeout
            )
        except asyncio.TimeoutError:
            print("⏰ Request timeout after 670 seconds")
            # Return a timeout response
            result = {
                "content": "The solution is not found or not valid. Request timed out after 670 seconds.",
                "model_used": request.model,
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


@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to view available models"""
    return {
        "available_models": FILTERED_AVAILABLE_MODELS,
        "model_count": len(FILTERED_AVAILABLE_MODELS)
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Direct Model API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8800, help="Port to bind to")
    parser.add_argument("--log-level", default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="Log level")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Direct Model API Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Available Models: {len(FILTERED_AVAILABLE_MODELS)}")
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
