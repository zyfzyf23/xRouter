#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elegant LLM Router Tools using LiteLLM

This module provides a clean, unified router system for multiple LLM providers:
- Single call_model() function for all models
- Rich model specifications as the single source of truth  
- Automatic tool function generation
- Cost tracking and usage statistics
- Support for OpenAI, Together AI, and Google Gemini models

Usage:
    from llm_router_tools import call_model, get_available_models, create_openai_tools
    
    # Direct model calls
    response, metadata = call_model("gpt-4o-mini", messages, {"max_tokens": 100})
    response, metadata = call_model("gemini-2.5-pro", messages, {"max_tokens": 8192})
    
    # Get OpenAI function calling tools
    tools = create_openai_tools()
"""
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelSpec:
    """Comprehensive specification for an LLM model"""
    name: str
    api_alias: str
    provider: str
    input_price_per_million: float  # USD per million tokens
    output_price_per_million: float  # USD per million tokens
    context_window: int
    max_output_tokens: int  # Maximum number of output tokens the model can generate
    capabilities: List[str]  # e.g., ["reasoning", "coding", "math", "general"]
    quality_tier: str  # "budget", "standard", "premium", "specialized"
    description: str  # Comprehensive description with capabilities, cost, best use cases

@dataclass
class UsageStats:
    """Track usage statistics for a model"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    last_used: Optional[str] = None
    
    def update(self, input_tokens: int, output_tokens: int, cost: float):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.last_used = datetime.now().isoformat()

# Model specifications - Single source of truth
MODEL_SPECS = {
    # OpenAI Models
    "gpt-5": ModelSpec(
        name="GPT-5",
        api_alias="gpt-5",
        provider="openai",
        input_price_per_million=1.25,
        output_price_per_million=10.00,
        context_window=400000,
        max_output_tokens=128000,
        capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "tool_calling"],
        quality_tier="premium",
        description="""Latest flagship model with built-in reasoning capabilities and expert-level intelligence.

        Capabilities: Advanced reasoning with invisible thinking, coding, math, general tasks, multimodal (text+image input), long context, enhanced tool calling
        Quality: Premium tier with state-of-the-art performance
        Cost: $1.25/M input tokens, $10.00/M output tokens (plus $0.125/M for cached input)
        Context: 400K tokens
        Max Output: 128K tokens (includes reasoning tokens)

        Best for: Mission-critical applications, complex agentic workflows, and tasks requiring high level of accuracy and reasoning.

        Key Features:
        - Built-in reasoning with 'minimal' reasoning mode
        - Significant reduction in hallucinations
        - Improved instruction following and reduced sycophancy
        - Enhanced personality and steerability

        Benchmark Performance:
        - Scale MultiChallenge (instruction following): 69.6%
        - MMLU-Pro: 87.3%   
        - GPQA Diamond: 85.7%
        - AIME 2025: 94.6%
        - HealthBench Hard: 46.2%
        - LMSYS Arena Elo: 1481"""
    ),

    "gpt-5-mini": ModelSpec(
        name="GPT-5 Mini",
        api_alias="gpt-5-mini", 
        provider="openai",
        input_price_per_million=0.25,
        output_price_per_million=2.00,
        context_window=400000,
        max_output_tokens=128000,
        capabilities=["reasoning", "coding", "math", "general", "multimodal"],
        quality_tier="standard",
        description="""Smaller, faster, and more cost-effective version of GPT-5 with strong performance.

        Capabilities: Reasoning, coding, math, general tasks, multimodal (text+image input)
        Quality: Mid-tier with excellent performance-to-cost ratio
        Cost: $0.25/M input tokens, $2.00/M output tokens
        Context: 400K tokens
        Max Output: 128K tokens

        Best for: High-volume applications, cost-sensitive workflows, general-purpose tasks where full GPT-5 capability isn't required, and applications needing good performance at lower cost.

        Benchmark Performance:
        - MMLU-Pro: 82.8%
        - GPQA Diamond: 82.3%
        - AIME 2025: 91.1%
        - LMSYS Arena Elo: 1375"""
    ),

    # Local Models
    "qwen2.5-1.5b-local": ModelSpec(
        name="Qwen2.5-1.5B-Instruct",
        api_alias="qwen2.5-1.5b-local",
        provider="local",
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        context_window=32768,
        max_output_tokens=2048,
        capabilities=["math", "reasoning", "coding", "general"],
        quality_tier="budget",
        description="""本地部署的Qwen2.5-1.5B-Instruct模型，使用4-bit量化优化。

        Capabilities: 数学推理、逻辑推理、编程、通用任务
        Quality: Budget tier，轻量级高效模型
        Cost: 免费（本地部署）
        Context: 32K tokens
        Max Output: 2K tokens

        Best for: 快速原型验证、资源受限环境、作为DPO训练的基座模型。

        Key Features:
        - 4-bit量化，显存占用约3GB
        - 支持中英文双语
        - 优秀的数学推理能力
        - 适合在RTX 4060等消费级显卡运行"""
    ),

    "qwen": ModelSpec(
        name="qwen3-max",
        api_alias="qwen3-max", 
        provider="dashscope",  
        input_price_per_million=3.2,
        output_price_per_million=12.8,
        context_window=256000,
        max_output_tokens=64000,
        capabilities=["reasoning", "coding", "math", "general", "multimodal"],
        quality_tier="standard",
        description="""Alibaba's most capable large language model, optimized for complex reasoning and long-context understanding.

        Capabilities: Advanced reasoning, complex coding, mathematics, and long-document analysis (up to 256k context).
        Quality: High-end tier (Top-tier performance in standard benchmarks).
        Cost: ¥3.2/M input tokens, ¥12.8/M output tokens (CNY).
        Context: 256K tokens (Input up to 252K).
        Max Output: 64K tokens.

        Best for: Complex problem solving, large-scale document analysis, financial/legal consulting, and advanced coding agents.

        Benchmark Performance:
        - MMLU-Pro: 84.6%  
        - AIME 2025: 92.3%
        - LiveCodeBench: 70.7% 
        - LMSYS Arena Elo: 1370+ 
        - Tau2-Bench: 74.8%"""
)
}
    

class LLMRouter:
    """用于管理多个LLM模型的统一路由器系统
    Unified router system for managing multiple LLM models"""
    
    def __init__(self):
        self.usage_stats = {model_id: UsageStats() for model_id in MODEL_SPECS.keys()}
        self._lock = threading.Lock()
        
        # Set up LiteLLM
        litellm.set_verbose = False  # Reduce logging noise
        
        # Verify API keys
        self._check_api_keys()
    
    def _check_api_keys(self):
        """Verify that necessary API keys are available"""
        openai_key = os.getenv("OPENAI_API_KEY")
        together_key = os.getenv("TOGETHER_API_KEY")
        google_key = os.getenv("GEMINI_API_KEY")
        qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

        if not openai_key:
            print("WARNING: OPENAI_API_KEY not found. OpenAI models will not work.")
        else:
            print("✓ OPENAI_API_KEY found")

        if not together_key:
            print("WARNING: TOGETHER_API_KEY not found. Together AI models will not work.")
        else:
            print("✓ TOGETHER_API_KEY found")

        if not google_key:
            print("WARNING: GEMINI_API_KEY not found. Google Gemini models will not work.")
        else:
            print("✓ GEMINI_API_KEY found")

        if not qwen_key:
            print("WARNING: QWEN_API_KEY or DASHSCOPE_API_KEY not found. Qwen models will not work.")
        else:
            print("✓ QWEN_API_KEY found")

        if not (openai_key or together_key or google_key or qwen_key):
            raise ValueError("At least one API key (OPENAI_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY, or QWEN_API_KEY) must be set")
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough estimation of token count for cost calculation"""
        text = " ".join([msg.get("content", "") for msg in messages])
        return len(text) // 4  # Rough approximation: 4 chars per token
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call"""
        spec = MODEL_SPECS[model_id]
        input_cost = (input_tokens / 1_000_000) * spec.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * spec.output_price_per_million
        return input_cost + output_cost
    
    def call_model(self, model_id: str, messages: List[Dict[str, str]], 
                   sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        使用统一接口，通过 ID 调用任何模型。

        参数：

        model_id: 来自 MODEL_SPECS 的模型标识符

        messages: 包含 'role' 和 'content' 键的消息字典列表

        sampling_params: 可选的字典，包含 'temperature'、'max_tokens' 和 'top_p' 三个参数

        返回值：

        (response_content, metadata_dict) 元组

        Call any model by ID with unified interface.
        
        Args:
            model_id: Model identifier from MODEL_SPECS
            messages: List of message dicts with 'role' and 'content' keys
            sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
            
        Returns:
            Tuple of (response_content, metadata_dict)
        """
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_SPECS.keys())}")
            
        spec = MODEL_SPECS[model_id]
        

        params = {
            "model": spec.api_alias,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 1.0) if sampling_params else 1.0,
            # "max_completion_tokens": max_tokens,
            "drop_params": True,  # for litellm, always drop params that are not supported by the model
        }
        
        # Only add top_p for models that support it (not o3/o4 models)
        if not spec.api_alias.startswith("o3") and not spec.api_alias.startswith("o4"):
            params["top_p"] = sampling_params.get("top_p", 1.0) if sampling_params else 1.0
        
        if spec.api_alias.startswith("o3") or spec.api_alias.startswith("o4") or spec.api_alias.startswith("gpt-5"):
            params["temperature"] = 1.0

        # default_max_tokens = 16384
        # Only add max_completion_tokens for thinking models
        # if "thinking" in spec.capabilities:
        #     if "gpt-5" in spec.api_alias:
        #         params["max_completion_tokens"] = sampling_params.get("max_tokens", default_max_tokens) if sampling_params else default_max_tokens,  # Very high token limit to account for extensive internal reasoning
        #     else:
        #         params["max_tokens"] = sampling_params.get("max_tokens", default_max_tokens) if sampling_params else default_max_tokens,  # Very high token limit to account for extensive internal reasoning
        
        # if "thinking" in spec.capabilities:
        #     params["max_tokens"] = 32768
        
        # If using Salesforce Research Internal API
        if spec.provider == "openai":
            params["api_key"] = "dummy"
            params["base_url"] = "https://gateway.salesforceresearch.ai/openai/process/v1/"
            params["custom_llm_provider"] = "openai"
            params["extra_headers"] = {"X-Api-Key": os.getenv("X_API_KEY")}

        # Add provider-specific parameters
        elif spec.provider == "together":
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "")
        elif spec.provider == "google":
            # Ensure Google API key is available for Gemini models
            if os.getenv("GEMINI_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        elif spec.provider == "DASHSCOPE":
            # Ensure Qwen API key is available
            qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if qwen_key:
                os.environ["QWEN_API_KEY"] = qwen_key
            # 告诉 LiteLLM：这是一个 OpenAI 兼容 API
            params["api_key"] = qwen_key
            params["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            params["custom_llm_provider"] = "openai"  # ← 关键！走 OpenAI 通道
        elif spec.provider == "local":
            # Local models should not be called through LiteLLM
            raise ValueError(f"Local model {model_id} should be handled by LocalModelManager, not LLMRouter")
        
        start_time = time.time()
        
        try:
            response = litellm.completion(**params)
            
            # Extract response details
            content = response.choices[0].message.content
            usage = response.usage
            
            # Handle thinking models and empty responses
            if content is None:
                # Check if this is a thinking model with reasoning content
                message = response.choices[0].message
                thinking_content = getattr(message, 'reasoning_content', None)
                
                if thinking_content:
                    # For thinking models, use reasoning content if available
                    content = f"[Thinking: {thinking_content}]"
                elif hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                    # Model used reasoning tokens but no visible output
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
                    if reasoning_tokens > 0:
                        content = f"[Thinking model used {reasoning_tokens} reasoning tokens but no visible output. Use max_tokens=8192+ to allow space for both thinking and response.]"
                    else:
                        content = "[Empty response - thinking model may need different parameters or much higher token limits (8192+)]"
                else:
                    content = "[Empty response - model returned None content. For Gemini Pro, try max_tokens=8192+ to account for internal reasoning.]"
            
            # Get token counts
            input_tokens = getattr(usage, 'prompt_tokens', self._estimate_tokens(messages))
            output_tokens = getattr(usage, 'completion_tokens', len(content) // 4)
            
            # Calculate cost and update stats
            cost = self._calculate_cost(model_id, input_tokens, output_tokens)
            
            with self._lock:
                self.usage_stats[model_id].update(input_tokens, output_tokens, cost)
            
            metadata = {
                "model_id": model_id,
                "model_name": spec.name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency": time.time() - start_time,
                "provider": spec.provider
            }
            
            return content, metadata
            
        except Exception as e:
            assert False, f"[Internal Error in call_model] Error calling {spec.name} (model_id: {model_id}): {str(e)}"

    async def acall_model(self, model_id: str, messages: List[Dict[str, str]], 
                         sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Async version: Call any model by ID with unified interface.
        
        Args:
            model_id: Model identifier from MODEL_SPECS
            messages: List of message dicts with 'role' and 'content' keys
            sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
            
        Returns:
            Tuple of (response_content, metadata_dict)
        """
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_SPECS.keys())}")
            
        spec = MODEL_SPECS[model_id]
        
        params = {
            "model": spec.api_alias,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 1.0) if sampling_params else 1.0,
            "drop_params": True,  # for litellm, always drop params that are not supported by the model
            "timeout": 600 # sets a 10 minute timeout
        }
        # override temperature for o3/o4/gpt-5 models because they don't support temperature
        if spec.api_alias.startswith("o3") or spec.api_alias.startswith("o4") or spec.api_alias.startswith("gpt-5"):
            params["temperature"] = 1.0
        
        # Only add top_p for models that support it (not o3/o4 models)
        if not spec.api_alias.startswith("o3") and not spec.api_alias.startswith("o4") and not spec.api_alias.startswith("gpt-5"):
            params["top_p"] = sampling_params.get("top_p", 1.0) if sampling_params else 1.0

        # default_max_tokens = 16384
        # # Only add max_completion_tokens for thinking models
        # if "thinking" in spec.capabilities or "gemini-2.5-pro" in spec.api_alias:
        #     params["max_completion_tokens"] = sampling_params.get("max_tokens", default_max_tokens) if sampling_params else default_max_tokens,  # Very high token limit to account for extensive internal reasoning
        
        # if "thinking" in spec.capabilities:
        #     params["max_tokens"] = 32768
            
        # Add provider-specific parameters
        if spec.provider == "together":
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "")
        elif spec.provider == "google":
            # Ensure Google API key is available for Gemini models
            if os.getenv("GEMINI_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        elif spec.provider == "qwen":
            # Ensure Qwen API key is available
            qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if qwen_key:
                os.environ["QWEN_API_KEY"] = qwen_key
        elif spec.provider == "local":
            # Local models should not be called through LiteLLM
            raise ValueError(f"Local model {model_id} should be handled by LocalModelManager, not LLMRouter")
        
        start_time = time.time()
        
        try:
            response = await litellm.acompletion(**params)
            
            # Extract response details
            content = response.choices[0].message.content
            usage = response.usage
            
            # Handle thinking models and empty responses
            if content is None:
                # Check if this is a thinking model with reasoning content
                message = response.choices[0].message
                thinking_content = getattr(message, 'reasoning_content', None)
                
                if thinking_content:
                    # For thinking models, use reasoning content if available
                    content = f"[Thinking: {thinking_content}]"
                elif hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                    # Model used reasoning tokens but no visible output
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
                    if reasoning_tokens > 0:
                        content = f"[Thinking model used {reasoning_tokens} reasoning tokens but no visible output. Use max_tokens=8192+ to allow space for both thinking and response.]"
                    else:
                        content = "[Empty response - thinking model may need different parameters or much higher token limits (8192+)]"
                else:
                    content = "[Empty response - model returned None content. For Gemini Pro, try max_tokens=8192+ to account for internal reasoning.]"
            
            # Get token counts
            input_tokens = getattr(usage, 'prompt_tokens', self._estimate_tokens(messages))
            output_tokens = getattr(usage, 'completion_tokens', len(content) // 4)
            
            # Calculate cost and update stats
            cost = self._calculate_cost(model_id, input_tokens, output_tokens)
            
            with self._lock:
                self.usage_stats[model_id].update(input_tokens, output_tokens, cost)
            
            metadata = {
                "model_id": model_id,
                "model_name": spec.name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency": time.time() - start_time,
                "provider": spec.provider
            }
            
            return content, metadata
            
        except Exception as e:
            assert False, f"[Internal Error in acall_model] Error calling {spec.name} (model_id: {model_id}): {str(e)}"

def call_model(router: LLMRouter, model_id: str, messages: List[Dict[str, str]], 
               sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Unified function to call any model by ID with 3 retry attempts.
    
    Args:
        model_id: Model identifier from MODEL_SPECS
        messages: List of message dicts with 'role' and 'content' keys
        sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
        
    Returns:
        Tuple of (response_content, metadata_dict)
    """
    last_error = None
    for attempt in range(3):
        try:
            return router.call_model(model_id, messages, sampling_params)
        except Exception as e:
            print(f"Error calling {model_id} in attempt {attempt}: {str(e)}")
            last_error = e
            if attempt < 2:  # Don't sleep on the last attempt
                import time
                time.sleep(5)  # Brief delay between retries
    
    assert False, f"Failed to call model {model_id} after 3 attempts. Last error: {last_error}"

async def acall_model(router: LLMRouter, model_id: str, messages: List[Dict[str, str]], 
                      sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Async unified function to call any model by ID with 3 retry attempts.
    
    Args:
        model_id: Model identifier from MODEL_SPECS
        messages: List of message dicts with 'role' and 'content' keys
        sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
        
    Returns:
        Tuple of (response_content, metadata_dict)
    """
    # Add random jitter delay at the beginning (0.01s - 5s)
    import asyncio
    import random
    jitter_delay = random.uniform(0.01, 3.0)
    await asyncio.sleep(jitter_delay)
    
    last_error = None
    # Start with debug off
    litellm.set_verbose = False
    for attempt in range(3):
        try:
            result = await router.acall_model(model_id, messages, sampling_params)
            # Success - ensure debug is off
            litellm.set_verbose = False
            return result
        except Exception as e:
            # Error occurred - turn on debug
            litellm.set_verbose = True
            print(f"Error calling {model_id} in attempt {attempt}: {str(e)}")
            last_error = e
            if attempt < 2:  # Don't sleep on the last attempt
                import asyncio
                jitter_delay = random.uniform(1.0, 4.0)
                await asyncio.sleep(jitter_delay)  # Brief delay between retries
    
    # All attempts failed - ensure debug is on for debugging
    litellm.set_verbose = True
    assert False, f"Failed to call model {model_id} after 3 attempts. Last error: {last_error}"

def get_available_models() -> Dict[str, ModelSpec]:
    """Get all available model specifications"""
    return MODEL_SPECS.copy()

def get_models_by_capability(capability: str) -> Dict[str, ModelSpec]:
    """Get models that have a specific capability"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if capability in spec.capabilities
    }

def get_models_by_tier(tier: str) -> Dict[str, ModelSpec]:
    """Get models in a specific quality tier"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if spec.quality_tier == tier
    }

def get_models_by_provider(provider: str) -> Dict[str, ModelSpec]:
    """Get models from a specific provider"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if spec.provider == provider
    }


async def simple_acall_test():
    """Ultra simple test for acall_model - just tests basic functionality"""
    router = LLMRouter()
    try:
        response, metadata = await acall_model(router, "gpt-4o", [{"role": "user", "content": "Say hello"}], {"max_tokens": 20})
        print(f"✅ acall_model works! Response: {response}")
        return True
    except Exception as e:
        print(f"❌ acall_model failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio

    # Test async function
    print("\n=== Running Async Test ===")
    asyncio.run(simple_acall_test())