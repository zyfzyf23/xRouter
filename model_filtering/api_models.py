#!/usr/bin/env python3
"""
API model interface for external model providers (OpenAI, Together AI, etc.)
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import openai
import requests
from dotenv import load_dotenv
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

console = Console()

@dataclass 
class APIResponse:
    """Response structure matching vLLM's RequestOutput format"""
    prompt: str
    outputs: List[str]
    request_id: Optional[str] = None

@dataclass
class ModelConfig:
    """Configuration for different model providers"""
    provider: str
    model_id: str
    context_window: int
    input_price_per_1k: float  # $ per 1K tokens
    output_price_per_1k: float  # $ per 1K tokens
    base_url: Optional[str] = None

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": ModelConfig("openai", "gpt-4o", 128000, 0.0025, 0.010),
    "gpt-4o-mini": ModelConfig("openai", "gpt-4o-mini", 128000, 0.00015, 0.00060),
    "gpt-4.1": ModelConfig("openai", "gpt-4.1", 1047576, 0.002, 0.008),
    "gpt-4.1-mini": ModelConfig("openai", "gpt-4.1-mini", 1047576, 0.00040, 0.00160),
    "gpt-4.1-nano": ModelConfig("openai", "gpt-4.1-nano", 1047576, 0.00010, 0.00140),
    "o3": ModelConfig("openai", "o3", 200000, 0.002, 0.008),
    "o3-pro": ModelConfig("openai", "o3-pro", 200000, 0.020, 0.080),
    "o4-mini": ModelConfig("openai", "o4-mini", 200000, 0.00110, 0.00440),
    
    # Together AI models
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": ModelConfig("together", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", 262144, 0.20/1000, 0.60/1000),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": ModelConfig("together", "Qwen/Qwen3-235B-A22B-Thinking-2507", 262144, 0.65/1000, 3.00/1000),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": ModelConfig("together", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 262144, 2.00/1000, 2.00/1000),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelConfig("together", "Qwen/Qwen2.5-72B-Instruct-Turbo", 131072, 1.20/1000, 1.20/1000),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelConfig("together", "Qwen/Qwen2.5-7B-Instruct-Turbo", 32768, 0.30/1000, 0.30/1000),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelConfig("together", "Qwen/Qwen2.5-Coder-32B-Instruct", 32768, 0.80/1000, 0.80/1000),
    "moonshotai/Kimi-K2-Instruct": ModelConfig("together", "moonshotai/Kimi-K2-Instruct", 131072, 1.00/1000, 3.00/1000),
    "deepseek-ai/DeepSeek-R1": ModelConfig("together", "deepseek-ai/DeepSeek-R1", 163840, 3.00/1000, 7.00/1000),
    "deepseek-ai/DeepSeek-R1-0528-tput": ModelConfig("together", "deepseek-ai/DeepSeek-R1-0528-tput", 163840, 0.55/1000, 2.19/1000),
    "deepseek-ai/DeepSeek-V3": ModelConfig("together", "deepseek-ai/DeepSeek-V3", 131072, 1.25/1000, 1.25/1000),
}

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.acquire()
            
            self.requests.append(now)

class APIModelInterface:
    """Interface for API-based models (OpenAI, Together AI, etc.)"""
    
    def __init__(self, model_name: str, max_requests_per_minute: int = 50):
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name)
        
        if not self.config:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_CONFIGS.keys())}")
        
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Initialize API clients
        if self.config.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.config.provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY") 
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            self.together_api_key = api_key
            self.together_base_url = "https://api.together.xyz/v1/chat/completions"
    
    def get_column_name(self) -> str:
        """Generate appropriate column name for pass rates"""
        if self.config.provider == "openai":
            # gpt-4o -> gpt4o_pass_rate
            # gpt-4.1-mini -> gpt41_mini_pass_rate
            clean_name = self.model_name.replace("-", "_").replace(".", "")
            return f"{clean_name}_pass_rate"
        else:
            # For together models, use last part of model ID
            # Qwen/Qwen2.5-7B-Instruct-Turbo -> qwen25_7b_instruct_turbo_pass_rate
            model_part = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            clean_name = model_part.lower().replace("-", "_").replace(".", "")
            return f"{clean_name}_pass_rate"
    
    def _call_openai_api(self, prompt: str, sampling_params: Dict[str, Any]) -> List[str]:
        """Call OpenAI API"""
        self.rate_limiter.acquire()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=sampling_params.get("max_tokens", 2048),
                temperature=sampling_params.get("temperature", 1.0),
                top_p=sampling_params.get("top_p", 1.0),
                n=sampling_params.get("n", 1),
            )
            
            # Track usage
            usage = response.usage
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            
            input_cost = usage.prompt_tokens * self.config.input_price_per_1k / 1000
            output_cost = usage.completion_tokens * self.config.output_price_per_1k / 1000
            self.total_cost += input_cost + output_cost
            
            return [choice.message.content for choice in response.choices]
            
        except Exception as e:
            console.print(f"[red]OpenAI API error: {e}[/red]")
            # Return empty responses to match expected count
            return [""] * sampling_params.get("n", 1)
    
    def _call_together_api(self, prompt: str, sampling_params: Dict[str, Any]) -> List[str]:
        """Call Together AI API"""
        self.rate_limiter.acquire()
        
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": sampling_params.get("max_tokens", 2048),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "n": sampling_params.get("n", 1),
        }
        
        try:
            response = requests.post(self.together_base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Track usage
            if "usage" in result:
                usage = result["usage"]
                self.total_input_tokens += usage["prompt_tokens"]
                self.total_output_tokens += usage["completion_tokens"]
                
                input_cost = usage["prompt_tokens"] * self.config.input_price_per_1k / 1000
                output_cost = usage["completion_tokens"] * self.config.output_price_per_1k / 1000
                self.total_cost += input_cost + output_cost
            
            return [choice["message"]["content"] for choice in result["choices"]]
            
        except Exception as e:
            console.print(f"[red]Together AI API error: {e}[/red]")
            # Return empty responses to match expected count
            return [""] * sampling_params.get("n", 1)
    
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[APIResponse]:
        """Generate responses for a batch of prompts"""
        console.print(f"[cyan]Generating {len(prompts)} prompts with {self.model_name}...[/cyan]")
        
        responses = []
        n_responses = sampling_params.get("n", 1)
        
        # Use ThreadPoolExecutor for concurrent API calls
        max_workers = min(10, len(prompts))  # Limit concurrent requests
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_prompt = {}
            for i, prompt in enumerate(prompts):
                if self.config.provider == "openai":
                    future = executor.submit(self._call_openai_api, prompt, sampling_params)
                elif self.config.provider == "together":
                    future = executor.submit(self._call_together_api, prompt, sampling_params)
                future_to_prompt[future] = (i, prompt)
            
            # Collect results in order
            results = [None] * len(prompts)
            for future in as_completed(future_to_prompt):
                i, prompt = future_to_prompt[future]
                try:
                    outputs = future.result()
                    results[i] = APIResponse(prompt=prompt, outputs=outputs)
                except Exception as e:
                    console.print(f"[red]Error processing prompt {i}: {e}[/red]")
                    results[i] = APIResponse(prompt=prompt, outputs=[""] * n_responses)
        
        return results
    
    def print_usage_stats(self):
        """Print usage statistics"""
        console.print(f"\n[bold]API Usage Statistics for {self.model_name}:[/bold]")
        console.print(f"  Input tokens: {self.total_input_tokens:,}")
        console.print(f"  Output tokens: {self.total_output_tokens:,}")
        console.print(f"  Total cost: ${self.total_cost:.4f}")
        
        if self.total_input_tokens > 0:
            console.print(f"  Average input cost per 1K: ${(self.total_cost / (self.total_input_tokens + self.total_output_tokens) * 1000):.6f}")

def get_supported_models() -> Dict[str, List[str]]:
    """Get list of supported models by provider"""
    models_by_provider = {}
    for model_name, config in MODEL_CONFIGS.items():
        if config.provider not in models_by_provider:
            models_by_provider[config.provider] = []
        models_by_provider[config.provider].append(model_name)
    return models_by_provider

def is_api_model(model_name: str) -> bool:
    """Check if a model name corresponds to an API model"""
    return model_name in MODEL_CONFIGS