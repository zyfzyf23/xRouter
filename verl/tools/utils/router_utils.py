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

    "gpt-5-nano": ModelSpec(
        name="GPT-5 Nano",
        api_alias="gpt-5-nano",
        provider="openai", 
        input_price_per_million=0.05,
        output_price_per_million=0.40,
        context_window=400000,
        max_output_tokens=128000,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="budget",
        description="""Ultra-efficient and cost-effective GPT-5 variant for high-volume applications.

        Capabilities: Basic reasoning, coding, math, general tasks
        Quality: Budget tier with optimized efficiency
        Cost: $0.05/M input tokens, $0.40/M output tokens
        Context: 400K tokens
        Max Output: 128K tokens

        Best for: High-volume, low-latency applications, developer tools, and real-time interactions.

        Benchmark Performance:
        - GPQA Diamond: 71.2%
        - AIME 2025: 85.2%
        - SWE-bench Verified: 34.8%"""
    ),
    "gpt-4o": ModelSpec(
        name="GPT-4o (Omni)",
        api_alias="gpt-4o",
        provider="openai",
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=["reasoning", "coding", "math", "general", "multimodal"],
        quality_tier="premium",
        description="""Advanced multimodal model with excellent reasoning and coding capabilities.
        
        Capabilities: Advanced reasoning, coding, math, general tasks, multimodal
        Quality: Premium tier  
        Cost: $2.50/M input tokens, $10.00/M output tokens
        Context: 128K tokens
        Max Output: 16,384 tokens
        
        Best for: High-quality multimodal applications, general-purpose tasks requiring a blend of speed and intelligence, and interactive chat where it is not critical to have the absolute best performance in a specialized domain.
        
        Benchmark Performance:
        - MMLU: 88.7%
        - GPQA Diamond: 79.1%
        - SWE-bench Verified: 29.8%
        - Arena-Hard v2: 61.9%
        - LiveCodeBench v6: 35.8%
        - LMSYS Arena Elo: 1391"""
    ),
    
    "gpt-4o-mini": ModelSpec(
        name="GPT-4o Mini",
        api_alias="gpt-4o-mini",
        provider="openai",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        context_window=128000,
        max_output_tokens=16384,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="budget",
        description="""Cost-effective model with a strong balance of performance, speed, and affordability.
        
        Capabilities: Reasoning, coding, math, general tasks
        Quality: Budget tier with good performance
        Cost: $0.15/M input tokens, $0.60/M output tokens
        Context: 128K tokens
        Max Output: 16,384 tokens
        
        Best for: High-volume applications, cost-sensitive agentic workflows with tool-calling, and general-purpose tasks where premium quality is not required.
        
        Benchmark Performance:
        - MMLU: 82.0%
        - MGSM (Math): 87.0%
        - HumanEval (Coding): 87.2%
        - MMMU (Multimodal): 59.4%
        - SWE-bench Verified (Agentless): 26.0%
        - LiveCodeBench v5: 38.4%"""
    ),
    
    "gpt-4.1": ModelSpec(
        name="GPT-4.1",
        api_alias="gpt-4.1",
        provider="openai",
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        context_window=1047576,
        max_output_tokens=32768,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="premium",
        description="""Latest GPT-4 variant with a massive context window and state-of-the-art coding and instruction-following abilities.
        
        Capabilities: Advanced reasoning, coding, math, general tasks, long context processing
        Quality: Premium tier
        Cost: $2.00/M input tokens, $8.00/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        Max Output: 32,768 tokens
        
        Best for: Production developer workflows, agentic systems requiring high reliability, complex reasoning over large documents, and structured instruction-following.
        
        Benchmark Performance:
        - MMLU: 90.4%
        - MMLU-Redux: 92.4%
        - MMLU-Pro: 81.8%
        - GPQA Diamond: 66.3%
        - AIME 2024: 46.5%
        - MATH-500: 92.4%
        - SWE-bench Verified (Agentic): 54.6%
        - SWE-bench Verified (Agentless): 40.8%
        - LiveCodeBench v6: 44.7%
        - Terminal-Bench (Terminus agent): 30.3%
        - IFEval: 88.0%
        - Tau-bench (Retail/Airline tool use): 74.8% / 54.5%
        - LMSYS Arena Elo: 1408"""
    ),
    
    "gpt-4.1-mini": ModelSpec(
        name="GPT-4.1 Mini",
        api_alias="gpt-4.1-mini",
        provider="openai",
        input_price_per_million=0.40,
        output_price_per_million=1.60,
        context_window=1047576,
        max_output_tokens=32768,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="budget",
        description="""Balanced model with a large context window and an exceptional performance-to-cost ratio.
        
        Capabilities: Reasoning, coding, math, general tasks, long context processing
        Quality: Budget tier
        Cost: $0.40/M input tokens, $1.60/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        Max Output: 32,768 tokens
        
        Best for: Scalable applications requiring high-quality, long-context processing at a reduced cost, such as RAG systems and in-editor assistants.
        
        Benchmark Performance:
        - MMLU: 87.5%
        - GPQA Diamond: 65.0%
        - AIME 2024: 49.6%
        - SWE-bench Verified: 23.6%
        - LiveBench (coding overall): 55.55
        - IFEval: 84.1%
        - Tau-bench (Airline/Retail tool use): 36.0% / 55.8%
        - LMSYS Arena Elo: 1372"""
    ),
    
    "gpt-4.1-nano": ModelSpec(
        name="GPT-4.1 Nano",
        api_alias="gpt-4.1-nano",
        provider="openai",
        input_price_per_million=0.10,
        output_price_per_million=1.40,
        context_window=1047576,
        max_output_tokens=32768,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="budget",
        description="""Ultra-low cost model with a large context window, optimized for speed in high-volume tasks.
        
        Capabilities: Basic reasoning, coding, math, general tasks, long context processing
        Quality: Budget tier
        Cost: $0.10/M input tokens, $1.40/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        Max Output: 32,768 tokens
        
        Best for: Ultra-efficient tasks like classification, summarization, or simple reasoning with massive context windows.
        
        Benchmark Performance:
        - MMLU: 80.1%
        - GPQA Diamond: 50.3%
        - Aider Polyglot (Code Editing): 9.8%
        - IFEval: 74.5%
        - LiveCodeBench: 42.7%
        - Tau-bench: 14.0%–22.6%"""
    ),
    
    "o3": ModelSpec(
        name="o3",
        api_alias="o3",
        provider="openai",
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        context_window=200000,
        max_output_tokens=100000,
        capabilities=["reasoning", "math", "science", "thinking"],
        quality_tier="specialized",
        description="""Advanced reasoning model optimized for mathematical and scientific tasks, trained to "think" before answering.
        
        Capabilities: Advanced reasoning, mathematics, scientific analysis
        Quality: Specialized for reasoning tasks
        Cost: $2.00/M input tokens, $8.00/M output tokens, its reasoning tokens could be very long, so the cost could be very high
        Context: 200K tokens
        Max Output: 100,000 tokens
        
        Best for: Analytical applications requiring deep multi-step reasoning and STEM precision.
        
        Note: O-series models only support temperature=1.0 and do not support top_p.
        Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
        Benchmark Performance:
        - MMLU-Pro: 85.3%
        - GPQA Diamond: 82.7%
        - SWE-bench Verified: 71.7%
        - Codeforces Elo: 2727
        - LiveCodeBench v6: 78.4%
        - Humanity's Last Exam (HLE): 20.0%
        - AIME 2025: 67%
        - LMSYS Arena Elo: 1452
        - SimpleQA: 51%
        - Terminal-Bench: 43.2%"""
    ),
    
    # "o3-pro": ModelSpec(
    #     name="o3 Pro",
    #     api_alias="o3-pro",
    #     provider="openai",
    #     input_price_per_million=20.00,
    #     output_price_per_million=80.00,
    #     context_window=200000,
    #     max_output_tokens=100000,
    #     capabilities=["reasoning", "math", "science", "research", "thinking"],
    #     quality_tier="premium",
    #     description="""Premium reasoning model for the most challenging problems, using more compute to "think harder".
        
    #     Capabilities: Premium reasoning, advanced mathematics, scientific research
    #     Quality: Top-tier specialized model
    #     Cost: $20.00/M input tokens, $80.00/M output tokens, its reasoning tokens could be very long, so the cost could be very high
    #     Context: 200K tokens
    #     Max Output: 100,000 tokens
        
    #     Best for: Mission-critical research, frontier scientific problems, and complex analytical tasks where achieving the highest accuracy is paramount.
        
    #     Note: O-series models only support temperature=1.0 and do not support top_p.
    #     Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
    #     Benchmark Performance:
    #     - AIME 2025: 68%
    #     - GPQA Diamond: 84.5%
    #     - Codeforces Elo: 2748
    #     - Terminal-Bench: 43.2%"""
    # ),
    
    "o4-mini": ModelSpec(
        name="o4 Mini",
        api_alias="o4-mini",
        provider="openai",
        input_price_per_million=1.10,
        output_price_per_million=4.40,
        context_window=200000,
        max_output_tokens=100000,
        capabilities=["reasoning", "math", "science", "thinking"],
        quality_tier="standard",
        description="""High-performance reasoning model with an excellent performance-to-cost ratio.
        
        Capabilities: Reasoning, mathematics, scientific analysis
        Quality: Standard tier for reasoning
        Cost: $1.10/M input tokens, $4.40/M output tokens, its reasoning tokens could be very long, so the cost could be very high
        Context: 200K tokens
        Max Output: 100,000 tokens
        
        Best for: Cost-sensitive development, math/coding tasks, and visual reasoning.
        
        Note: O-series models only support temperature=1.0 and do not support top_p.
        Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
        Benchmark Performance:
        - MMLU-Pro: 83.2%
        - GPQA Diamond: 78.4%
        - LiveCodeBench v6: 80.4%
        - AIME 2025: 65%
        - Humanity's Last Exam (HLE): 17.5%
        - SWE-bench Verified: 68.1%
        - Codeforces Elo: 2719
        - LMSYS Arena Elo: 1400+
        - SimpleQA: 20%"""
    ),
    
    # Together AI Models
    "gpt-oss-120b": ModelSpec(
        name="GPT-OSS 120B",
        api_alias="together_ai/openai/gpt-oss-120b",
        provider="together",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        context_window=131072,
        max_output_tokens=131072,
        capabilities=["reasoning", "coding", "math", "general", "chain_of_thought"],
        quality_tier="standard",
        description="""Open-weight MoE model from OpenAI, designed for efficient on-premise deployment and strong reasoning.
        
        Capabilities: Reasoning, coding, mathematics, general tasks, chain-of-thought
        Quality: Standard tier.
        Cost: $0.15/M input tokens, $0.60/M output tokens
        Context: 131,072 tokens (128K)
        Max Output: 131,072 tokens
        
        Best for: On-premise or private cloud deployments, especially in privacy-sensitive domains like healthcare, and for building cost-effective, open-weight agentic systems.
        
        Benchmark Performance:
        - GPQA Diamond (no tools): 67.1%
        - MMLU: 85.9%
        - HLE (no tools): 5.2%
        - Codeforces Elo (with tools): 1595
        - SWE-Bench Verified: 47.9%
        - HealthBench: 53.0%"""
    ),
    "gpt-oss-20b": ModelSpec(
        name="GPT-OSS 20B",
        api_alias="together_ai/openai/gpt-oss-20b",
        provider="together",
        input_price_per_million=0.05,
        output_price_per_million=0.20,
        context_window=131072,
        max_output_tokens=131072,
        capabilities=["reasoning", "math", "general", "chain_of_thought"],
        quality_tier="budget",
        description="""Open-weight MoE model from OpenAI, designed for highly efficient on-premise deployment and strong reasoning.
        
        Capabilities: Reasoning, mathematics, general tasks, chain-of-thought
        Quality: Budget tier.
        Cost: $0.05/M input tokens, $0.20/M output tokens
        Context: 131,072 tokens (128K)
        Max Output: 131,072 tokens

        Best for: Cost-effective on-premise, particularly for applications needing a balance of performance and efficiency.
        
        Benchmark Performance:
        - GPQA Diamond (no tools): 56.8%
        - MMLU: 80.4%
        - HLE (no tools): 4.2%
        - Codeforces Elo (with tools): 1366
        - SWE-Bench Verified: 37.4%
        - HealthBench: 40.0%"""
    ),
    "qwen3-235b-instruct": ModelSpec(
        name="Qwen3 235B Instruct",
        api_alias="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        provider="together",
        input_price_per_million=0.20,
        output_price_per_million=0.60,
        context_window=262144,
        max_output_tokens=262144,
        capabilities=["reasoning", "coding", "math", "general", "multilingual"],
        quality_tier="premium",
        description="""Large-scale, open-weight MoE model with excellent multilingual capabilities and strong all-around performance.
        
        Capabilities: Reasoning, coding, math, general tasks, multilingual
        Quality: Premium tier with excellent multilingual support
        Cost: $0.20/M input tokens, $0.60/M output tokens
        Context: 262,144 tokens
        Max Output: 262,144 tokens
        
        Best for: High-quality, cost-effective inference for multilingual applications and general-purpose tasks requiring strong instruction-following.
        
        Benchmark Performance (Instruct-2507, Non-thinking):
        - MMLU: 87.0%
        - MMLU-Pro: 77.3%
        - GPQA-Diamond: 62.9%
        - AIME 2025: 24.7%
        - MATH-500: 91.2%
        - SWE-bench Verified (Agentless): 39.4%
        - LiveCodeBench v6: 37.0%
        - IFEval: 83.2%
        - Tau2 (Retail/Airline): 57.0% / 26.5%
        - LMSYS Arena Elo: 1430"""
    ),
    
    "qwen3-235b-thinking": ModelSpec(
        name="Qwen3 235B Thinking",
        api_alias="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        provider="together",
        input_price_per_million=0.65,
        output_price_per_million=3.00,
        context_window=262144,
        max_output_tokens=262144,
        capabilities=["reasoning", "math", "analysis", "chain_of_thought", "thinking"],
        quality_tier="specialized",
        description="""Specialized open-weight reasoning model with built-in, transparent chain-of-thought capabilities.
        
        Capabilities: Advanced reasoning, mathematics, analysis, chain-of-thought
        Quality: Specialized for reasoning with built-in CoT
        Cost: $0.65/M input tokens, $3.00/M output tokens, its reasoning tokens could be very long, so the cost could be very high
        Context: 262,144 tokens
        Max Output: 262,144 tokens
        
        Best for: Complex analytical problems, mathematical proofs, and logical analysis requiring transparent, step-by-step reasoning chains.
        
        Note: Thinking model requires high token limits (8192+) to account for internal reasoning.

        Benchmark Performance (Thinking-2507 version):
        - LiveCodeBench v6: 74.1%
        - HMMT25 (Math): 83.9%
        - SuperGPQA (Science): 64.9%
        - AIME25 (Math): 92.3%
        - HLE (no tools): 18.2%
        - LMSYS Arena Elo: 1399"""
    ),
    
    "qwen3-coder-480b": ModelSpec(
        name="Qwen3 Coder 480B",
        api_alias="together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        provider="together",
        input_price_per_million=2.00,
        output_price_per_million=2.00,
        context_window=262144,
        max_output_tokens=262144,
        capabilities=["coding", "debugging", "code_review", "programming"],
        quality_tier="specialized",
        description="""Massive 480B parameter, open-weight MoE model hyper-specialized for coding.
        
        Capabilities: Advanced coding, debugging, code review, programming
        Quality: Specialized coding model.
        Cost: $2.00/M input tokens, $2.00/M output tokens
        Context: 262,144 tokens
        Max Output: 262,144 tokens
        
        Best for: The most demanding, end-to-end software engineering tasks, including complex code generation, large-scale refactoring, and agentic development workflows.
        
        Benchmark Performance:
        - SWE‑bench Verified: 67.0%
        - SWE‑bench Multilingual: 54.7%
        - Multi‑SWE‑bench mini: 25.8%
        - Multi‑SWE‑bench flash: 27.0%
        - Aider‑Polyglot: 61.8%
        - Spider2: 31.1%
        - Agentic Coding (Terminal‑Bench): 37.5
        - Agentic Browser Use (WebArena): 49.9%
        - Agentic Browser Use (Mind2Web): 55.8%
        - TAU‑Bench Retail: 77.5%
        - TAU‑Bench Airline: 60.0%"""
    ),
    
    # "qwen2.5-72b": ModelSpec(
    #     name="Qwen2.5 72B Turbo",
    #     api_alias="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    #     provider="together",
    #     input_price_per_million=1.20,
    #     output_price_per_million=1.20,
    #     context_window=131072,
    #     capabilities=["reasoning", "coding", "math", "general"],
    #     quality_tier="standard",
    #     description="""Fast and efficient open-weight model with balanced performance and a particular strength in mathematics.
        
    #     Capabilities: Reasoning, coding, math, general tasks
    #     Quality: Standard tier with fast inference
    #     Cost: $1.20/M input tokens, $1.20/M output tokens
    #     Context: 131,072 tokens
        
    #     Best for: General-purpose applications requiring balanced performance and speed, especially production workloads with advanced math/coding needs.

    #     Benchmark Performance:
    #     - MATH (w/ TIR): 83.1%
    #     - GSM8K: 91.6%
    #     - MMLU-Pro: 75.4%
    #     - MBPP: 88.2%
    #     - MultiPL-E: 75.1%
    #     - Arena-Hard (chat): 81.2
    #     - MT-Bench: 9.35
    #     - LiveCodeBench (2305-2409): 55.5"""
    # ),

    # "qwen2.5-coder-32b": ModelSpec(
    #     name="Qwen2.5 Coder 32B",
    #     api_alias="together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
    #     provider="together",
    #     input_price_per_million=0.80,
    #     output_price_per_million=0.80,
    #     context_window=32768,
    #     capabilities=["coding", "debugging", "programming"],
    #     quality_tier="specialized",
    #     description="""Coding-focused open-weight model with strong programming capabilities.
        
    #     Capabilities: Coding, debugging, programming
    #     Quality: Specialized tier for coding tasks
    #     Cost: $0.80/M input tokens, $0.80/M output tokens
    #     Context: 32,768 tokens
        
    #     Best for: Standard coding tasks, code repair, debugging, and moderate-complexity software development.
        
    #     Benchmark Performance:
    #     - Aider (Code Repair): 73.7%
    #     - HumanEval: 92.7%
    #     - MBPP: 82.6%
    #     - CRUXEval: 50.1%
    #     - LiveCodeBench: 43.1%
    #     - BigCodeBench: 29.5%"""
    # ),
    
    "kimi-k2": ModelSpec(
        name="Kimi K2 Instruct",
        api_alias="together_ai/moonshotai/Kimi-K2-Instruct",
        provider="together",
        input_price_per_million=1.00,
        output_price_per_million=3.00,
        context_window=131072,
        max_output_tokens=131072,
        capabilities=["reasoning", "math", "analysis", "multilingual", "coding", "thinking"],
        quality_tier="premium",
        description="""Advanced 1T parameter open-weight MoE model, uniquely trained for agentic workflows and tool use.
        
        Capabilities: Advanced reasoning, mathematics, analysis, multilingual, agentic coding
        Quality: Premium tier with strong analytical and tool-use capabilities
        Cost: $1.00/M input tokens, $3.00/M output tokens
        Context: 131,072 tokens
        Max Output: 131,072 tokens
        
        Best for: Agentic coding workflows requiring interaction with external tools, complex analytical tasks, and multilingual applications.
        
        Benchmark Performance:
        - MMLU: 89.5%
        - GPQA-Diamond: 75.1%
        - MATH-500: 97.4%
        - AIME 2025: 49.5%
        - SWE-bench Verified (Agentic): 65.8%
        - SWE-bench Verified (Agentless): 51.8%
        - LiveCodeBench v6: 53.7%
        - TerminalBench (Inhouse): 30.0%
        - Tau2 (Retail/Airline): 70.6% / 56.5%
        - LMSYS Arena Elo: 1400+"""
    ),
    
    "deepseek-r1": ModelSpec(
        name="DeepSeek R1",
        api_alias="together_ai/deepseek-ai/DeepSeek-R1",
        provider="together",
        input_price_per_million=3.00,
        output_price_per_million=7.00,
        context_window=163840,
        max_output_tokens=131072,
        capabilities=["reasoning", "math", "science", "research", "thinking"],
        quality_tier="premium",
        description="""Research-focused strong reasoning model.

        Capabilities: Reasoning, mathematics, science, research
        Quality: Premium tier with strong research capabilities
        Cost: $3.00/M input tokens, $7.00/M output tokens
        Context: 163,840 tokens
        Max Output: 131,072 tokens

        Best for: Creative Writing,scientific reasoning and analytical problem solving.

        Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
        Benchmark Performance:
        - AIME 2024: 79.8%
        - MATH-500: 97.3%
        - HLE (no tools): 17.7%
        - LiveCodeBench v6: 68.7%
        - Codeforces percentile: 96.3%
        - SuperGPQA: 61.82%
        - Terminal-Bench: 52% (Warp agent), 30% (Terminus agent)
        - LMSYS Arena Elo: 1394"""
    ),
    
    "deepseek-r1-tput": ModelSpec(
        name="DeepSeek R1 Throughput",
        api_alias="together_ai/deepseek-ai/DeepSeek-R1-0528-tput",
        provider="together",
        input_price_per_million=0.55,
        output_price_per_million=2.19,
        context_window=163840,
        max_output_tokens=131072,
        capabilities=["reasoning", "math", "science", "thinking"],
        quality_tier="standard",
        description="""High-throughput version of DeepSeek R1 with optimized cost.
        
        Capabilities: Reasoning, mathematics, science with optimized speed
        Quality: Standard tier with better cost-performance
        Cost: $0.55/M input tokens, $2.19/M output tokens
        Context: 163,840 tokens
        Max Output: 131,072 tokens
        
        Best for: Scaling reasoning tasks that require good performance at a lower cost, particularly for mathematical problems where speed and cost are key factors.

        Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
        Benchmark Performance:
        - Performance metrics are not publicly available for this throughput-optimized variant of DeepSeek R1."""
    ),
    
    # "deepseek-v3": ModelSpec(
    #     name="DeepSeek V3",
    #     api_alias="together_ai/deepseek-ai/DeepSeek-V3",
    #     provider="together",
    #     input_price_per_million=1.25,
    #     output_price_per_million=1.25,
    #     context_window=131072,
    #     capabilities=["reasoning", "coding", "math", "general"],
    #     quality_tier="standard",
    #     description="""Balanced open-weight MoE model with strong performance across multiple domains.
        
    #     Capabilities: Reasoning, coding, mathematics, general tasks
    #     Quality: Standard tier with balanced performance
    #     Cost: $1.25/M input tokens, $1.25/M output tokens
    #     Context: 131,072 tokens
        
    #     Best for: General-purpose applications requiring a consistent balance of reasoning and coding capabilities across multiple domains.
        
    #     Benchmark Performance:
    #     - MMLU: 89.4%
    #     - GPQA-Diamond: 68.4%
    #     - MATH-500: 94.0%
    #     - AIME 2024: 59.4%
    #     - SWE-bench Verified (Agentless): 36.6%
    #     - LiveCodeBench v6: 46.9%
    #     - IFEval: 81.1%
    #     - Tau2 (Retail/Airline): 69.1% / 39.0%
    #     - LMSYS Arena Elo: 1391"""
    # ),
    
    # Google Gemini Models
    # "gemini-2.5-pro": ModelSpec(
    #     name="Gemini 2.5 Pro",
    #     api_alias="gemini/gemini-2.5-pro",
    #     provider="google",
    #     input_price_per_million=1.25,  # ≤200K tokens, $2.50 for >200K
    #     output_price_per_million=10.00,  # ≤200K tokens, $15.00 for >200K
    #     context_window=1048576,  # 1M tokens
    #     max_output_tokens=65535,
    #     capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "thinking"],
    #     quality_tier="premium",
    #     description="""Google's most advanced model with elite reasoning, controllable "thinking" capabilities, and a massive context window.
        
    #     Capabilities: Advanced reasoning, coding, math, general, multimodal, long context, thinking
    #     Quality: Premium tier with thinking capabilities
    #     Cost: $1.25/M input tokens, $10.00/M output tokens (≤200K), higher for >200K
    #     Context: 1,048,576 tokens (~1M tokens)
    #     Max Output: 65,535 tokens
        
    #     Best for: Frontier reasoning tasks, deep analysis of long documents or codebases, and complex multimodal applications requiring the highest quality.
        
    #     Note: Thinking model requires high token limits (8192+) to account for internal reasoning.
        
    #     Benchmark Performance (Thinking mode):
    #     - GPQA Diamond: 86.4%
    #     - AIME 2025: 88.0%
    #     - LiveCodeBench: 69.0%
    #     - SWE-bench Verified (single attempt): 59.6%
    #     - HLE (no tools): 21.6%
    #     - MRCR (Long Context): 91.5% at 128K
    #     - LMSYS Arena Elo: 1459"""
    # ),
    
    # "gemini-2.5-flash-lite": ModelSpec(
    #     name="Gemini 2.5 Flash Lite",
    #     api_alias="gemini/gemini-2.5-flash-lite",
    #     provider="google",
    #     input_price_per_million=0.10,
    #     output_price_per_million=0.40,
    #     context_window=1048576,  # 1M tokens
    #     max_output_tokens=65535,
    #     capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "high_throughput"],
    #     quality_tier="budget",
    #     description="""High-throughput Gemini model optimized for cost-efficiency with an optional, controllable reasoning mode.
        
    #     Capabilities: Reasoning, coding, math, general, multimodal, long context, high throughput
    #     Quality: Budget tier optimized for cost and throughput
    #     Cost: $0.10/M input tokens, $0.40/M output tokens
    #     Context: 1,048,576 tokens (~1M tokens)
    #     Max Output: 65,535 tokens
        
    #     Best for: High-volume applications and cost-sensitive projects requiring large context, where developers can dynamically enable reasoning for higher quality on specific tasks.
        
    #     Benchmark Performance (Non-thinking):
    #     - MMLU: 90.1%
    #     - GPQA-Diamond: 68.2%
    #     - MATH-500: 95.4%
    #     - AIME 2024: 61.3%
    #     - SWE-bench Verified (Agentless): 32.6%
    #     - LiveCodeBench v6: 44.7%
    #     - IFEval: 84.3%
    #     - Tau2 (Retail/Airline): 64.3% / 42.5%"""
    # ),
}


    # unavaible for now
    # "gemini-2.5-flash": ModelSpec(
    #     name="Gemini 2.5 Flash",
    #     api_alias="gemini/gemini-2.5-flash",
    #     provider="google",
    #     input_price_per_million=0.30,
    #     output_price_per_million=2.50,
    #     context_window=1048576,  # 1M tokens
    #     capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "hybrid_reasoning"],
    #     quality_tier="standard",
    #     description="""Fast and balanced Gemini model with hybrid reasoning capabilities.
        
    #     Capabilities: Reasoning, coding, math, general, multimodal, long context, hybrid reasoning
    #     Quality: Standard tier with balanced performance
    #     Cost: $0.30/M input tokens, $2.50/M output tokens
    #     Context: 1,048,576 tokens (~1M tokens)
        
    #     Best for: Balanced applications requiring good performance with large context,
    #     hybrid reasoning tasks, general purpose with cost efficiency."""
    # ),
    

class LLMRouter:
    """Unified router system for managing multiple LLM models"""
    
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
            
        if not (openai_key or together_key or google_key):
            raise ValueError("At least one API key (OPENAI_API_KEY, TOGETHER_API_KEY, or GEMINI_API_KEY) must be set")
    
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