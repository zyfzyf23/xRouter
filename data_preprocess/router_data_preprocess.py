import argparse
import json
import os
import random
import re
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from datasets import Dataset
from tqdm import tqdm

from verl.tools.utils.router_utils import MODEL_SPECS, get_models_by_tier
from verl.utils.hdfs_io import copy, makedirs

# SYSTEM_PROMPT = f"""You are a helpful assistant — an intelligent LLM router and prompt engineer. Your job is to:

# 1. Analyze the user query and conversation history.
# 2. If the task is simple and clearly within your capabilities, answer the user directly.
# 3. If not, select the most appropriate model from the available tools (in the format call_<model_name>).
# 4. Craft an optimized system prompt for that model and task.
# 5. Set suitable sampling parameters.
# 6. After receiving the response from the model, either repeat step 2-5 to refine the response or select the best response from the model by using the select_response tool or summarize the responses and directly respond to user.

# ## MODEL SELECTION RULES:
# - Your primary goal: Solve the user query at the *lowest cost possible*.
# - The cheapest option is answering the user yourself. However, you'll be penalized if your response is inadequate.
# - You may call multiple models if needed. Weigh the benefit of better answers vs. increased cost.
# - For complex queries, explore and compare responses from different models.

# ## NOTE
# You are a weak chat model. All other models are stronger. If you are uncertain or the task is complex, call a stronger model. 
# For thinking models, you should set large max_tokens (at least 8192).

# ## SYSTEM PROMPT OPTIMIZATION GUIDELINES
# - Define role & scope: start with a single-sentence role and a 1-line capability boundary.
# - Be concrete: add simple, explicit format example if the user query is not clear.
# - Match model to task: design prompts that exploit a model’s strengths (reasoning, creativity, code).
# - Minimize context: DO NOT repeat conversation history (like in the user query); rely on the runtime to supply it.
# - Add a short reminder in delegated-model system prompts that encourages structured thinking, validation, and confident final outputs. Keep it ≤ 3 lines.

# ## OTHER GUIDELINES
# - Use low temperature for precise/factual tasks, high temperature for creative tasks.
# - Be cost-efficient: choose the cheapest capable model that solves the task.
# - Delegate complex reasoning to strong thinking models.

# ## IMPORTANT:
# - When using call_<model_name>, only pass the optimized system prompt and sampling parameters as arguments.
# - The conversation history and user query will be automatically provided to the selected model — you do *not* need to include these information in the system prompt.
# - The tools will also be passed to the model automatically after invocation.

# ## RESPONSE FORMAT
# Respond in this format:
# <think>
# Your reasoning. First, assess if you can solve the task directly. If not, choose the best model for it.
# </think>
# `your_content`

# - If `your_content` is a function call (e.g., call_<model_name>), it will be executed and you’ll receive the model’s response.
# - You can directly respond to user or call select_response.
# - If use select_response, the selected model's response will be directly returned to user.
# - If you use select_response, you should not use other tools.
# - If it’s plain text, it will be returned to the user.
# - After receiving a model’s response, decide if it’s sufficient. If yes, reply to the user; if not, call another model.
# - When replying to the user, follow any format they request.
# """

SYSTEM_PROMPT = f"""You are a helpful assistant — an intelligent LLM router and prompt engineer. Your job is to:

1. Analyze the user query and conversation history.
2. If the task is simple and clearly within your capabilities, like casual chat, who are you, or 1+1=? type questions, answer the user directly.
3. Otherwise, choose one or more models from the available tools (tools are named call_<model_name>).
4. For each selected model, craft an optimized system prompt and sampling parameters.
5. Execute model calls, then either:
   a) SUMMARIZE: synthesize multiple model responses into one validated reply to the user, or
   b) SELECT: return *one* model's response verbatim using the select_response tool (see rules below).

## CALL / SELECT RULES
- call_<model_name> arguments must contain only:
  - the optimized system prompt (string), and
  - sampling parameters (e.g., temperature).
- Do NOT include conversation history or the user query in the system prompt — the runtime supplies them automatically.
- select_response(call_id) returns the selected model call's response verbatim to the user (copy-paste).
- You may select only **one** model response with select_response.
- If you use select_response, you **must not** perform parallel tool use in the same turn.
- Prefer SELECT when one model’s output is clearly superior and requires no synthesis; prefer SUMMARIZE when answers must be combined, reconciled, or edited.

## MODEL SELECTION GUIDELINES
- Primary objective: solve the user query at the *lowest cost possible*.
- If both model A and model B can solve the task, choose the cheaper one.
- Cheapest valid option: answer yourself. But you'll be penalized for inadequate answers.
- You may call multiple models to improve quality; balance expected accuracy gains vs cost.
- You are a weak chat model; if uncertain or the task is complex, call stronger models.

## SYSTEM PROMPT ENGINEERING
- Role & scope: 1-sentence role + 1-line capability to define the scope.
- Be concrete: include an example if ambiguity exists in the user query.
- Specify requirements & constraints: schema, style, output format, if any applicable.
- Match model to task: making the prompt better exploit model strengths (reasoning, creativity, code).
- Try your best to make the prompt unlock the selected model's capabilities.

## SAMPLING RECOMMENDATIONS
- Temperature: Use low temperature for precise/factual tasks, high temperature for creative tasks.

## RESPONSE FORMAT
Respond exactly in this format:
<think>
Your detailed step by step reasoning. For instance, think and decide if you can answer directly. If not, think carefully if the task requires heavy reasoning, and which model you plan to call and why.
</think>
`your_content`

- If `your_content` is a function call (e.g., call_<model_name>), it will be executed.
- After model responses:
  - If SUMMARIZING: `your_content` would be plain text which summarizes the responses. It will be returned directly to user. Follow any format the user requested. Use it when you must reconcile or refine multiple outputs.
  - If SELECTING: call select_response(call_id). That selected model output will be returned directly to user; it can saves the cost and time, so this is the preferred way to respond to the user.
"""

SYSTEM_PROMPT_SIMPLE = f"""
You are a helpful assistant — an intelligent LLM router and prompt engineer.  
Your responsibility is to analyze the user's query and choose the best response strategy.  
You must select **exactly one** of the following options:

1. **Answer Directly**  
   - If the query is simple and clearly within your own capabilities (e.g., casual chat, "who are you?", "what is 1+1?"), respond directly.  
   - This is always the cheapest option, but you will be penalized for inadequate answers.  

2. **Call a Model**  
   - If the query is complex or requires specialized skills, delegate to the most appropriate model (tools are named `call_<model_name>`).  
   - For the chosen model, you must craft an optimized system prompt and select suitable sampling parameters.  
   - The model’s output will be returned directly to the user.  

### Model Selection Guidelines
- **Primary goal:** Solve the user’s query at the **lowest possible cost**.  
- If multiple models can handle the task, prefer the cheaper one.  
- When uncertain or if the task seems challenging, delegate to a more capable model.  

### Tool Call Rules
- You may call **only one model per turn**.  
- **Do not** include the user query or conversation history in the `system_prompt`.  
  (The runtime environment provides these automatically.)  

### System Prompt Engineering
When writing the system prompt for the selected model:  
- **Role & Scope:** Define the model’s role and capabilities.  
- **Be Concrete:** If the query is ambiguous, add a guiding example.  
- **Specify Constraints:** If output needs a schema, style, or format (e.g., JSON), define it explicitly.  
- **Match Strengths:** Tailor the prompt to the chosen model’s abilities (reasoning, creativity, coding, etc.).  
- **Goal:** Unlock the model’s full potential for the specific task.
- **Never** include the details of the user query or conversation history in the system prompt. The runtime environment provides these to the selected model automatically.

### Response Format
You must always respond in the following structure:  

<think>  
Step-by-step analysis of the user’s query:  
- Decide if you can answer directly.  
- If not, justify your choice of model.  
- Explain your system prompt design.  
</think>  
`your_content`

- If answering directly → `your_content` is plain text for the user.  
- If delegating → `your_content` is a single function call, e.g. `call_<model_name>(...)`.  
"""


# Model sampling ranges for each tier
MODEL_SAMPLING_RANGES = {
    "premium": {"min": 1, "max": 3},
    "budget": {"min": 2, "max": 5},
    "standard": {"min": 2, "max": 4},
    "specialized": {"min": 1, "max": 2}
}

# Fixed model sets for special data portions
# FIXED_MODEL_SET_1 = [
#     "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
#     "o3", "o3-pro", "o4-mini", "gpt-oss-120b", "gpt-oss-20b", 
#     "qwen3-235b-instruct", "qwen3-235b-thinking", "qwen3-coder-480b", 
#     "kimi-k2", "deepseek-r1", "deepseek-r1-tput", "gemini-2.5-pro", "gemini-2.5-flash-lite"
# ]

FIXED_MODEL_SET_1 = [
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o",
    "gpt-4.1", "o3", "o3-pro", "o4-mini", "gpt-oss-120b", 
    "gpt-oss-20b" #, "gemini-2.5-pro", "gemini-2.5-flash-lite"
]

FIXED_MODEL_SET_2 = [
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini", 
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o3-pro", "o4-mini", 
    "gpt-oss-120b", "gpt-oss-20b", "qwen3-coder-480b", 
    "kimi-k2"#, "gemini-2.5-pro", "gemini-2.5-flash-lite"
]

FIXED_MODEL_SET_3 = [
    "gpt-4.1-nano",
    "gpt-oss-120b", "gpt-oss-20b",
    "kimi-k2" #, "gemini-2.5-pro", "gemini-2.5-flash-lite"
]

CHEMEVAL_MODEL_SET = [
    "gemini-2.5-pro", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-oss-20b", "kimi-k2", "deepseek-r1", "qwen3-235b-instruct", "qwen3-235b-thinking"
]

def create_openai_tools(selected_models: List[str]) -> List[Dict]:
    """
    Create OpenAI function calling tools from model specifications.
    
    Args:
        selected_models: List of model IDs to include
        
    Returns:
        List of OpenAI function tool definitions
    """
    tools = []
    
    for model_id in selected_models:
        if model_id not in MODEL_SPECS:
            continue
            
        spec = MODEL_SPECS[model_id]
        func_name = f"call_{model_id.replace('-', '_').replace('.', '_')}"
        
        tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": f"Call {spec.name} model. {spec.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "optimized_system_prompt": {
                            "type": "string",
                            "description": "Optimized system prompt for this specific model and task"
                        },
                        # "max_tokens": {
                        #     "type": "integer",
                        #     "description": "Maximum tokens to generate",
                        #     "default": 8192 if "thinking" in spec.capabilities else 2048
                        # },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0.0 to 2.0)",
                            "default": 1.0
                        }
                    },
                    "required": ["optimized_system_prompt"]
                }
            }
        }
        tools.append(tool)
    
    return tools

def print_available_models():
    """Print available models by tier for debugging."""
    print("\nAvailable models by tier:")
    for tier in ["premium", "budget", "standard", "specialized"]:
        tier_models = list(get_models_by_tier(tier).keys())
        print(f"  {tier}: {len(tier_models)} models")
        for model in tier_models:
            print(f"    - {model}")
    print()

def sample_models_by_tier() -> List[str]:
    """
    Sample models from each tier based on the defined ranges.
    
    Returns:
        List of sampled model IDs
    """
    sampled_models = []
    
    for tier, ranges in MODEL_SAMPLING_RANGES.items():
        tier_models = list(get_models_by_tier(tier).keys())
        if tier_models:
            # Randomly select number of models for this tier
            num_models = random.randint(ranges["min"], ranges["max"])
            # Sample without replacement
            sampled = random.sample(tier_models, min(num_models, len(tier_models)))
            sampled_models.extend(sampled)
    
    return sampled_models

def get_model_set_for_sample(sample_idx: int, total_samples: int, use_fixed_sets: bool = False, fixed_set_percentage: float = 0.1) -> Tuple[List[str], str]:
    """
    Determine which model set to use for a given sample.
    
    Args:
        sample_idx: Index of the current sample
        total_samples: Total number of samples
        use_fixed_sets: Whether to use fixed model sets
        fixed_set_percentage: Percentage of samples to use fixed sets for (per set)
        
    Returns:
        Tuple of (model_list, set_type) where set_type is 'random', 'fixed_set_1', 'fixed_set_2', or 'fixed_set_3'
    """
    if not use_fixed_sets:
        return sample_models_by_tier(), 'random'
    
    # Calculate boundaries for fixed sets
    fixed_set_1_end = int(total_samples * fixed_set_percentage)
    fixed_set_2_end = int(total_samples * fixed_set_percentage * 2)
    fixed_set_3_end = int(total_samples * fixed_set_percentage * 3)
    
    if sample_idx < fixed_set_1_end:
        # Filter available models from FIXED_MODEL_SET_1
        available_models = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
        return available_models, 'fixed_set_1'
    elif sample_idx < fixed_set_2_end:
        # Filter available models from FIXED_MODEL_SET_2
        available_models = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
        return available_models, 'fixed_set_2'
    elif sample_idx < fixed_set_3_end:
        # Filter available models from FIXED_MODEL_SET_3
        available_models = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
        return available_models, 'fixed_set_3'
    else:
        # Use random sampling for the remaining 70%
        return sample_models_by_tier(), 'random'

def save_generation_config(output_dir: str, args, model_sampling_ranges: Dict, seed: int = 42, max_system_prompt_length: int = 2000):
    """
    Save generation configuration to a YAML file for reproducibility.
    
    Args:
        output_dir: Directory to save the config file
        args: Command line arguments
        model_sampling_ranges: Model sampling ranges used
        seed: Random seed used
        max_system_prompt_length: Maximum system prompt length
    """
    config = {
        "generation_config": {
            "input_directory": args.input_dir,
            "output_directory": args.output_dir,
            "num_repetitions": args.num_repetitions,
            "random_seed": seed,
            "model_sampling_ranges": model_sampling_ranges,
            "simple_mode": getattr(args, 'simple_mode', False),
            "system_prompt_length": len(SYSTEM_PROMPT_SIMPLE if getattr(args, 'simple_mode', False) else SYSTEM_PROMPT),
            "system_prompt": SYSTEM_PROMPT_SIMPLE if getattr(args, 'simple_mode', False) else SYSTEM_PROMPT,
            "max_system_prompt_length": max_system_prompt_length,
            "total_available_models": len(MODEL_SPECS),
            "available_models_by_tier": {
                tier: len(get_models_by_tier(tier)) for tier in ["premium", "budget", "standard", "specialized"]
            },
            "fixed_model_sets": {
                "use_fixed_sets": getattr(args, 'use_fixed_sets', False),
                "fixed_set_percentage": getattr(args, 'fixed_set_percentage', 0.1),  # deprecated
                "fixed_set_1_percentage": getattr(args, 'fixed_set_1_percentage', 0.1),
                "fixed_set_2_percentage": getattr(args, 'fixed_set_2_percentage', 0.1),
                "fixed_set_3_percentage": getattr(args, 'fixed_set_3_percentage', 0.1),
                "proportional_sampling": getattr(args, 'use_fixed_sets', False),
                "fixed_model_set_1": FIXED_MODEL_SET_1,
                "fixed_model_set_2": FIXED_MODEL_SET_2,
                "fixed_model_set_3": FIXED_MODEL_SET_3,
                "available_fixed_set_1_models": [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS],
                "available_fixed_set_2_models": [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS],
                "available_fixed_set_3_models": [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
            }
        },
        "command_line_args": {
            "premium_min": args.premium_min,
            "premium_max": args.premium_max,
            "budget_min": args.budget_min,
            "budget_max": args.budget_max,
            "standard_min": args.standard_min,
            "standard_max": args.standard_max,
            "specialized_min": args.specialized_min,
            "specialized_max": args.specialized_max,
            "use_fixed_sets": getattr(args, 'use_fixed_sets', False),
            "fixed_set_percentage": getattr(args, 'fixed_set_percentage', 0.1),  # deprecated
            "fixed_set_1_percentage": getattr(args, 'fixed_set_1_percentage', 0.1),
            "fixed_set_2_percentage": getattr(args, 'fixed_set_2_percentage', 0.1),
            "fixed_set_3_percentage": getattr(args, 'fixed_set_3_percentage', 0.1),
            "use_fixed_set_only": getattr(args, 'use_fixed_set_only', False),
            "fixed_set_choice": getattr(args, 'fixed_set_choice', 1),
            "simple_mode": getattr(args, 'simple_mode', False)
        },
        "generation_timestamp": {
            "created_at": str(np.datetime64('now')),
            "script_version": "1.0"
        }
    }
    
    config_file = os.path.join(output_dir, "generation_config.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Generation config saved to: {config_file}")
    return config_file

def process_data_sample_worker(data_tuple: Tuple[Dict, int, str, int, int, bool]) -> List[Dict]:
    """
    Multiprocessing wrapper for process_data_sample.
    
    Args:
        data_tuple: Tuple containing (example, idx, split, num_repetitions, max_system_prompt_length, simple_mode)
        
    Returns:
        List of processed samples with different model combinations
    """
    example, idx, split, num_repetitions, max_system_prompt_length, simple_mode = data_tuple
    return process_data_sample(example, idx, split, num_repetitions, max_system_prompt_length, simple_mode)

def process_data_sample(example: Dict, idx: int, split: str, num_repetitions: int, max_system_prompt_length: int, simple_mode: bool = False) -> List[Dict]:
    """
    Process a single data sample and create multiple versions with different model combinations.
    
    Args:
        example: Original data sample
        idx: Sample index
        split: Dataset split name
        num_repetitions: Number of repetitions for this sample
        max_system_prompt_length: Maximum system prompt length
        simple_mode: Whether to use simple mode (SYSTEM_PROMPT_SIMPLE and no select_response)
        
    Returns:
        List of processed samples with different model combinations
    """
    processed_samples = []
    
    # Extract messages and extra_info
    messages = example.pop("prompt", [])
    extra_info = example.pop("extra_info", {})
    
    # Choose system prompt based on simple_mode
    chosen_system_prompt = SYSTEM_PROMPT_SIMPLE if simple_mode else SYSTEM_PROMPT
    
    # Handle system prompt logic (keeping existing logic)
    if messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        assert messages[1]["role"] == "user"
        user_query = messages[1]["content"]
        # if system prompt is too long, we just regard it as a user query
        if len(system_prompt) > max_system_prompt_length:
            user_query = "USER QUERY: \n"+ system_prompt + "\n" + user_query
            messages[0]["content"] = chosen_system_prompt
            messages[1]["content"] = user_query
        # if system prompt is not too long, we keep it as the user query prompt as put it in the user query part
        else:
            messages[0]["content"] = chosen_system_prompt
            messages[1]["content"] = "USER SYSTEM: \n" + system_prompt + "\nUSER QUERY: \n" + user_query
    else:
        # First message is user message
        user_query = messages[0]["content"]
        messages.insert(0, {"role": "system", "content": chosen_system_prompt})
        messages[1]["content"] = "USER QUERY: \n" + user_query

    # Create multiple versions with different model combinations
    for version in range(num_repetitions):  # Create different versions
        # Sample models for this version
        sampled_models = sample_models_by_tier()
        
        # Create tools for the sampled models
        tools = create_openai_tools(sampled_models)
        
        # Create new extra_info with tools
        new_extra_info = extra_info.copy()
        new_extra_info["need_tools_kwargs"] = True
        
        # Create simplified tools_kwargs as {func_name: {"dummy": "placeholder"}}
        tools_kwargs = {}
        if not simple_mode:
            tools_kwargs["select_response"] = {"dummy": "placeholder"}
        for model_id in sampled_models:
            func_name = f"call_{model_id.replace('-', '_').replace('.', '_')}"
            tools_kwargs[func_name] = {"dummy": "placeholder"}
        
        new_extra_info["tools_kwargs"] = tools_kwargs
        
        # Create new sample
        new_sample = example.copy()
        new_sample["prompt"] = messages
        new_sample["extra_info"] = new_extra_info
        new_sample["id"] = f"{example.get('id', idx)}_{version}"
        
        processed_samples.append(new_sample)

    return processed_samples

def process_data_sample_with_fixed_models(example: Dict, idx: int, split: str, max_system_prompt_length: int, fixed_models: List[str], set_type: str, simple_mode: bool = False) -> List[Dict]:
    """
    Process a single data sample with a fixed set of models (no repetitions).
    
    Args:
        example: Original data sample
        idx: Sample index
        split: Dataset split name
        max_system_prompt_length: Maximum system prompt length
        fixed_models: Fixed list of models to use
        set_type: Type of fixed set ('fixed_set_1' or 'fixed_set_2')
        simple_mode: Whether to use simple mode (SYSTEM_PROMPT_SIMPLE and no select_response)
        
    Returns:
        List containing one processed sample with fixed model combination
    """
    processed_samples = []
    
    # Extract messages and extra_info
    messages = example.pop("prompt", [])
    extra_info = example.pop("extra_info", {})
    
    # Choose system prompt based on simple_mode
    chosen_system_prompt = SYSTEM_PROMPT_SIMPLE if simple_mode else SYSTEM_PROMPT
    
    # Handle system prompt logic (same as original function)
    if messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        assert messages[1]["role"] == "user"
        user_query = messages[1]["content"]
        # if system prompt is too long, we just regard it as a user query
        if len(system_prompt) > max_system_prompt_length:
            user_query = "USER QUERY: \n"+ system_prompt + "\n" + user_query
            messages[0]["content"] = chosen_system_prompt
            messages[1]["content"] = user_query
        # if system prompt is not too long, we keep it as the user query prompt as put it in the user query part
        else:
            messages[0]["content"] = chosen_system_prompt
            messages[1]["content"] = "USER SYSTEM: \n" + system_prompt + "\nUSER QUERY: \n" + user_query
    else:
        # First message is user message
        user_query = messages[0]["content"]
        messages.insert(0, {"role": "system", "content": chosen_system_prompt})
        messages[1]["content"] = "USER QUERY: \n" + user_query

    # Create single version with fixed models
    tools = create_openai_tools(fixed_models)
    
    # Create new extra_info with tools
    new_extra_info = extra_info.copy()
    new_extra_info["need_tools_kwargs"] = True
    
    # Create simplified tools_kwargs as {func_name: {"dummy": "placeholder"}}
    tools_kwargs = {}
    if not simple_mode:
        tools_kwargs["select_response"] = {"dummy": "placeholder"}
    for model_id in fixed_models:
        func_name = f"call_{model_id.replace('-', '_').replace('.', '_')}"
        tools_kwargs[func_name] = {"dummy": "placeholder"}
    
    new_extra_info["tools_kwargs"] = tools_kwargs
    
    # Create new sample
    new_sample = example.copy()
    new_sample["prompt"] = messages
    new_sample["extra_info"] = new_extra_info
    new_sample["id"] = f"{example.get('id', idx)}_{set_type}"
    new_sample["model_set_type"] = set_type
    
    processed_samples.append(new_sample)
    
    return processed_samples

def process_fixed_set_worker(data_tuple: Tuple[Dict, int, str, int, List[str], str, bool]) -> List[Dict]:
    """
    Multiprocessing wrapper for process_data_sample_with_fixed_models.
    
    Args:
        data_tuple: Tuple containing (example, idx, split, max_system_prompt_length, fixed_models, set_type, simple_mode)
        
    Returns:
        List containing one processed sample with fixed model combination
    """
    example, idx, split, max_system_prompt_length, fixed_models, set_type, simple_mode = data_tuple
    return process_data_sample_with_fixed_models(example, idx, split, max_system_prompt_length, fixed_models, set_type, simple_mode)

def create_fixed_set_samples(dataset, fixed_set_percentage: float, max_system_prompt_length: int, workers: int = 8, simple_mode: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create additional samples using fixed model sets.
    
    Args:
        dataset: Original dataset
        fixed_set_percentage: Percentage of original data to use for each fixed set
        max_system_prompt_length: Maximum system prompt length
        workers: Number of worker processes
        simple_mode: Whether to use simple mode (SYSTEM_PROMPT_SIMPLE and no select_response)
        
    Returns:
        Tuple of (fixed_set_1_samples, fixed_set_2_samples, fixed_set_3_samples)
    """
    total_samples = len(dataset)
    num_samples_per_set = int(total_samples * fixed_set_percentage)
    
    # Randomly select samples for each fixed set
    import random
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    set_1_indices = all_indices[:num_samples_per_set]
    set_2_indices = all_indices[num_samples_per_set:num_samples_per_set*2]
    set_3_indices = all_indices[num_samples_per_set*2:num_samples_per_set*3]
    
    # Filter available models
    available_set_1_models = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
    available_set_2_models = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
    available_set_3_models = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
    
    print(f"Creating additional fixed set samples:")
    print(f"  Fixed Set 1: {len(set_1_indices)} samples with {len(available_set_1_models)} models")
    print(f"  Fixed Set 2: {len(set_2_indices)} samples with {len(available_set_2_models)} models")
    print(f"  Fixed Set 3: {len(set_3_indices)} samples with {len(available_set_3_models)} models")
    
    # Prepare data for fixed set 1
    set_1_data_tuples = [
        (dict(dataset[idx]), idx, "train", max_system_prompt_length, available_set_1_models, "fixed_set_1", simple_mode)
        for idx in set_1_indices
    ]
    
    # Prepare data for fixed set 2
    set_2_data_tuples = [
        (dict(dataset[idx]), idx, "train", max_system_prompt_length, available_set_2_models, "fixed_set_2", simple_mode)
        for idx in set_2_indices
    ]
    
    # Prepare data for fixed set 3
    set_3_data_tuples = [
        (dict(dataset[idx]), idx, "train", max_system_prompt_length, available_set_3_models, "fixed_set_3", simple_mode)
        for idx in set_3_indices
    ]
    
    # Process fixed set 1
    set_1_samples = []
    if set_1_data_tuples:
        if len(set_1_data_tuples) < workers or workers == 1:
            for data_tuple in tqdm(set_1_data_tuples, desc="Processing Fixed Set 1"):
                processed = process_fixed_set_worker(data_tuple)
                set_1_samples.extend(processed)
        else:
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap(process_fixed_set_worker, set_1_data_tuples),
                    total=len(set_1_data_tuples),
                    desc="Processing Fixed Set 1"
                ))
                for processed in results:
                    set_1_samples.extend(processed)
    
    # Process fixed set 2
    set_2_samples = []
    if set_2_data_tuples:
        if len(set_2_data_tuples) < workers or workers == 1:
            for data_tuple in tqdm(set_2_data_tuples, desc="Processing Fixed Set 2"):
                processed = process_fixed_set_worker(data_tuple)
                set_2_samples.extend(processed)
        else:
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap(process_fixed_set_worker, set_2_data_tuples),
                    total=len(set_2_data_tuples),
                    desc="Processing Fixed Set 2"
                ))
                for processed in results:
                    set_2_samples.extend(processed)
    
    # Process fixed set 3
    set_3_samples = []
    if set_3_data_tuples:
        if len(set_3_data_tuples) < workers or workers == 1:
            for data_tuple in tqdm(set_3_data_tuples, desc="Processing Fixed Set 3"):
                processed = process_fixed_set_worker(data_tuple)
                set_3_samples.extend(processed)
        else:
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap(process_fixed_set_worker, set_3_data_tuples),
                    total=len(set_3_data_tuples),
                    desc="Processing Fixed Set 3"
                ))
                for processed in results:
                    set_3_samples.extend(processed)
    
    return set_1_samples, set_2_samples, set_3_samples

def process_dataset(input_dir: str, output_dir: str, num_repetitions: int = 3, workers: int = 8, args=None, model_sampling_ranges=None):
    """
    Process all datasets in the input directory using multiprocessing.
    
    Args:
        input_dir: Directory containing parquet files
        output_dir: Directory to save processed data
        num_repetitions: Number of times to repeat each sample with different model combinations
        workers: Number of worker processes to use for multiprocessing
        args: Command line arguments for config saving
        model_sampling_ranges: Model sampling ranges for config saving
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save generation config if args are provided
    if args and model_sampling_ranges:
        save_generation_config(output_dir, args, model_sampling_ranges, args.seed, args.max_system_prompt_length)
    
    # Get all parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    total_processed = 0
    
    for parquet_file in parquet_files:
        file_path = os.path.join(input_dir, parquet_file)
        print(f"Processing {parquet_file}...")
        
        try:
            # Load dataset
            dataset = Dataset.from_parquet(file_path)
            print(f"  Loaded {len(dataset)} samples")
            
            all_processed_samples = []
            
            # Check if using fixed set only mode
            use_fixed_set_only = getattr(args, 'use_fixed_set_only', False)
            
            if use_fixed_set_only:
                # Fixed set only mode - use only the selected fixed set
                fixed_set_choice = getattr(args, 'fixed_set_choice', 1)
                print(f"  Using fixed set only mode with Fixed Set {fixed_set_choice}")
                
                # Select the appropriate fixed model set
                if fixed_set_choice == 1:
                    selected_fixed_models = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
                    set_type = "fixed_set_1"
                elif fixed_set_choice == 2:
                    selected_fixed_models = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
                    set_type = "fixed_set_2"
                else:  # fixed_set_choice == 3
                    selected_fixed_models = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
                    set_type = "fixed_set_3"
                
                print(f"  Using {len(selected_fixed_models)} models from Fixed Set {fixed_set_choice}")
                
                # Prepare data for fixed set processing
                data_tuples = [
                    (dict(example), idx, "train", args.max_system_prompt_length, selected_fixed_models, set_type, args.simple_mode)
                    for idx, example in enumerate(dataset)
                ]
                
                # Process samples using fixed set worker
                if len(data_tuples) == 0:
                    print(f"  No samples to process in {parquet_file}")
                    continue
                    
                if len(data_tuples) < workers or workers == 1:
                    print(f"  Processing {len(data_tuples)} samples sequentially with Fixed Set {fixed_set_choice}...")
                    for data_tuple in tqdm(data_tuples, desc=f"Processing {parquet_file} (Fixed Set {fixed_set_choice})"):
                        processed_samples = process_fixed_set_worker(data_tuple)
                        all_processed_samples.extend(processed_samples)
                else:
                    print(f"  Processing {len(data_tuples)} samples using {workers} workers with Fixed Set {fixed_set_choice}...")
                    with Pool(processes=workers) as pool:
                        results = list(tqdm(
                            pool.imap(process_fixed_set_worker, data_tuples),
                            total=len(data_tuples),
                            desc=f"Processing {parquet_file} (Fixed Set {fixed_set_choice})"
                        ))
                        
                        for processed_samples in results:
                            all_processed_samples.extend(processed_samples)
                            
            else:
                # Proportional sampling mode
                use_fixed_sets = getattr(args, 'use_fixed_sets', False)
                
                if use_fixed_sets:
                    # Calculate sample allocation based on proportions
                    fixed_set_1_pct = getattr(args, 'fixed_set_1_percentage', 0.1)
                    fixed_set_2_pct = getattr(args, 'fixed_set_2_percentage', 0.1)
                    fixed_set_3_pct = getattr(args, 'fixed_set_3_percentage', 0.1)
                    
                    total_fixed_pct = fixed_set_1_pct + fixed_set_2_pct + fixed_set_3_pct
                    random_pct = 1.0 - total_fixed_pct
                    
                    if total_fixed_pct > 1.0:
                        raise ValueError(f"Total fixed set percentages ({total_fixed_pct:.2f}) exceed 1.0")
                    if random_pct < 0:
                        raise ValueError(f"Fixed set percentages sum to more than 100%")
                    
                    # Determine actual max samples based on dataset size and random sampling needs
                    dataset_size = len(dataset)
                    requested_max_samples = args.max_num_samples
                    
                    # For random samples, we need to account for num_repetitions
                    max_possible_random = dataset_size * num_repetitions
                    
                    # Calculate how many dataset samples we'd need for the fixed sets
                    requested_fixed_1 = int(requested_max_samples * fixed_set_1_pct)
                    requested_fixed_2 = int(requested_max_samples * fixed_set_2_pct)
                    requested_fixed_3 = int(requested_max_samples * fixed_set_3_pct)
                    requested_random = requested_max_samples - requested_fixed_1 - requested_fixed_2 - requested_fixed_3
                    
                    # Check if we have enough dataset samples for the fixed sets
                    total_fixed_dataset_samples_needed = requested_fixed_1 + requested_fixed_2 + requested_fixed_3
                    random_dataset_samples_needed = (requested_random + num_repetitions - 1) // num_repetitions
                    total_dataset_samples_needed = total_fixed_dataset_samples_needed + random_dataset_samples_needed
                    
                    if total_dataset_samples_needed > dataset_size:
                        print(f"  WARNING: Requested samples require {total_dataset_samples_needed} dataset samples, but only {dataset_size} available.")
                        print(f"  Scaling down proportionally to fit available data.")
                        
                        # Scale down proportionally
                        scale_factor = dataset_size / total_dataset_samples_needed
                        
                        # Apply scaling, ensuring we don't exceed dataset size
                        scaled_fixed_1_dataset = int(requested_fixed_1 * scale_factor)
                        scaled_fixed_2_dataset = int(requested_fixed_2 * scale_factor)
                        scaled_fixed_3_dataset = int(requested_fixed_3 * scale_factor)
                        scaled_random_dataset = dataset_size - scaled_fixed_1_dataset - scaled_fixed_2_dataset - scaled_fixed_3_dataset
                        
                        # Calculate final sample counts
                        num_fixed_1 = scaled_fixed_1_dataset
                        num_fixed_2 = scaled_fixed_2_dataset
                        num_fixed_3 = scaled_fixed_3_dataset
                        num_random = min(scaled_random_dataset * num_repetitions, max_possible_random)
                        
                        max_samples = num_fixed_1 + num_fixed_2 + num_fixed_3 + num_random
                        
                        print(f"  Adjusted allocation:")
                        print(f"    Dataset samples for Fixed Set 1: {scaled_fixed_1_dataset} -> {num_fixed_1} final samples")
                        print(f"    Dataset samples for Fixed Set 2: {scaled_fixed_2_dataset} -> {num_fixed_2} final samples")
                        print(f"    Dataset samples for Fixed Set 3: {scaled_fixed_3_dataset} -> {num_fixed_3} final samples")
                        print(f"    Dataset samples for Random: {scaled_random_dataset} -> {num_random} final samples")
                    else:
                        # Original logic when we have enough samples
                        max_samples = requested_max_samples
                        num_fixed_1 = requested_fixed_1
                        num_fixed_2 = requested_fixed_2
                        num_fixed_3 = requested_fixed_3
                        num_random = requested_random
                    
                    print(f"  Proportional sampling configuration:")
                    print(f"    Fixed Set 1: {num_fixed_1} samples ({fixed_set_1_pct:.1%})")
                    print(f"    Fixed Set 2: {num_fixed_2} samples ({fixed_set_2_pct:.1%})")
                    print(f"    Fixed Set 3: {num_fixed_3} samples ({fixed_set_3_pct:.1%})")
                    print(f"    Random: {num_random} samples ({random_pct:.1%})")
                    print(f"    Total: {max_samples} samples")
                    
                    # Create non-overlapping index allocation
                    all_indices = list(range(len(dataset)))
                    random.shuffle(all_indices)  # Shuffle to randomize which samples go where
                    
                    # Allocate indices for each portion
                    random_dataset_samples_needed = (num_random + num_repetitions - 1) // num_repetitions if num_random > 0 else 0
                    
                    idx_start = 0
                    random_indices = all_indices[idx_start:idx_start + random_dataset_samples_needed]
                    idx_start += len(random_indices)
                    
                    fixed_1_indices = all_indices[idx_start:idx_start + num_fixed_1] if num_fixed_1 > 0 else []
                    idx_start += len(fixed_1_indices)
                    
                    fixed_2_indices = all_indices[idx_start:idx_start + num_fixed_2] if num_fixed_2 > 0 else []
                    idx_start += len(fixed_2_indices)
                    
                    fixed_3_indices = all_indices[idx_start:idx_start + num_fixed_3] if num_fixed_3 > 0 else []
                    
                    print(f"  Index allocation:")
                    print(f"    Random: {len(random_indices)} dataset samples -> {num_random} final samples")
                    print(f"    Fixed Set 1: {len(fixed_1_indices)} dataset samples -> {num_fixed_1} final samples")
                    print(f"    Fixed Set 2: {len(fixed_2_indices)} dataset samples -> {num_fixed_2} final samples")
                    print(f"    Fixed Set 3: {len(fixed_3_indices)} dataset samples -> {num_fixed_3} final samples")
                    
                    # Process random samples first
                    random_samples = []
                    if num_random > 0 and len(random_indices) > 0:
                        random_data_tuples = [
                            (dict(dataset[idx]), idx, "train", num_repetitions, args.max_system_prompt_length, args.simple_mode)
                            for idx in random_indices
                        ]
                        
                        print(f"    Processing {len(random_data_tuples)} samples for random sampling...")
                        if len(random_data_tuples) < workers or workers == 1:
                            for data_tuple in tqdm(random_data_tuples, desc="Processing Random Samples"):
                                processed_samples = process_data_sample_worker(data_tuple)
                                random_samples.extend(processed_samples)
                        else:
                            with Pool(processes=workers) as pool:
                                results = list(tqdm(
                                    pool.imap(process_data_sample_worker, random_data_tuples),
                                    total=len(random_data_tuples),
                                    desc="Processing Random Samples"
                                ))
                                for processed_samples in results:
                                    random_samples.extend(processed_samples)
                        
                        # Shuffle and take only what we need
                        random.shuffle(random_samples)
                        random_samples = random_samples[:num_random]
                    
                    # Process fixed set samples
                    all_processed_samples = random_samples.copy()
                    
                    # Generate fixed set samples with specific counts
                    available_set_1_models = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
                    available_set_2_models = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
                    available_set_3_models = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
                    
                    # Fixed Set 1
                    if num_fixed_1 > 0 and len(fixed_1_indices) > 0:
                        set_1_data_tuples = [
                            (dict(dataset[idx]), idx, "train", args.max_system_prompt_length, available_set_1_models, "fixed_set_1", args.simple_mode)
                            for idx in fixed_1_indices
                        ]
                        
                        set_1_samples = []
                        for data_tuple in tqdm(set_1_data_tuples, desc="Processing Fixed Set 1"):
                            processed = process_fixed_set_worker(data_tuple)
                            set_1_samples.extend(processed)
                        all_processed_samples.extend(set_1_samples)
                    
                    # Fixed Set 2
                    if num_fixed_2 > 0 and len(fixed_2_indices) > 0:
                        set_2_data_tuples = [
                            (dict(dataset[idx]), idx, "train", args.max_system_prompt_length, available_set_2_models, "fixed_set_2", args.simple_mode)
                            for idx in fixed_2_indices
                        ]
                        
                        set_2_samples = []
                        for data_tuple in tqdm(set_2_data_tuples, desc="Processing Fixed Set 2"):
                            processed = process_fixed_set_worker(data_tuple)
                            set_2_samples.extend(processed)
                        all_processed_samples.extend(set_2_samples)
                    
                    # Fixed Set 3
                    if num_fixed_3 > 0 and len(fixed_3_indices) > 0:
                        set_3_data_tuples = [
                            (dict(dataset[idx]), idx, "train", args.max_system_prompt_length, available_set_3_models, "fixed_set_3", args.simple_mode)
                            for idx in fixed_3_indices
                        ]
                        
                        set_3_samples = []
                        for data_tuple in tqdm(set_3_data_tuples, desc="Processing Fixed Set 3"):
                            processed = process_fixed_set_worker(data_tuple)
                            set_3_samples.extend(processed)
                        all_processed_samples.extend(set_3_samples)
                    
                    # Calculate actual generated samples
                    actual_random = len(random_samples)
                    actual_fixed_1 = len(fixed_1_indices) if len(fixed_1_indices) > 0 else 0
                    actual_fixed_2 = len(fixed_2_indices) if len(fixed_2_indices) > 0 else 0  
                    actual_fixed_3 = len(fixed_3_indices) if len(fixed_3_indices) > 0 else 0
                    
                    print(f"  Final generated samples:")
                    print(f"    Random samples: {actual_random}")
                    print(f"    Fixed Set 1 samples: {actual_fixed_1}")
                    print(f"    Fixed Set 2 samples: {actual_fixed_2}")
                    print(f"    Fixed Set 3 samples: {actual_fixed_3}")
                    print(f"    Total samples: {len(all_processed_samples)}")
                    
                else:
                    # Original logic - process all samples with random model sampling
                    all_processed_samples = []
                    data_tuples = [
                        (example, idx, "train", num_repetitions, args.max_system_prompt_length, args.simple_mode)
                        for idx, example in enumerate(dataset)
                    ]
                
                    # Process samples using multiprocessing (original logic)
                    if len(data_tuples) == 0:
                        print(f"  No samples to process in {parquet_file}")
                        continue
                        
                    # Use single processing for small datasets or when workers=1
                    if len(data_tuples) < workers or workers == 1:
                        print(f"  Processing {len(data_tuples)} samples sequentially (small dataset or workers=1)...")
                        for data_tuple in tqdm(data_tuples, desc=f"Processing {parquet_file}"):
                            processed_samples = process_data_sample_worker(data_tuple)
                            all_processed_samples.extend(processed_samples)
                    else:
                        print(f"  Processing {len(data_tuples)} samples using {workers} workers...")
                        with Pool(processes=workers) as pool:
                            # Use imap for progress tracking
                            results = list(tqdm(
                                pool.imap(process_data_sample_worker, data_tuples),
                                total=len(data_tuples),
                                desc=f"Processing {parquet_file}"
                            ))
                            
                            # Flatten results
                            for processed_samples in results:
                                all_processed_samples.extend(processed_samples)

            # Shuffle all samples
            random.shuffle(all_processed_samples)
                
            # Create dataset for this file
            if all_processed_samples:
                # For proportional sampling, we already have the right number of samples
                # For non-proportional sampling, apply the max_num_samples limit
                use_fixed_sets = getattr(args, 'use_fixed_sets', False)
                if use_fixed_sets and not getattr(args, 'use_fixed_set_only', False):
                    # Proportional sampling already handled max_num_samples
                    final_dataset = Dataset.from_list(all_processed_samples)
                else:
                    # Original behavior or fixed_set_only mode - apply max_num_samples limit
                    final_dataset = Dataset.from_list(all_processed_samples[:args.max_num_samples])
                
                # Generate new filename with updated trajectory number
                # Extract base name and trajectory number
                base_name = parquet_file.replace('.parquet', '')
                
                # Try to find the last numeric part (including k, m, etc.)
                import re

                # Pattern to match numbers at the end (including k, m, etc.)
                pattern = r'_(\d+(?:\.\d+)?[km]?)$'
                match = re.search(pattern, base_name)
                
                if match:
                    # Found a numeric suffix, replace it with new count
                    new_trajectory_count = len(final_dataset)
                    new_base_name = re.sub(pattern, f'_{new_trajectory_count}', base_name)
                else:
                    # No numeric suffix found, append the new count
                    new_base_name = base_name + f"_{len(final_dataset)}"
                
                output_file = os.path.join(output_dir, f"{new_base_name}.parquet")
                
                # Add error handling for Parquet writing
                try:
                    final_dataset.to_parquet(output_file)
                    print(f"  Completed processing {parquet_file}")
                    print(f"  Saved {len(final_dataset)} samples to {output_file}")
                    total_processed += len(final_dataset)
                except Exception as e:
                    print(f"  Error saving {output_file}: {e}")
                    # Try to save as JSON as fallback
                    try:
                        json_file = output_file.replace('.parquet', '.json')
                        with open(json_file, 'w') as f:
                            json.dump(all_processed_samples, f, indent=2)
                        print(f"  Saved as JSON fallback: {json_file}")
                    except Exception as json_e:
                        print(f"  Failed to save as JSON: {json_e}")
            else:
                print(f"  No samples processed for {parquet_file}")
            
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Total processed samples across all files: {total_processed}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets with model sampling and tool generation")
    parser.add_argument("--input_dir", default="./data/XXX",
                       help="Directory containing input parquet files")
    parser.add_argument("--output_dir", default="./data/XXX",
                       help="Directory to save processed data")
    parser.add_argument("--num_repetitions", type=int, default=3,
                       help="Number of times to repeat each sample with different model combinations")
    parser.add_argument("--premium_min", type=int, default=1,
                       help="Minimum number of premium models to sample")
    parser.add_argument("--premium_max", type=int, default=7,
                       help="Maximum number of premium models to sample")
    parser.add_argument("--budget_min", type=int, default=1,
                       help="Minimum number of budget models to sample")
    parser.add_argument("--budget_max", type=int, default=5,
                       help="Maximum number of budget models to sample")
    parser.add_argument("--standard_min", type=int, default=0,
                       help="Minimum number of standard models to sample")
    parser.add_argument("--standard_max", type=int, default=5,
                       help="Maximum number of standard models to sample")
    parser.add_argument("--specialized_min", type=int, default=0,
                       help="Minimum number of specialized models to sample")
    parser.add_argument("--specialized_max", type=int, default=3,
                       help="Maximum number of specialized models to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--max_system_prompt_length", type=int, default=2000,
                       help="Maximum length of system prompt")
    parser.add_argument("--max_num_samples", type=int, default=5000,
                       help="Maximum number of samples to process")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of worker processes for multiprocessing")
    parser.add_argument("--use_fixed_sets", action="store_true", default=False,
                       help="Enable fixed model sets with proportional sampling (replaces additive behavior)")
    parser.add_argument("--fixed_set_percentage", type=float, default=0.1,
                       help="Percentage of data to use for each fixed model set (default: 0.1 = 10%) - DEPRECATED, use individual set percentages")
    parser.add_argument("--fixed_set_1_percentage", type=float, default=0.1,
                       help="Percentage of total samples for Fixed Set 1 (default: 0.1 = 10%)")
    parser.add_argument("--fixed_set_2_percentage", type=float, default=0.1,
                       help="Percentage of total samples for Fixed Set 2 (default: 0.1 = 10%)")
    parser.add_argument("--fixed_set_3_percentage", type=float, default=0.1,
                       help="Percentage of total samples for Fixed Set 3 (default: 0.1 = 10%)")
    parser.add_argument("--use_fixed_set_only", action="store_true", default=False,
                       help="Use only fixed model sets (no random sampling). Overrides --use_fixed_sets")
    parser.add_argument("--fixed_set_choice", type=int, default=1, choices=[1, 2, 3],
                       help="Which fixed set to use when --use_fixed_set_only is enabled (1, 2, or 3, default: 1)")
    parser.add_argument("--simple_mode", action="store_true", default=False,
                       help="Enable simple mode - uses SYSTEM_PROMPT_SIMPLE and excludes select_response from tools_kwargs")

    args = parser.parse_args()

    # Validate proportional sampling arguments
    if args.use_fixed_sets and not args.use_fixed_set_only:
        total_fixed_pct = args.fixed_set_1_percentage + args.fixed_set_2_percentage + args.fixed_set_3_percentage
        if total_fixed_pct > 1.0:
            parser.error(f"Total fixed set percentages ({total_fixed_pct:.2f}) cannot exceed 1.0")
        if total_fixed_pct < 0:
            parser.error("Fixed set percentages cannot be negative")

    # example usage:
    # debug purpose (original logic only)
    # python router_data_process.py --output_dir data/router_train_data_filter_03_debug --num_repetitions 1 --premium_min 1 --premium_max 2 --budget_min 1 --budget_max 3 --standard_min 0 --standard_max 2 --specialized_min 0 --specialized_max 1 --seed 42 --max_system_prompt_length 2000 --workers 8
    # with proportional fixed model sets (new behavior):
    # python router_data_process.py --output_dir data/router_train_data_filter_03_proportional --use_fixed_sets --fixed_set_1_percentage 0.5 --fixed_set_2_percentage 0.1 --fixed_set_3_percentage 0.1 --max_num_samples 5000 --seed 42 --workers 8
    # with equal proportional fixed sets:
    # python router_data_process.py --output_dir data/router_train_data_filter_03_equal --use_fixed_sets --fixed_set_1_percentage 0.33 --fixed_set_2_percentage 0.33 --fixed_set_3_percentage 0.34 --max_num_samples 5000 --seed 42 --workers 8
    # with fixed model sets only (no random sampling):
    # python router_data_process.py --output_dir data/router_train_data_filter_03_fixed_only --use_fixed_set_only --fixed_set_choice 1 --seed 42 --workers 8 (choices: 1, 2, or 3)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Update sampling ranges based on command line arguments
    MODEL_SAMPLING_RANGES["premium"] = {"min": args.premium_min, "max": args.premium_max}
    MODEL_SAMPLING_RANGES["budget"] = {"min": args.budget_min, "max": args.budget_max}
    MODEL_SAMPLING_RANGES["standard"] = {"min": args.standard_min, "max": args.standard_max}
    MODEL_SAMPLING_RANGES["specialized"] = {"min": args.specialized_min, "max": args.specialized_max}
    
    print("Model sampling ranges:")
    for tier, ranges in MODEL_SAMPLING_RANGES.items():
        print(f"  {tier}: {ranges['min']}-{ranges['max']} models")
    
    # Print fixed model sets configuration if enabled
    if args.use_fixed_set_only:
        print(f"\nFixed set ONLY mode enabled:")
        print(f"  Using Fixed Set {args.fixed_set_choice} exclusively")
        if args.fixed_set_choice == 1:
            available_models = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
            print(f"  Fixed Set 1 models ({len(available_models)} available): {', '.join(available_models)}")
        elif args.fixed_set_choice == 2:
            available_models = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
            print(f"  Fixed Set 2 models ({len(available_models)} available): {', '.join(available_models)}")
        else:  # args.fixed_set_choice == 3
            available_models = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
            print(f"  Fixed Set 3 models ({len(available_models)} available): {', '.join(available_models)}")
        print(f"  No random sampling will be performed")
    elif args.use_fixed_sets:
        print(f"\nFixed model sets (proportional mode) enabled:")
        fixed_set_1_pct = getattr(args, 'fixed_set_1_percentage', 0.1)
        fixed_set_2_pct = getattr(args, 'fixed_set_2_percentage', 0.1)
        fixed_set_3_pct = getattr(args, 'fixed_set_3_percentage', 0.1)
        total_fixed_pct = fixed_set_1_pct + fixed_set_2_pct + fixed_set_3_pct
        random_pct = 1.0 - total_fixed_pct
        
        print(f"  Fixed Set 1: {fixed_set_1_pct:.1%} of total samples")
        print(f"  Fixed Set 2: {fixed_set_2_pct:.1%} of total samples")
        print(f"  Fixed Set 3: {fixed_set_3_pct:.1%} of total samples")
        print(f"  Random sampling: {random_pct:.1%} of total samples")
        print(f"  Total samples will be exactly: {args.max_num_samples}")
        
        available_set_1 = [model for model in FIXED_MODEL_SET_1 if model in MODEL_SPECS]
        available_set_2 = [model for model in FIXED_MODEL_SET_2 if model in MODEL_SPECS]
        available_set_3 = [model for model in FIXED_MODEL_SET_3 if model in MODEL_SPECS]
        print(f"  Fixed set 1 models ({len(available_set_1)} available): {', '.join(available_set_1)}")
        print(f"  Fixed set 2 models ({len(available_set_2)} available): {', '.join(available_set_2)}")
        print(f"  Fixed set 3 models ({len(available_set_3)} available): {', '.join(available_set_3)}")
    
    # Print simple mode configuration
    if args.simple_mode:
        print(f"\nSimple mode enabled:")
        print(f"  Using SYSTEM_PROMPT_SIMPLE instead of regular SYSTEM_PROMPT")
        print(f"  select_response function will be excluded from tools_kwargs")
    else:
        print(f"\nSimple mode disabled (using regular SYSTEM_PROMPT with select_response)")
    
    print(f"\nUsing {args.workers} worker processes for multiprocessing")
    
    # Print available models
    print_available_models()
    
    # Process the dataset
    process_dataset(args.input_dir, args.output_dir, args.num_repetitions, args.workers, args, MODEL_SAMPLING_RANGES)
