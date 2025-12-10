#!/usr/bin/env python3
"""
Debug script to understand Guru dataset structure and find math/code samples
"""

import os
import json
from datasets import load_dataset

# Set HF endpoint for China network
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def print_sample_structure(example, idx):
    """Print detailed structure of a sample"""
    print(f"\n=== Sample {idx} ===")
    print(f"Keys: {list(example.keys())}")

    # Print key fields
    if 'ability' in example:
        print(f"Ability: {example['ability']}")
    if 'data_source' in example:
        print(f"Data Source: {example['data_source']}")

    # Check prompt structure
    if 'prompt' in example:
        prompt = example['prompt']
        print(f"\nPrompt type: {type(prompt)}")
        if isinstance(prompt, list):
            print(f"Prompt length: {len(prompt)}")
            if prompt:
                first_msg = prompt[0] if prompt else None
                if first_msg and isinstance(first_msg, dict):
                    print(f"First message role: {first_msg.get('role')}")
                    content = first_msg.get('content', '')
                    print(f"First message content (first 100 chars): {str(content)[:100]}...")

    # Check response
    if 'response' in example and example['response']:
        print(f"\nResponse exists, type: {type(example['response'])}")
        print(f"Response (first 100 chars): {str(example['response'])[:100]}...")

    if 'completion' in example and example['completion']:
        print(f"\nCompletion exists, type: {type(example['completion'])}")
        print(f"Completion (first 100 chars): {str(example['completion'])[:100]}...")

def main():
    print("Loading Guru dataset for debugging...")

    # Load dataset
    try:
        ds = load_dataset("LLM360/guru-RL-92k", split="train", streaming=True)
        print("Dataset loaded successfully!\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Track domains
    abilities = {}
    data_sources = {}
    math_samples = []
    code_samples = []

    print("Scanning first 100 samples to understand structure...")

    for i, example in enumerate(ds):
        if i >= 100:
            break

        # Print first few samples in detail
        if i < 5:
            print_sample_structure(example, i)

        # Track abilities and data sources
        ability = str(example.get('ability', '')).lower()
        data_source = str(example.get('data_source', '')).lower()

        if ability:
            abilities[ability] = abilities.get(ability, 0) + 1
        if data_source:
            data_sources[data_source] = data_sources.get(data_source, 0) + 1

        # Check for math/code keywords
        math_keywords = ["math", "gsm8k", "reasoning", "arithmetic", "algebra", "geometry", "calculus"]
        code_keywords = ["codegen", "python", "code", "programming", "coding", "leetcode", "java", "javascript"]

        is_math = any(kw in ability or kw in data_source for kw in math_keywords)
        is_code = any(kw in ability or kw in data_source for kw in code_keywords)

        if is_math and len(math_samples) < 3:
            math_samples.append((i, ability, data_source, example.get('prompt', '')[:200]))

        if is_code and len(code_samples) < 3:
            code_samples.append((i, ability, data_source, example.get('prompt', '')[:200]))

    # Print statistics
    print("\n\n" + "="*60)
    print("ABILITY DISTRIBUTION (Top 10)")
    print("="*60)
    sorted_abilities = sorted(abilities.items(), key=lambda x: x[1], reverse=True)[:10]
    for ability, count in sorted_abilities:
        print(f"{ability}: {count}")

    print("\n" + "="*60)
    print("DATA SOURCE DISTRIBUTION (Top 10)")
    print("="*60)
    sorted_sources = sorted(data_sources.items(), key=lambda x: x[1], reverse=True)[:10]
    for source, count in sorted_sources:
        print(f"{source}: {count}")

    print("\n" + "="*60)
    print("MATH SAMPLES FOUND")
    print("="*60)
    for idx, ability, source, prompt in math_samples:
        print(f"\nSample {idx}:")
        print(f"  Ability: {ability}")
        print(f"  Source: {source}")
        print(f"  Prompt: {prompt}...")

    print("\n" + "="*60)
    print("CODE SAMPLES FOUND")
    print("="*60)
    for idx, ability, source, prompt in code_samples:
        print(f"\nSample {idx}:")
        print(f"  Ability: {ability}")
        print(f"  Source: {source}")
        print(f"  Prompt: {prompt}...")

if __name__ == "__main__":
    main()