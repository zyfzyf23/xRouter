#!/usr/bin/env python3
"""
Example script showing how to use API models for difficulty filtering.

This demonstrates the complete workflow:
1. Run inference with an API model (OpenAI or Together AI)
2. Calculate rewards 
3. Generate pass rate JSON
4. Add pass rates to training data

Prerequisites:
- Set OPENAI_API_KEY and/or TOGETHER_API_KEY in .env file
- Have training data in data/train/ directory

Usage:
python examples/api_model_filtering_example.py --model gpt-4o-mini --dataset math__gsm8k_1k
"""

import argparse
import os
import sys
sys.path.append('Reasoning360')

from rich.console import Console
from model_filtering.api_models import get_supported_models

console = Console()

def main():
    parser = argparse.ArgumentParser(description="API Model Filtering Example")
    parser.add_argument("--model", type=str, required=True, 
                        help="API model name (use 'list' to see supported models)")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name (without .parquet extension)")
    parser.add_argument("--generations", type=int, default=8, 
                        help="Number of generations per prompt")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size (keep small for API models)")
    parser.add_argument("--max_requests_per_minute", type=int, default=30, 
                        help="API rate limit")
    
    args = parser.parse_args()
    
    # Show supported models
    if args.model == "list":
        supported = get_supported_models()
        console.print("\n[bold]Supported API Models:[/bold]")
        for provider, models in supported.items():
            console.print(f"\n[cyan]{provider.upper()}:[/cyan]")
            for model in models:
                console.print(f"  • {model}")
        return
    
    # Check environment variables
    console.print("🔍 Checking API keys...")
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    
    if not openai_key:
        console.print("[yellow]⚠️ OPENAI_API_KEY not found in environment[/yellow]")
    else:
        console.print("✅ OPENAI_API_KEY found")
    
    if not together_key:
        console.print("[yellow]⚠️ TOGETHER_API_KEY not found in environment[/yellow]")
    else:
        console.print("✅ TOGETHER_API_KEY found")
    
    dataset_path = f"data/train/{args.dataset}.parquet"
    if not os.path.exists(dataset_path):
        console.print(f"❌ Dataset not found: {dataset_path}")
        console.print("Available datasets:")
        train_dir = "data/train"
        if os.path.exists(train_dir):
            for file in sorted(os.listdir(train_dir)):
                if file.endswith(".parquet"):
                    console.print(f"  • {file[:-8]}")  # Remove .parquet
        return
    
    console.print(f"📂 Using dataset: {dataset_path}")
    
    # Build commands to run
    console.print("\n[bold]Example commands to run:[/bold]\n")
    
    # Step 1: Full pipeline
    console.print("1️⃣ [cyan]Complete pipeline (recommended):[/cyan]")
    console.print(f"python model_filtering/run_full_pipeline.py \\")
    console.print(f"    --model_path {args.model} \\")
    console.print(f"    --dataset_parquet_path {dataset_path} \\")
    console.print(f"    --output_dir ./api_filter_output \\")
    console.print(f"    --n {args.generations} \\")
    console.print(f"    --batch_size {args.batch_size} \\")
    console.print(f"    --max_requests_per_minute {args.max_requests_per_minute}")
    
    # Step 2: Individual steps
    console.print("\n2️⃣ [cyan]Individual steps:[/cyan]")
    
    console.print("\n  🔄 Inference:")
    console.print(f"python model_filtering/run_inference.py \\")
    console.print(f"    --model_path {args.model} \\")
    console.print(f"    --dataset_parquet_path {dataset_path} \\")
    console.print(f"    --output_dir ./api_filter_output \\")
    console.print(f"    --n {args.generations} \\")
    console.print(f"    --batch_size {args.batch_size} \\")
    console.print(f"    --max_requests_per_minute {args.max_requests_per_minute}")
    
    console.print("\n  💯 Reward calculation:")
    console.print(f"python model_filtering/run_reward.py \\")
    console.print(f"    --model_path {args.model} \\")
    console.print(f"    --dataset_parquet_path {dataset_path} \\")
    console.print(f"    --output_dir ./api_filter_output")
    
    console.print("\n  📊 Add pass rates to training data:")
    console.print(f"python model_filtering/run_add_pr.py \\")
    console.print(f"    --parquet_in {dataset_path} \\")
    console.print(f"    --model_path {args.model} \\")
    console.print(f"    --pass_rate_json ./api_filter_output/{args.dataset}_{args.model.split('/')[-1]}_pass_rates.json \\")
    console.print(f"    --parquet_out_dir ./filtered_data")
    
    # Cost estimation
    console.print("\n[bold]💰 Estimated cost (rough):[/bold]")
    
    # Read dataset to estimate size
    try:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=dataset_path, split="train")
        num_samples = len(ds)
        
        # Rough token estimates
        avg_prompt_tokens = 200  # Conservative estimate
        avg_response_tokens = 100  # Conservative estimate
        
        total_input_tokens = num_samples * args.generations * avg_prompt_tokens
        total_output_tokens = num_samples * args.generations * avg_response_tokens
        
        # Cost estimates for popular models
        cost_estimates = {
            "gpt-4o-mini": (0.00015, 0.00060),  # input, output per 1K tokens
            "gpt-4o": (0.0025, 0.010),
            "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30/1000, 0.30/1000),
        }
        
        if args.model in cost_estimates:
            input_price, output_price = cost_estimates[args.model]
            estimated_cost = (total_input_tokens * input_price / 1000 + 
                            total_output_tokens * output_price / 1000)
            
            console.print(f"  Dataset samples: {num_samples}")
            console.print(f"  Estimated input tokens: {total_input_tokens:,}")
            console.print(f"  Estimated output tokens: {total_output_tokens:,}")
            console.print(f"  Estimated cost: ${estimated_cost:.2f}")
        else:
            console.print("  [yellow]Cost estimation not available for this model[/yellow]")
            
    except Exception as e:
        console.print(f"  [yellow]Could not estimate cost: {e}[/yellow]")
    
    console.print("\n[green]Ready to run! Choose one of the commands above.[/green]")

if __name__ == "__main__":
    main()