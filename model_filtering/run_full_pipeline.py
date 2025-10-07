#!/usr/bin/env python3
"""
Complete difficulty filtering pipeline that runs inference, calculates rewards, and generates pass rates.
Supports both API models (OpenAI, Together AI) and local models (vLLM).

Example usage:
python run_full_pipeline.py --model_path gpt-4o --dataset_parquet_path data/train/math__gsm8k_1k.parquet --output_dir ./output
"""

import argparse
import json
import os
import glob
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel

from verl.tools.utils.router_utils import MODEL_SPECS

def is_api_model(model_path: str) -> bool:
    """Check if model is an API model from router_utils"""
    return model_path in MODEL_SPECS

def get_supported_models():
    """Get supported models organized by provider"""
    models_by_provider = {}
    for model_id, spec in MODEL_SPECS.items():
        provider = spec.provider
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model_id)
    return models_by_provider

console = Console()

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    console.print(f"🔄 {description}")
    console.print(f"   Command: [cyan]{' '.join(cmd)}[/cyan]")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        console.print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"❌ {description} failed with exit code {e.returncode}")
        console.print(f"   STDOUT: {e.stdout}")
        console.print(f"   STDERR: {e.stderr}")
        return False

def collect_pass_rates(output_dir: str, model_name: str, dataset_name: str) -> Dict[str, float]:
    """Collect pass rates from all rank directories"""
    console.print("📊 Collecting pass rates from all ranks...")
    
    base_dir = os.path.join(output_dir, dataset_name, model_name)
    rank_dirs = sorted(glob.glob(os.path.join(base_dir, "dp*")))
    
    all_pass_rates = {}
    
    for rank_dir in rank_dirs:
        final_results_file = os.path.join(rank_dir, "final_results.json")
        if not os.path.exists(final_results_file):
            console.print(f"⚠️ Missing final_results.json in {rank_dir}")
            continue
            
        with open(final_results_file, 'r') as f:
            rank_data = json.load(f)
            
        for key, sample in rank_data["results"].items():
            if "pass_rate" in sample:
                # Extract index from extra_info
                extra_info = sample.get("extra_info", {})
                idx = None
                for field in ("idx", "index", "id"):
                    if field in extra_info:
                        idx = str(extra_info[field])
                        break
                
                if idx is not None:
                    all_pass_rates[idx] = sample["pass_rate"]
    
    console.print(f"✅ Collected {len(all_pass_rates)} pass rates")
    return all_pass_rates

def get_model_defaults(model_path: str) -> dict:
    """Get default sampling parameters optimized for specific model types"""
    if model_path in MODEL_SPECS:
        spec = MODEL_SPECS[model_path]
        
        # Default parameters based on model capabilities and type
        defaults = {
            "max_new_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.9
        }
        
        # Adjust for thinking models (need more tokens for reasoning)
        if "thinking" in spec.capabilities:
            defaults["max_new_tokens"] = 8192
            
        # Adjust for reasoning models
        if "reasoning" in spec.capabilities:
            defaults["max_new_tokens"] = 4096
            
        # Adjust for coding models
        if "coding" in spec.capabilities:
            defaults["max_new_tokens"] = 4096
            
        # Provider-specific adjustments
        if spec.provider == "openai":
            if spec.api_alias.startswith(("o3", "o4", "gpt-5")):
                defaults["temperature"] = 1.0  # Required for these models
                
        elif spec.provider == "google":
            defaults["max_new_tokens"] = 8192  # Gemini benefits from higher limits
            
        return defaults
    else:
        # Local model defaults
        return {
            "max_new_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.9
        }

def main():
    parser = argparse.ArgumentParser(description="Complete difficulty filtering pipeline")
    
    # Core arguments
    parser.add_argument("--model_path", type=str, required=True, help="Model path or API model name")
    parser.add_argument("--dataset_parquet_path", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output_dir", type=str, default="./diff_filter_output", help="Output directory")
    
    # Inference arguments (use None as default to enable model-specific defaults)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--n", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max tokens to generate (default: model-optimized)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: model-optimized)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling (default: model-optimized)")
    parser.add_argument("--use_default_sampling", action="store_true", help="Use model-optimized default sampling parameters")
    
    # API model arguments
    parser.add_argument("--max_requests_per_minute", type=int, default=50, 
                        help="Rate limit for API models")
    
    # Parallel processing arguments
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--reward_workers", type=int, default=16, help="CPU workers for reward calculation")
    
    # Control arguments
    parser.add_argument("--debug", action="store_true", help="Debug mode with smaller data")
    parser.add_argument("--force_regenerate", action="store_true", 
                        help="Force regeneration ignoring existing results")
    parser.add_argument("--skip_inference", action="store_true", 
                        help="Skip inference step (use existing results)")
    parser.add_argument("--skip_reward", action="store_true", 
                        help="Skip reward calculation (use existing results)")
    
    args = parser.parse_args()
    
    # Apply model-optimized defaults if requested or if parameters are None
    if args.use_default_sampling or args.max_new_tokens is None or args.temperature is None or args.top_p is None:
        defaults = get_model_defaults(args.model_path)
        
        if args.max_new_tokens is None:
            args.max_new_tokens = defaults["max_new_tokens"]
        if args.temperature is None:
            args.temperature = defaults["temperature"]
        if args.top_p is None:
            args.top_p = defaults["top_p"]
            
        console.print(f"[dim]Using model-optimized defaults: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}[/dim]")
    
    # Display configuration
    console.rule("[bold]Difficulty Filtering Pipeline", style="blue")
    
    model_type = "API Model" if is_api_model(args.model_path) else "Local Model"
    console.print(
        Panel(
            f"[bold]Model:[/bold] {args.model_path} ({model_type})\n"
            f"[bold]Dataset:[/bold] {args.dataset_parquet_path}\n"
            f"[bold]Output:[/bold] {args.output_dir}\n"
            f"[bold]Generations:[/bold] {args.n}\n"
            f"[bold]Batch size:[/bold] {args.batch_size}\n"
            f"[bold]Max tokens:[/bold] {args.max_new_tokens}\n"
            f"[bold]Temperature:[/bold] {args.temperature}\n"
            f"[bold]Top-p:[/bold] {args.top_p}",
            title="📋 Configuration",
            border_style="blue"
        )
    )
    
    # Show supported API models if requested
    if args.model_path == "list":
        supported = get_supported_models()
        console.print("\n[bold]Supported API Models:[/bold]")
        for provider, models in supported.items():
            console.print(f"\n[cyan]{provider.upper()}:[/cyan]")
            for model in models:
                console.print(f"  • {model}")
        return
    
    dataset_name = os.path.basename(args.dataset_parquet_path).rsplit(".parquet", 1)[0]
    model_name = args.model_path.split("/")[-1]
    
    success = True
    
    # Step 1: Run inference
    if not args.skip_inference:
        console.rule("[bold]Step 1: Running Inference", style="green")
        
        inference_cmd = [
            sys.executable, "model_filtering/run_inference.py",
            "--model_path", args.model_path,
            "--dataset_parquet_path", args.dataset_parquet_path,
            "--output_dir", args.output_dir,
            "--batch_size", str(args.batch_size),
            "--n", str(args.n),
            "--max_new_tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--dp_size", str(args.dp_size),
            "--tp_size", str(args.tp_size),
            "--max_requests_per_minute", str(args.max_requests_per_minute),
        ]
        
        if args.debug:
            inference_cmd.append("--debug")
        if args.force_regenerate:
            inference_cmd.append("--force_regenerate")
            
        success = run_command(inference_cmd, "Running inference")
        if not success:
            console.print("❌ Inference failed, stopping pipeline")
            return
    else:
        console.print("⏩ Skipping inference step")
    
    # Step 2: Calculate rewards
    if not args.skip_reward:
        console.rule("[bold]Step 2: Calculating Rewards", style="yellow")
        
        reward_cmd = [
            sys.executable, "model_filtering/run_reward.py",
            "--model_path", args.model_path,
            "--dataset_parquet_path", args.dataset_parquet_path,
            "--output_dir", args.output_dir,
            "--reward_workers", str(args.reward_workers),
        ]
        
        if args.debug:
            reward_cmd.append("--debug")
            
        success = run_command(reward_cmd, "Calculating rewards")
        if not success:
            console.print("❌ Reward calculation failed, stopping pipeline")
            return
    else:
        console.print("⏩ Skipping reward calculation step")
    
    # Step 3: Generate pass rate JSON
    console.rule("[bold]Step 3: Generating Pass Rate JSON", style="cyan")
    
    pass_rates = collect_pass_rates(args.output_dir, model_name, dataset_name)
    
    if pass_rates:
        # Save pass rates JSON
        pass_rates_file = os.path.join(args.output_dir, f"{dataset_name}_{model_name}_pass_rates.json")
        with open(pass_rates_file, 'w') as f:
            json.dump(pass_rates, f, indent=2)
        
        console.print(f"✅ Saved pass rates to [cyan]{pass_rates_file}[/cyan]")
        
        # Display statistics
        rates = list(pass_rates.values())
        avg_rate = sum(rates) / len(rates) if rates else 0
        console.print(f"📊 Average pass rate: [cyan]{avg_rate:.3f}[/cyan]")
        console.print(f"📊 Total samples: [cyan]{len(rates)}[/cyan]")
        
        # Show distribution
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(thresholds) - 1):
            count = sum(1 for r in rates if thresholds[i] <= r < thresholds[i+1])
            console.print(f"📊 Pass rate [{thresholds[i]:.1f}, {thresholds[i+1]:.1f}): {count} samples")
            
    else:
        console.print("❌ No pass rates collected")
        success = False
    
    # Summary
    console.rule("[bold]Pipeline Complete", style="green" if success else "red")
    if success:
        console.print("🎉 Difficulty filtering pipeline completed successfully!")
        console.print(f"📄 Results saved in: [cyan]{args.output_dir}[/cyan]")
        console.print(f"📊 Pass rates JSON: [cyan]{pass_rates_file}[/cyan]")
    else:
        console.print("❌ Pipeline failed - check errors above")

if __name__ == "__main__":
    main()