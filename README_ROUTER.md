# xRouter: Cost-Aware Multi-Model Routing System

This README provides step-by-step instructions for setting up, training, and using the xRouter system - an intelligent LLM routing system that dynamically selects optimal models for tasks while optimizing both performance and cost.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training the Router](#training-the-router)
- [Using the Router](#using-the-router)
- [Testing and Evaluation](#testing-and-evaluation)
- [Troubleshooting](#troubleshooting)

## Overview

xRouter is a cost-aware multi-model routing system that:
- **Intelligently routes** queries to 20+ available LLMs across multiple providers
- **Optimizes costs** while maintaining high performance through reinforcement learning
- **Supports multi-model orchestration** with response synthesis and selection
- **Provides production-ready** APIs compatible with OpenAI standards

## Prerequisites

- **Hardware**: NVIDIA GPUs with CUDA 12.4+ (8 GPUs recommended for training)
- **Python**: 3.12+
- **Storage**: ~100GB for training data and models
- **API Keys**: OpenAI, Together AI, and/or Google Gemini API keys

## Installation

### 1. Clone and Setup Base Environment

```bash
# Clone the repository
git clone git@github.com:LLM360/Reasoning360.git
cd Reasoning360

# Create conda environment
conda create -n Reasoning360 python=3.12
conda activate Reasoning360

# Install CUDA toolkit
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit cuda-nvcc

# Install core dependencies
pip install uv  # for faster package installation
uv pip install torch==2.6.0
uv pip install flash-attn==2.7.3 --no-build-isolation
uv pip install -e .[gpu,math,vllm,test]
```

### 2. Install Additional Router Dependencies

```bash
# Install router-specific dependencies
pip install litellm
pip install rich  # for console output
pip install python-dotenv  # for environment variables
```

## Environment Setup

### 1. API Keys Configuration

Create a `.env` file in the project root or set environment variables:

```bash
# Required API keys (at least one needed)
export OPENAI_API_KEY="your_openai_api_key_here"
export TOGETHER_API_KEY="your_together_ai_api_key_here"
export GEMINI_API_KEY="your_google_gemini_api_key_here"

# Optional: WandB for training logging
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_DISABLED=false  # Set to true to disable logging
```

### 2. GPU Environment Setup

```bash
# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Clean up conflicting GPU variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
```

### 3. SandboxFusion Setup (Optional for Code Execution)

For secure code execution during training/evaluation:

**Option 1: Local Installation**
```bash
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion
poetry install
make run-online
```

**Option 2: SLURM Container (Production)**
```bash
enroot import docker://varad0309/code_sandbox:server
sbatch scripts/sandbox/run_server.sbatch
```

Configure sandbox servers:
```bash
export SANDBOX_FUSION_SERVERS="your-server-hostname"
```

## Data Preparation

### 1. Download Base Training Data

Download the Guru RL dataset or prepare your own:

```bash
# Download Guru data (recommended)
python scripts/tools/download_guru.py
```

This creates:
- `./data/train/` - Training files
- `./data/online_eval/` - Online evaluation files
- `./data/offline_eval/` - Offline evaluation files

### 2. Create Filtered Training Data

Create difficulty-filtered datasets for router training:

```bash
# Run the data filtering pipeline from model_filtering
python model_filtering/run_inference.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_parquet_path "data/train/your_dataset.parquet" \
    --output_dir "./diff_filter_output"

python model_filtering/run_reward.py \
    --model_path "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_parquet_path "data/train/your_dataset.parquet" \
    --output_dir "./diff_filter_output"
```

This creates filtered datasets like:
- `data/train_filter_05/` (pass rate < 0.5, hard tasks)
- `data/train_filter_06_1/` (0.6 < pass rate < 1.0, medium tasks)

### 3. Generate Router Training Data

Use the data preprocessing script to create router-specific training data:

```bash
# Set current date for versioning
date_str=$(date +%m%d)

# Process hard tasks (low pass rate = challenging)
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.3 \
    --fixed_set_2_percentage 0.05 \
    --fixed_set_3_percentage 0.05 \
    --num_repetitions 2 \
    --premium_min 1 --premium_max 5 \
    --budget_min 1 --budget_max 5 \
    --standard_min 1 --standard_max 5 \
    --specialized_min 0 --specialized_max 3 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/015_train_hard_${date_str} \
    --input_dir data/train_filter_015 \
    --max_num_samples 400

# Process math-specific tasks
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.5 \
    --fixed_set_2_percentage 0.1 \
    --fixed_set_3_percentage 0.02 \
    --num_repetitions 1 \
    --premium_min 1 --premium_max 5 \
    --budget_min 1 --budget_max 5 \
    --standard_min 1 --standard_max 5 \
    --specialized_min 0 --specialized_max 3 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/015_train_hard_math_${date_str} \
    --input_dir data/train_filter_015_math \
    --max_num_samples 1500

# Process medium difficulty tasks
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.5 \
    --fixed_set_2_percentage 0.1 \
    --fixed_set_3_percentage 0.05 \
    --num_repetitions 1 \
    --premium_min 1 --premium_max 5 \
    --budget_min 1 --budget_max 5 \
    --standard_min 1 --standard_max 5 \
    --specialized_min 0 --specialized_max 3 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/1565_train_medium_${date_str} \
    --input_dir data/train_filter_1565 \
    --max_num_samples 400

# Process simple tasks
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.5 \
    --fixed_set_2_percentage 0.1 \
    --fixed_set_3_percentage 0.05 \
    --num_repetitions 1 \
    --premium_min 1 --premium_max 5 \
    --budget_min 1 --budget_max 5 \
    --standard_min 1 --standard_max 5 \
    --specialized_min 0 --specialized_max 3 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/06_1_train_simple_${date_str} \
    --input_dir data/train_filter_06_1 \
    --max_num_samples 200
```

### 4. Combine and Filter Training Data

```bash
# Create combined training directory
mkdir -p data/combined_train_${date_str}

# Combine all processed data
mv data/1565_train_medium_${date_str}/* data/combined_train_${date_str}/
mv data/015_train_hard_${date_str}/* data/combined_train_${date_str}/
mv data/015_train_hard_math_${date_str}/* data/combined_train_${date_str}/
mv data/06_1_train_simple_${date_str}/* data/combined_train_${date_str}/

# Clean up intermediate directories
rm -rf data/1565_train_medium_${date_str}
rm -rf data/015_train_hard_${date_str}
rm -rf data/015_train_hard_math_${date_str}
rm -rf data/06_1_train_simple_${date_str}

# Analyze token distribution and filter
python scripts/token_distribution_analysis.py \
    --data_folder data/combined_train_${date_str}/ \
    --max_tokens 12000

# Clean up after filtering (script creates filtered version)
rm -rf data/combined_train_${date_str}
```

### 5. Prepare Evaluation Data

```bash
# Generate evaluation dataset
python data_preprocess/router_data_preprocess.py \
    --use_fixed_set_only \
    --fixed_set_choice 1 \
    --num_repetitions 1 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/processed_eval_500_${date_str} \
    --input_dir data/offline_eval/ \
    --max_num_samples 500

# Filter evaluation data
python scripts/token_distribution_analysis.py \
    --data_folder data/processed_eval_500_${date_str} \
    --max_tokens 12000

# Clean up unfiltered version
rm -rf data/processed_eval_500_${date_str}
```

## Training the Router

### 1. Prepare Ray Cluster

```bash
# Setup Ray temporary directory
mkdir -p $HOME/ray_tmp_0825
export RAY_TMPDIR=$HOME/ray_tmp_0825
export TMPDIR=$HOME/ray_tmp_0825
export RAY_SESSION_DIRECTORY=$HOME/ray_tmp_0825

# Stop any existing Ray cluster
ray stop || true
pkill -f "ray" || true
sleep 2

# Start Ray head node
head_node_ip=$(hostname -I | awk '{print $1}')
port=6595
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --include-dashboard=False --block &
sleep 5
```

### 2. Configure Training Parameters

Update the training script `scripts/train/train-0902-7b.sh` with your data paths:

```bash
# Data paths - update these to your generated data
TRAIN_DATA_DIR=/path/to/your/data/combined_train_MMDD_filtered_12k/
ONLINE_EVAL_DATA_DIR=/path/to/your/data/processed_eval_500_MMDD_filtered_12k/

# Model configuration
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct  # or your preferred base model

# Cost-aware reward parameters
reward_lambda=2.0  # Cost penalty coefficient
reward_K=1.0       # Reward threshold
cost_max=0.6       # Maximum normalized cost
simple_mode=False  # Use full agent mode vs simple mode

# Tool configuration
tool_config_path="./examples/sglang_multiturn/config/tool_config/router_tool_config.yaml"
max_turns=3
```

### 3. Start Training

```bash
# Make training script executable
chmod +x scripts/train/train-0902-7b.sh

# Start training (modify paths in script first)
./scripts/train/train-0902-7b.sh
```

The training script will:
- Use GRPO (Group Relative Policy Optimization) algorithm
- Train for 2 epochs with cost-aware rewards
- Save checkpoints every epoch
- Log to WandB (if enabled)
- Use 8 GPUs with FSDP optimization

### 4. Monitor Training

```bash
# Check Ray status
ray status

# Monitor GPU usage
nvidia-smi

# Check training logs
tail -f /path/to/training/logs

# View WandB dashboard (if enabled)
# Navigate to your WandB project URL
```

## Using the Router

### 1. Test Router Components

First, test the router utilities:

```bash
# Test the router utilities
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/test_router_simple.py
```

### 2. Start Router Server

```bash
# Start the router server (simple mode)
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/serve_router_simple_mode.py \
    --port 8800 \
    --router-model gpt-5-nano \
    --log-level info
```

For production routing with trained model:
```bash
# Use your trained router model
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/serve_router_simple_mode.py \
    --port 8800 \
    --router-model /path/to/your/trained/router/model \
    --log-level info
```

### 3. Test Router Server

```bash
# Test the router server
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/test_simple_router.py
```

### 4. Use Router via API

**OpenAI-Compatible Client:**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="dummy"  # API key not required for local router
)

response = client.chat.completions.create(
    model="router",
    messages=[
        {"role": "user", "content": "Write a Python function to reverse a linked list"}
    ]
)

print(response.choices[0].message.content)
```

**Direct Router Usage:**
```python
from verl.tools.utils.router_utils import LLMRouter

router = LLMRouter()

# Call specific model
response, metadata = await router.acall_model(
    model_id="gpt-5-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    sampling_params={"temperature": 0.7, "max_tokens": 1024}
)

print(f"Response: {response}")
print(f"Cost: ${metadata['cost']:.6f}")
print(f"Tokens: {metadata['input_tokens']} + {metadata['output_tokens']}")
```

## Testing and Evaluation

### 1. Unit Tests

```bash
# Run router-specific tests
python -m pytest tests/router/ -v

# Test individual components
python verl/tools/router_tool_conversion_demo.py
```

### 2. Model Performance Testing

```bash
# Test model performance on coding tasks
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/test_router_simple.py

# Run comprehensive router tests
PYTHONPATH=$HOME/1_project/Reasoning360 python scripts/router/test_comprehensive_router.py
```

### 3. Evaluation on Benchmarks

```bash
# Run offline evaluation on 17 benchmarks
python scripts/offline_eval/run_evaluation.py \
    --model_path /path/to/your/trained/router \
    --eval_data_dir data/offline_eval/ \
    --output_dir ./evaluation_results
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Check API keys are set
echo $OPENAI_API_KEY
echo $TOGETHER_API_KEY
echo $GEMINI_API_KEY

# Test API connectivity
python -c "from verl.tools.utils.router_utils import LLMRouter; router = LLMRouter()"
```

**2. Memory Issues During Training**
```bash
# Reduce batch sizes in training script
train_prompt_bsz=32  # Reduce from 64
n_resp_per_prompt=4  # Reduce from 8

# Enable CPU offloading
offload=True
```

**3. Ray Cluster Issues**
```bash
# Clean Ray environment
ray stop
pkill -f "ray" || true
rm -rf /tmp/ray/ray_current_cluster
rm -rf /tmp/ray/session_*

# Restart Ray
ray start --head --node-ip-address="$(hostname -I | awk '{print $1}')" --port=6595
```

**4. Model Loading Issues**
```bash
# Check model availability
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# Check GPU memory
nvidia-smi

# Use smaller model if needed
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
```

**5. Data Processing Issues**
```bash
# Check data integrity
python -c "import pandas as pd; df = pd.read_parquet('data/train/your_file.parquet'); print(df.columns, len(df))"

# Verify required columns
# Expected: prompt, completion, data_source, apply_chat_template, reward_model
```

### Performance Optimization

**Training Speed:**
- Use `uv pip` instead of `pip` for faster installation
- Enable FSDP parameter/optimizer offloading
- Use gradient checkpointing
- Optimize batch sizes for your GPU memory

**Inference Speed:**
- Use vLLM backend for faster inference
- Enable chunked prefill for long contexts
- Adjust `gpu_memory_utilization` parameter

**Cost Optimization:**
- Tune `reward_lambda` parameter (higher = more cost penalty)
- Use `simple_mode=True` for basic routing
- Monitor cost metrics in training logs

## Directory Structure

After setup, your directory structure should look like:

```
Reasoning360/
├── data/
│   ├── train_filter_05/          # Hard tasks (pass rate < 0.5)
│   ├── train_filter_06_1/        # Medium tasks (0.6 < pass rate < 1.0)
│   ├── combined_train_MMDD_filtered_12k/  # Final training data
│   ├── processed_eval_500_MMDD_filtered_12k/  # Evaluation data
│   └── offline_eval/             # Benchmark evaluation data
├── scripts/
│   ├── train/train-0902-7b.sh    # Main training script
│   └── router/                   # Router server and test scripts
├── examples/sglang_multiturn/config/tool_config/
│   └── router_tool_config.yaml   # Tool configuration
├── verl/tools/utils/
│   └── router_utils.py           # Core router implementation
├── data_preprocess/
│   └── router_data_preprocess.py # Data preprocessing script
└── checkpoints/                  # Training checkpoints (created during training)
```

This completes the comprehensive setup guide for xRouter. The system provides intelligent, cost-aware routing across 20+ language models with production-ready APIs and extensive customization options.