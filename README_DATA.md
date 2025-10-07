# Simple Data Processing for Router Training

This guide explains how to process the `simple_samples.json` data for router training using the provided scripts.

## Available Scripts

### 1. `process_simple_data_standalone.py` (Recommended)

A standalone script that includes all necessary processing logic without heavy dependencies.

**Features:**
- No dependency on the full `verl` package 
- Includes basic model specifications for router training
- Self-contained processing logic
- Full configuration options

**Usage:**

```bash
# Basic processing (all samples, 3 repetitions per sample)
python3 process_simple_data_standalone.py \
    --input_file data/new_data/simple_samples.json \
    --output_dir data/simple_router_processed

# Quick test with limited samples
python3 process_simple_data_standalone.py \
    --max_samples 100 \
    --num_repetitions 2 \
    --output_dir data/simple_router_test

# Simple mode (using simplified system prompt, no select_response)
python3 process_simple_data_standalone.py \
    --simple_mode \
    --num_repetitions 2 \
    --output_dir data/simple_router_simple

# Custom model sampling ranges
python3 process_simple_data_standalone.py \
    --premium_min 1 --premium_max 2 \
    --budget_min 1 --budget_max 3 \
    --standard_min 1 --standard_max 2 \
    --specialized_min 0 --specialized_max 1
```

### 2. `process_simple_data.py` (Requires Dependencies)

Uses the full router preprocessing pipeline from `router_data_preprocess.py`.

**Requirements:**
- Full `verl` package installation
- All dependencies (tensordict, litellm, etc.)

**Usage:**

```bash
# Only use if you have the full environment set up
python3 process_simple_data.py --input_file data/new_data/simple_samples.json
```

## Input Data Format

The script expects `simple_samples.json` with the following structure:

```json
[
  {
    "question": "What is an apple?",
    "model_answer": "An apple is a small fruit...",
    "ground_truth": "An apple is an edible fruit..."
  },
  ...
]
```

## Output Format

The processed data will be in router training format with:

- **System prompt**: Router instructions for model selection and prompt engineering
- **User query**: The original question from simple_samples.json  
- **Tools**: Function definitions for calling various models
- **Extra info**: Metadata including original answers and ground truth

### Output Files

1. `simple_samples_router.parquet` - Main processed dataset
2. `generation_config.yaml` - Configuration used for processing

## Configuration Options

### Model Sampling

Control how many models from each tier are included:

- `--premium_min/max`: High-quality models (gpt-4o, o3, etc.)
- `--budget_min/max`: Cost-effective models (deepseek-r1, etc.)  
- `--standard_min/max`: Balanced models (gemini-2.5-pro, etc.)
- `--specialized_min/max`: Task-specific models (qwen3-coder, etc.)

### Processing Options

- `--num_repetitions`: How many versions to create per sample (default: 3)
- `--max_samples`: Limit number of input samples to process
- `--simple_mode`: Use simplified system prompt without select_response tool
- `--seed`: Random seed for reproducible results

### Example: Quick Processing for Testing

```bash
python3 process_simple_data_standalone.py \
    --max_samples 50 \
    --num_repetitions 1 \
    --simple_mode \
    --output_dir quick_test \
    --seed 42
```

### Example: Full Production Processing  

```bash
python3 process_simple_data_standalone.py \
    --input_file data/new_data/simple_samples.json \
    --output_dir data/production_router_data \
    --num_repetitions 3 \
    --premium_min 2 --premium_max 4 \
    --budget_min 3 --budget_max 6 \
    --seed 42
```

## Model Information

The standalone script includes the following models:

**Premium Tier:**
- gpt-4o, gpt-4o-mini  
- o3, o3-pro

**Budget Tier:**  
- gpt-4.1-nano
- deepseek-r1
- qwen3-235b-instruct

**Standard Tier:**
- gemini-2.5-pro, gemini-2.5-flash-lite
- kimi-k2

**Specialized Tier:**
- qwen3-coder-480b (coding tasks)

## Troubleshooting

### Import Errors
- Use `process_simple_data_standalone.py` to avoid dependency issues
- Ensure you have `datasets`, `tqdm`, `numpy`, and `yaml` installed

### Memory Issues
- Use `--max_samples` to process smaller batches
- Reduce `--num_repetitions` to create fewer variants per sample

### File Not Found
- Check that `data/new_data/simple_samples.json` exists
- Use absolute paths if running from different directories

## Verifying Output

Check the processed data:

```bash
python3 -c "
from datasets import Dataset
ds = Dataset.from_parquet('your_output_dir/simple_samples_router.parquet')
print(f'Processed {len(ds)} samples')
print('Sample keys:', list(ds[0].keys()))
print('Tools available:', list(ds[0]['extra_info']['tools_kwargs'].keys()))
"
```
