# xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2510.08439-b31b1b.svg)](https://arxiv.org/abs/2510.08439)
[![GitHub](https://img.shields.io/badge/GitHub-SalesforceAIResearch%2FxRouter-blue?logo=github)](https://github.com/SalesforceAIResearch/xRouter)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

xRouter is an intelligent LLM routing system trained with reinforcement learning to dynamically select optimal models from 20+ available LLMs while optimizing for both performance and cost.

<!-- ![Router Architecture](docs/_static/router_architecture.png) -->

## 🎯 Overview

xRouter enables intelligent, cost-aware routing across multiple LLM providers through:

- **Intelligent Model Selection**: Routes queries to 20+ LLMs across OpenAI, Together AI, and other providers
- **Cost-Performance Optimization**: Balances quality and cost through RL-trained routing policies
- **Agentic Prompt Engineering**: Automatically optimizes system prompts for target models
- **Multi-Turn Reasoning**: Supports complex workflows with response synthesis and selection
- **Production-Ready API**: OpenAI-compatible endpoints for seamless integration

## 🏗️ Architecture

### Core Components

```
rl_router/
├── verl/
│   ├── tools/
│   │   ├── router_tool.py              # Router and selection tool implementations
│   │   └── utils/router_utils.py       # Model specs, routing logic, LiteLLM integration
│   └── workers/
│       ├── rollout/sglang_rollout/     # Modified rollout for router training
│       └── reward_manager/async_dapo.py # Cost-aware reward shaping
├── data_preprocess/
│   └── router_data_preprocess.py        # Training data generation pipeline
├── examples/sglang_multiturn/config/
│   └── tool_config/
│       └── router_tool_config.yaml      # 20+ model tool definitions
├── train/                               # DAPO training scripts
├── evaluation/                          # Evaluation and serving pipeline
└── tests/router/                        # Router unit tests
```

### Key Features

- **🤖 20+ Model Support**: Access to GPT-5, GPT-4.1, o3/o4, DeepSeek R1, Qwen3, Kimi K2, and more
- **💰 Cost Optimization**: RL-trained policies that minimize costs while maintaining quality
- **🎯 Adaptive Routing**: Dynamic model selection based on query complexity and requirements
- **⚡ High Performance**: SGLang backend with optimized inference pipeline
- **📊 Rich Observability**: Comprehensive cost tracking and routing decision logs

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU(s) (8 GPUs recommended for training)
- API keys for model providers: OpenAI, Together AI, and/or Google Gemini

### Installation

1. **Clone and Setup Environment**
   ```bash
   git clone https://github.com/SalesforceAIResearch/xRouter.git
   cd xRouter
   
   conda create -n xrouter python=3.12
   conda activate xrouter
   
   # Install core dependencies
   pip install uv
   uv pip install torch==2.6.0
   uv pip install flash-attn==2.7.3 --no-build-isolation
   uv pip install -e .[gpu,math,vllm,test]
   
   # Install router-specific dependencies
   pip install litellm rich python-dotenv
   ```

2. **Configure API Keys**
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export TOGETHER_API_KEY="your_together_key"
   export GEMINI_API_KEY="your_gemini_key"  # optional
   ```

3. **Test Router Components**
   ```bash
   # Test model connections
   python tests/router/test_simple_connection.py
   
   # Verify all 20+ models are accessible
   python -c "from verl.tools.utils.router_utils import MODEL_SPECS; print(f'{len(MODEL_SPECS)} models available')"
   ```

## 📊 Data Preparation

### 1. Download Base Dataset

```bash
# Download training data (from Reasoning360)
python scripts/tools/download_guru.py
```

### 2. Generate Router Training Data

The data preprocessing pipeline creates training samples with dynamic model pools and optimized system prompts:

```bash
date_str=$(date +%m%d)

# Process hard tasks (pass rate < 0.15)
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.5 \
    --fixed_set_2_percentage 0.1 \
    --fixed_set_3_percentage 0.05 \
    --num_repetitions 2 \
    --premium_min 1 --premium_max 5 \
    --budget_min 1 --budget_max 5 \
    --standard_min 1 --standard_max 5 \
    --specialized_min 0 --specialized_max 3 \
    --seed 42 \
    --max_system_prompt_length 2000 \
    --output_dir data/train_hard_${date_str} \
    --input_dir data/train_filter_015 \
    --max_num_samples 400

# Process medium difficulty tasks
python data_preprocess/router_data_preprocess.py \
    --use_fixed_sets \
    --fixed_set_1_percentage 0.5 \
    --fixed_set_2_percentage 0.1 \
    --fixed_set_3_percentage 0.05 \
    --num_repetitions 1 \
    --output_dir data/train_medium_${date_str} \
    --input_dir data/train_filter_1565 \
    --max_num_samples 400
```

**Key Parameters:**
- `--use_fixed_sets`: Enable proportional sampling with fixed model sets
- `--fixed_set_X_percentage`: Proportion of samples using predefined model pools
- `--num_repetitions`: Create multiple versions with different model combinations
- `--premium/budget/standard/specialized_min/max`: Model sampling ranges per tier

### 3. Combine and Filter Data

```bash
# Combine all processed datasets
mkdir -p data/combined_train_${date_str}
mv data/train_*_${date_str}/* data/combined_train_${date_str}/

# Filter by token length
python scripts/token_distribution_analysis.py \
    --data_folder data/combined_train_${date_str}/ \
    --max_tokens 12000
```

## 🏋️ Training

### Overview

xRouter uses DAPO (Distributional Advantage Policy Optimization) with cost-aware reward shaping to learn optimal routing policies.

### Setup and Launch

```bash
# Configure Ray cluster
export RAY_TMPDIR=/path/to/ray_tmp
ray stop || true
head_node_ip=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address="$head_node_ip" --port=6595 --include-dashboard=False --block &

# Launch training
bash train/example_singlenode_router1.sh
```

**Key Training Configuration:**

```bash
# In train/example_singlenode_router1.sh
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA_DIR="data/combined_train_MMDD_filtered_12k/"

# Cost-aware reward parameters
reward_lambda=2.0        # Cost penalty coefficient
reward_K=1.0            # Reward threshold
cost_max=0.6            # Maximum normalized cost

# Tool configuration
tool_config_path="examples/sglang_multiturn/config/tool_config/router_tool_config.yaml"
max_turns=3             # Maximum agentic turns
```

**Training Features:**
- **DAPO Algorithm**: Distributional advantage estimation with cost awareness
- **Cost Shaping**: Penalizes expensive routing decisions
- **Multi-Turn Credit Assignment**: Proper reward attribution across turns
- **Tool-Augmented Training**: Learns to use 20+ model tools effectively

## 📈 Evaluation & Deployment

### 1. Host Router Model

Start the trained router model with tool calling enabled:

```bash
cd evaluation
bash host_router.sh  # Serves on port 8000
```

### 2. Launch Router API

Start the OpenAI-compatible API server:

```bash
bash serve_router.sh  # Serves on port 8800
```

### 3. Run Benchmarks

```bash
# Comprehensive testing
python test_serve.py --test all

# Specific test categories
python test_serve.py --test math_problem
python test_serve.py --test coding_task
python test_serve.py --test reasoning_task

# Benchmark on 17 evaluation datasets
python benchmark_router.py \
    --eval_data_dir data/offline_eval/ \
    --output_dir evaluation/outputs/
```

### API Usage Example

```python
import openai

# Initialize client
client = openai.OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="dummy"  # API key not required for local deployment
)

# Send request
response = client.chat.completions.create(
    model="router-tool-rl",
    messages=[
        {"role": "user", "content": "Solve this complex math problem: ..."}
    ],
    max_tokens=1000
)

# Get response
print(response.choices[0].message.content)

# Access routing metadata
metadata = response.router_metadata
print(f"Model used: {metadata['model_used']}")
print(f"Total cost: ${metadata['total_cost']:.6f}")
print(f"Routing strategy: {metadata['routing_strategy']}")
```

## 🔧 Advanced Configuration

### Routing Modes

**Agent Mode** (default)  
The router can call multiple models, compare responses, and use the `select_response` tool for ensemble decisions:
```bash
--simple_mode False  # In data preprocessing
```

**Simple Mode**  
The router selects exactly one model per turn, minimizing latency for simpler tasks:
```bash
python data_preprocess/router_data_preprocess.py --simple_mode
```

### Cost Tuning

Adjust cost sensitivity in training:

```bash
# Higher λ = stronger cost penalty
reward_lambda=2.0    # Default: balanced
reward_lambda=5.0    # More cost-conscious
reward_lambda=0.5    # Prioritize quality
```

### Model Pool Customization

Edit fixed model sets in `data_preprocess/router_data_preprocess.py`:

```python
FIXED_MODEL_SET_1 = [
    "gpt-5", "gpt-5-mini", "o3", "o4-mini",
    "gpt-oss-120b", "gpt-oss-20b"
]
```

## 📋 Available Models

| Tier | Models | Best For |
|------|--------|----------|
| **Premium** | GPT-5, GPT-4.1, o3, Qwen3-235B-Instruct, Kimi K2 | Mission-critical tasks |
| **Standard** | GPT-5-Mini, GPT-4.1-Mini, o4-Mini, GPT-OSS-120B | Balanced performance |
| **Budget** | GPT-5-Nano, GPT-4.1-Nano, GPT-4o-Mini, GPT-OSS-20B | High-volume tasks |
| **Specialized** | o3, DeepSeek-R1, Qwen3-235B-Thinking, Qwen3-Coder-480B | Domain-specific |

See `verl/tools/utils/router_utils.py` for full specifications.

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Unit tests
pytest tests/router/ -v

# Test model API connections
python tests/router/test_simple_connection.py

# Integration tests
cd evaluation
python test_host.py      # Test hosted router model
python test_serve.py     # Test full router API pipeline
```

## 📖 Documentation

For detailed documentation on specific components:
- [Data Preprocessing Guide](data_preprocess/README.md)
- [Evaluation & Serving Guide](evaluation/README.md)
- [Router Implementation Details](README_ROUTER.md)

## 📝 Key Innovations

### Data Processing
- **Dynamic Model Pools**: Each training sample contains a unique set of available models
- **Prompt Optimization**: Router learns to engineer optimal system prompts for each target model
- **Curriculum Learning**: Three-tier fixed model sets for progressive difficulty (simple → hard tasks)

### Training
- **Cost-Aware Rewards**: Implements `reward = quality - λ × cost` with normalized cost tracking
- **Tool Augmentation**: Learns to effectively use `call_<model_name>` and `select_response` tools
- **Enhanced Rollout**: Extracts routing decisions and cost metrics from multi-turn conversations

### Serving
- **LiteLLM Integration**: Unified interface supporting 20+ models across multiple providers
- **Context Tracking**: Maintains multi-turn conversation state and routing history
- **Rich Observability**: Comprehensive logs for cost, latency, and routing decisions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find xRouter useful for your research or applications, please cite our paper:

```bibtex
@article{qian2025xrouter,
  title={xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning},
  author={Qian, Cheng and Liu, Zuxin and Kokane, Shirley and Prabhakar, Akshara and Qiu, Jielin and Chen, Haolin and Liu, Zhiwei and Ji, Heng and Yao, Weiran and Heinecke, Shelby and Savarese, Silvio and Xiong, Caiming and Wang, Huan},
  journal={arXiv preprint arXiv:2510.08439},
  year={2025}
}
```

## 🙏 Acknowledgements

This project builds upon exceptional work from the open-source community:

- **[Reasoning360](https://github.com/LLM360/Reasoning360)**: Foundational RL training framework and base infrastructure
- **[VERL](https://github.com/volcengine/verl)**: Robust reinforcement learning infrastructure for distributed LLM training
- **[SGLang](https://github.com/sgl-project/sglang)**: High-performance LLM serving backend with efficient inference
- **[LiteLLM](https://github.com/BerriAI/litellm)**: Unified API interface supporting 20+ LLM providers

We are grateful to the maintainers and contributors of these projects for their invaluable contributions to the AI/ML ecosystem.

## 📮 Contact

For questions, issues, or collaboration opportunities, please [open an issue on GitHub](https://github.com/SalesforceAIResearch/xRouter/issues).
