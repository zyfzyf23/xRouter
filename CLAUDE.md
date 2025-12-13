# CLAUDE.md - XRouter Lightweight Refactoring Guide
请使用简体中文回答问题。
在给出代码时，考虑是否急剧占用了大量资源，比如不可以突然占用大量内存。
项目文件scripts下的代码是重构代码，并不是xRouter 的原生代码，如果任务要求复现xRouter的代码，尽可能忽略scripts文件夹下的代码。

**🚨 注意：当前项目正在进行特定方向的重构。请优先遵循以下“当前开发任务”中的指示，忽略下方关于 Ray 集群、vLLM 和分布式训练的旧指令。**

## 1. 🎯 当前开发任务：轻量化离线 DPO 重构

### 项目目标
我们要将 XRouter 从原本的“在线强化学习 (DAPO)”架构修改为适合单卡笔记本 (RTX 4060 8GB) 运行的 **“离线缓存 + DPO”** 架构。

### 硬件限制 (Hard Constraints)
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **Environment**: WSL2 (Ubuntu 22.04)
- **禁止项**: 
    - ❌ 禁止使用 vLLM 或 SGLang (显存不足)。
    - ❌ 禁止使用 Ray 分布式训练。
    - ❌ 禁止加载 7B 以上的本地模型进行训练。
- **必须项**:
    - ✅ 训练必须使用 `bitsandbytes` (4-bit Quantization)。
    - ✅ 训练必须使用 `peft` (LoRA)。
    - ✅ 基座模型锁定为 `Qwen/Qwen2.5-1.5B-Instruct`。

### 环境限制
- **conda环境**: py310

---

## 2. 🗺️ 开发路线图 (Step-by-Step)

请按以下五个阶段协助我完成代码编写。

### 阶段一：核心定位 (Discovery)
- **目标**: 找到“编排引擎 (Orchestration Engine)”逻辑，即负责调用 `litellm` 或外部 API 的核心函数。
- **关键文件**: 重点关注 `verl/tools/utils/router_utils.py`。
- **任务**: 理解 `call_model` 接口，准备将其剥离出来用于造数据。

### 阶段二：构建离线缓存 (Offline Data Gen)
- **脚本目标**: `scripts/generate_offline_cache.py`
- **逻辑**: 
    1. 读取训练数据集 (如 GSM8K)。
    2. **强制遍历 (Forced Traversal)**: 不使用 Router 决策，而是对每个问题，强制调用模型池中的所有模型 (如 `["gpt-4o", "qwen-turbo", "gpt-3.5"]`)。
    3. **数据记录**: 必须保存 `prompt`, `model_name`, `response`, `is_correct` (需复现原评估逻辑), `token_usage`。
- **注意**: 仅调用 API 或轻量级推理，不加载 RL Actor 模型。

### 阶段三：DPO 数据构建 (Preprocessing)
- **脚本目标**: `scripts/preprocess_dpo.py`
- **逻辑**: 
    1. 读取 `offline_cache.jsonl`。
    2. **奖励公式**: 实现论文公式 $R = R_{binary} \times (K - \lambda C)$。
       - 若 `is_correct` 为 False，Reward = 0。
       - 若 Correct，Reward = $1.0 - 0.1 \times Cost$ (示例系数)。
    3. **配对生成**: 对同一 Prompt，比较不同模型的 Reward。
       - `Reward(A) > Reward(B)` -> `chosen=A`, `rejected=B`。
    4. **输出**: HuggingFace Dataset 格式 (`dpo_train_data.json`)。

### 阶段四：轻量化训练 (Lightweight Training)
- **脚本目标**: `scripts/train_dpo_light.py`
- **工具栈**: 使用 `trl` (DPOTrainer) + `peft` + `bitsandbytes`。
- **配置**:
    - Base Model: `Qwen/Qwen2.5-1.5B-Instruct`
    - Quantization: `load_in_4bit=True` (关键！防止 OOM)
    - LoRA: `r=16`, `target_modules=["q_proj", "v_proj", ...]`
    - Batch Size: 1 (配合 Gradient Accumulation)

### 阶段五：集成与验证 (Evaluation)
- **脚本目标**: `scripts/evaluate_lora.py`
- **逻辑**:
    1. 加载 Base Model (1.5B) + 训练好的 LoRA Adapter。
    2. 恢复 Router 的自主决策模式 (不再强制遍历)。
    3. 运行测试集，统计 **Accuracy** 和 **Total Cost**。

---

## 3. 🛠️ 当前环境配置 (Current Env)

我们使用的是轻量化环境，与原文档不同：
```bash
python: 3.10
torch: 2.5.1+cu121
libraries: flash_attn (pre-compiled), bitsandbytes, peft, trl, litellm

### 核心架构组件

```
xRouter/
├── verl/                           # VERL 强化学习框架
│   ├── tools/
│   │   ├── utils/router_utils.py     # 核心路由器功能和模型规范
│   │   ├── router_tool.py           # 路由工具实现
│   │   └── schemas/                # 工具模式定义
│   ├── workers/
│   │   ├── rollout/sglang_rollout/  # SGLang 推理后端
│   │   └── reward_manager/         # 成本感知奖励塑造
│   └── recipe/dapo/                # DAPO 训练算法
├── data_preprocess/
│   └── router_data_preprocess.py    # 训练数据生成管道
├── examples/
│   └── sglang_multiturn/config/tool_config/
│       └── router_tool_config.yaml # 20+ 模型工具定义
├── train/                          # 训练脚本和配置
├── evaluation/                     # 评估和服务部署
└── tests/router/                    # 路由器单元测试
```

## 常用开发命令

### 环境设置
```bash
# 创建基础环境
conda create -n xrouter python=3.12
conda activate xrouter

# 安装核心依赖
pip install uv
uv pip install torch==2.6.0
uv pip install flash-attn==2.7.3 --no-build-isolation
uv pip install -e .[gpu,math,vllm,test]

# 路由器特定依赖
pip install litellm rich python-dotenv

# API 密钥配置（至少需要一个）
export OPENAI_API_KEY="your_openai_key"
export TOGETHER_API_KEY="your_together_key"
export GEMINI_API_KEY="your_gemini_key"
```

### 测试和验证
```bash
# 测试模型连接
python tests/router/test_simple_connection.py

# 验证所有模型可访问性
python -c "from verl.tools.utils.router_utils import MODEL_SPECS; print(f'{len(MODEL_SPECS)} models available')"

# 单元测试
pytest tests/router/ -v
```

### 数据预处理
```bash
# 下载基础训练数据
python scripts/tools/download_guru.py

# 生成路由器训练数据（困难任务）
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
    --output_dir data/train_hard_MMDD \
    --input_dir data/train_filter_015 \
    --max_num_samples 400
```

### 训练配置和启动
```bash
# 设置 Ray 集群
export RAY_TMPDIR=$HOME/ray_tmp
ray stop || true
head_node_ip=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address="$head_node_ip" --port=6595 --include-dashboard=False --block &

# 启动训练
bash train/example_singlenode_router1.sh
```

关键训练参数：
- `BASE_MODEL`: 基础模型（通常是 Qwen/Qwen2.5-7B-Instruct）
- `TRAIN_DATA_DIR`: 预处理的路由器训练数据
- `reward_lambda`: 成本惩罚系数（默认 2.0）
- `tool_config_path`: 路由器工具配置路径
- `max_turns`: 最大智能体轮次（默认 3）

### 部署和服务
```bash
# 启动路由器模型服务器
cd evaluation
bash host_router.sh  # 端口 8000

# 启动 OpenAI 兼容 API
bash serve_router.sh  # 端口 8800

# 运行基准测试
python benchmark_router.py \
    --eval_data_dir data/offline_eval/ \
    --output_dir evaluation/outputs/
```

## 路由系统架构

### 模型分层和选择策略

**高级模型**：GPT-5、GPT-4.1、o3、Qwen3-235B-Instruct、Kimi K2
- 用于关键任务、复杂智能体工作流

**标准模型**：GPT-5-Mini、GPT-4.1-Mini、o4-Mini、GPT-OSS-120B
- 用于成本敏感工作流、性能平衡应用

**预算模型**：GPT-5-Nano、GPT-4.1-Nano、GPT-4o-Mini、GPT-OSS-20B
- 用于大量应用、实时交互

**专业模型**：o3、DeepSeek-R1、Qwen3-235B-Thinking、Qwen3-Coder-480B
- 用于数学推理、科学研究、编程任务

### DAPO 训练算法

- **分布式优势估计**：使用多个推估计优势分布
- **成本感知奖励**：`reward = quality - λ × cost` 形式的奖励函数
- **多轮信用分配**：跨智能体轮次的正确奖励归属
- **工具使用学习**：路由器学习最优的 `call_<model_name>` 和 `select_response` 工具使用模式

### 路由策略

**简单模式**：每轮只选择一个模型（最小延迟）
```bash
python data_preprocess/router_data_preprocess.py --simple_mode
```

**智能体模式**：可调用多个模型并使用 `select_response` 进行集成决策（默认）
```bash
# 在数据预处理中默认启用
python data_preprocess/router_data_preprocess.py  # 默认为智能体模式
```

## 关键配置文件

### 工具配置
- `examples/sglang_multiturn/config/tool_config/router_tool_config.yaml`：定义 20+ 个模型工具
- 每个工具包含模型描述、参数和成本信息
- 支持 `call_<model_name>` 函数和 `select_response` 选择机制

### 训练配置
- `train/example_singlenode_router1.sh`：主要训练脚本
- `data_preprocess/router_data_preprocess.py`：训练数据生成管道
- `verl/tools/utils/router_utils.py`：模型规范和统一路由器接口

## 模型规范和 API 集成

核心模型注册表在 `verl/tools/utils/router_utils.py` 中定义，包含 20+ 个模型：

```python
# 使用示例
from verl.tools.utils.router_utils import LLMRouter, MODEL_SPECS

router = LLMRouter()
response, metadata = router.call_model(
    "gpt-4o",
    [{"role": "user", "content": "解释量子计算"}],
    {"temperature": 0.7, "max_tokens": 1024}
)
```

每个模型规范包含：
- 定价信息（输入/输出每百万 token 美元）
- 上下文窗口和最大输出长度
- 能力标签（推理、编程、数学等）
- 性能基准测试结果

## 推理后端支持

### SGLang 集成
```bash
# SGLang 后端用于训练时推理
export SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
python -m verl.third_party.sglang.srt_runner \
    --model-path $model_path \
    --port $SGLANG_HTTP_PORT
```

### vLLM 集成
```bash
# vLLM 后端用于生产部署
python -m verl.third_party.vllm.vllm_v_0_6_3.llm_engine \
    --model $model_path \
    --gpu-memory-utilization 0.6
```

## 成本感知训练

### 奖励函数设计
- `reward_lambda` 控制成本敏感性（越高 = 越强的成本惩罚）
- `reward_K` 设置奖励阈值
- `cost_max` 最大归一化成本
- 支持多轮信用分配和工具使用跟踪

### 课程学习
- **固定模型集**：三个预定义模型集用于渐进难度训练
- **动态模型池**：每个训练样本包含唯一的模型组合
- **提示优化**：路由器学习为每个目标模型设计最优系统提示

## API 使用示例

### OpenAI 兼容客户端
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="dummy"  # 本地部署不需要 API 密钥
)

response = client.chat.completions.create(
    model="router-tool-rl",
    messages=[
        {"role": "user", "content": "编写一个 Python 函数来反转链表"}
    ],
    max_tokens=1000
)

# 访问路由元数据
metadata = response.router_metadata
print(f"使用模型: {metadata['model_used']}")
print(f"总成本: ${metadata['total_cost']:.6f}")
print(f"路由策略: {metadata['routing_strategy']}")
```

### 直接路由器使用
```python
from verl.tools.utils.router_utils import LLMRouter

router = LLMRouter()

# 调用特定模型
response, metadata = await router.acall_model(
    model_id="gpt-5-mini",
    messages=[{"role": "user", "content": "解释机器学习"}],
    sampling_params={"temperature": 0.7, "max_tokens": 1024}
)

print(f"响应: {response}")
print(f"成本: ${metadata['cost']:.6f}")
print(f"Token: {metadata['input_tokens']} + {metadata['output_tokens']}")
```

## 评估和基准测试

### 离线评估
```bash
# 在 17 个基准数据集上评估
python evaluation/benchmark_router.py \
    --model_path /path/to/your/trained/router \
    --eval_data_dir data/offline_eval/ \
    --output_dir ./evaluation_results
```

### 在线评估
```bash
# 运行全面测试套件
python evaluation/test_serve.py --test all

# 特定测试类别
python evaluation/test_serve.py --test math_problem
python evaluation/test_serve.py --test coding_task
python evaluation/test_serve.py --test reasoning_task
```

## 开发注意事项

- 所有路由器工具遵循 `call_<model_name>` 命名约定
- 模型规范在 `MODEL_SPECS` 中是单一真实来源
- 训练使用课程学习进行渐进模型池难度
- 系统支持简单路由（单模型）和智能体模式（多模型 + 选择）
- 成本跟踪集成在训练和推理管道中
- 思维模型（o3、o4、DeepSeek-R1）需要高 token 限制（8192+）以支持内部推理
- 使用 FSDP 和参数/优化器卸载进行内存效率训练