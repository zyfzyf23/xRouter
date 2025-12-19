#!/usr/bin/env python3
"""
服务器版 DPO 训练脚本 - 12GB显存限制版
适用于 RTX 3090 24GB 显存，但限制使用不超过12GB
"""

import os
import json
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 核心库导入
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


# 强制限制 PyTorch 只能看到 50% 的显存 (约12GB)
torch.cuda.set_per_process_memory_fraction(0.5, 0)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@dataclass
class DPOServerConfig:
    """服务器 DPO 训练配置参数 - 12GB显存限制"""

    # 模型路径 - Docker容器内路径
    model_name_or_path: str = "/workspace/Qwen2.5-1.5B-Instruct"

    # 数据路径 - Docker容器内路径
    train_data_path: str = "/workspace/xRouter/data/dpo_train_math.jsonl"

    # 输出路径 - Docker容器内路径
    output_dir: str = "/workspace/xRouter/outputs/dpo_lora_server"

    # 训练参数 - 12GB显存优化，优于8GB本地版
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # 比本地8GB版大1倍
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # 等效batch_size=16
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # 序列长度 - 与本地相同或更好
    max_length: int = 2304  # 与本地相同
    max_prompt_length: int = 1152
    max_target_length: int = 1280

    # LoRA 配置 - 与本地相同或更好
    lora_r: int = 16  # 与本地相同
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None  # 自动设置

    # 量化配置 - 可以选择不量化，因为12GB比8GB宽松
    load_in_4bit: bool = True  
    load_in_8bit: bool = False   
    bnb_8bit_compute_dtype: str = "bfloat16"
    bnb_4bit_compute_dtype: str = "bfloat16" # 虽然权重是 4-bit，但计算时提升到 bfloat16 精度
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True # 对量化常数（scale/zero-point）也进行 8-bit 量化

    # 保存和评估
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2  # 减少保存的模型数量
    logging_steps: int = 10
    eval_steps: int = 500

    # 内存优化设置
    gradient_checkpointing: bool = True  # 启用梯度检查点
    optim: str = "paged_adamw_8bit"  # 使用8bit优化器
    dataloader_pin_memory: bool = False  # 避免额外的内存占用

    # DPO 特定参数
    beta: float = 0.1
    loss_type: str = "sigmoid"

    # 内存限制
    max_memory_gb: float = 12.0  # 最大显存使用限制

    def __post_init__(self):
        """初始化后处理"""
        # 设置 Qwen2.5 的 LoRA target modules
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def monitor_memory(max_memory_gb: float = 12.0):
    """监控 GPU 内存使用，确保不超过限制"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.1f}GB")

        # 如果显存使用超过限制，触发垃圾回收
        if allocated > max_memory_gb:
            logger.warning(f"Memory usage ({allocated:.2f}GB) exceeds limit ({max_memory_gb}GB), triggering cleanup...")
            torch.cuda.empty_cache()
            # 重新检查
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"After cleanup: {allocated_after:.2f}GB allocated")


def load_and_preprocess_data(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    split_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    加载并预处理 DPO 数据
    """
    logger.info(f"Loading data from {data_path}")

    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 读取 JSONL 数据
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(data)} samples")

    # 设置随机种子
    torch.manual_seed(seed)

    # 转换为 DPO 所需格式
    processed_data = []
    for idx, sample in enumerate(data):
        try:
            # 将对话格式转换为字符串
            prompt_messages = sample['prompt']
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Chosen: prompt + chosen response
            chosen_messages = prompt_messages + sample['chosen']
            chosen = tokenizer.apply_chat_template(
                chosen_messages,
                tokenize=False
            )

            # Rejected: prompt + rejected response
            rejected_messages = prompt_messages + sample['rejected']
            rejected = tokenizer.apply_chat_template(
                rejected_messages,
                tokenize=False
            )

            # 检查长度
            prompt_tokens = tokenizer(prompt)['input_ids']
            chosen_tokens = tokenizer(chosen)['input_ids']
            rejected_tokens = tokenizer(rejected)['input_ids']

            # 跳过过长的样本
            if (len(prompt_tokens) > max_prompt_length or
                len(chosen_tokens) > max_length or
                len(rejected_tokens) > max_length):
                logger.warning(f"Skipping sample {idx} due to excessive length")
                continue

            processed_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            continue

    logger.info(f"Successfully processed {len(processed_data)} samples")

    # 随机打乱数据
    import random
    random.seed(seed)
    random.shuffle(processed_data)

    # 划分训练集和验证集
    if split_ratio > 0:
        split_idx = int(len(processed_data) * (1 - split_ratio))
        train_data = processed_data[:split_idx]
        eval_data = processed_data[split_idx:]

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        logger.info(f"Train dataset: {len(train_data)} samples")
        logger.info(f"Eval dataset: {len(eval_data)} samples")
    else:
        train_dataset = Dataset.from_list(processed_data)
        eval_dataset = None
        logger.info(f"Train dataset: {len(processed_data)} samples")

    return train_dataset, eval_dataset


def load_model_and_tokenizer(config: DPOServerConfig):
    """加载模型和分词器 - 12GB显存限制版"""

    # 加载分词器
    logger.info(f"Loading tokenizer from {config.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="left"
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 量化配置
    if config.load_in_4bit:
        logger.info("Using 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    elif config.load_in_8bit:
        logger.info("Using 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        logger.info("Loading model without quantization...")
        bnb_config = None

    # 加载模型
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        use_cache=False,  # 训练时关闭缓存
        max_memory={0: f"{int(config.max_memory_gb)}GB"}  # 限制显存使用
    )

    # 配置 gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA 配置 - 较小的rank以节省显存
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        inference_mode=False
    )

    # 应用 LoRA
    logger.info("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} "
        f"|| All params: {all_params:,} "
        f"|| Trainable%: {100 * trainable_params / all_params:.2f}%"
    )

    # 监控显存
    monitor_memory(config.max_memory_gb)

    return model, tokenizer


def get_training_arguments(config: DPOServerConfig) -> TrainingArguments:
    """获取训练参数 - 12GB显存优化版"""

    return TrainingArguments(
        # 基本参数
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,

        # 批次和梯度
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # 优化器参数
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,

        # 保存和日志
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,

        # 内存优化
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=config.dataloader_pin_memory,
        optim=config.optim,

        # 其他
        remove_unused_columns=False,
        max_grad_norm=config.max_grad_norm,

        # 报告
        report_to="none",

        # 精度
        fp16=False,
        bf16=True,

        # 分布式训练
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,

        # 额外的内存优化
        dataloader_num_workers=0,  # 避免多进程
        save_on_each_node=False,
    )


class MemoryMonitorCallback(transformers.TrainerCallback):
    """内存监控回调函数 - 12GB限制版"""

    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """每个步骤结束时监控内存"""
        monitor_memory(self.max_memory_gb)
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """每个epoch结束时强制垃圾回收"""
        torch.cuda.empty_cache()
        monitor_memory(self.max_memory_gb)
        return control


def main():
    """主训练函数"""
    # 加载配置
    config = DPOServerConfig()

    # 从环境变量读取配置（可选）
    if os.getenv("MODEL_PATH"):
        config.model_name_or_path = os.getenv("MODEL_PATH")
    if os.getenv("DATA_PATH"):
        config.train_data_path = os.getenv("DATA_PATH")
    if os.getenv("OUTPUT_DIR"):
        config.output_dir = os.getenv("OUTPUT_DIR")

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    # 设置PyTorch内存分配策略 - 根据错误提示添加
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

    # 启用内存高效的注意力机制（如果可用）
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    # 设置随机种子
    transformers.set_seed(42)

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 检查 CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        logger.info(f"Memory limit set to: {config.max_memory_gb}GB")
    else:
        logger.warning("CUDA not available!")
        return

    # 在加载模型前清理显存
    torch.cuda.empty_cache()
    monitor_memory(config.max_memory_gb)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)

    # 加载数据 - 不限制样本数，使用完整数据集
    train_dataset, eval_dataset = load_and_preprocess_data(
        config.train_data_path,
        tokenizer,
        config.max_length,
        config.max_prompt_length,
        split_ratio=0.1
    )

    # 创建 DPO 配置
    dpo_config = DPOConfig(
        # 基本训练参数
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=config.dataloader_pin_memory,
        optim=config.optim,
        remove_unused_columns=False,
        max_grad_norm=config.max_grad_norm,
        report_to="none",
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,

        # DPO 特定参数
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_length - config.max_prompt_length,
        generate_during_eval=False,
    )

    # 创建 DPO 训练器
    # DPO会自动复制模型作为参考模型，但使用PEFT时会更节省内存
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT会自动处理参考模型
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 添加内存监控回调
    dpo_trainer.add_callback(MemoryMonitorCallback(config.max_memory_gb))

    # 开始训练
    logger.info("Starting DPO training (12GB memory limit)...")
    train_result = dpo_trainer.train()

    # 保存最终模型
    logger.info("Saving final model...")
    dpo_trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 保存训练状态
    dpo_trainer.log_metrics("train", train_result.metrics)
    dpo_trainer.save_state()

    # 评估
    if eval_dataset is not None:
        logger.info("Running evaluation...")
        eval_result = dpo_trainer.evaluate()
        dpo_trainer.log_metrics("eval", eval_result)

        logger.info(f"Final eval loss: {eval_result.get('eval_loss', 'N/A')}")

    logger.info("Training completed!")

    # 打印模型信息
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")

    # 显存清理
    del model, dpo_trainer
    torch.cuda.empty_cache()
    monitor_memory(config.max_memory_gb)


if __name__ == "__main__":
    main()