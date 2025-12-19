import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
import logging
import numpy as np
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 配置 ---
MODEL_NAME = "/workspace/Qwen2.5-1.5B-Instruct"
DATA_FILE = "data/sft_train_balanced.jsonl"
OUTPUT_DIR = "outputs/sft_1218_with_low_lr"


def main():
    logger.info(f"Loading model: {MODEL_NAME}")

    # 检查 GPU 环境
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"CUDA device: {gpu_name}")
        logger.info(f"CUDA memory: {total_memory_gb:.1f}GB")

        # 检查BFloat16支持
        if torch.cuda.is_bf16_supported():
            logger.info("GPU supports BFloat16")
            use_bf16 = True
        else:
            logger.warning("GPU does not support BFloat16, using FP16")
            use_bf16 = False

        # 设置内存限制（最多12GB）
        max_memory_gb = min(12.0, total_memory_gb * 0.9)  # 不超过90%的总内存
        memory_fraction = max_memory_gb / total_memory_gb

        logger.info(f"Memory limit set to: {max_memory_gb:.1f}GB")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)

        # 清理GPU缓存
        torch.cuda.empty_cache()

    else:
        logger.warning("CUDA not available!")
        return

    # 1. 量化配置 (适配 8GB 显存)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Using 4-bit quantization with {'BFloat16' if use_bf16 else 'FP16'}")

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFT通常right padding

    # 4. LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = prepare_model_for_kbit_training(model)

    # 内存监控函数
    def monitor_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # 分层抽样函数，确保本地和云端模型比例1:1
    def stratified_split(dataset, test_size=0.1, seed=42):
        """
        对数据集进行分层抽样，确保本地和云端模型的比例保持1:1
        """
        # 首先将数据分成本地模型处理和云端模型处理的两类
        local_samples = []
        cloud_samples = []

        logger.info("Analyzing data distribution...")
        for sample in dataset:
            # 检查assistant的回复内容来判断是本地还是云端处理
            assistant_message = None
            for msg in sample['messages']:
                if msg['role'] == 'assistant':
                    assistant_message = msg['content']
                    break

            if assistant_message:
                if "I can solve this" in assistant_message:
                    local_samples.append(sample)
                elif "This is beyond my capability" in assistant_message:
                    cloud_samples.append(sample)

        logger.info(f"Found {len(local_samples)} local model samples")
        logger.info(f"Found {len(cloud_samples)} cloud model samples")

        # 对每类数据进行分层抽样
        from sklearn.model_selection import train_test_split

        # 本地模型样本划分
        local_train, local_eval = train_test_split(
            local_samples,
            test_size=test_size,
            random_state=seed
        )

        # 云端模型样本划分
        cloud_train, cloud_eval = train_test_split(
            cloud_samples,
            test_size=test_size,
            random_state=seed
        )

        # 合并训练集和验证集
        train_data = local_train + cloud_train
        eval_data = local_eval + cloud_eval

        # 打乱数据顺序
        import random
        random.seed(seed)
        random.shuffle(train_data)
        random.shuffle(eval_data)

        # 转换为Dataset对象
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        # 验证比例
        train_local = sum(1 for s in train_data if "I can solve this" in s['messages'][-1]['content'])
        train_cloud = sum(1 for s in train_data if "This is beyond my capability" in s['messages'][-1]['content'])
        eval_local = sum(1 for s in eval_data if "I can solve this" in s['messages'][-1]['content'])
        eval_cloud = sum(1 for s in eval_data if "This is beyond my capability" in s['messages'][-1]['content'])

        logger.info("Train set distribution:")
        logger.info(f"  Local: {train_local} samples")
        logger.info(f"  Cloud: {train_cloud} samples")
        logger.info(f"  Ratio (Local:Cloud) = {train_local}:{train_cloud}")

        logger.info("Eval set distribution:")
        logger.info(f"  Local: {eval_local} samples")
        logger.info(f"  Cloud: {eval_cloud} samples")
        logger.info(f"  Ratio (Local:Cloud) = {eval_local}:{eval_cloud}")

        return train_dataset, eval_dataset

    # 监控模型加载后的内存
    monitor_memory()

    # 5. 加载并划分数据集
    logger.info(f"Loading data from {DATA_FILE}")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    logger.info(f"Loaded {len(full_dataset)} samples")

    # 使用分层抽样划分训练集和验证集，确保本地和云端模型比例均为1:1
    train_dataset, eval_dataset = stratified_split(full_dataset, test_size=0.1, seed=42)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Eval dataset: {len(eval_dataset)} samples")

    # 6. 训练参数 - 添加验证相关配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,           
        per_device_train_batch_size=2,# 显存允许的话可以是 2 或 4
        per_device_eval_batch_size=2, # 验证时可以稍大的 batch size
        gradient_accumulation_steps=16,
        learning_rate=5e-5,          
        weight_decay=0.01,
        bf16=use_bf16,               # 根据GPU支持情况选择
        fp16=not use_bf16,           # 如果不支持bf16则使用fp16
        logging_steps=10,             # 每10步记录一次日志
        save_strategy="epoch",        # 每个 epoch 保存一次
        eval_strategy="epoch",  # 每个 epoch 评估一次
        eval_steps=500,               # 评估步数（配合evaluation_strategy使用）
        save_total_limit=2,           # 最多保存2个模型
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 使用验证loss作为最佳模型指标
        greater_is_better=False,      # loss越小越好
        report_to="none",
        optim="paged_adamw_32bit"
    )

    # 记录训练参数
    logger.info("Training parameters:")
    logger.info(f"  - Epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Eval batch size: {training_args.per_device_eval_batch_size}")
    logger.info(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    logger.info(f"  - Max sequence length: 2048")
    logger.info(f"  - Evaluation strategy: {training_args.eval_strategy}")
    logger.info(f"  - Save strategy: {training_args.save_strategy}")

    # 统计数据集信息
    logger.info("Analyzing dataset...")
    sample_lengths = []
    truncated_count = 0
    max_length = 2048

    # 统计前100个样本的长度（避免遍历整个数据集）
    sample_size = min(100, len(train_dataset))
    for i in range(sample_size):
        messages = train_dataset[i]['messages']
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(text, truncation=False)
        length = len(tokenized['input_ids'])
        sample_lengths.append(length)
        if length > max_length:
            truncated_count += 1

    avg_length = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
    max_sample_length = max(sample_lengths) if sample_lengths else 0

    logger.info(f"Dataset statistics (based on {sample_size} samples):")
    logger.info(f"  - Total train samples: {len(train_dataset)}")
    logger.info(f"  - Total eval samples: {len(eval_dataset)}")
    logger.info(f"  - Average length: {avg_length:.0f} tokens")
    logger.info(f"  - Max length: {max_sample_length} tokens")
    logger.info(f"  - Estimated truncated samples: {truncated_count * len(train_dataset) // sample_size}")
    logger.info(f"  - Max sequence length: {max_length}")

    # 计算每个epoch的步数
    steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.num_train_epochs
    logger.info(f"Training steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_steps}")

    def formatting_prompts_func(example):
        output_texts = []
        for messages in example['messages']:
            # 使用 apply_chat_template，但不加 generation prompt，因为这是训练
            # messages 中已包含 system prompt，apply_chat_template 会自动处理格式
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    # 7. SFT Trainer - 添加验证集
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 添加验证集
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,  # 使用新版本参数名
        # 注意：SFT 训练时，不需要 compute_metrics，因为 Trainer 会自动计算 loss
        # 如果需要，可以在评估后手动计算
    )

    trainer.formatting_func = formatting_prompts_func

    # 训练前的内存监控
    monitor_memory()
    logger.info("Starting training...")
    train_result = trainer.train()

    # 训练完成统计
    logger.info("Training completed!")

    # 训练后的内存监控
    monitor_memory()

    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    logger.info(f"Total training steps: {train_result.global_step}")

    # 计算实际处理的样本数
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    total_samples_processed = train_result.global_step * effective_batch_size
    samples_per_epoch = len(train_dataset)
    epochs_completed = total_samples_processed / samples_per_epoch

    logger.info("Training summary:")
    logger.info(f"  - Total samples processed: {total_samples_processed}")
    logger.info(f"  - Samples per epoch: {samples_per_epoch}")
    logger.info(f"  - Epochs completed: {epochs_completed:.2f}")
    logger.info(f"  - Effective batch size: {effective_batch_size}")

    # 运行最终验证
    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()

    # 计算困惑度
    if 'eval_loss' in eval_result:
        eval_loss = eval_result['eval_loss']
        perplexity = np.exp(eval_loss) if eval_loss > 0 else float('inf')
        eval_result['eval_perplexity'] = perplexity

    logger.info("Final evaluation results:")
    for key, value in eval_result.items():
        if isinstance(value, float):
            logger.info(f"  - {key}: {value:.4f}")
        else:
            logger.info(f"  - {key}: {value}")

    # 保存模型
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存训练状态和指标
    trainer.log_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", eval_result)
    trainer.save_state()

    # 清理GPU缓存
    torch.cuda.empty_cache()
    monitor_memory()

    # 输出最终总结
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    logger.info(f"Final eval loss: {eval_result.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"Final perplexity: {eval_result.get('eval_perplexity', 'N/A'):.2f}")
    logger.info("="*50)

if __name__ == "__main__":
    main()