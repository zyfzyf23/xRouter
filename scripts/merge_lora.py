#!/usr/bin/env python3
"""
合并 LoRA 权重脚本
将训练好的 LoRA 适配器权重合并到底座模型中，生成完整模型用于 vLLM 推理
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    clean_cache: bool = True
):
    """
    合并 LoRA 权重到底座模型

    Args:
        base_model_path: 底座模型路径
        adapter_path: LoRA 适配器路径
        output_path: 输出路径
        device_map: 设备映射
        torch_dtype: 模型数据类型
        clean_cache: 是否清理 GPU 缓存
    """

    # 清理 GPU 缓存（如果需要）
    if clean_cache and torch.cuda.is_available():
        logger.info("清理 GPU 缓存...")
        torch.cuda.empty_cache()

    # 创建输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir.absolute()}")

    # 检查输入路径
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"底座模型路径不存在: {base_model_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA 适配器路径不存在: {adapter_path}")

    logger.info(f"底座模型: {base_model_path}")
    logger.info(f"LoRA 适配器: {adapter_path}")

    # 设置默认数据类型
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    # 步骤 1: 加载底座模型
    logger.info("正在加载底座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 步骤 2: 加载 LoRA 适配器
    logger.info("正在加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    # 打印模型信息
    logger.info(f"模型类型: {type(model)}")
    logger.info(f"底座模型类型: {type(model.base_model)}")

    # 步骤 3: 合并权重
    logger.info("正在合并 LoRA 权重...")
    model = model.merge_and_unload()

    # 验证合并后的模型
    logger.info(f"合并后模型类型: {type(model)}")
    logger.info("LoRA 权重已成功合并到底座模型")

    # 步骤 4: 加载并保存 tokenizer
    logger.info("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 步骤 5: 保存合并后的模型和 tokenizer
    logger.info(f"正在保存合并后的模型到: {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="10GB"
    )

    logger.info(f"正在保存 tokenizer 到: {output_path}")
    tokenizer.save_pretrained(output_path)

    # 步骤 6: 验证输出
    output_files = os.listdir(output_path)
    required_files = [
        "config.json",
        "pytorch_model.bin" if torch.cuda.is_available() else "pytorch_model.bin.index.json",
        "tokenizer_config.json",
        "tokenizer.json"
    ]

    missing_files = [f for f in required_files if not any(f in file for file in output_files)]
    if missing_files:
        logger.warning(f"警告: 以下文件可能缺失: {missing_files}")
    else:
        logger.info("所有必需文件已保存")

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"模型大小: ~{total_params * 4 / 1024**3:.2f} GB (FP32)")

    # 检查输出目录大小
    dir_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    logger.info(f"输出目录大小: {dir_size / 1024**3:.2f} GB")

    # 最终清理
    if clean_cache and torch.cuda.is_available():
        logger.info("最终清理 GPU 缓存...")
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"清理后 GPU 显存使用: {allocated:.2f} GB")

    logger.info("✅ LoRA 权重合并完成!")
    logger.info(f"合并后的模型已保存到: {output_path}")

    return model, tokenizer


def main():
    """主函数"""
    # 默认路径配置
    base_model_path = "../models/Qwen2.5-1.5B-Instruct"
    adapter_path = "outputs/sft_1218_with_low_lr"
    output_path = "outputs/sft_merged_1218_with_low_lr"

    # 从环境变量读取路径（可选）
    if os.getenv("BASE_MODEL_PATH"):
        base_model_path = os.getenv("BASE_MODEL_PATH")
    if os.getenv("ADAPTER_PATH"):
        adapter_path = os.getenv("ADAPTER_PATH")
    if os.getenv("OUTPUT_PATH"):
        output_path = os.getenv("OUTPUT_PATH")

    # 打印配置信息
    logger.info("=" * 60)
    logger.info("LoRA 权重合并配置")
    logger.info("=" * 60)
    logger.info(f"底座模型路径: {base_model_path}")
    logger.info(f"LoRA 适配器路径: {adapter_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info("=" * 60)

    try:
        # 执行合并
        model, tokenizer = merge_lora_weights(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_path=output_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            clean_cache=True
        )

        # 简单测试合并后的模型
        logger.info("\n测试合并后的模型...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"测试输入: {test_text}")
        logger.info(f"模型输出: {response}")

        logger.info("\n✅ 模型测试成功! 可以使用 vLLM 加载此模型进行推理。")

    except Exception as e:
        logger.error(f"❌ 合并过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()