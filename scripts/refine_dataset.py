#!/usr/bin/env python3
"""
数据集清洗脚本 v2：从 ModelScope 下载 GSM8K 数据集并合并到 raw_prompts.jsonl
- 保留现有的 429 条数据
- 从 ModelScope 下载 GSM8K 数据集
- 随机提取 2072 条数据
- ID 从 429 开始
- 从 answer 字段中提取 #### 后面的答案作为 ground_truth
"""

import os
import json
import random
import re
from typing import Dict, Any, List
from tqdm import tqdm
# from modelscope import MsDataset  # 由于版本兼容问题，使用 datasets 替代
from datasets import load_dataset

def extract_answer_from_gsm8k(answer_text: str) -> str:
    """
    从 GSM8K 的 answer 字段中提取 #### 后面的答案

    Args:
        answer_text: GSM8K 的完整答案文本

    Returns:
        提取的最终答案（#### 后面的内容）
    """
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    # 查找 #### 后面的内容
    match = re.search(r'####\s*([^\n]+)', answer_text)
    if match:
        return match.group(1).strip()

    # 如果没有找到 ####，尝试其他模式
    # 有些答案可能使用 "The answer is" 或类似的模式
    patterns = [
        r'(?:The answer is|Answer:|Result:)\s*([^\n]+)',
        r'=\s*([^\n]+)$',
        r'([0-9]+(?:\.[0-9]+)?)\s*$'
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # 如果都没找到，返回原始文本的最后部分
    lines = answer_text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('<<') and len(line) < 50:
            return line

    # 默认返回原始文本
    return answer_text.strip()

def load_existing_data(raw_file: str) -> List[Dict]:
    """
    加载现有的 raw_prompts.jsonl 数据

    Args:
        raw_file: 原始数据文件路径

    Returns:
        现有数据列表
    """
    if not os.path.exists(raw_file):
        print(f"❌ 错误: 找不到原始数据文件 {raw_file}")
        return []

    print(f"📖 正在读取原始数据文件 {raw_file}...")
    existing_data = []

    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取现有数据"):
            try:
                data = json.loads(line)
                existing_data.append(data)
            except Exception as e:
                print(f"⚠️  解析行失败: {e}")

    print(f"✅ 读取了 {len(existing_data)} 条现有数据")
    return existing_data

def download_and_sample_gsm8k(num_samples: int = 2072, seed: int = 42) -> List[Dict]:
    """
    从 Hugging Face 下载 GSM8K 数据集并随机采样

    Args:
        num_samples: 需要采样的数据量
        seed: 随机种子

    Returns:
        采样后的 GSM8K 数据列表
    """
    print(f"\n📥 正在从 Hugging Face 下载 GSM8K 数据集...")

    try:
        # 下载 GSM8K 数据集
        dataset = load_dataset('openai/gsm8k', 'main')

        # 获取训练集
        gsm8k_train = dataset['train']
        print(f"✅ 成功下载 GSM8K 数据集，训练集共 {len(gsm8k_train)} 条数据")

        # 转换为列表以便采样
        gsm8k_data = list(gsm8k_train)

    except Exception as e:
        print(f"❌ 下载 GSM8K 数据集失败: {e}")
        # 尝试备用数据集
        try:
            dataset = load_dataset('gsm8k', 'main')
            gsm8k_train = dataset['train']
            gsm8k_data = list(gsm8k_train)
            print(f"✅ 成功下载 GSM8K 数据集（备用源），共 {len(gsm8k_data)} 条数据")
        except Exception as e2:
            print(f"❌ 所有尝试都失败: {e2}")
            return []

    # 随机采样
    if len(gsm8k_data) < num_samples:
        print(f"⚠️  警告: GSM8K 数据不足 {num_samples} 条，仅使用 {len(gsm8k_data)} 条")
        sampled_data = gsm8k_data
    else:
        print(f"🎲 正在随机采样 {num_samples} 条数据...")
        random.seed(seed)
        sampled_data = random.sample(gsm8k_data, num_samples)

    # 处理采样数据
    processed_data = []
    print(f"\n💾 正在处理 GSM8K 数据...")

    for idx, sample in enumerate(tqdm(sampled_data, desc="处理数据")):
        # 提取问题
        question = sample.get('question', '')

        # 提取答案
        answer_text = sample.get('answer', '')
        ground_truth = extract_answer_from_gsm8k(answer_text)

        # 生成 ID（从 429 开始）
        new_id = str(429 + idx)

        processed_data.append({
            "id": new_id,
            "domain": "math",
            "source": "gsm8k",
            "question": question,
            "ground_truth": ground_truth
        })

    print(f"✅ 成功处理 {len(processed_data)} 条 GSM8K 数据")
    return processed_data

def save_combined_data(existing_data: List[Dict], gsm8k_data: List[Dict], output_file: str):
    """
    保存合并后的数据到文件

    Args:
        existing_data: 现有的数据
        gsm8k_data: 新的 GSM8K 数据
        output_file: 输出文件路径
    """
    # 合并数据
    all_data = existing_data + gsm8k_data
    print(f"\n🔗 数据合并完成，总计 {len(all_data)} 条")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存到文件
    print(f"\n💾 正在保存到 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data, desc="保存数据"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\n✅ 数据已成功保存！")
    print(f"   - 保留现有数据: {len(existing_data)} 条")
    print(f"   - 新增 GSM8K 数据: {len(gsm8k_data)} 条")
    print(f"   - 总计: {len(all_data)} 条")

def main():
    """主函数"""
    # 配置参数
    RAW_FILE = "data/raw_prompts.jsonl"
    OUTPUT_FILE = "data/raw_prompts.jsonl"
    NUM_GSM8K_SAMPLES = 2072
    RANDOM_SEED = 42

    print("=" * 60)
    print("数据集清洗脚本 v2")
    print("=" * 60)
    print(f"目标：保留现有数据 + 添加 {NUM_GSM8K_SAMPLES} 条 GSM8K 数据")

    # 1. 加载现有数据
    existing_data = load_existing_data(RAW_FILE)

    # 2. 下载并采样 GSM8K 数据
    gsm8k_data = download_and_sample_gsm8k(NUM_GSM8K_SAMPLES, RANDOM_SEED)

    # 3. 合并并保存数据
    if gsm8k_data:
        save_combined_data(existing_data, gsm8k_data, OUTPUT_FILE)

        # 显示输出示例
        print("\n📄 新增数据格式示例 (前3条):")
        for i, data in enumerate(gsm8k_data[:3]):
            print(f"\nGSM8K {i+1} (ID: {data['id']}):")
            q_preview = data['question'][:150] + "..." if len(data['question']) > 150 else data['question']
            print(f"  question: {q_preview}")
            print(f"  answer: {data['ground_truth']}")
    else:
        print("\n❌ 未能获取 GSM8K 数据，请检查网络连接或 ModelScope 配置")

if __name__ == "__main__":
    main()