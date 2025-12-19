#!/usr/bin/env python3
"""
构建测试数据集脚本
从 GSM8K 测试集和本地 parquet 文件中提取未在训练集中出现的题目
生成独立的测试集 data/test_set.jsonl
"""

import os
import json
import random
import re
from typing import Dict, Any, List, Set
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

def load_training_questions(raw_file: str) -> Set[str]:
    """
    加载训练集中的所有问题，构建黑名单

    Args:
        raw_file: 训练集文件路径

    Returns:
        包含所有训练集问题的集合
    """
    if not os.path.exists(raw_file):
        print(f"❌ 错误: 找不到训练集文件 {raw_file}")
        return set()

    print(f"📖 正在读取训练集 {raw_file}，构建问题黑名单...")
    questions = set()

    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取训练集"):
            try:
                data = json.loads(line)
                question = data.get('question', '').strip()
                if question:
                    questions.add(question)
            except Exception as e:
                print(f"⚠️  解析行失败: {e}")

    print(f"✅ 构建黑名单完成，包含 {len(questions)} 个问题")
    return questions

def extract_content_from_prompt(prompt: Any) -> str:
    """
    从 prompt 字段中提取用户问题内容
    兼容多种格式（字符串、列表、JSON等）
    """
    import numpy as np
    import ast

    # 预处理：处理 numpy 数组或非字符串类型
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()

    # 如果已经是列表，直接处理
    if isinstance(prompt, list):
        return _extract_from_list_obj(prompt)

    # 如果不是字符串，强转字符串
    if not isinstance(prompt, str):
        prompt = str(prompt)

    prompt = prompt.strip()

    # 如果不是以列表开头，直接返回
    if not prompt.startswith('['):
        return prompt

    # 尝试 JSON 解析
    try:
        parsed_obj = json.loads(prompt)
        if isinstance(parsed_obj, list):
            return _extract_from_list_obj(parsed_obj)
    except:
        pass

    # 尝试 Python AST 解析
    try:
        parsed_obj = ast.literal_eval(prompt)
        if isinstance(parsed_obj, list):
            return _extract_from_list_obj(parsed_obj)
    except (ValueError, SyntaxError):
        pass

    # 正则表达式强制提取
    try:
        contents = re.findall(r"'content':\s*(['\"])(.*?)\1", prompt, re.DOTALL)
        roles = re.findall(r"'role':\s*(['\"])(.*?)\1", prompt, re.DOTALL)

        if len(contents) == len(roles):
            for i, (_, role_val) in enumerate(roles):
                if role_val == 'user':
                    return contents[i][1]

        # 查找 "content": "...", "role": "user" 组合
        match = re.search(r"'content':\s*(['\"])(.*?)\1,\s*'role':\s*'user'", prompt, re.DOTALL)
        if match:
            return match.group(2)
    except Exception:
        pass

    return prompt

def _extract_from_list_obj(data_list: list) -> str:
    """辅助函数：从列表对象中提取 content"""
    # 优先找 user
    for item in data_list:
        if isinstance(item, dict) and item.get('role') == 'user':
            content = item.get('content', '')
            if 'Please output the final answer' in content:
                content = content.split('Please output the final answer')[0].strip()
            return content

    # 没找到 user，返回第一个非空
    for item in data_list:
        if isinstance(item, dict):
            content = item.get('content', '')
            if content:
                return content.strip()
    return ""

def extract_ground_truth_from_parquet(sample: pd.Series) -> str:
    """从 parquet 样本中提取标准答案"""
    if 'reward_model' in sample and pd.notna(sample['reward_model']):
        reward_model = sample['reward_model']
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth')
            if gt: return str(gt)
    if 'extra_info' in sample and pd.notna(sample['extra_info']):
        extra_info = sample['extra_info']
        if isinstance(extra_info, dict):
            ans = extra_info.get('answer')
            if ans: return str(ans)
    return ""

def extract_answer_from_gsm8k(answer_text: str) -> str:
    """
    从 GSM8K 的 answer 字段中提取 #### 后面的答案
    """
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    # 查找 #### 后面的内容
    match = re.search(r'####\s*([^\n]+)', answer_text)
    if match:
        return match.group(1).strip()

    # 如果没有找到 ####，尝试其他模式
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

def load_gsm8k_test_set(blacklist: Set[str]) -> List[Dict]:
    """
    加载 GSM8K 测试集，过滤掉训练集中已有的题目

    Args:
        blacklist: 训练集问题黑名单

    Returns:
        GSM8K 测试集数据列表
    """
    print(f"\n📥 正在从 Hugging Face 下载 GSM8K 测试集...")

    try:
        # 下载 GSM8K 测试集
        dataset = load_dataset('openai/gsm8k', 'main', split='test')
        print(f"✅ 成功下载 GSM8K 测试集，共 {len(dataset)} 条数据")
    except Exception as e:
        print(f"❌ 下载 GSM8K 测试集失败: {e}")
        return []

    # 过滤数据
    test_data = []
    skipped_count = 0

    for idx, sample in enumerate(tqdm(dataset, desc="过滤 GSM8K 测试集")):
        question = sample.get('question', '').strip()
        answer = sample.get('answer', '')

        # 检查是否在黑名单中
        if question in blacklist:
            skipped_count += 1
            continue

        # 提取答案
        ground_truth = extract_answer_from_gsm8k(answer)

        test_data.append({
            "id": f"test_gsm8k_{idx}",
            "domain": "math",
            "source": "gsm8k_test",
            "question": question,
            "ground_truth": ground_truth
        })

    print(f"✅ GSM8K 测试集处理完成：保留 {len(test_data)} 条，跳过 {skipped_count} 条")
    return test_data

def load_hard_test_questions(parquet_file: str, blacklist: Set[str], num_samples: int = 500, seed: int = 42) -> List[Dict]:
    """
    从 parquet 文件中采样困难的测试题

    Args:
        parquet_file: parquet 文件路径
        blacklist: 训练集问题黑名单
        num_samples: 需要采样的数量
        seed: 随机种子

    Returns:
        困难测试题数据列表
    """
    if not os.path.exists(parquet_file):
        print(f"❌ 错误: 找不到 parquet 文件 {parquet_file}")
        return []

    print(f"\n📖 正在读取 parquet 文件 {parquet_file}...")

    try:
        df = pd.read_parquet(parquet_file)
        print(f"✅ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"❌ 读取 Parquet 文件失败: {e}")
        return []

    # 显示数据源分布
    print("\n📊 数据源分布:")
    source_counts = df['data_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  - {source}: {count} 条")

    # 筛选包含 math 或 olympiad 的数据源
    print("\n🔍 正在筛选困难题目...")
    mask = df['data_source'].str.contains('math', case=False, na=False)
    filtered_df = df[mask]
    print(f"✅ 找到 {len(filtered_df)} 条候选题目")

    # 进一步过滤：提取问题并检查黑名单
    valid_samples = []
    for _, sample in filtered_df.iterrows():
        question = extract_content_from_prompt(sample['prompt']).strip()
        if question and question not in blacklist:
            sample_dict = sample.to_dict()
            sample_dict['extracted_question'] = question
            valid_samples.append(sample_dict)

    print(f"✅ 过滤黑名单后剩余 {len(valid_samples)} 条题目")

    # 随机采样
    if len(valid_samples) < num_samples:
        print(f"⚠️  可用题目不足 {num_samples} 条，仅使用 {len(valid_samples)} 条")
        sampled_samples = valid_samples
    else:
        print(f"🎲 正在随机采样 {num_samples} 条题目...")
        random.seed(seed)
        sampled_samples = random.sample(valid_samples, num_samples)

    # 处理采样数据
    hard_test_data = []
    for idx, sample in enumerate(sampled_samples):
        question = sample['extracted_question']
        ground_truth = extract_ground_truth_from_parquet(pd.Series(sample))

        hard_test_data.append({
            "id": f"test_hard_{idx}",
            "domain": "math",
            "source": "hard_test",
            "question": question,
            "ground_truth": ground_truth
        })

    print(f"✅ 成功处理 {len(hard_test_data)} 条困难测试题")
    return hard_test_data

def save_test_set(gsm8k_data: List[Dict], hard_data: List[Dict], output_file: str):
    """
    保存测试集到文件

    Args:
        gsm8k_data: GSM8K 测试集数据
        hard_data: 困难测试题数据
        output_file: 输出文件路径
    """
    # 合并数据
    all_data = gsm8k_data + hard_data
    print(f"\n🔗 测试集合并完成，总计 {len(all_data)} 条")
    print(f"   - GSM8K 测试题: {len(gsm8k_data)} 条")
    print(f"   - 困难测试题: {len(hard_data)} 条")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存到文件
    print(f"\n💾 正在保存到 {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data, desc="保存测试集"):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"\n✅ 测试集已成功保存到 {output_file}！")

    # 显示输出示例
    print("\n📄 测试集格式示例:")
    print("\nGSM8K 测试题示例:")
    if gsm8k_data:
        example = gsm8k_data[0]
        print(f"  - ID: {example['id']}")
        print(f"  - Source: {example['source']}")
        print(f"  - Question: {example['question'][:100]}...")
        print(f"  - Answer: {example['ground_truth']}")

    print("\n困难测试题示例:")
    if hard_data:
        example = hard_data[0]
        print(f"  - ID: {example['id']}")
        print(f"  - Source: {example['source']}")
        print(f"  - Question: {example['question'][:100]}...")
        print(f"  - Answer: {example['ground_truth']}")

def main():
    """主函数"""
    # 配置参数
    TRAIN_FILE = "data/raw_prompts.jsonl"
    PARQUET_FILE = "data/train/math__combined_54.4k.parquet"
    OUTPUT_FILE = "data/test_set.jsonl"
    NUM_HARD_SAMPLES = 500
    RANDOM_SEED = 42

    print("=" * 60)
    print("构建测试数据集脚本")
    print("=" * 60)
    print("目标：构建独立的测试集，确保不与训练集重叠")

    # 1. 构建训练集黑名单
    blacklist = load_training_questions(TRAIN_FILE)

    # 2. 加载 GSM8K 测试集
    gsm8k_data = load_gsm8k_test_set(blacklist)

    # 3. 加载困难测试题
    hard_data = load_hard_test_questions(PARQUET_FILE, blacklist, NUM_HARD_SAMPLES, RANDOM_SEED)

    # 4. 保存测试集
    if gsm8k_data or hard_data:
        save_test_set(gsm8k_data, hard_data, OUTPUT_FILE)
    else:
        print("\n❌ 未能获取任何测试数据，请检查输入文件和网络连接")

if __name__ == "__main__":
    main()