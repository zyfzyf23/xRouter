#!/usr/bin/env python3
"""
从本地的 Guru 数学数据集中提取数学题目
读取 data/train/math__combined_54.4k.parquet 文件并生成 data/raw_prompts.jsonl
"""

import os
import json
import random
import ast
import re
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np  # 确保处理 numpy 类型数据

def extract_content_from_prompt(prompt: Any) -> str:
    """
    从 prompt 字段中提取用户问题内容
    改进逻辑：多级降级策略 (JSON -> AST -> 修复后AST -> 正则提取)
    """
    
    # 0. 预处理：处理 numpy 数组或非字符串类型
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

    # === 策略 1: 尝试 JSON 解析 (最标准) ===
    try:
        parsed_obj = json.loads(prompt)
        if isinstance(parsed_obj, list):
            return _extract_from_list_obj(parsed_obj)
    except:
        pass

    # === 策略 2: 尝试 Python AST 解析 (标准) ===
    try:
        parsed_obj = ast.literal_eval(prompt)
        if isinstance(parsed_obj, list):
            return _extract_from_list_obj(parsed_obj)
    except (ValueError, SyntaxError):
        pass

    # === 策略 3: 尝试修复 LaTeX 转义符后 AST 解析 ===
    # LaTeX 中的 \ 经常导致 Python 字符串解析失败，尝试将 \ 替换为 \\
    # 注意：这可能会破坏原本已经转义的字符，所以仅作为失败后的尝试
    try:
        # 简单的启发式修复：如果字符串包含 \ 但不是 \\，尝试替换
        # 这是一个激进的操作，仅在上面失败时使用
        fixed_prompt = prompt.replace('\\', '\\\\')
        parsed_obj = ast.literal_eval(fixed_prompt)
        if isinstance(parsed_obj, list):
            return _extract_from_list_obj(parsed_obj)
    except:
        pass

    # === 策略 4: 正则表达式强制提取 (核武器) ===
    # 既然解析不了结构，就直接用正则抓取 'content': '...' 中的内容
    # 匹配模式：寻找 'role': 'user' 附近的 'content'
    try:
        # 模式 A: {'content': '抓取这里', 'role': 'user'}
        # 使用非贪婪匹配，同时也允许 content 在 role 之后
        
        # 尝试匹配 content 内容。注意处理转义引号。
        # 这里使用一个简化的逻辑：找到 role='user' 的那个字典块
        
        # 1. 这种复杂的嵌套用正则很难完美匹配，我们尝试提取所有 content
        # 假设格式是 standard python repr: 'content': '...'
        
        # 查找所有 content 块
        contents = re.findall(r"'content':\s*(['\"])(.*?)\1", prompt, re.DOTALL)
        roles = re.findall(r"'role':\s*(['\"])(.*?)\1", prompt, re.DOTALL)
        
        # 如果能对应上，找到 user 对应的 content
        if len(contents) == len(roles):
            for i, (_, role_val) in enumerate(roles):
                if role_val == 'user':
                    return contents[i][1] # 返回 content 的内容组
        
        # 如果上面没对齐，直接暴力匹配第一个看起来像 user content 的
        # 查找 "content": "...", "role": "user" 组合
        match = re.search(r"'content':\s*(['\"])(.*?)\1,\s*'role':\s*'user'", prompt, re.DOTALL)
        if match:
            return match.group(2)
            
        # 翻转顺序查找 "role": "user", "content": "..."
        match_reverse = re.search(r"'role':\s*'user'.*?'content':\s*(['\"])(.*?)\1", prompt, re.DOTALL)
        if match_reverse:
            return match_reverse.group(2)

    except Exception:
        pass

    # === 放弃治疗 ===
    # 如果所有解析都失败，说明数据格式极其破碎，返回原始字符串以便人工检查
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

def extract_ground_truth(sample: pd.Series) -> str:
    """提取标准答案"""
    if 'response' in sample and pd.notna(sample['response']):
        return str(sample['response'])
    if 'completion' in sample and pd.notna(sample['completion']):
        return str(sample['completion'])
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

def process_sample(sample: pd.Series, idx: int) -> Dict:
    """处理单个数据样本"""
    question = extract_content_from_prompt(sample['prompt'])
    ground_truth = extract_ground_truth(sample)
    source = sample.get('data_source', 'unknown')

    return {
        "id": str(idx),
        "domain": "math",
        "source": source,
        "question": question,
        "ground_truth": ground_truth
    }

def main():
    # 配置参数
    INPUT_FILE = "data/train/math__combined_54.4k.parquet"
    OUTPUT_FILE = "data/raw_prompts_simple.jsonl"
    # todo
    NUM_SAMPLES = 10
    RANDOM_SEED = 42

    print("=" * 60)
    print("数学数据提取脚本 (多级解析增强版)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    random.seed(RANDOM_SEED)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("\n📖 正在读取数据...")
    try:
        df = pd.read_parquet(INPUT_FILE)
        print(f"✅ 成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"❌ 读取 Parquet 文件失败: {e}")
        return

    # 调试：打印一个失败样本的原始字符串
    print("\n🔍 数据预检 (第一条 prompt):")
    if len(df) > 0:
        p1 = df.iloc[0]['prompt']
        print(f"  Raw type: {type(p1)}")
        print(f"  Raw content prefix: {str(p1)[:50]}...")

    print(f"\n🎲 正在随机采样 {NUM_SAMPLES} 条数据...")
    if len(df) > NUM_SAMPLES:
        sampled_df = df.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED)
    else:
        sampled_df = df

    print(f"\n💾 正在处理并保存数据到 {OUTPUT_FILE}...")
    
    success_count = 0
    fail_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, (_, sample) in enumerate(tqdm(sampled_df.iterrows(), total=len(sampled_df))):
            processed = process_sample(sample, idx)
            
            # 简单统计解析是否成功 (如果 question 仍然以 [ 开头，说明解析可能失败了)
            if processed['question'].strip().startswith('['):
                fail_count += 1
            else:
                success_count += 1
                
            f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

    print(f"\n✅ 处理完成！")
    print(f"   解析成功(预估): {success_count}")
    print(f"   解析失败(保留原样): {fail_count}")
    
    # 显示输出示例
    print("\n📄 输出格式示例 (前2条):")
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if i >= 2: break
                data = json.loads(line)
                print(f"\nJSON {i}:")
                q_preview = data['question']
                if len(q_preview) > 150: q_preview = q_preview[:150] + "..."
                print(f"  question: {q_preview}") 

if __name__ == "__main__":
    main()