import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import re


# 定义全局 System Prompt
ROUTER_SYSTEM_PROMPT = (
    "You are an intelligent router agent. "
    "Your task is to analyze the difficulty of the user's question and decide whether to solve it yourself or route it to a more powerful cloud model. "
    "- If the question is simple or within your capabilities, answer it directly starting with <think>I can solve this.</think>. "
    "- If the question is complex, requires multi-step reasoning that you are unsure about, or is beyond your capability, route it to the cloud model using the format: <think>This is beyond my capability.</think>\n\n<tool_code>call_remote_model(prompt)</tool_code>"
)

@dataclass
class Stats:
    """统计信息类，用于跟踪数据处理各阶段的数量"""
    total_raw: int = 0      # 原始数据总条数
    local_samples: int = 0  # 本地模型能处理的样本数
    cloud_samples: int = 0  # 需要云端模型处理的样本数
    final_samples: int = 0  # 最终平衡后的训练样本数

def clean_answer(answer: str) -> str:
    """
    清理答案文本，去除多余的空行
    Args:
        answer: 原始答案文本
    Returns:
        清理后的答案文本
    """
    return re.sub(r'\n+', '\n', answer.strip())

def ensure_boxed(question: str) -> str:
    """
    确保数学问题要求答案在 \boxed{} 格式中
    这是为了统一答案格式，便于模型学习
    Args:
        question: 原始问题文本
    Returns:
        添加了 boxed 要求的问题文本
    """
    if r'\boxed{' not in question and "Please answer within" not in question:
        if question.strip().endswith('.'):
            question = question.strip()[:-1] + '. Please answer within \\boxed{}.'
        elif question.strip().endswith('?'):
            question = question.strip()[:-1] + '? Please answer within \\boxed{}.'
        else:
            question = question.strip() + ' Please answer within \\boxed{}.'
    return question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/offline_cache_math.jsonl")
    parser.add_argument("--output_file", type=str, default="data/sft_train_balanced.jsonl")
    args = parser.parse_args()
    
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    
    # 临时列表用于存储分类后的样本
    local_data = []  # Case 1 & 3
    cloud_data = []  # Case 2

    print(f"📖 读取原始数据: {args.input_file}")
    
    with open(args.input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                stats.total_raw += 1
                
                weak_correct = item.get('weak_correct', False)
                strong_correct = item.get('strong_correct', False)
                weak_ans = item.get('weak_ans', '')
                question = ensure_boxed(item.get('question', '')) # 统一加 boxed
                
                # --- 构造 SFT 样本 ---
                # 格式: {"prompt": "...", "completion": "..."}
                
                # 1. 本地能做 (Case 1 & 3) -> 训练目标: <think>I can solve...</think> Answer
                if weak_correct: 
                    target = f"<think>I can solve this.</think>\n\n{clean_answer(weak_ans)}"
                    sample = {
                        "prompt": question,  # 只有 User 内容，SFTTrainer 会自动加模板
                        "completion": target,
                        "type": "local"
                    }
                    local_data.append(sample)
                    
                # 2. 本地不能做但云端能做 (Case 2) -> 训练目标: <think>Too hard...</think> <tool>
                elif not weak_correct and strong_correct:
                    # 使用优化后的短 Tool Call
                    target = f"<think>This is beyond my capability.</think>\n\n<tool_code>call_remote_model(prompt)</tool_code>"
                    sample = {
                        "prompt": question,
                        "completion": target,
                        "type": "cloud"
                    }
                    cloud_data.append(sample)
                    
                # Case 4 (双输) 依然丢弃
                
            except Exception as e:
                continue

    stats.local_samples = len(local_data)
    stats.cloud_samples = len(cloud_data)
    
    print("-" * 40)
    print(f"原始分布 -> 本地(Local): {stats.local_samples} | 云端(Cloud): {stats.cloud_samples}")
    
    # --- ⚖️ 强制平衡逻辑 (Under-sampling) ---
    min_count = min(stats.local_samples, stats.cloud_samples)
    
    if min_count == 0:
        print("❌ 错误：某一类样本数为 0，无法平衡！")
        return

    # 随机采样，让两者数量一致
    balanced_local = random.sample(local_data, min_count)
    balanced_cloud = random.sample(cloud_data, min_count)
    
    final_data = balanced_local + balanced_cloud
    random.shuffle(final_data) # 打乱顺序
    
    stats.final_samples = len(final_data)
    
    # --- 写入文件 ---
    with open(args.output_file, 'w') as f:
        for sample in final_data:
            # 转换为 HuggingFace SFTTrainer 需要的格式
            # SFTTrainer 通常接受 text 字段，或者 messages 字段
            # 这里我们直接存 messages 格式，方便后续处理
            output_obj = {
                "messages": [
                    # ✅ 1. 加入 System Prompt
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    # 2. User Question
                    {"role": "user", "content": sample["prompt"]},
                    # 3. Assistant Answer (Target)
                    {"role": "assistant", "content": sample["completion"]}
                ]
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            
    print("-" * 40)
    print(f"✅ 平衡后 -> 本地: {min_count} | 云端: {min_count}")
    print(f"🚀 总训练样本: {stats.final_samples}")
    print(f"💾 已保存至: {args.output_file}")

if __name__ == "__main__":
    main()