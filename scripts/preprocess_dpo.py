#!/usr/bin/env python3
"""
DPO 数据预处理脚本
从 offline_cache.jsonl 生成适用于训练 Router 的 DPO 格式数据
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
from dataclasses import dataclass


"""- ✅ 处理了四种场景的数据：
    - 场景 1(省钱):Weak ✅
  Strong ✅ → 选直接回答
    - 场景 2(求稳):Weak ❌
  Strong ✅ → 选路由云端
    - 场景 3(捡漏):Weak ✅
  Strong ❌ → 选直接回答
    - 场景 4(双输):Both ❌ →
  丢弃"""
@dataclass
class Stats:
    """统计信息类"""
    total_samples: int = 0
    dpo_samples: int = 0
    save_money_cases: int = 0  # 场景1 + 场景3（自己能做对）
    seek_stability_cases: int = 0  # 场景2（自己做不对，需要求助云端）
    discarded_cases: int = 0  # 场景4（双输）


def clean_answer(answer: str) -> str:
    """清理答案文本，移除多余的换行和空格"""
    # 移除开头的空白
    answer = answer.strip()
    # 将多个连续换行替换为单个换行
    answer = re.sub(r'\n+', '\n', answer)
    return answer


def has_boxed_answer(text: str) -> bool:
    """检查文本中是否已包含 \boxed{} 格式的答案"""
    return r'\boxed{' in text


def ensure_boxed_requirement(question: str) -> str:
    """确保问题中包含 \boxed{} 要求"""
    if not has_boxed_answer(question) and "Please answer within" not in question:
        # 在问题末尾添加 \boxed{} 要求
        if question.endswith('.'):
            question = question[:-1] + '. Please answer within \boxed{}.'
        elif question.endswith('?'):
            question = question[:-1] + '? Please answer within \boxed{}.'
        else:
            question = question + ' Please answer within \boxed{}.'
    return question


def construct_direct_action(weak_answer: str) -> str:
    """构造直接回答的动作（省钱模式）"""
    weak_answer = clean_answer(weak_answer)
    return f"<think>I can solve this.</think>\n\n{weak_answer}"


def construct_route_action(question: str) -> str:
    """构造路由到云端模型的动作（求稳模式）"""
    # 确保 question 中有 \boxed{} 要求
    # question = ensure_boxed_requirement(question)
    return f"<think>This is beyond my capability.</think>\n\n<tool_code>call_remote_model(prompt)</tool_code>"


def process_sample(sample: Dict[str, Any], stats: Stats) -> Dict[str, Any]:
    """
    处理单个样本，生成 DPO 数据

    返回:
        - DPO 样本字典（如果有效）
        - None（如果应丢弃）
    """
    weak_correct = sample.get('weak_correct', False)
    strong_correct = sample.get('strong_correct', False)
    weak_ans = sample.get('weak_ans', '')
    raw_question = sample.get('question', '')

    stats.total_samples += 1

    # 构造两种动作
    question = ensure_boxed_requirement(raw_question)
    action_direct = construct_direct_action(weak_ans)
    action_route = construct_route_action(question)

    # 场景判定
    if weak_correct and strong_correct:
        # 场景 1: 省钱（Weak Correct ✅ Strong Correct ✅）
        # 两个都对，选本地的（省钱）
        dpo_sample = {
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": action_direct}],
            "rejected": [{"role": "assistant", "content": action_route}]
        }
        stats.save_money_cases += 1
        stats.dpo_samples += 1
        return dpo_sample

    elif not weak_correct and strong_correct:
        # 场景 2: 求稳（Weak Wrong ❌ Strong Correct ✅）
        # 本地做不对，必须求助云端
        dpo_sample = {
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": action_route}],
            "rejected": [{"role": "assistant", "content": action_direct}]
        }
        stats.seek_stability_cases += 1
        stats.dpo_samples += 1
        return dpo_sample

    elif weak_correct and not strong_correct:
        # 场景 3: 捡漏（Weak Correct ✅ Strong Wrong ❌）
        # 本地对了，云端反而错了，当然选本地
        dpo_sample = {
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": action_direct}],
            "rejected": [{"role": "assistant", "content": action_route}]
        }
        stats.save_money_cases += 1
        stats.dpo_samples += 1
        return dpo_sample

    else:
        # 场景 4: 双输（Weak Wrong ❌ Strong Wrong ❌）
        # 两个都做不对，丢弃
        stats.discarded_cases += 1
        return None


def main():
    parser = argparse.ArgumentParser(description="将 offline_cache 转换为 DPO 训练格式")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/offline_cache_math.jsonl",
        help="输入的离线缓存文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dpo_train_math.jsonl",
        help="输出的 DPO 训练数据文件路径"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试）"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化统计
    stats = Stats()

    # 处理数据
    print(f"📖 读取离线缓存文件: {args.input_file}")
    print(f"💾 输出 DPO 数据到: {args.output_file}")
    print("-" * 50)

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:

        for idx, line in enumerate(f_in):
            if args.max_samples and idx >= args.max_samples:
                break

            try:
                sample = json.loads(line.strip())
                dpo_sample = process_sample(sample, stats)

                if dpo_sample:
                    f_out.write(json.dumps(dpo_sample, ensure_ascii=False) + '\n')

                # 打印进度
                if (idx + 1) % 100 == 0:
                    print(f"✅ 已处理 {idx + 1} 行...")

            except json.JSONDecodeError as e:
                print(f"⚠️  第 {idx + 1} 行 JSON 解析错误: {e}")
                continue

    # 打印统计信息
    print("-" * 50)
    print("📊 处理完成！统计信息:")
    print(f"   总数据量: {stats.total_samples}")
    print(f"   生成的 DPO 样本: {stats.dpo_samples}")
    print(f"   └─ 省钱样本 (Case 1 + 3): {stats.save_money_cases}")
    print(f"   └─ 求稳样本 (Case 2): {stats.seek_stability_cases}")
    print(f"   丢弃样本 (Case 4): {stats.discarded_cases}")
    print("-" * 50)

    # 计算比例
    if stats.total_samples > 0:
        print("📈 样本比例:")
        print(f"   DPO 样本比例: {stats.dpo_samples/stats.total_samples*100:.1f}%")
        if stats.dpo_samples > 0:
            print(f"   └─ 省钱样本占比: {stats.save_money_cases/stats.dpo_samples*100:.1f}%")
            print(f"   └─ 求稳样本占比: {stats.seek_stability_cases/stats.dpo_samples*100:.1f}%")


if __name__ == "__main__":
    main()