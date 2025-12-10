#!/usr/bin/env python3
"""
将 data/train/ 中的 Guru 数据集（parquet 格式）处理成 XRouter 训练格式
严格模仿 XRouter 原项目的数据预处理逻辑
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# XRouter 系统提示词（从 router_data_preprocess.py 复制）
SYSTEM_PROMPT = """You are a helpful assistant — an intelligent LLM router and prompt engineer. Your job is to:

1. Analyze the user query and conversation history.
2. If the task is simple and clearly within your capabilities, like casual chat, who are you, or 1+1=? type questions, answer the user directly.
3. Otherwise, choose one or more models from the available tools (tools are named call_<model_name>).
4. For each selected model, craft an optimized system prompt and sampling parameters.
5. Execute model calls, then either:
   a) SUMMARIZE: synthesize multiple model responses into one validated reply to the user, or
   b) SELECT: return *one* model's response verbatim using the select_response tool (see rules below).

## CALL / SELECT RULES
- call_<model_name> arguments must contain only:
  - the optimized system prompt (string), and
  - sampling parameters (e.g., temperature).
- Do NOT include conversation history or the user query in the system prompt — the runtime supplies them automatically.
- select_response(call_id) returns the selected model call's response verbatim to the user (copy-paste).
- You may select only **one** model response with select_response.
- If you use select_response, you **must not** perform parallel tool use in the same turn.
- Prefer SELECT when one model's output is clearly superior and requires no synthesis; prefer SUMMARIZE when answers must be combined, reconciled, or edited.

## MODEL SELECTION GUIDELINES
- Primary objective: solve the user query at the *lowest cost possible*.
- If both model A and model B can solve the task, choose the cheaper one.
- Cheapest valid option: answer yourself. But you'll be penalized for inadequate answers.
- You may call multiple models to improve quality; balance expected accuracy gains vs cost.
- You are a weak chat model; if uncertain or the task is complex, call stronger models.

## SYSTEM PROMPT ENGINEERING
- Role & scope: 1-sentence role + 1-line capability to define the scope.
- Be concrete: include an example if ambiguity exists in the user query.
- Specify requirements & constraints: schema, style, output format, if any applicable.
- Match model to task: making the prompt better exploit model strengths (reasoning, creativity, code).
- Try your best to make the prompt unlock the selected model's capabilities.

## SAMPLING RECOMMENDATIONS
- Temperature: Use low temperature for precise/factual tasks, high temperature for creative tasks.

## RESPONSE FORMAT
Respond exactly in this format:
</think>
Your detailed step by step reasoning. For instance, think and decide if you can answer directly. If not, think carefully if the task requires heavy reasoning, and which model you plan to call and why.
</think>
`your_content`

- If `your_content` is a function call (e.g., call_<model_name>), it will be executed.
- After model responses:
  - If SUMMARIZING: `your_content` would be plain text which summarizes the responses. It will be returned directly to user. Follow any format the user requested. Use it when you must reconcile or refine multiple outputs.
  - If SELECTING: call select_response(call_id). That selected model output will be returned directly to user; it can saves the cost and time, so this is the preferred way to respond to the user.
"""


def process_math_sample(sample: pd.Series, idx: int) -> Dict:
    """
    处理数学样本（参考 data_preprocess/math/dapo_or1_merge_deduped.py）

    Args:
        sample: pandas Series，包含数学题数据
        idx: 样本索引

    Returns:
        Dict: 处理后的样本
    """
    # 获取原始问题
    # prompt 字段是消息列表，需要提取
    if isinstance(sample['prompt'], list) and len(sample['prompt']) > 0:
        # 如果 prompt 是消息列表格式
        original_question = sample['prompt'][0].get('content', '')
        # 移除末尾的 "Please output the final answer" 部分（如果存在）
        if 'Please output the final answer' in original_question:
            original_question = original_question.split('Please output the final answer')[0].strip()
    else:
        # 备用方案：从 extra_info 中获取
        extra_info = sample.get('extra_info', {})
        if isinstance(extra_info, dict):
            original_question = extra_info.get('original_question', str(sample.get('prompt', '')))
        else:
            original_question = str(sample.get('prompt', ''))

    # 获取答案
    reward_model = sample.get('reward_model', {})
    if isinstance(reward_model, dict):
        answer = reward_model.get('ground_truth', '')
    else:
        answer = ''

    # 构建问题（模仿 dapo_or1_merge_deduped.py）
    question = original_question + " Please output the final answer within \\boxed{}."

    # 构建 XRouter 格式的 messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    processed_sample = {
        "domain": "math",
        "source_id": f"guru_math_{sample.get('data_source', 'unknown')}_{idx}",
        "original_question": original_question,
        "original_answer": answer,
        "xrouter_messages": messages,
        "extra_info": {
            "data_source": sample.get('data_source', ''),
            "ability": sample.get('ability', 'math'),
            "reward_model": reward_model,
            "is_unique": sample.get('is_unique', False),
            "qwen2.5_7b_pass_rate": sample.get('qwen2.5_7b_pass_rate', 0.0),
            "qwen3_30b_pass_rate": sample.get('qwen3_30b_pass_rate', 0.0)
        }
    }

    return processed_sample


def process_code_sample(sample: pd.Series, idx: int) -> Optional[Dict]:
    """
    处理代码样本（参考 data_preprocess/codegen/ 下的多个文件）

    Args:
        sample: pandas Series，包含代码题数据
        idx: 样本索引

    Returns:
        Optional[Dict]: 处理后的样本，如果处理失败返回 None
    """
    # 获取原始问题
    query = sample.get('query', '')

    # 数据清洗（参考 primeintellect.py 和 humaneval.py）
    # 1. 移除图像引用
    if "<image>" in query.lower() or "[image]" in query.lower():
        print(f"Skipping sample {idx}: contains image reference")
        return None

    # 2. 移除 "### Answer:" 等无关提示（参考 leetcode2k.py）
    query = query.replace("### Answer: (use the provided format with backticks)", "").strip()
    query = query.replace("### Format: ", "### Format:\n")

    # 3. 标准化问题格式（参考 livecodebench.py 的 prompt 格式）
    # 如果问题不包含标准的 Python 程序员提示，则添加
    if "You are an expert Python programmer" not in query:
        # 提取核心问题描述
        if "### Question:" in query:
            question_part = query.split("### Question:")[1].strip()
        else:
            question_part = query

        # 重新构建标准格式
        query = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

### Question:
{question_part}"""

    # 获取答案和测试信息
    original_answer = sample.get('completion', '')
    entry_point = sample.get('entry_point', '')
    test_code = sample.get('test', '')

    # 验证测试代码（参考 humaneval.py 的验证逻辑）
    # 如果有测试代码和入口点，构建完整的测试
    if test_code and entry_point:
        # 添加调用检查
        full_test = f"{test_code}\n\ncheck({entry_point})"
    else:
        full_test = test_code

    # 构建 XRouter 格式的 messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    # 处理 meta 字段
    meta = sample.get('meta', {})
    if isinstance(meta, dict):
        # 创建可序列化的 meta 副本
        serializable_meta = {}
        for k, v in meta.items():
            if hasattr(v, 'isoformat'):
                serializable_meta[k] = v.isoformat()
            else:
                serializable_meta[k] = v
        meta = serializable_meta

    # 提取难度信息（如果有）
    difficulty = ""
    if isinstance(meta, dict) and 'difficulty' in meta:
        difficulty = meta['difficulty']

    processed_sample = {
        "domain": "code",
        "source_id": f"guru_code_{sample.get('data_source', 'unknown')}_{idx}",
        "original_question": query,
        "original_answer": original_answer,
        "xrouter_messages": messages,
        "extra_info": {
            "data_source": sample.get('data_source', ''),
            "ability": sample.get('ability', 'codegen'),
            "entry_point": entry_point,
            "test": full_test,
            "meta": meta,
            "difficulty": difficulty,
            "question_title": meta.get('question_title', '') if isinstance(meta, dict) else '',
            "tags": meta.get('tags', []) if isinstance(meta, dict) else [],
            "is_unique": sample.get('is_unique', False),
            "qwen2.5_7b_pass_rate": sample.get('qwen2.5_7b_pass_rate', 0.0),
            "qwen3_30b_pass_rate": sample.get('qwen3_30b_pass_rate', 0.0)
        }
    }

    return processed_sample


def process_guru_dataset(
    output_file: str,
    max_math_samples: int = 1000,
    max_code_samples: int = 1000,
    train_dir: str = "data/train"
):
    """
    处理 Guru 数据集（从本地 parquet 文件）

    Args:
        output_file: 输出文件路径
        max_math_samples: 最大数学题样本数
        max_code_samples: 最大代码题样本数
        train_dir: 训练数据目录
    """
    print("Processing Guru dataset from local parquet files...")

    math_samples = []
    code_samples = []

    # 处理数学数据
    math_file = os.path.join(train_dir, "math__combined_54.4k.parquet")
    if os.path.exists(math_file):
        print(f"Loading math data from {math_file}...")
        df_math = pd.read_parquet(math_file)

        print(f"Found {len(df_math)} math samples")

        # 随机采样
        if len(df_math) > max_math_samples:
            df_math = df_math.sample(n=max_math_samples, random_state=42)

        print(f"Processing {len(df_math)} math samples...")
        for idx, (_, sample) in enumerate(tqdm(df_math.iterrows(), total=len(df_math))):
            processed = process_math_sample(sample, idx)
            math_samples.append(processed)

    # 处理代码数据
    code_files = [
        "codegen__leetcode2k_1.3k.parquet",
        "codegen__livecodebench_440.parquet",
        "codegen__primeintellect_7.5k.parquet",
        "codegen__taco_8.8k.parquet"
    ]

    total_code_samples = 0
    for code_file in code_files:
        file_path = os.path.join(train_dir, code_file)
        if os.path.exists(file_path):
            print(f"\nLoading code data from {file_path}...")
            df_code = pd.read_parquet(file_path)

            # 计算还可以采集多少样本
            remaining_samples = max_code_samples - total_code_samples
            if remaining_samples <= 0:
                break

            # 如果样本数超过所需，随机采样
            if len(df_code) > remaining_samples:
                df_code = df_code.sample(n=remaining_samples, random_state=42)

            print(f"Processing {len(df_code)} code samples from {code_file}...")
            for idx, (_, sample) in enumerate(tqdm(df_code.iterrows(), total=len(df_code))):
                processed = process_code_sample(sample, idx + total_code_samples)
                if processed is not None:
                    code_samples.append(processed)

            total_code_samples += len(df_code)

    # 合并所有样本
    all_samples = math_samples + code_samples

    print(f"\nProcessing complete!")
    print(f"Math samples: {len(math_samples)}")
    print(f"Code samples: {len(code_samples)}")
    print(f"Total samples: {len(all_samples)}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存为 JSONL 格式
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False, default=str) + '\n')

    print("Done!")

    # 打印一些示例
    print("\n=== Example Math Sample ===")
    if math_samples:
        sample = math_samples[0]
        print(f"Domain: {sample['domain']}")
        print(f"Source: {sample['source_id']}")
        print(f"Question: {sample['original_question'][:200]}...")
        print(f"Messages: {json.dumps(sample['xrouter_messages'], indent=2)[:500]}...")

    print("\n=== Example Code Sample ===")
    if code_samples:
        sample = code_samples[0]
        print(f"Domain: {sample['domain']}")
        print(f"Source: {sample['source_id']}")
        print(f"Question: {sample['original_question'][:200]}...")
        print(f"Messages: {json.dumps(sample['xrouter_messages'], indent=2)[:500]}...")


def main():
    parser = argparse.ArgumentParser(description='Process Guru dataset for XRouter training')
    parser.add_argument(
        '--output-file',
        default='data/guru_xrouter_processed.jsonl',
        help='Output file path'
    )
    parser.add_argument(
        '--max-math-samples',
        type=int,
        default=1000,
        help='Maximum number of math samples to collect'
    )
    parser.add_argument(
        '--max-code-samples',
        type=int,
        default=1000,
        help='Maximum number of code samples to collect'
    )
    parser.add_argument(
        '--train-dir',
        default='data/train',
        help='Directory containing the parquet files'
    )

    args = parser.parse_args()

    process_guru_dataset(
        output_file=args.output_file,
        max_math_samples=args.max_math_samples,
        max_code_samples=args.max_code_samples,
        train_dir=args.train_dir
    )


if __name__ == '__main__':
    main()