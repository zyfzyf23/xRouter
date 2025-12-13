#!/usr/bin/env python3
"""
将 data/train/ 中的 Guru 数据集（parquet 格式）处理成 XRouter 训练格式
严格模仿 XRouter 原项目的数据预处理逻辑
"""

import os
import json
import argparse
import gc  # 引入垃圾回收模块
import random
from typing import Dict, List, Optional, Generator
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

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
    max_math_samples: int = 2500,
    train_dir: str = "data/train",
    seed: int = 42
):
    print("Processing Guru dataset (Aggressive Memory Saving Mode)...")
    
    # 【优化 1】配置 PyArrow 内存分配器
    # 尝试让 PyArrow 只要有空闲内存就立即归还给 OS
    if hasattr(pa, 'jemalloc_set_decay_ms'):
        pa.jemalloc_set_decay_ms(0) 
    
    # 设置随机种子
    random.seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_count = 0
    math_count = 0
    code_count = 0

    # ======================
    # 处理数学数据
    # ======================
    math_file = os.path.join(train_dir, "math__combined_54.4k.parquet")
    if os.path.exists(math_file):
        print(f"\n[1/2] Processing math data: {math_file}")
        
        parquet_file = pq.ParquetFile(math_file)
        total_rows = parquet_file.metadata.num_rows
        
        if total_rows > max_math_samples:
            target_indices = set(random.sample(range(total_rows), max_math_samples))
        else:
            target_indices = set(range(total_rows))

        with open(output_file, 'w', encoding='utf-8') as f_out:
            pbar = tqdm(total=len(target_indices), desc="Math samples")
            current_global_idx = 0
            
            # 【优化 2】减小 Batch Size，降低瞬时压力
            batch_size = 1000 
            
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                batch_rows = batch.num_rows
                
                # 筛选索引
                batch_indices_to_keep = []
                for local_i in range(batch_rows):
                    if (current_global_idx + local_i) in target_indices:
                        batch_indices_to_keep.append(local_i)
                
                if batch_indices_to_keep:
                    # 转换为 Pandas
                    relevant_batch = batch.take(batch_indices_to_keep)
                    df_batch = relevant_batch.to_pandas()
                    
                    # 转换为 dict 列表再迭代，通常比 iterrows 更省内存
                    records = df_batch.to_dict('records')
                    
                    for row_dict in records:
                        # 临时构造 Series 以适配你的旧函数接口
                        # (虽然有些浪费，但为了不改动 process_math_sample 逻辑)
                        sample_series = pd.Series(row_dict)
                        processed = process_math_sample(sample_series, total_count)
                        f_out.write(json.dumps(processed, ensure_ascii=False, default=str) + '\n')
                        
                        math_count += 1
                        total_count += 1
                        pbar.update(1)
                    
                    # 显式删除临时变量
                    del relevant_batch, df_batch, records
                
                current_global_idx += batch_rows
                
                # 显式删除 PyArrow batch 对象
                del batch
                
                # 【优化 3】强制每轮循环都进行垃圾回收
                # 虽然会稍微变慢，但能保命
                gc.collect()

            pbar.close()
        gc.collect() # 大循环结束后再次回收
    else:
        print(f"⚠ Math file not found: {math_file}")

    # ======================
    # 处理代码数据
    # ======================
    code_datasets = [
        ("codegen__leetcode2k_1.3k.parquet", 1.0, "LeetCode"),
        ("codegen__livecodebench_440.parquet", 1.0, "LiveCodeBench"),
        ("codegen__primeintellect_7.5k.parquet", 0.1, "PrimeIntellect"),
        ("codegen__taco_8.8k.parquet", 0.1, "TACO")
    ]

    expected_total = 0
    actual_collected = 0
    # code_count 已经在上面初始化过了

    write_mode = 'a' if os.path.exists(output_file) else 'w'
    
    with open(output_file, write_mode, encoding='utf-8') as f_out:
        for code_file, sample_ratio, name in code_datasets:
            file_path = os.path.join(train_dir, code_file)
            if not os.path.exists(file_path):
                continue

            print(f"\n[2/2] Processing: {name}")

            try:
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                
                target_count = int(total_rows * sample_ratio) if sample_ratio < 1.0 else total_rows
                expected_total += target_count
                
                target_indices = set()
                if sample_ratio < 1.0:
                    target_indices = set(random.sample(range(total_rows), target_count))
                
                current_file_collected = 0
                current_global_idx = 0
                
                pbar = tqdm(total=target_count, desc=f"{name}")
                
                # 【优化 2】减小 Batch Size
                batch_size = 1000

                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    batch_rows = batch.num_rows
                    
                    if sample_ratio >= 1.0:
                        relevant_batch = batch
                    else:
                        local_indices_to_keep = []
                        for local_i in range(batch_rows):
                            if (current_global_idx + local_i) in target_indices:
                                local_indices_to_keep.append(local_i)
                        
                        if not local_indices_to_keep:
                            current_global_idx += batch_rows
                            del batch
                            gc.collect() # 即使跳过也要回收
                            continue
                        
                        relevant_batch = batch.take(local_indices_to_keep)

                    df_batch = relevant_batch.to_pandas()
                    records = df_batch.to_dict('records') # 转为字典处理
                    
                    for row_dict in records:
                        sample_series = pd.Series(row_dict)
                        processed = process_code_sample(sample_series, total_count)
                        
                        if processed is not None:
                            f_out.write(json.dumps(processed, ensure_ascii=False, default=str) + '\n')
                            current_file_collected += 1
                            code_count += 1
                            total_count += 1
                            pbar.update(1)
                            
                            if sample_ratio < 1.0 and current_file_collected >= target_count:
                                break
                    
                    current_global_idx += batch_rows
                    
                    # 清理内存
                    del batch, relevant_batch, df_batch, records
                    
                    # 【优化 3】强制回收
                    gc.collect()
                    
                    if sample_ratio < 1.0 and current_file_collected >= target_count:
                        break

                pbar.close()
                actual_collected += current_file_collected
                gc.collect()

            except Exception as e:
                print(f"❌ Error processing {code_file}: {e}")
                # 打印 traceback 方便调试
                import traceback
                traceback.print_exc()

    # ======================
    # 总结
    # ======================
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    print(f"Math samples:     {math_count}")
    print(f"Code samples:     {code_count}")
    print(f"Total samples:    {total_count}")
    print(f"Expected code:    {expected_total}")
    print(f"Actual code:      {actual_collected}")
    print(f"Output file:      {output_file}")
    print(f"{'='*60}")

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
        default=2500,
        help='Maximum number of math samples to collect'

    )
    parser.add_argument(
        '--train-dir',
        default='data/train',
        help='Directory containing the parquet files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling'
    )

    args = parser.parse_args()

    process_guru_dataset(
        output_file=args.output_file,
        max_math_samples=args.max_math_samples,
        train_dir=args.train_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()