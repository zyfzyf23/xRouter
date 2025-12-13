#!/usr/bin/env python3
"""
仅处理 Guru 数据集中的 Math 部分
将 data/train/math__combined_54.4k.parquet 处理成 XRouter 训练格式
"""

import os
import json
import argparse
import gc
import random
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

# XRouter 系统提示词
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
    """
    # 获取原始问题
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

    # 构建问题
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


def process_math_dataset(
    output_file: str,
    max_math_samples: int = 2500,
    train_dir: str = "data/train",
    seed: int = 42
):
    print("Processing Guru Math dataset only...")
    
    # 配置 PyArrow 内存分配器
    if hasattr(pa, 'jemalloc_set_decay_ms'):
        pa.jemalloc_set_decay_ms(0) 
    
    # 设置随机种子
    random.seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    math_count = 0

    # ======================
    # 处理数学数据
    # ======================
    math_file = os.path.join(train_dir, "math__combined_54.4k.parquet")
    
    if not os.path.exists(math_file):
        print(f"❌ Error: Math file not found at {math_file}")
        return

    print(f"Processing math data: {math_file}")
    
    parquet_file = pq.ParquetFile(math_file)
    total_rows = parquet_file.metadata.num_rows
    
    # 确定要采样的索引
    if total_rows > max_math_samples:
        target_indices = set(random.sample(range(total_rows), max_math_samples))
    else:
        target_indices = set(range(total_rows))

    # 使用 'w' 模式重新写入文件（覆盖）
    with open(output_file, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(total=len(target_indices), desc="Math samples")
        current_global_idx = 0
        
        # 批量处理以节省内存
        batch_size = 1000 
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_rows = batch.num_rows
            
            # 筛选当前 batch 中需要的索引
            batch_indices_to_keep = []
            for local_i in range(batch_rows):
                if (current_global_idx + local_i) in target_indices:
                    batch_indices_to_keep.append(local_i)
            
            if batch_indices_to_keep:
                relevant_batch = batch.take(batch_indices_to_keep)
                df_batch = relevant_batch.to_pandas()
                records = df_batch.to_dict('records')
                
                for row_dict in records:
                    sample_series = pd.Series(row_dict)
                    processed = process_math_sample(sample_series, math_count)
                    f_out.write(json.dumps(processed, ensure_ascii=False, default=str) + '\n')
                    
                    math_count += 1
                    pbar.update(1)
                
                del relevant_batch, df_batch, records
            
            current_global_idx += batch_rows
            del batch
            gc.collect() # 强制垃圾回收

        pbar.close()
    gc.collect()

    # ======================
    # 总结
    # ======================
    print(f"\n{'='*60}")
    print("Processing Summary (Math Only)")
    print(f"{'='*60}")
    print(f"Total Math samples extracted: {math_count}")
    print(f"Output file:                  {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Process Guru Math dataset for XRouter training')
    parser.add_argument(
        '--output-file',
        default='data/guru_math_only.jsonl',
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

    process_math_dataset(
        output_file=args.output_file,
        max_math_samples=args.max_math_samples,
        train_dir=args.train_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()