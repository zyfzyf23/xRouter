#!/usr/bin/env python3
"""
vLLM 推理脚本 - 优化版本，处理所有 1819 个样本
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import time

# 添加项目根目录到 path
sys.path.append(str(Path(__file__).parent.parent))

# 定义和训练时一模一样的 System Prompt
ROUTER_SYSTEM_PROMPT = (
    "You are an intelligent router agent. "
    "Your task is to analyze the difficulty of the user's question and decide whether to solve it yourself or route it to a more powerful cloud model. "
    "- If the question is simple or within your capabilities, answer it directly starting with <think>I can solve this.</think>. "
    "- If the question is complex, requires multi-step reasoning that you are unsure about, or is beyond your capability, route it to the cloud model using the format: <think>This is beyond my capability.</think>\n\n<tool_code>call_remote_model(prompt)</tool_code>"
)

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"错误：缺少依赖: {e}")
    print("请运行: pip install vllm==0.6.3 transformers")
    sys.exit(1)

# 指向你合并后的模型路径 todo
tokenizer = AutoTokenizer.from_pretrained("outputs/sft_merged_1218_with_low_lr", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("outputs/sft_merged_1218_with_low_lr", trust_remote_code=True)

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """加载测试数据集"""
    test_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    return test_data

def ensure_boxed_requirement(question: str) -> str:
    """
    [关键] 复刻训练时的预处理逻辑
    确保 Prompt 与 DPO 训练时 100% 一致
    """
    # 简单的判断，避免重复添加
    if r'\boxed{' not in question and "Please answer within" not in question:
        if question.strip().endswith('.'):
            question = question.strip()[:-1] + '. Please answer within \\boxed{}.'
        elif question.strip().endswith('?'):
            question = question.strip()[:-1] + '? Please answer within \\boxed{}.'
        else:
            question = question.strip() + ' Please answer within \\boxed{}.'
    return question

def format_prompts(questions: List[str]) -> List[str]:
    prompts = []
    for q in questions:
        # 1. 确保 Boxed 后缀 (保持不变)
        processed_q = ensure_boxed_requirement(q)
        
        # 2. 构造 Prompt
        # ✅ 手动拼接 System Prompt
        # ✅ 依然保留 <think> 强制预填充，双重保险
        prompt = (
            f"<|im_start|>system\n{ROUTER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{processed_q}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )
        prompts.append(prompt)
    return prompts


def parse_router_logic(output: str) -> tuple[str, str]:
    """
    解析 Router 逻辑（增强版）
    能够识别标准标签、幻觉标签以及思维链中的拒答声明。
    返回：(routed_to, final_answer)
    """
    
    # 1. 定义所有可能代表“去云端”的信号列表
    cloud_indicators = [
        "<tool_code>",           # 标准训练目标
        "call_remote_model",    # 可能直接输出了函数调用
        "beyond my capability",  # System Prompt 中定义的拒答关键词
        "route to the cloud"     # 其他可能的语义表述
    ]

    # 2. 检查输出中是否包含任意一个信号
    # 使用 output.lower() 可以避免大小写不一致的问题（可选）
    if any(indicator in output for indicator in cloud_indicators):
        # 只要命中一个，就认为是云端任务
        # 返回空字符串作为 answer，因为后续流程会调用云端模型生成真正的 answer
        return "cloud", ""
    
    else:
        # 如果没有任何云端信号，则认为是本地回答
        # 这里直接返回 output，你可能后续需要清洗掉 <think> 标签
        return "local", output


def save_results(results: List[Dict], output_path: str):
    """保存结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    # 路径配置 todo
    model_path = "outputs/sft_merged_1218_with_low_lr"
    test_data_path = "data/test_set.jsonl"
    output_path = "data/test_results_stage_a.jsonl"
    checkpoint_path = "data/checkpoint_results_stagea.jsonl"

    print("=" * 60)
    print("vLLM 推理脚本启动（优化版）")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        sys.exit(1)

    # 检查测试数据
    if not os.path.exists(test_data_path):
        print(f"错误：测试数据不存在: {test_data_path}")
        sys.exit(1)

    # 加载测试数据
    print(f"加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"测试样本数: {len(test_data)}")

    # 使用全局 tokenizer
    print(f"\n使用已初始化的 tokenizer")
    print("Tokenizer 已就绪！")

    # 提取问题并格式化
    questions = [item["question"] for item in test_data]
    prompts = format_prompts(questions)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    start_time = time.time()

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,  # 提高到 0.9
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=False,  # 启用 CUDA graphs
        )
        print(f"模型加载成功！耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"错误：模型加载失败: {e}")
        print("尝试使用 eager 模式...")
        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                trust_remote_code=True,
                max_model_len=4096,
                enforce_eager=True,
            )
            print("模型加载成功（eager 模式）！")
        except Exception as e2:
            print(f"错误：模型加载失败: {e2}")
            sys.exit(1)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,  
        max_tokens=2048,  # 固定为 2048
        stop=[],  # 不设置停止词
        skip_special_tokens=False,  # 保留特殊token
    )

    # 批量推理
    print(f"\n开始批量推理...")
    batch_size = 64  # 更大的批处理大小
    results = []
    local_count = 0
    cloud_count = 0

    # 检查是否有检查点
    start_idx = 0
    if os.path.exists(checkpoint_path):
        print(f"发现检查点文件，从断点继续...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        start_idx = len(results)
        print(f"已处理 {start_idx} 个样本")

    # 分批处理
    total_batches = (len(prompts) - start_idx + batch_size - 1) // batch_size
    progress_bar = tqdm(
        range(start_idx, len(prompts), batch_size),
        desc="推理进度",
        initial=start_idx // batch_size,
        total=total_batches
    )

    save_every = 5  # 每 5 批保存一次检查点

    for batch_idx, i in enumerate(progress_bar):
        batch_prompts = prompts[i:i + batch_size]
        batch_data = test_data[i:i + batch_size]

        # 生成
        outputs = llm.generate(batch_prompts, sampling_params)

        # 处理结果
        for output, item in zip(outputs, batch_data):
            generated_text = output.outputs[0].text.strip()

            # 解析 Router 逻辑
            routed_to, final_answer = parse_router_logic(generated_text)

            if routed_to == "local":
                local_count += 1
            else:
                cloud_count += 1

            # 保存结果
            result = {
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "domain": item.get("domain", ""),
                "source": item.get("source", ""),
                "model_output": generated_text,
                "routed_to": routed_to,
                "final_answer": final_answer,
            }
            results.append(result)

        # 定期保存检查点
        if (batch_idx + 1) % save_every == 0:
            save_results(results, checkpoint_path)
            print(f"\n检查点已保存 ({len(results)}/{len(test_data)})")

    # 保存最终结果
    print(f"\n保存结果到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(results, output_path)

    # 删除检查点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 统计信息
    total_time = time.time() - start_time
    avg_time_per_sample = total_time / len(test_data)

    print("\n" + "=" * 60)
    print("推理完成！")
    print(f"总样本数: {len(results)}")
    print(f"本地处理: {local_count} ({local_count/len(results)*100:.1f}%)")
    print(f"云端路由: {cloud_count} ({cloud_count/len(results)*100:.1f}%)")
    print(f"总耗时: {total_time/60:.2f} 分钟")
    print(f"平均每样本: {avg_time_per_sample:.2f} 秒")
    print(f"结果已保存到: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()