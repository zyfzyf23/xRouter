#!/usr/bin/env python3
"""
使用qwen2.5-1.5b-instruct模型处理test_set.jsonl文件
为每个问题生成答案并验证正确性，添加weak_output、cost、correct字段
"""

import os
import json
import argparse
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入xRouter模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 额外的推理指导语
extra_instruction = (
    "Please reason step by step to solve the problem. "
    "You must use strict LaTeX formatting for all mathematical expressions and symbols throughout your reasoning (e.g., use $x^2$ instead of x^2). "
    "At the very end of your solution, output the final answer inside \\boxed{}. "
    "The content inside \\boxed{} must be the simplest mathematical value, with NO units, text, or variables. "
    "For example: \\boxed{42} or \\boxed{3\\pi} or \\boxed{\\frac{1}{2}}."
)


class LocalModelManager:
    """管理本地模型的加载和推理"""

    def __init__(self, model_path: str = "/home/zyf/models/Qwen2.5-1.5B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型和分词器"""
        print(f"Loading local model: {self.model_path}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型（不使用量化，使用 FP16 精度）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配到 GPU
            low_cpu_mem_usage=True
        )

        print(f"Model loaded successfully on {self.device}")

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, add_instruction: bool = True) -> Tuple[str, Dict]:
        """
        生成响应并返回token统计
        :param add_instruction: 是否添加 extra_instruction
        """
        if self.model is None:
            self.load_model()

        # 格式化输入
        do_sample = temperature > 0

        # 根据用途决定是否拼接 extra_instruction
        content = prompt + extra_instruction if add_instruction else prompt

        messages = [{"role": "user", "content": content}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32768
        ).to(self.device)

        input_tokens = inputs.input_ids.shape[1]

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        generated_tokens = outputs[0][input_tokens:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 计算 token 统计
        output_tokens = len(generated_tokens)

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": 0.0  # 本地模型成本为 0
        }

        return response, metadata


class TestSetProcessor:
    """测试集处理器 - 使用qwen2.5-1.5b-instruct模型"""

    def __init__(self, weak_model: str = "qwen2.5-1.5b-instruct", model_path: str = "../Qwen2.5-1.5B-Instruct"):
        self.weak_model = weak_model
        self.local_model = LocalModelManager(model_path=model_path)
        print(f"使用本地模型: Qwen2.5-1.5B-Instruct")

    def extract_answer_robust(self, text: str) -> str:
        """
        稳健提取器：
        1. 优先提取 \boxed{...} (支持嵌套)
        2. 如果没有 boxed，尝试正则找 Answer: (兜底)
        """
        # --- 策略 1: 堆栈法提取 \boxed ---
        if "\\boxed{" in text:
            idx = text.rfind("\\boxed{")
            if idx >= 0:
                depth = 0
                start = idx + 7  # len("\\boxed{")
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        if depth == 0:
                            return text[start:i]  # 成功提取
                        depth -= 1

        # --- 策略 2: 正则兜底 (处理没写 boxed 的情况) ---
        match = re.search(r'(?:Answer|is):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def verify_math_equivalence(self, pred_str: str, gt_str: str) -> bool:
        """
        数学等价性验证（简化版）
        """
        # 1. 字符串完全相等直接返回 True
        if pred_str.strip() == gt_str.strip():
            return True

        # 2. 尝试转换为浮点数比较
        try:
            # 提取数字
            pred_num = float(re.findall(r'[-+]?\d*\.?\d+', pred_str)[0])
            gt_num = float(re.findall(r'[-+]?\d*\.?\d+', gt_str)[0])
            return abs(pred_num - gt_num) < 1e-6
        except:
            # 3. 归一化文本比较
            pred_norm = pred_str.replace(" ", "").replace("\\,", "").replace(r"\frac", "").replace("/", "")
            gt_norm = gt_str.replace(" ", "").replace("\\,", "").replace(r"\frac", "").replace("/", "")
            return pred_norm == gt_norm

    def verify_answer(self, response: str, ground_truth: str) -> bool:
        """
        主验证入口
        """
        # 1. 提取答案
        pred_content = self.extract_answer_robust(response)
        if not pred_content:
            return False  # 提取不到答案，直接判错

        # 2. 验证
        return self.verify_math_equivalence(pred_content, ground_truth)

    def call_weak_model(self, question: str, max_retries: int = 3) -> Tuple[str, float]:
        """
        调用弱模型生成答案
        返回: (response, cost)
        """
        try:
            print(f"调用本地模型 {self.weak_model}...")
            response, metadata = self.local_model.generate(
                question,
                max_tokens=2048,
                temperature=0.0,
                add_instruction=True
            )

            # 获取成本信息（本地模型成本为 0）
            cost = metadata.get("cost", 0.0)

            return response, cost

        except Exception as e:
            print(f"本地模型调用失败: {e}")
            # 返回错误信息
            return f"ERROR: {str(e)}", 0.0

    def process_single_item(self, item: Dict) -> Dict:
        """处理单个测试项"""
        # 复制原始数据
        result = item.copy()

        # 获取问题和答案
        question = item["question"]
        ground_truth = item.get("ground_truth", item.get("answer", ""))

        print(f"\n处理问题 {item.get('id', '')}")

        # 调用弱模型
        weak_output, cost = self.call_weak_model(question)

        # 验证答案
        correct = self.verify_answer(weak_output, ground_truth)

        # 添加新字段
        result["weak_output"] = weak_output
        result["cost"] = cost
        result["correct"] = correct

        # 打印结果
        print(f"  弱模型: Qwen2.5-1.5B-Instruct (Local)")
        print(f"  成本: ${cost:.6f}")
        print(f"  正确性: {'✓' if correct else '✗'}")

        return result

    def get_processed_ids(self, output_path: str) -> set:
        """
        获取已处理的ID集合（支持断点续传）
        """
        processed_ids = set()
        if not os.path.exists(output_path):
            return processed_ids

        print(f"扫描已处理的输出: {output_path} ...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    processed_ids.add(str(data.get("id")))
                except:
                    continue
        print(f"找到 {len(processed_ids)} 个已处理的项目")
        return processed_ids

    
    def process_test_set(self, input_path: str, output_path: str, max_items: Optional[int] = None):
        """处理整个测试集"""

        # 获取已处理的ID
        processed_ids = self.get_processed_ids(output_path)

        # 统计总行数
        print(f"统计输入文件行数: {input_path}...")
        total_lines = sum(1 for _ in open(input_path, 'rb'))
        if max_items:
            total_lines = min(total_lines, max_items)
        print(f"总问题数: {total_lines}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        correct_count = 0
        total_cost = 0.0

        # 流式处理
        with open(output_path, "a", encoding="utf-8") as f_out:
            with open(input_path, "r", encoding="utf-8") as f_in:

                for line in tqdm(f_in, total=total_lines, desc="Processing"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    current_id = str(item.get("id"))

                    # 跳过已处理的
                    if current_id in processed_ids:
                        continue

                    # 处理项目
                    result = self.process_single_item(item)

                    # 更新统计
                    processed_count += 1
                    if result.get("correct", False):
                        correct_count += 1
                    total_cost += result.get("cost", 0.0)

                    # 写入结果
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

                    # 打印进度
                    if processed_count % 10 == 0:
                        accuracy = correct_count / processed_count * 100 if processed_count > 0 else 0
                        print(f"\n进度: {processed_count}/{total_lines}")
                        print(f"正确率: {accuracy:.1f}%")
                        print(f"总成本: ${total_cost:.6f}\n")

                    # 检查是否达到最大限制
                    if max_items and processed_count >= max_items:
                        break

        # 最终统计
        print("\n" + "="*50)
        print("处理完成!")
        print(f"处理项目数: {processed_count}")
        print(f"正确答案数: {correct_count}")
        print(f"正确率: {correct_count / processed_count * 100:.2f}%" if processed_count > 0 else "正确率: 0%")
        print(f"总成本: ${total_cost:.6f}")
        print(f"输出文件: {output_path}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="使用本地qwen2.5-1.5b-instruct模型处理测试集")
    parser.add_argument("--input_path", type=str, default="data/test_set.jsonl", help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="data/test_set_with_weak_model.jsonl", help="输出数据路径")
    parser.add_argument("--weak_model", type=str, default="qwen2.5-1.5b-instruct", help="弱模型ID")
    parser.add_argument("--model_path", type=str, default="../Qwen2.5-1.5B-Instruct", help="本地模型路径")
    parser.add_argument("--max_items", type=int, default=None, help="最大处理项目数（用于测试）")

    args = parser.parse_args()

    print("="*50)
    print("测试集本地模型处理脚本")
    print("="*50)
    print(f"输入文件: {args.input_path}")
    print(f"输出文件: {args.output_path}")
    print(f"模型路径: {args.model_path}")
    if args.max_items:
        print(f"最大处理数: {args.max_items}")
    print("="*50)

    # 创建处理器
    try:
        processor = TestSetProcessor(weak_model=args.weak_model, model_path=args.model_path)
    except Exception as e:
        print(f"初始化错误: {e}")
        return

    # 处理测试集
    processor.process_test_set(
        input_path=args.input_path,
        output_path=args.output_path,
        max_items=args.max_items
    )


if __name__ == "__main__":
    main()