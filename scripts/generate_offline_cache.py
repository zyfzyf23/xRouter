#!/usr/bin/env python3
"""
生成xRouter离线缓存数据脚本
强制遍历模型池中的所有模型，为每个问题生成答案并验证正确性
"""

import os
import json
import argparse
import gc
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sympy import parse_expr, simplify
from sympy.parsing.latex import parse_latex # 需安装: pip install antlr4-python3-runtime

# 导入xRouter模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.tools.utils.router_utils import LLMRouter, MODEL_SPECS
from verl.utils.reward_score.math_dapo import verify


extra_instruction = (
    "Please reason step by step to solve the problem. "
    "You must use strict LaTeX formatting for all mathematical expressions and symbols throughout your reasoning (e.g., use $x^2$ instead of x^2). "
    "At the very end of your solution, output the final answer inside \\boxed{}. "
    "The content inside \\boxed{} must be the simplest mathematical value, with NO units, text, or variables. "
    "For example: \\boxed{42} or \\boxed{3\\pi} or \\boxed{\\frac{1}{2}}."
)

class LocalModelManager:
    """管理本地模型的加载和推理"""

    def __init__(self, model_id: str, model_path: Optional[str] = None, quantization: str = "4bit"):
        self.model_id = model_id
        self.model_path = model_path or "/home/zyf/models/Qwen2.5-1.5B-Instruct"
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型和分词器"""
        print(f"Loading local model: {self.model_id}")

        # 量化配置
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto"
        )

        print(f"Model {self.model_id} loaded successfully")

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.0, add_instruction: bool = True) -> Tuple[str, Dict]:
        """
        生成响应并返回token统计
        :param add_instruction: 是否添加 extra_instruction (解题时为True, 验证/判题时为False)
        """
        if self.model is None:
            self.load_model()

        # 格式化输入
        do_sample = temperature > 0
        
        # [修改点] 根据用途决定是否拼接 extra_instruction
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
                top_p=None,      # 避免自动启用
                top_k=None, 
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        output_tokens = outputs.shape[1] - input_tokens
        response = self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True
        )

        # 返回响应和元数据
        metadata = {
            "model_id": self.model_id,
            "model_name": MODEL_SPECS[self.model_id].name if self.model_id in MODEL_SPECS else self.model_id,
            "provider": "local",
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(input_tokens + output_tokens),
            "cost": 0.0  # 本地模型无成本
        }

        return response, metadata


class OfflineCacheGenerator:
    """离线缓存生成器"""

    def __init__(self, config: Dict):
        self.config = config
        self.router = LLMRouter()
        self.local_models = {}
        self.weak_model = config["weak_model"]
        self.strong_model = config["strong_model"]

        # 初始化本地模型管理器
        for model_name in [self.weak_model, self.strong_model]:
            if model_name in MODEL_SPECS and MODEL_SPECS[model_name].provider == "local":
                self.local_models[model_name] = LocalModelManager(model_name)

    def call_model_with_fallback(self, model_name: str, question: str) -> Tuple[str, Dict]:
        """统一的模型调用接口"""
        if model_name in self.local_models:
            # 本地模型调用
            response, metadata = self.local_models[model_name].generate(
                question,
                max_tokens=self.config.get("max_tokens", 2048),
                temperature=self.config.get("temperature", 0.7)
            )
            return response, metadata
        else:
            # 云端模型通过 LLMRouter 调用
            messages = [{"role": "user", "content": question + extra_instruction}]
            sampling_params = {
                "max_tokens": self.config.get("max_tokens", 2048),
                "temperature": self.config.get("temperature", 0.7)
            }
            response, metadata = self.router.call_model(model_name, messages, sampling_params)
            return response, metadata

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
                start = idx + 7 # len("\\boxed{")
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        if depth == 0:
                            return text[start:i] # 成功提取
                        depth -= 1
        
        # --- 策略 2: 正则兜底 (处理没写 boxed 的情况) ---
        match = re.search(r'(?:Answer|is):\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        return ""

    def verify_math_equivalence(self, pred_str: str, gt_str: str) -> bool:
        """
        数学等价性验证 (SymPy)
        """
        # 1. 字符串完全相等直接返回 True (最快)
        if pred_str.strip() == gt_str.strip():
            return True

        # 2. 尝试 SymPy 数学验证
        try:
            # 预处理：移除 LaTeX 中的某些非数学符号
            clean_pred = pred_str.replace(r'\dfrac', r'\frac').replace(r'\,', '')
            clean_gt = gt_str.replace(r'\dfrac', r'\frac').replace(r'\,', '')

            # 解析 LaTeX
            expr_pred = parse_latex(clean_pred)
            expr_gt = parse_latex(clean_gt)

            # 判断差值是否为 0
            # simplify(expr1 - expr2) == 0 是最严谨的判断
            diff = simplify(expr_pred - expr_gt)
            return diff == 0
            
        except Exception:
            # 如果解析失败 (比如包含复杂文本)，回退到归一化字符串比对
            # 这里你可以调用你原来的 normalize_final_answer
            return self.normalize_text(pred_str) == self.normalize_text(gt_str)

    def normalize_text(self, text: str) -> str:
        """
        加强版归一化：处理 LaTeX 的常见变体，防止字符串比对误判
        """
        # 1. 去除首尾空白和所有空格
        text = text.strip().replace(" ", "")
        
        # 2. 统一分数符号
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        
        # 3. 统一乘号 (可选，视情况而定)
        text = text.replace(r"\cdot", "").replace(r"\times", "")
        
        # 4. 【关键修复】处理指数/下标的花括号
        # 将 ^{n} 转换为 ^n (仅针对单字符)
        # 同样适用于 _{n} -> _n
        text = re.sub(r'\^\{([a-zA-Z0-9])\}', r'^\1', text)
        text = re.sub(r'_\{([a-zA-Z0-9])\}', r'_\1', text)
        
        # 5. 移除外层可能的额外括号 (视情况开启，有时候模型会输出 (answer))
        # if text.startswith("(") and text.endswith(")"):
        #     text = text[1:-1]
            
        return text

    def verify_answer(self, response: str, ground_truth: str) -> bool:
        """
        主验证入口
        """
        # 1. 提取
        pred_content = self.extract_answer_robust(response)
        if not pred_content:
            return False # 提取不到答案，直接判错
            
        # 2. 验证
        return self.verify_math_equivalence(pred_content, ground_truth)
    
    def process_single_question(self, item: Dict) -> Dict:
        """处理单个问题"""
        question = item["question"]
        ground_truth = item.get("ground_truth", item.get("answer", ""))

        # 初始化结果
        weak_model_spec = MODEL_SPECS.get(self.weak_model)
        strong_model_spec = MODEL_SPECS.get(self.strong_model)

        results = {
            "id": str(item.get("id", "")),
            "question": question,
            "ground_truth": ground_truth,
            "weak_model": weak_model_spec.name if weak_model_spec else self.weak_model,
            "weak_ans": "",
            "weak_token": "",
            "weak_correct": False,
            "strong_model": strong_model_spec.name if strong_model_spec else self.strong_model,
            "strong_ans": "",
            "strong_token": "",
            "strong_cost":"",
            "strong_correct": False
        }

        # 调用弱模型
        try:
            print(f"Processing question {item.get('id', '')} with {self.weak_model}...")
            weak_response, weak_metadata = self.call_model_with_fallback(self.weak_model, question)
            results["weak_ans"] = weak_response
            results["weak_token"] = str(weak_metadata.get("total_tokens", 0))
            results["weak_correct"] = self.verify_answer(weak_response, ground_truth)
        except Exception as e:
            print(f"弱模型 {self.weak_model} 处理问题 {item.get('id', '')} 时出错: {e}")
            results["weak_ans"] = f"ERROR: {str(e)}"
            results["weak_token"] = "0"
            results["weak_correct"] = False

        # 调用强模型
        try:
            print(f"Processing question {item.get('id', '')} with {self.strong_model}...")
            strong_response, strong_metadata = self.call_model_with_fallback(self.strong_model, question)
            input_tokens = strong_metadata.get("input_tokens", 0)
            output_tokens = strong_metadata.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            results["strong_ans"] = strong_response
            results["strong_token"] = str(total_tokens) 
            results["strong_cost"] = strong_metadata.get("cost", 0.0)
            results["strong_correct"] = self.verify_answer(strong_response, ground_truth)
        except Exception as e:
            print(f"强模型 {self.strong_model} 处理问题 {item.get('id', '')} 时出错: {e}")
            results["strong_ans"] = f"ERROR: {str(e)}"
            results["strong_token"] = "0"
            results["strong_correct"] = False

        return results

    def get_processed_ids(self, output_path: str) -> set:
        """
        高性能获取已处理ID:
        不加载整个文件,而是扫描文件提取ID,使用 Set 保证 O(1) 查找
        """
        processed_ids = set()
        if not os.path.exists(output_path):
            return processed_ids
            
        print(f"Scanning existing output: {output_path} ...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    # 尝试快速解析
                    # 只需要解析 "id" 字段，避免 full json load (如果行很大)
                    # 但为了稳健，这里还是用 json.loads
                    data = json.loads(line)
                    processed_ids.add(str(data.get("id")))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items.")
        return processed_ids
    
    def generate_cache(self, input_path: str, output_path: str):
        """生成离线缓存 (流式处理版)"""
        
        # 1. 获取已处理的 ID 集合 (修复断点续传)
        processed_ids = self.get_processed_ids(output_path)
        
        # 2. 统计输入文件总行数 (为了 tqdm 进度条，不读取内容)
        print(f"Counting input lines in {input_path}...")
        total_lines = sum(1 for _ in open(input_path, 'rb'))
        print(f"Total input questions: {total_lines}")

        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 预加载模型
        for model_manager in self.local_models.values():
            model_manager.load_model()

        processed_count = 0
        
        # 3. 流式读取输入 + 追加写入输出
        with open(output_path, "a", encoding="utf-8") as f_out:
            # 使用原生 open 读取，极大降低内存
            with open(input_path, "r", encoding="utf-8") as f_in:
                
                # 使用 tqdm 包装文件迭代器
                for line in tqdm(f_in, total=total_lines, desc="Processing"):
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    current_id = str(item.get("id"))

                    # 核心修复：检查 ID 是否已存在
                    if current_id in processed_ids:
                        continue

                    # 处理问题
                    result = self.process_single_question(item)

                    # 写入
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush() # 确保写入磁盘

                    processed_count += 1
                    
                    # 显存清理
                    if processed_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print(f"Done. Processed {processed_count} new items.")

def main():
    parser = argparse.ArgumentParser(description="生成离线缓存")
    parser.add_argument("--input_path", type=str, default="/home/zyf/xRouter/data/raw_prompts_simple.jsonl", help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="/home/zyf/xRouter/data/offline_cache_simple.jsonl", help="输出缓存路径")
    # parser.add_argument("--start_index", type=int, default=0, help="起始索引（用于断点续传）")
    parser.add_argument("--weak_model", type=str, default="qwen2.5-1.5b-local", help="弱模型ID")
    parser.add_argument("--strong_model", type=str, default="qwen", help="强模型ID") # 这里不改，去模型字典里改
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")

    args = parser.parse_args()

    # 验证模型配置
    if args.weak_model not in MODEL_SPECS:
        print(f"错误：弱模型 {args.weak_model} 未在 MODEL_SPECS 中配置")
        return

    if args.strong_model not in MODEL_SPECS:
        print(f"错误：强模型 {args.strong_model} 未在 MODEL_SPECS 中配置")
        return

    # 配置
    config = {
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature
    }

    print("="*50)
    print("xRouter 离线缓存生成器")
    print("="*50)
    print(f"弱模型: {args.weak_model}")
    print(f"强模型: {args.strong_model}")
    print(f"输入文件: {args.input_path}")
    print(f"输出文件: {args.output_path}")
    print("="*50)

    # 创建生成器
    generator = OfflineCacheGenerator(config)

    # 生成缓存
    generator.generate_cache(
        input_path=args.input_path,
        output_path=args.output_path,
        #start_index=args.start_index
    )


if __name__ == "__main__":
    main()