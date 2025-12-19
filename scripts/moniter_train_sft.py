import re
import csv
import ast
import pandas as pd

def parse_training_log(log_file_path):
    # --- 存储数据的容器 ---
    training_steps = []    # 存储 {'loss': ..., 'epoch': ...}
    eval_steps = []        # 存储 {'eval_loss': ..., 'epoch': ...}
    config_params = {}     # 存储初始配置 (如 Batch size)
    final_summary = {}     # 存储最后的 Summary (如 train_runtime)

    # --- 正则表达式 ---
    # 匹配类似 {'loss': 0.8178, ...} 的 Python 字典字符串
    dict_pattern = re.compile(r"\{'.*?':.*?\}")
    
    # 匹配配置参数，例如: "INFO -   - Train batch size: 2"
    config_pattern = re.compile(r"INFO -   - (.*?): (.*)")
    
    # 匹配最终摘要，例如: "  train_loss               =     0.1616"
    summary_pattern = re.compile(r"\s+(.*?)\s+=\s+(.*)")

    print(f"正在读取文件: {log_file_path} ...")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 1. 提取字典数据 (Training 和 Eval 的过程数据)
        dict_match = dict_pattern.search(line)
        if dict_match:
            try:
                # 使用 ast.literal_eval 安全地将字符串转为 Python 字典
                data_dict = ast.literal_eval(dict_match.group(0))
                
                if 'loss' in data_dict:
                    # 这是一个训练步
                    training_steps.append(data_dict)
                elif 'eval_loss' in data_dict:
                    # 这是一个评估步
                    eval_steps.append(data_dict)
            except Exception as e:
                # 忽略解析错误的行（通常是进度条截断导致的）
                continue

        # 2. 提取初始配置参数
        if "INFO -   -" in line:
            config_match = config_pattern.search(line)
            if config_match:
                key = config_match.group(1).strip()
                val = config_match.group(2).strip()
                config_params[key] = val

        # 3. 提取最终摘要 (通常在文件末尾 ***** train metrics ***** 之后)
        if "=" in line and ("train_" in line or "eval_" in line or "epoch" in line):
            summary_match = summary_pattern.match(line)
            if(summary_match):
                key = summary_match.group(1).strip()
                val = summary_match.group(2).strip()
                final_summary[key] = val

    # --- 数据处理与保存 ---

    # 1. 保存详细的历史数据 (Training History)
    if training_steps or eval_steps:
        # 将列表转换为 DataFrame
        df_train = pd.DataFrame(training_steps)
        df_eval = pd.DataFrame(eval_steps)
        
        # 为了方便在一张表中看，我们将 eval 数据合并到同一个 CSV
        # 通常 eval 只有 epoch 信息，我们可以把它们拼接到一起
        df_all = pd.concat([df_train, df_eval], ignore_index=True)
        
        # 按照 epoch 排序
        if 'epoch' in df_all.columns:
            df_all = df_all.sort_values(by='epoch')
            
        # 调整列顺序，让关键指标在前面
        cols = list(df_all.columns)
        priority_cols = ['epoch', 'step', 'loss', 'eval_loss', 'learning_rate', 'eval_mean_token_accuracy']
        new_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]
        df_all = df_all[new_cols]

        output_history = "training_history.csv"
        df_all.to_csv(output_history, index=False)
        print(f"✅ 详细训练/评估历史已保存至: {output_history} ({len(df_all)} 条记录)")

    # 2. 保存配置和摘要 (Summary)
    # 合并 Config 和 Final Summary
    all_summary = {**config_params, **final_summary}
    
    if all_summary:
        df_summary = pd.DataFrame(list(all_summary.items()), columns=['Parameter', 'Value'])
        output_summary = "training_summary.csv"
        df_summary.to_csv(output_summary, index=False)
        print(f"✅ 训练参数与最终结果已保存至: {output_summary}")

if __name__ == "__main__":
    # 指定你的日志文件名
    log_file = "my_run.log"
    
    try:
        parse_training_log(log_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file}，请确保该文件在当前目录下。")
    except Exception as e:
        print(f"发生错误: {e}")