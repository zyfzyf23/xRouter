import json
from datasets import load_dataset

def truncate_long_values(obj):
    """递归截断长字符串，防止刷屏"""
    if isinstance(obj, dict):
        return {k: truncate_long_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_long_values(v) for v in obj]
    elif isinstance(obj, str):
        # 如果字符串超过 100 字符，只显示前 100 个
        return obj[:10] + "..." if len(obj) > 100 else obj
    else:
        return str(obj)

def main():
    print("正在流式加载 LLM360/guru-RL-92k ...")
    # 记得加 trust_remote_code=True 防止某些数据集报错，虽然 Guru 可能不需要
    ds = load_dataset("LLM360/guru-RL-92k", split="train", streaming=True)
    
    print("\n=== 前 3 条数据结构预览 (已截断长文本) ===")
    for i, item in enumerate(ds):
        if i >= 3: break
        
        # 1. 打印所有字段名
        print(f"\n[Sample {i+1}] Keys: {list(item.keys())}")
        
        # 2. 重点查看我们关心的字段值
        print(f"  -> Domain (ability): {item.get('ability')}")
        print(f"  -> Source (data_source): {item.get('data_source')}")
        print(f"  -> Question (prompt): {str(item.get('prompt'))[:50]}...")
        print(f"  -> Answer (response): {str(item.get('response'))[:50]}...")
        
        # 3. 如果你想看完整结构但不想刷屏，用这个截断函数
        print(json.dumps(truncate_long_values(item), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()