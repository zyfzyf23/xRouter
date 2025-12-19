import json

def count_weak_correct_true(file_path):
    total = 0
    correct_count = 0
    s_correct_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total += 1
                if data.get("weak_correct") is True:
                    correct_count += 1
                if data.get("strong_correct") is True:
                    s_correct_count += 1
            except json.JSONDecodeError:
                continue
    return total, correct_count,s_correct_count

if __name__ == "__main__":
    # 👇 指定 JSONL 文件路径
    file_path = "data/offline_cache_math.jsonl"  
    
    total, correct,s_correct = count_weak_correct_true(file_path)
    print(f"Total valid entries: {total}")
    print(f"Entries with weak_correct=true: {correct}")
    print(f"Entries with strong_correct=true: {s_correct}")