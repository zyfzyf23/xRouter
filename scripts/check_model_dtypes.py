# check_model_dtypes.py
import os
import safetensors.torch as safetensors

def get_dtype_from_safetensors(path):
    """从 .safetensors 文件中读取第一个张量的 dtype"""
    if not os.path.exists(path):
        return None, f"文件不存在: {path}"
    try:
        tensors = safetensors.load_file(path)
        if not tensors:
            return None, "文件中无张量"
        first_key = next(iter(tensors.keys()))
        dtype = tensors[first_key].dtype
        return dtype, first_key
    except Exception as e:
        return None, f"加载失败: {e}"

def main():
    # 配置路径（请根据你的实际路径修改）
    lora_dir = "outputs/sft_1218_with_low_lr"
    merged_dir = "outputs/sft_merged_1218_with_low_lr"

    lora_path = os.path.join(lora_dir, "adapter_model.safetensors")
    merged_path = os.path.join(merged_dir, "model.safetensors")

    print("🔍 检查模型精度 (dtype)...\n")

    # 检查 LoRA 权重
    print("1. LoRA 适配器权重:")
    dtype, info = get_dtype_from_safetensors(lora_path)
    if dtype is not None:
        print(f"   - 文件: {lora_path}")
        print(f"   - 示例张量: {info}")
        print(f"   - 精度 (dtype): {dtype}")
    else:
        print(f"   ❌ {info}")

    print()

    # 检查合并后的模型
    print("2. 合并后的完整模型:")
    dtype, info = get_dtype_from_safetensors(merged_path)
    if dtype is not None:
        print(f"   - 文件: {merged_path}")
        print(f"   - 示例张量: {info}")
        print(f"   - 精度 (dtype): {dtype}")
    else:
        print(f"   ❌ {info}")

    print("\n✅ 检查完成。")

if __name__ == "__main__":
    main()