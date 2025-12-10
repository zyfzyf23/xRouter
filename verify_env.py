import sys
import torch
import importlib.util

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        if importlib.util.find_spec(import_name) is not None:
            module = __import__(import_name)
            version = getattr(module, '__version__', '未知版本')
            print(f"✅ {name:<15} 安装成功 (v{version})")
            return True
        else:
            print(f"❌ {name:<15} 未找到")
            return False
    except Exception as e:
        print(f"❌ {name:<15} 导入报错: {e}")
        return False

print("="*30)
print("正在进行深度学习环境自检...")
print("="*30)

# 1. 检查 Python 版本
print(f"Python 版本: {sys.version.split()[0]}")

# 2. 检查 PyTorch 与 CUDA
if check_package("torch"):
    print(f"   CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        # 简单张量测试
        try:
            x = torch.tensor([1.0]).cuda()
            print("   GPU 张量计算测试: 通过")
        except Exception as e:
            print(f"   GPU 张量计算测试: 失败 ({e})")
    else:
        print("   ❌ 警告: 未检测到 GPU，无法进行训练！")

# 3. 检查 Flash Attention (关键)
check_package("flash_attn")

# 4. 检查 Bitsandbytes (4060 救星)
check_package("bitsandbytes")

# 5. 检查 XRouter 核心依赖
check_package("litellm")
check_package("verl")  # 如果你只安装了[gpu,math]，这个可能在 xRouter 目录下被识别

# 6. 检查 训练工具链 (HuggingFace)
check_package("transformers")
check_package("peft")
check_package("trl")
check_package("accelerate")

print("="*30)
print("检查完成！")
