#!/usr/bin/env python3
"""测试所有必要的导入是否正常"""

print("🔍 测试导入...")

try:
    import torch
    print("✅ PyTorch 导入成功")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")

try:
    import torchvision
    print("✅ TorchVision 导入成功")
except ImportError as e:
    print(f"❌ TorchVision 导入失败: {e}")

try:
    from einops import rearrange
    print("✅ Einops 导入成功")
except ImportError as e:
    print(f"❌ Einops 导入失败: {e}")

try:
    from PIL import Image
    print("✅ PIL 导入成功")
except ImportError as e:
    print(f"❌ PIL 导入失败: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib 导入成功")
except ImportError as e:
    print(f"❌ Matplotlib 导入失败: {e}")

try:
    import config.configTrain as cfg
    print("✅ 配置模块导入成功")
    print(f"   模型名称: {cfg.model_name}")
    print(f"   数据路径: {cfg.data_path}")
except ImportError as e:
    print(f"❌ 配置模块导入失败: {e}")

try:
    from algorithm import Algorithm
    print("✅ Algorithm 导入成功")
except ImportError as e:
    print(f"❌ Algorithm 导入失败: {e}")

print("\n🎯 导入测试完成！")

