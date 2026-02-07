# 环境设置指南

## 概述

本项目支持两个独立的环境：

- **venv** (CPU): 用于开发、CI、文档生成
- **venv-gpu** (CUDA): 用于 GPU 实验、剪枝评测

## 快速切换

### 切换到 CPU 环境（开发/CI）

```bash
cd /home/lsc/vibe-research
source venv/bin/activate
```

### 切换到 GPU 环境（实验/评测）

```bash
cd /home/lsc/vibe-research
source venv-gpu/bin/activate
```

## 环境详情

### CPU 环境 (venv)

| 组件 | 版本 |
|------|------|
| Python | 3.12.11 |
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| torchaudio | 2.10.0+cpu |

**用途**: 代码检查、单元测试、文档构建、CI

### GPU 环境 (venv-gpu)

| 组件 | 版本 |
|------|------|
| Python | 3.12.11 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA RTX A6000 x2 |

**用途**: YOLOv8 剪枝实验、GPU 加速评测

## 创建 GPU 环境（首次）

```bash
cd /home/lsc/vibe-research

# 1. 创建虚拟环境
python -m venv venv-gpu
source venv-gpu/bin/activate
python -m pip install -U pip

# 2. 安装 CUDA 版 PyTorch
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# 3. 安装项目和依赖
pip install -e ".[dev]"
pip install -U ultralytics torch-pruning

# 4. 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 常见问题

### Q: 为什么需要两个环境？

**A**: 
- CPU 环境不依赖 NVIDIA driver，适合 CI 和开发
- GPU 环境需要特定 CUDA 版本，用于实际实验
- 分离避免版本冲突（如 ultralytics 的 torchvision 兼容性问题）

### Q: 测试在不同环境表现如何？

**A**:
- **CPU 环境**: YOLOv8 集成测试会 skip（捕获任意异常）
- **GPU 环境**: 所有测试都能运行（如果依赖正确安装）

### Q: 如何检查当前环境？

```bash
which python                    # 查看 Python 路径
python -c "import torch; print(torch.__version__)"  # 查看 torch 版本
python -c "print(torch.cuda.is_available())"        # 检查 CUDA
```

## 参考

- [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)
- [CUDA 兼容性矩阵](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
