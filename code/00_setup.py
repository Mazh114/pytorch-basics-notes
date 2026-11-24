# 第0章 准备环境 · Chapter 0: Setup
# 运行 / run:  .venv/bin/python code/00_setup.py
import torch

print("torch 版本 / version:", torch.__version__)

# 官方教程: Every operation can run on CPU or an accelerator (CUDA, MPS, ...)
# macOS Apple Silicon 的加速器是 MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("加速器 / accelerator:", device)

# 张量默认建在 CPU 上，需要 .to() 显式搬运 / tensors are created on the CPU by default
x = torch.ones(3)
print("初始设备 / initial device:", x.device)

x = x.to(device)
print("搬移后 / after .to():  ", x.device)

y = (x * 2).cpu()  # numpy 只能接收 CPU 张量，回去之前必须 .cpu()
print("搬回后 / after .cpu():", y.device, "| values:", y.numpy())
