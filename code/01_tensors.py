# 第1章 Tensors 张量 · Chapter 1: Tensors
# 官方教程: "Tensors are a specialized data structure... we use tensors to encode
# the inputs and outputs of a model, as well as the model's parameters."
# 运行 / run:  .venv/bin/python code/01_tensors.py
import numpy as np
import torch

torch.manual_seed(0)  # 固定随机种子，输出可复现 / reproducible random numbers

print("=== 1. 初始化 Initializing ===")
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)                     # 从数据创建, dtype 自动推断
np_array = np.array(data)
x_np = torch.from_numpy(np_array)               # 从 numpy 创建
x_ones = torch.ones_like(x_data)                # 保留形状/类型 keeps properties
x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖类型 overrides dtype
print("from data: ", x_data.tolist(), "| from numpy:", x_np.tolist())
print("ones_like:", x_ones.tolist(), "| rand_like:", x_rand.tolist())

shape = (2, 3)                                  # shape 是维度元组 / tuple of dimensions
print("rand:", torch.rand(shape).shape, "ones:", torch.ones(shape).shape,
      "zeros:", torch.zeros(shape).shape)

print("\n=== 2. 三大属性 Attributes: shape / dtype / device ===")
t = torch.rand(3, 4)
print("shape:", t.shape, "| dtype:", t.dtype, "| device:", t.device)

print("\n=== 3. 索引与切片 Indexing & slicing (numpy 风格) ===")
tensor = torch.ones(4, 4)
print("first row:   ", tensor[0].tolist())
print("first col:   ", tensor[:, 0].tolist())
print("last col:    ", tensor[..., -1].tolist())
tensor[:, 1] = 0                                # 直接给切片赋值 / assign to a slice
print("after assign:", tensor.tolist())

print("\n=== 4. 拼接 Joining: cat vs stack ===")
a = torch.ones(2, 2)
cat = torch.cat([a, a], dim=0)                  # 沿已有维度接 / concatenate
stack = torch.stack([a, a], dim=0)              # 新增一维 / add a new dimension
print("cat dim=0:  ", cat.shape)   # [4,2]
print("stack dim=0:", stack.shape) # [2,2,2]

print("\n=== 5. 算术 @ 矩阵乘 vs * 逐元素 ===")
m = tensor.clone()
m[:, 1] = 0
y1 = m @ m.T          # 矩阵乘 / matrix multiplication
y2 = m * m            # 逐元素乘 / element-wise product
print("m @ m.T row0:", y1[0].tolist())
print("m * m   row0:", y2[0].tolist())

print("\n=== 6. 聚合与 .item() ===")
print("sum:", m.sum().item(), type(m.sum().item()).__name__)  # 单元素张量 -> Python 数值
print("max:", m.max().item(), "| mean:", m.mean().item())

print("\n=== 7. in-place 操作（_ 后缀）===")
print("before add_:", m[0].tolist())
m.add_(5)             # 原地修改，会改变 m 本身 / in-place modifies m
print("after add_: ", m[0].tolist())
# 官方警告: "In-place operations save some memory, but can be problematic when
# computing derivatives because of an immediate loss of history."

print("\n=== 8. 变形与广播 Reshape & broadcast ===")
flat = torch.arange(12)                # [0..11]
print("view(3,4) shape:", flat.view(3, 4).shape)
print("view(-1) shape: ", flat.view(-1).shape)   # -1 = 自动推算这一维
big = torch.ones(3, 4)
row = torch.tensor([10.0, 20.0, 30.0, 40.0])     # (4,) 广播成 (3,4)
print("broadcast (3,4)+(4,):", (big + row)[0].tolist())
col = torch.tensor([[1.0], [2.0], [3.0]])        # (3,1) 广播成 (3,4)
print("broadcast (3,4)+(3,1):", (big * col)[1].tolist())

print("\n=== 9. 与 numpy 共享内存 Bridge with NumPy ===")
n = np.ones(5)
t2 = torch.from_numpy(n)
n += 1
print("numpy 改了，张量也变:", t2.tolist())  # 共享同一块内存 / shared memory
t3 = torch.ones(5)
n2 = t3.numpy()
t3.add_(1)
print("张量改了，numpy 也变:", n2.tolist())
print("(想要独立副本请用 .clone() 或 torch.tensor(n))")
