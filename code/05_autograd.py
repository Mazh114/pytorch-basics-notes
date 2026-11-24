# 第5章 Autograd 自动求导 · Chapter 5: Automatic Differentiation
# 官方教程: "When training neural networks, the most frequently used algorithm is
# back propagation. In this algorithm, parameters (model weights) are adjusted
# according to the gradient of the loss function with respect to the given parameter."
# 运行 / run:  .venv/bin/python code/05_autograd.py
import torch

# ---------- 1. 最简单的求导：y = x^2, dy/dx = 2x ----------
x = torch.tensor(3.0, requires_grad=True)   # requires_grad=True 开始追踪 / track
y = x * x
print("y =", y.item(), "| y.grad_fn =", y.grad_fn)   # grad_fn 记录"我是怎么算出来的"
y.backward()                                 # 从输出往回走一遍计算图
print("dy/dx at x=3:", x.grad.item(), "(手算 2x = 6)")   # 只有叶节点有 .grad

# ---------- 2. 线性回归模型的手算梯度验证 ----------
# 模型 z = w·x + b，loss = MSE(z, y)（单样本）
# 手推: dz/dw = x, dz/db = 1, loss = (z-y)^2
#   -> dloss/dz = 2(z-y) -> dloss/dw = 2(z-y)x, dloss/db = 2(z-y)
x1 = torch.tensor([1.0, 2.0])
y1 = torch.tensor([0.0, 1.0])
w = torch.tensor([0.5, -0.5], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
z = w @ x1 + b
loss = ((z - y1) ** 2).mean()
loss.backward()
print("\nautograd dloss/dw:", w.grad.tolist(), "dloss/db:", b.grad.item())
z_val = (w @ x1 + b).detach()                # detach: 拿出数值，不再追踪
# loss = mean(u^2), u = z - y1 (广播). ∂loss/∂z = Σ_i u_i (mean 把 2u_i 除回 2)
manual_w = (z_val - y1).sum() * x1           # ∂loss/∂w_j = (∂loss/∂z)·x1_j
manual_b = (z_val - y1).sum()                # ∂loss/∂b  =  ∂loss/∂z
print("手算   dloss/dw:", manual_w.tolist(), "dloss/db:", manual_b.item(), "（应一致）")

# ---------- 3. 梯度是累加的！Gradients accumulate ----------
# 官方: "PyTorch accumulates the gradients... If you want to compute the proper
# gradients, you need to zero out the grad property before."
x2 = torch.tensor(3.0, requires_grad=True)
f = x2 * x2
f.backward()
print("\n第一次 backward, x2.grad =", x2.grad.item())       # 6
f = x2 * x2
f.backward()
print("第二次 backward, x2.grad =", x2.grad.item(), "(6+6 累加了！)")
x2.grad.zero_()
f = x2 * x2
f.backward()
print("zero_() 之后 backward, x2.grad =", x2.grad.item(), "(干净了)")
# 这就是训练循环里 optimizer.zero_grad() 存在的原因。

# ---------- 4. 只有叶节点(leaf)有 .grad ----------
x3 = torch.tensor(2.0, requires_grad=True)
h = x3 * 3          # h 是中间节点（非叶）
out = h * h
out.backward()
print("\nx3.grad:", x3.grad.item(), "| h.grad:", h.grad, "(中间节点没有 grad；"
      "上一条 UserWarning 正是 PyTorch 在提醒这件事，属预期输出)")

# ---------- 5. 关闭追踪 no_grad / detach ----------
# 官方: "when we have trained the model and just want to apply it to some input
# data, i.e. we only want to do forward computations... stop tracking with
# torch.no_grad()"。也用于冻结参数 frozen parameters、推理提速。
w5 = torch.randn(3, requires_grad=True)
print("\noutside no_grad:", (w5 @ w5).requires_grad)          # True
with torch.no_grad():
    print("inside  no_grad:", (w5 @ w5).requires_grad)        # False
print("detach():", (w5.detach() @ w5.detach()).requires_grad) # False
