# 第7章 保存与加载 · Chapter 7: Save & Load the Model
# 官方教程: "PyTorch models store the learned parameters in an internal state
# dictionary, called state_dict."
# 运行 / run:  .venv/bin/python code/07_save_load.py
import os
import torch
from torch import nn

torch.manual_seed(0)
TMP = os.path.join(os.path.dirname(__file__), "..", "data", "ckpt")
os.makedirs(TMP, exist_ok=True)

# ---------- 1. 训练一个小模型备用 ----------
x = torch.linspace(-1, 1, 64).reshape(-1, 1)
y = 2 * x + 1
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
for _ in range(10):                                # 故意只训 10 步，留给 checkpoint 续训
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()

print("=== state_dict 里有什么 ===")
for k, v in model.state_dict().items():
    print(f"  {k}: shape {tuple(v.shape)}")

path = os.path.join(TMP, "linear_weights.pth")
torch.save(model.state_dict(), path)          # 官方推荐：只保存参数 / best practice

# ---------- 2. 加载：先实例化同结构模型，再 load_state_dict ----------
loaded = nn.Linear(1, 1)                       # 同结构！ / same architecture
loaded.load_state_dict(torch.load(path, weights_only=True))
loaded.eval()   # 官方: "be sure to call model.eval() before inferencing..."
xt = torch.tensor([[0.3], [-0.7]])
same = torch.allclose(model(xt), loaded(xt))
print("\n加载后与原模型输出一致:", same)

# ---------- 3. 结构不匹配会报错（state_dict 按名字+形状匹配）----------
bigger = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
try:
    bigger.load_state_dict(torch.load(path, weights_only=True))
except RuntimeError as e:
    print("\n加载到不同结构 -> RuntimeError:", str(e).split("\n")[0])

# ---------- 4. checkpoint：保存"现场"以便继续训练 ----------
# 官方相关教程: 保存 epoch / 模型参数 / 优化器状态 / loss，恢复后可接着训
ckpt = os.path.join(TMP, "checkpoint.pth")
torch.save({
    "epoch": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "loss": loss.item(),
}, ckpt)

resume = nn.Linear(1, 1)
resume_opt = torch.optim.SGD(resume.parameters(), lr=0.05)
checkpoint = torch.load(ckpt, weights_only=True)
resume.load_state_dict(checkpoint["model_state_dict"])
resume_opt.load_state_dict(checkpoint["optimizer_state_dict"])
print(f"\n恢复 checkpoint: epoch={checkpoint['epoch']} loss={checkpoint['loss']:.6f}")

# 再训 100 步，loss 继续下降 -> 说明优化器动量/状态也被恢复了
for _ in range(100):
    resume_opt.zero_grad()
    l2 = ((resume(x) - y) ** 2).mean()
    l2.backward()
    resume_opt.step()
print(f"继续训练 100 步后 loss: {l2.item():.6f}")

# ---------- 5. 整模型保存（legacy，不推荐）----------
whole = os.path.join(TMP, "whole_model.pth")
torch.save(model, whole)                       # 用 pickle 连结构一起存
m2 = torch.load(whole, weights_only=False)     # 加载时依赖类的定义代码可用
print("整模型保存加载成功，输出一致:", torch.allclose(model(xt), m2(xt)))
