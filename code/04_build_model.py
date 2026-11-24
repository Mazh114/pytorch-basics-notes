# 第4章 搭建神经网络 · Chapter 4: Build the Neural Network
# 官方教程: "Every module in PyTorch subclasses the nn.Module. A neural network is
# a module itself that consists of other modules (layers)."
# 运行 / run:  .venv/bin/python code/04_build_model.py
import torch
from torch import nn

torch.manual_seed(0)

# ---------- 1. 定义网络：__init__ 建层, forward 定义前向 ----------
class NeuralNetwork(nn.Module):
    """官方 FashionMNIST 网络：784 -> 512 -> 512 -> 10."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):          # 前向：数据怎么流过网络
        x = self.flatten(x)        # [B,28,28] -> [B,784]
        logits = self.linear_relu_stack(x)
        return logits              # 返回 logits（未归一化的分数）raw scores

model = NeuralNetwork()
print(model)

# 官方警告: "Do not call model.forward() directly!" 用 model(x) 触发完整流程
X = torch.rand(1, 28, 28)
logits = model(X)
print("\nlogits shape:", logits.shape)             # [1, 10]

pred_probab = nn.Softmax(dim=1)(logits)            # logits -> 概率 [0,1]
y_pred = pred_probab.argmax(1)                     # 概率最大的类 / predicted class
print("prob sum:", pred_probab.sum().item(), "| predicted class:", y_pred.item())

# ---------- 2. 逐层看形状 Layers: Flatten -> Linear -> ReLU ----------
input_image = torch.rand(3, 28, 28)                # 一个 batch 3 张图
print("\ninput:          ", tuple(input_image.shape))
flat = nn.Flatten()(input_image)
print("after Flatten:  ", tuple(flat.shape))       # [3, 784]
hidden = nn.Linear(28 * 28, 20)(flat)
print("after Linear:   ", tuple(hidden.shape))     # [3, 20]
activated = nn.ReLU()(hidden)
print("after ReLU:     negatives -> 0 | count zero:", (activated == 0).sum().item())

# 官方: "Non-linear activations are what create the complex mappings between the
# model's inputs and outputs." 没有激活函数，多层线性叠加仍是线性映射：
# lin2(lin1(x)) == (W2 @ W1) x + (W2 b1 + b2)，永远等价于一层线性层。
lin1, lin2 = nn.Linear(6, 8), nn.Linear(8, 4)
x0 = torch.randn(3, 6)
with torch.no_grad():
    merged = nn.Linear(6, 4)
    merged.weight.copy_(lin2.weight @ lin1.weight)
    merged.bias.copy_(lin2.bias + lin2.weight @ lin1.bias)
    diff = (lin2(lin1(x0)) - merged(x0)).abs().max().item()
print("\n两层线性(无激活) 合并成一层后最大差异:", diff, "-> 仍是线性映射")

# ---------- 3. 参数 parameters / named_parameters ----------
print("\n=== parameters（训练时被优化的 weight 和 bias）===")
total = 0
for name, param in model.named_parameters():
    print(f"{name:38s} {tuple(param.shape)}")
    total += param.numel()
print("总参数量 / total parameters:", total)

# ---------- 4. nn.Sequential 等价写法（用 state_dict 保证完全一致）----------
seq_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)
for p_new, p_old in zip(seq_model.parameters(), model.parameters()):
    with torch.no_grad():
        p_new.copy_(p_old)                      # 按顺序把参数搬过去（结构完全相同）
x_test = torch.rand(2, 28, 28)
diff = (model(x_test) - seq_model(x_test)).abs().max().item()
print("\nSequential 与自定义类输出最大差异:", diff)   # 0.0 -> 两种写法等价

# ---------- 5. 搬到加速器 ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
x_dev = torch.rand(1, 28, 28, device=device)
print("model on:", next(model.parameters()).device, "| pred:", model(x_dev).argmax(1).item())
