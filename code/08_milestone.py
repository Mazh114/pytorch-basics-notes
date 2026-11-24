# 第8章 里程碑：完整训练闭环 · Chapter 8: Milestone — the full loop
# 数据 -> 模型 -> 损失+优化器 -> train_loop -> test_loop -> 保存
# 运行 / run:  .venv/bin/python code/08_milestone.py
#   可选参数 --fashion 会训练真正的 FashionMNIST（需 torchvision，首次下载 26.4M）
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---------- 合成二分类：两个同心圆（纯 torch，无需下载任何数据）----------
def make_circles(n=800, noise=0.12):
    n1 = n // 2
    ang1 = torch.rand(n1) * 2 * torch.pi
    ang2 = torch.rand(n - n1) * 2 * torch.pi
    inner = torch.stack([torch.cos(ang1), torch.sin(ang1)], 1) * 0.6
    outer = torch.stack([torch.cos(ang2), torch.sin(ang2)], 1) * 1.6
    X = torch.cat([inner, outer]) + noise * torch.randn(n, 2)
    y = torch.cat([torch.zeros(n1), torch.ones(n - n1)]).long()
    return X, y

X, y = make_circles()
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
ntr = int(0.8 * len(X))                          # 80% 训练 / 20% 测试
train_ds = TensorDataset(X[:ntr], y[:ntr])
test_ds = TensorDataset(X[ntr:], y[ntr:])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=128)

# ---------- 模型：官方同款思路的 MLP ----------
class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),                # 2 个类别 -> 2 个 logits
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()                  # CrossEntropyLoss = LogSoftmax + NLLLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ---------- 官方同款 train_loop / test_loop ----------
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (Xb, yb) in enumerate(dataloader):
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()                    # 清空旧梯度
        loss.backward()                          # 反向传播
        optimizer.step()                         # 更新参数
        if batch % 5 == 0:
            print(f"  loss: {loss.item():>7f}  [{batch * len(Xb):>5d}/{len(dataloader.dataset):>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()                                 # 关掉 dropout/batchnorm 的训练行为
    correct, total = 0, 0
    with torch.no_grad():                        # 测试不需要梯度
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            correct += (pred.argmax(1) == yb).type(torch.float).sum().item()
            total += len(yb)
    print(f"  Test Error: Accuracy: {100 * correct / total:>0.1f}%")
    return correct / total

print("\n=== 训练 5 个 epoch ===")
epochs = 5
best = 0.0
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dl, model, loss_fn, optimizer)
    acc = test_loop(test_dl, model, loss_fn)
    if acc > best:                               # 挑战题答案：跟踪并保存最佳模型
        best = acc
        torch.save(model.state_dict(), "best_mlp.pth")

print(f"\n最佳 test accuracy: {best:.3f}（已保存到 best_mlp.pth）")

# ---------- 加载最佳模型并做一次推理 ----------
fresh = MLP().to(device)
fresh.load_state_dict(torch.load("best_mlp.pth", weights_only=True))
fresh.eval()
with torch.no_grad():
    sample = torch.tensor([[0.2, 0.1], [1.8, 0.0]], device=device)  # 内圈/外圈各一点
    print("sample logits:", fresh(sample).cpu().tolist())

# ---------- 可选：FashionMNIST 真实数据完整训练 ----------
if "--fashion" in sys.argv:
    from torchvision import datasets
    from torchvision.transforms import v2
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    train_dl = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=64)

    fm = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512), nn.ReLU(),
        nn.Linear(512, 10),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fm.parameters(), lr=1e-3)  # 官方超参数
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dl, fm, loss_fn, optimizer)
        test_loop(test_dl, fm, loss_fn)
    print("Done! (官方教程 10 epoch 可到 ~70% accuracy)")
