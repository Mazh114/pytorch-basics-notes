# 第6章 训练循环 · Chapter 6: Optimizing Model Parameters
# 官方教程: "in each iteration the model makes a guess about the output, calculates
# the error in its guess (loss), collects the derivatives of the error with respect
# to its parameters, and optimizes these parameters using gradient descent."
# 训练三步: optimizer.zero_grad() -> loss.backward() -> optimizer.step()
# 运行 / run:  .venv/bin/python code/06_optimization.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# ---------- 数据：拟合 y = 2x + 1（第2章的合成数据集思路）----------
x = torch.linspace(-1, 1, 64).reshape(-1, 1)     # 64 个样本, 特征 [64,1]
y = 2 * x + 1
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

def train(model, lr, epochs, zero_grads=True, loss_fn=nn.MSELoss(), opt="sgd"):
    if opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)                     # 1. 猜 / guess
            loss = loss_fn(pred, yb)             # 2. 算误差 / loss
            if zero_grads:
                optimizer.zero_grad()            # 3a. 清空旧梯度（不清会累加！）
            loss.backward()                      # 3b. 反向传播求梯度
            optimizer.step()                     # 3c. 沿梯度更新参数
    return loss.item()

print("=== 1. 基本训练：线性层拟合 y=2x+1 ===")
model = nn.Linear(1, 1)                          # 学到的应为 w≈2, b≈1
final = train(model, lr=0.05, epochs=300)
w = model.weight.item(); b = model.bias.item()
print(f"learned: w={w:.4f}, b={b:.4f} | final loss={final:.6f}")

print("\n=== 2. 学习率 learning rate 的影响 ===")
# 官方: "Smaller values yield slow learning speed, while large values may result
# in unpredictable behavior during training."
for lr in (1e-5, 0.05, 10.0):
    m = nn.Linear(1, 1)
    torch.manual_seed(1)
    final = train(m, lr=lr, epochs=300)
    print(f"lr={lr:<7g} final loss={final:.6f}")   # 1e-5 几乎不学; 10 发散/NaN

print("\n=== 3. 忘记 zero_grad 会怎样 ===")
m_bad = nn.Linear(1, 1)
final = train(m_bad, lr=0.05, epochs=300, zero_grads=False)
print(f"不 zero_grad, final loss={final:.6f}（收敛差/震荡，因为梯度越滚越大）")

print("\n=== 4. SGD vs Adam ===")
m1 = nn.Linear(1, 1); torch.manual_seed(2)
l1 = train(m1, lr=0.05, epochs=20, opt="sgd")
m2 = nn.Linear(1, 1); torch.manual_seed(2)
l2 = train(m2, lr=0.05, epochs=20, opt="adam")
print(f"只训 20 步: SGD  final loss={l1:.6f} | Adam final loss={l2:.6f}"
      "（Adam 自适应步长，这里收敛快得多）")

print("\n=== 5. 分类版 test_loop（官方写法）===")
# 官方: "with torch.no_grad() ensures that no gradients are computed during test
# mode... also serves to reduce unnecessary gradient computations and memory usage"
torch.manual_seed(0)
Xc = torch.randn(200, 2)
yc = ((Xc[:, 0] * Xc[:, 1]) > 0).long()          # 简单异或形分布 XOR-ish
clf = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.SGD(clf.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()                  # "combines nn.LogSoftmax and nn.NLLLoss"
dl = DataLoader(TensorDataset(Xc, yc), batch_size=32, shuffle=True)
for epoch in range(50):
    clf.train()
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(clf(xb), yb)
        loss.backward()
        opt.step()
clf.eval()
with torch.no_grad():
    pred = clf(Xc)
    acc = (pred.argmax(1) == yc).type(torch.float).sum().item() / len(yc)
print(f"分类训练后 accuracy = {acc:.3f}")
