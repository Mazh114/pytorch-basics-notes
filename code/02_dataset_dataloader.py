# 第2章 Dataset 与 DataLoader · Chapter 2: Datasets & DataLoaders
# 官方教程: "Dataset stores the samples and their corresponding labels, and
# DataLoader wraps an iterable around the Dataset to enable easy access to the samples."
# 运行 / run:  .venv/bin/python code/02_dataset_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

torch.manual_seed(0)  # 固定随机种子，输出可复现 / reproducible random numbers

# ---------- 1. 自定义 Dataset：必须实现三件套 __init__ / __len__ / __getitem__ ----------
class InMemoryDataset(Dataset):
    """合成数据集 y = 2x + 1，x 为 2 维向量 / synthetic linear dataset."""
    def __init__(self, n=100):
        torch.manual_seed(0)
        self.x = torch.randn(n, 2)               # 特征 features
        self.y = self.x @ torch.tensor([2.0, 3.0]) + 1.0  # 标签 labels

    def __len__(self):
        return len(self.x)          # 样本个数 / number of samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]  # 返回 (特征, 标签) 元组

ds = InMemoryDataset(n=100)
print("len(ds) =", len(ds))                       # 100
x3, y3 = ds[2]
print("ds[2] ->", x3.tolist(), y3.item())

# ---------- 2. DataLoader：小批量 minibatch + 打乱 shuffle ----------
dl = DataLoader(ds, batch_size=7, shuffle=True)
print("\nlen(dl) =", len(dl))                     # ceil(100/7) = 15 个 batch
xb, yb = next(iter(dl))                           # 取第一个 batch
print("batch shapes:", xb.shape, yb.shape)        # [7,2] [7]

print("\n=== shuffle 对比（同一 DataLoader 迭代两个 epoch）===")
dl3 = DataLoader(ds, batch_size=7, shuffle=True)
for epoch in range(2):
    first = next(iter(dl3))   # 每个 epoch 重新洗牌，第一批样本不同
    print(f"epoch {epoch} 第一个样本 x[0]:", first[0][0].tolist())
print("(把 shuffle=False 试试 -> 两个 epoch 完全相同)")

print("\n=== 不打乱时最后一队人数 / last batch size when n % batch != 0 ===")
dl2 = DataLoader(ds, batch_size=33, shuffle=False)
print("len(dl2) =", len(dl2))
last_x, last_y = list(dl2)[-1]
print("last batch shape:", last_x.shape)          # 100 = 33*3 + 1 -> [1,2]

print("\n=== 3. 现成的 TensorDataset（ Often you don't need a custom class）===")
tds = TensorDataset(torch.arange(10.0).reshape(-1, 1), torch.arange(10.0))
tdl = DataLoader(tds, batch_size=4)
fx, fy = next(iter(tdl))
print("TensorDataset batch:", fx.ravel().tolist(), "|", fy.tolist())

print("\n=== 4. 内置数据集 FashionMNIST（需要 torchvision，首次会下载 26.4M）===")
try:
    from torchvision import datasets
    from torchvision.transforms import v2
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    imgs, labels = next(iter(train_loader))
    # 官方输出: Feature batch shape: torch.Size([64, 1, 28, 28])
    print("Feature batch shape:", imgs.shape)
    print("Labels batch shape: ", labels.shape, "| label[0] =", labels[0].item())
except Exception as e:  # noqa: BLE001
    print("跳过 FashionMNIST（未装 torchvision 或下载失败）:", e)
