# 第3章 Transforms 数据变换 · Chapter 3: Transforms
# 官方教程: "Data does not always come in its final processed form that is required
# for training machine learning algorithms. We use transforms to perform some
# manipulation of the data and make it suitable for training."
# 运行 / run:  .venv/bin/python code/03_transforms.py
import torch
import torch.nn.functional as F

# ---------- 1. transform / target_transform 的分工 ----------
# transform 改特征 features，target_transform 改标签 labels。
# 任何可调用对象（函数/lambda/nn.Module）都可以当 transform。

# ---------- 2. 手写一个归一化 transform（不用 torchvision 也能学思想）----------
mean, std = torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])

def normalize(x: torch.Tensor) -> torch.Tensor:
    # 经典公式 x' = (x - mean) / std，把数据压到均值 0 附近
    return (x - mean) / std

x = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
print("raw:      ", x.tolist())
print("normalize:", normalize(x).tolist())

# ---------- 3. Lambda 变换：标签转 one-hot ----------
# 官方示例: target_transform = v2.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float())
target_transform = lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
print("\nlabel 3 ->", target_transform(3).tolist())   # [0,0,0,1,0,0,0,0,0,0]

# ---------- 4. Compose：把多个变换串成流水线 ----------
def compose(*fns):
    def run(v):
        for fn in fns:
            v = fn(v)
        return v
    return run

scale01 = lambda t: t / 255.0          # 像素 0..255 -> 0..1（模拟 v2.ToDtype(scale=True)）
pipeline = compose(scale01, lambda t: t.flatten())  # 先归一化再拉平 flatten
img = torch.tensor([[[255], [128], [0]]])            # 模拟 1x3 图像
print("\npipeline(raw image) =", pipeline(img).tolist())  # [1.0, 0.5019..., 0.0]

# ---------- 5. torchvision 真实体验（如果已安装）----------
try:
    from torchvision import datasets
    from torchvision.transforms import v2
    ds = datasets.FashionMNIST(
        root="data", train=True, download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform=v2.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()),
    )
    img, label = ds[0]
    print("\nFashionMNIST[0]: image", img.shape, img.dtype,
          "min/max:", img.min().item(), img.max().item())
    print("one-hot label:", label.tolist())
except Exception as e:  # noqa: BLE001
    print("跳过 torchvision 部分:", e)
