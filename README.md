# PyTorch 基础 · 双语学习打卡 / Learn PyTorch Basics (bilingual check-in site)

Based on the official PyTorch tutorial series
[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
(Tensors → Datasets & DataLoaders → Transforms → Build Model → Autograd → Optimization → Save & Load),
rebuilt as a bilingual (English essence + 中文注释) hands-on course with **more concrete examples and runnable drills**.

**在线使用 / Live page**: https://mazh114.github.io/pytorch-basics-notes/

## 功能 / Features

- 侧边栏目录，点击直达，可收起 / Collapsible sidebar TOC with scrollspy
- 多用户打卡进度（localStorage 按用户分开）/ Multi-user check-in progress (per-browser localStorage)
- 每章大打卡 + 每个知识点/习题小打卡，带日期戳 / Big chapter check-ins + per-item check-ins with date stamps
- 导出/导入 JSON 备份 / Export & import progress backup
- 速查表 + 12 条自查清单 / Cheat sheet + 12-item self-check checklist

## 本地跑代码 / Run the code locally

All example & exercise solutions are in [`code/`](code/), verified with Python 3.14 + torch 2.15.0 on Apple Silicon (MPS).

```bash
git clone https://github.com/Mazh114/pytorch-basics-notes.git
cd pytorch-basics-notes
python3 -m venv .venv && .venv/bin/pip install torch torchvision numpy
.venv/bin/python code/00_setup.py        # 环境自检
.venv/bin/python code/08_milestone.py    # 里程碑：完整训练闭环（100% accuracy）
.venv/bin/python code/08_milestone.py --fashion   # FashionMNIST 完整训练（首次下载 26.4M）
```

| 脚本 | 内容 |
|---|---|
| `00_setup.py` | 版本/设备检测，CPU↔MPS 往返 |
| `01_tensors.py` | 创建/属性/索引/拼接/算术/in-place/广播/numpy 桥 |
| `02_dataset_dataloader.py` | 自定义 Dataset、batch/shuffle、FashionMNIST |
| `03_transforms.py` | 归一化、Lambda one-hot、Compose |
| `04_build_model.py` | nn.Module、逐层形状、参数量、Sequential 等价性 |
| `05_autograd.py` | 手推梯度验证、梯度累加、no_grad |
| `06_optimization.py` | 训练循环、学习率实验、zero_grad bug、SGD vs Adam |
| `07_save_load.py` | state_dict、加载报错、checkpoint 续训 |
| `08_milestone.py` | 同心圆二分类端到端（可选 `--fashion`） |

## 打卡说明 / Check-in notes

打卡数据保存在**当前浏览器**的 localStorage，不会跨设备同步；换设备请用侧边栏「导出备份 / Import」。
Progress is stored in this browser only; use Export/Import to migrate.

## 来源 / Sources

- PyTorch official tutorial: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Sister site (Python basics, CS50P): https://mazh114.github.io/cs50p-pytorch-notes/
