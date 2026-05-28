# 从0开始基于transformer进行股价预测（pytorch版本）- 实现计划

## 博客概述
博客介绍如何仅用 Transformer 的 Encoder 部分对工商银行（601398）2014-2024 年日线行情做时间序列预测：用连续 5 天 6 维特征（open/close/high/low/volume/money）预测第 6 天收盘价。给出完整 PyTorch 数据管道、模型定义、训练与评估代码，并解释为何弃用 Decoder、为何用 Linear 代替 Embedding。

**技术栈：** PyTorch + 聚宽数据 + Matplotlib 可视化

---

## 任务清单

### Task 1: 环境准备与数据获取

**涉及文件：**
- 创建: `requirements.txt`（依赖列表）
- 创建: `data_download.py`（聚宽下载脚本）

**步骤：**
- [x] Step 1.1: 写入依赖：torch、pandas、numpy、matplotlib、jqresearch（或 akshare 替代）
- [x] Step 1.2: 实现 `get_price` 下载 601398 2014-01-01 至 2024-01-01 日线并保存为 `data/601398.XSHG.csv`

---

### Task 2: 数据预处理与加载器

**涉及文件：**
- 创建: `dataset.py`（`split_data` 函数及 `TensorDataset`/`DataLoader` 封装）

**步骤：**
- [x] Step 2.1: 读取 csv，提取 6 列并归一化到 0-1
- [x] Step 2.2: 用 5 天滑窗生成 `(seq_len, feature)` 样本与次日 close 标签
- [x] Step 2.3: 按 8:2 划分训练/验证，返回 `DataLoader` 及反归一化用的 min/max

---

### Task 3: Transformer Encoder 模型实现

**涉及文件：**
- 创建: `model.py`（`PositionalEncoding`、`EncoderLayer`、`Encoder`、`Transformer`）

**步骤：**
- [x] Step 3.1: 用 `nn.Linear(feature, d_model)` 替换词嵌入；`d_model=512, n_heads=8, n_layers=6`
- [x] Step 3.2: 实现 `PositionalEncoding` 与 `MultiHeadAttention`/`FeedForward`
- [x] Step 3.3: 堆叠 6 层 Encoder，最后加 `nn.Linear(d_model,1)` 并时序平均输出 `[batch]`

---

### Task 4: 训练与验证循环

**涉及文件：**
- 创建: `train.py`（主训练脚本）

**步骤：**
- [x] Step 4.1: 实例化模型、MSE loss、Adam(lr=1e-3)，梯度裁剪 0.5
- [x] Step 4.2: 50 epoch 训练，每 epoch 记录平均 loss，保存最佳模型 `best_model.pt`
- [x] Step 4.3: 每 epoch 在验证集计算 MAE、RMSE、PCC 并绘图保存到 `picture/`

---

### Task 5: 测试与结果可视化

**涉及文件：**
- 创建: `evaluate.py`（指标计算与绘图函数）

**步骤：**
- [x] Step 5.1: 实现反归一化、MAE/RMSE/PCC 计算
- [x] Step 5.2: 绘制真实 vs 预测曲线并保存 png
- [x] Step 5.3: 输出最终测试集指标

---

### Task 6: 项目验证与完成

**步骤：**
- [x] Step 6.1: 运行 `python train.py` 确认无报错，loss 下降
- [x] Step 6.2: 检查 `picture/` 生成效果图及控制台指标打印
- [x] Step 6.3: 清理临时文件，提交完整代码包