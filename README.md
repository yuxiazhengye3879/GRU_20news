# 20 Newsgroups 二分类（1层 GRU）

本项目使用 `20newsgroups` 中两个类别完成二分类任务：

- `alt.atheism`
- `soc.religion.christian`

并严格按要求使用：

1. `newsgroups_train` 与 `newsgroups_test`。
2. 将 `newsgroups_train` 按 `0.8/0.2` 划分为训练集与验证集。
3. 使用 **1层 GRU** 完成文本分类。

## 目录结构

- `20_news_data.py`：主程序（数据加载、预处理、训练、验证、测试）
- `requirements.txt`：依赖文件
- `README.md`：项目说明

## 模型结构与参数选择

### 模型结构

- Embedding：`vocab_size -> 128`
- GRU：`num_layers=1`，`hidden_size=128`，`batch_first=True`
- Dropout：`0.3`
- 全连接层：`128 -> 1`
- 输出：`BCEWithLogitsLoss`（二分类）

### 关键超参数

- `max_len=400`：保留更多上下文信息
- `batch_size=32`
- `lr=8e-4`
- `weight_decay=1e-4`
- `epochs=18`
- `patience=4`：验证集早停
- `min_freq=3`：过滤低频噪声词
- `hidden_dim=160`
- `dropout=0.4`

### 优化方法

- 使用 `AdamW` 优化器：
  - 对 NLP 小型分类任务收敛稳定
  - 配合 `weight_decay` 有助于减轻过拟合
- 训练时使用梯度裁剪（`max_norm=5.0`）提升训练稳定性。

## 调参与效果

在本地 `20news-bydate` 数据集上，对比“基线配置”和“调参后配置”（随机种子固定为 `42`）。

### 基线配置

- `min_freq=2, max_len=300`
- `hidden_dim=128, dropout=0.3`
- `batch_size=64, lr=1e-3`
- `epochs=15, patience=3`

基线结果：

- 验证集最佳准确率：`0.7500`
- 测试集准确率：`0.6862`

### 调参思路

1. 增加文本长度上限（`300 -> 400`），减少信息截断。
2. 提升词频阈值（`2 -> 3`），降低低频噪声。
3. 适度增强模型容量（`hidden_dim 128 -> 160`）。
4. 增强正则（`dropout 0.3 -> 0.4`）以缓解过拟合。
5. 调整优化节奏（`batch_size 64 -> 32`，`lr 1e-3 -> 8e-4`）。
6. 放宽早停（`patience 3 -> 4`），给模型更多收敛时间。

### 调参后配置（当前默认）

- `min_freq=3, max_len=400`
- `embed_dim=128, hidden_dim=160, dropout=0.4`
- `batch_size=32, lr=8e-4, weight_decay=1e-4`
- `epochs=18, patience=4`

调参后结果：

- 验证集最佳准确率：`0.8009`
- 测试集准确率：`0.7252`

最终测试准确率超过 `0.7`，满足项目要求。

## 为什么该设置通常能在测试集达到较优效果

这个二分类任务语义差异明显，1层 GRU 在以下配置下通常可以获得较好表现：

- 通过验证集调参（而不是直接看测试集）
- 使用早停防止过拟合
- 使用 `AdamW + dropout + weight_decay`

在当前本地数据和固定随机种子下，测试准确率从 `0.6862` 提升至 `0.7252`。

## 运行方式

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd gru
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行主程序

```bash
python 20_news_data.py
```

程序会输出：

- 词汇表大小
- 训练/验证/测试样本数量
- 每个 epoch 的训练与验证指标
- 最终测试集准确率
