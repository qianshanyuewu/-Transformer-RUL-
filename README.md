# Bearing-RUL-Transformer

基于多特征融合与编码器-解码器 Transformer 的滚动轴承剩余使用寿命预测（本科毕业设计）

## 概述

以 XJTU-SY 滚动轴承加速寿命试验数据集为基础，构建了从原始振动信号到 RUL 预测的完整流水线：

```
原始振动信号 → db4小波去噪 → 19维候选特征 → 筛选12维累积时域特征
→ 融合健康指标(FPT检测) → Transformer / LSTM / GRU 预测 → Optuna超参数优化
```

**主要结果：** Optuna 调参后，Transformer 在全部三种工况上均取得最优 RMSE，三工况平均测试集 RMSE 从 19.00 降至 9.25，逐工况较最优基线降低 6.9%~83.7%。

## 项目结构

```
├── data_processing/       # 信号去噪、特征提取、健康指标构建
│   ├── features.py        # 小波去噪 + 19维特征提取
│   ├── selection.py        # 单调性-相关性评分 + 冗余分析
│   ├── health.py           # 累积变换 + 融合健康指标 + FPT检测
│   └── chapter2_pipeline.py
├── modeling/              # 深度学习模型
│   ├── transformer.py     # 编码器-解码器 Transformer
│   ├── lstm.py            # LSTM 基线
│   ├── gru.py             # GRU 基线
│   ├── metrics.py         # RMSE / MAE / MAPE / PHM Score
│   └── trainer.py         # 统一训练器 (AdamW + Early Stopping)
├── scripts/
│   ├── experiments/       # 各章实验入口脚本
│   └── figures/           # 图表生成脚本
├── docs/                  # 论文各章 Markdown 源文件
├── config.py              # 全局配置 (数据路径、工况参数)
└── protocol.py            # 冻结协议 (特征列表、划分方案)
```

## 实验协议

| 项目 | 配置 |
|------|------|
| 数据集 | XJTU-SY (3工况 × 5轴承 = 15个全寿命样本) |
| 输入通道 | 水平振动信号 |
| 去噪 | db4 小波, level=1, 软阈值 |
| 建模特征 | 12维累积时域特征 (从19维筛选) |
| 标签 | 寿命比值 P = L_t / L |
| 滑动窗口 | size=10, step=1 |
| 优化器 | AdamW, 早停(patience=5) |
| 评估 | 3种子重复 (13, 42, 3407), 平均结果 |
| 调参 | Optuna TPE, 30 trials |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 XJTU-SY 数据集并解压到 extracted/ 目录
#    数据集来源: https://biaowang.tech/xjtu-sy-bearing-datasets/

# 3. 运行第2章: 特征构建与健康指标
python3 scripts/experiments/run_chapter2.py

# 4. 运行第3章: 基线对比实验
python3 scripts/experiments/run_chapter3_baseline.py

# 5. 运行第4章: Optuna 超参数优化
python3 scripts/experiments/run_chapter4_optuna.py
```

## 主要结论

1. **特征构建：** 12维累积时域特征 + 融合健康指标可稳定检测退化起始点 (15个轴承 FPT 平均索引 14.93)
2. **数据敏感性：** 固定配置下 Transformer 在大样本工况显著优于 LSTM/GRU，小样本工况则相反
3. **超参数优化：** Optuna 发现最优配置为单层结构 + 4头注意力 + 低 dropout，调参后全工况 RMSE 均最优

## 许可证

[MIT License](LICENSE)
