# 第3章步骤2：统一训练框架检查

## 1. 检查目标

验证第3章统一训练框架是否已经具备以下能力：

1. 读取第3章样本构造阶段生成的 `.npz` 数据
2. 以统一接口接入 Transformer、LSTM、GRU 三模型
3. 使用统一的损失函数、优化器和训练循环
4. 输出 MAE、RMSE、MAPE 指标
5. 至少在一个工况上完成最小可运行冒烟测试

## 2. 新增代码

### 2.1 数据加载

- `modeling/data.py`

功能：

1. 读取 `train.npz / val.npz / test.npz`
2. 封装为 PyTorch `Dataset`
3. 构造 `DataLoader`

### 2.2 指标计算

- `modeling/metrics.py`

当前统一指标：

1. `MAE`
2. `RMSE`
3. `MAPE`

其中 `MAPE` 使用分母下限保护，与第3章协议保持一致。

### 2.3 模型定义

- `modeling/models.py`

当前提供：

1. `TransformerRegressor`
2. `LSTMRegressor`
3. `GRURegressor`
4. `build_model()` 统一构造接口

### 2.4 训练与评估

- `modeling/trainer.py`

当前支持：

1. 统一随机种子设置
2. 自动选择设备
3. 统一训练循环
4. 基于验证集 RMSE 的 early stopping
5. 输出训练历史和验证/测试指标

### 2.5 冒烟脚本

- `run_chapter3_smoke.py`

用途：

1. 只在 `35Hz12kN` 上运行
2. 每个模型只训练极少 epoch
3. 目标仅为验证框架打通，不用于正式实验结论

## 3. 冒烟测试结果

本次最小冒烟测试采用：

1. 工况：`35Hz12kN`
2. 模型：Transformer、LSTM、GRU
3. epoch：`2`
4. 目的：验证训练循环和指标通路，而不是追求最终精度

结果如下：

### Transformer

- best_epoch: `2`
- val:
  - MAE = `45.937579`
  - RMSE = `53.846848`
  - MAPE = `91.515279`
- test:
  - MAE = `11.178046`
  - RMSE = `13.501422`
  - MAPE = `83.280903`

### LSTM

- best_epoch: `2`
- val:
  - MAE = `46.082966`
  - RMSE = `53.921098`
  - MAPE = `91.736287`
- test:
  - MAE = `11.206477`
  - RMSE = `13.532790`
  - MAPE = `83.189748`

### GRU

- best_epoch: `2`
- val:
  - MAE = `46.734884`
  - RMSE = `54.450796`
  - MAPE = `93.233051`
- test:
  - MAE = `11.695940`
  - RMSE = `14.041361`
  - MAPE = `83.195884`

## 4. 客观结论

当前可以确认：

1. 第3章样本能够被三模型共用的数据接口正确读取。
2. Transformer、LSTM、GRU 都可以完成前向传播、反向传播与验证评估。
3. 训练循环、early stopping 与指标计算链路已经打通。
4. 当前输出结果只是“冒烟级别结果”，不能写入论文作为正式实验结论。

## 5. 当前保留意见

### 5.1 冒烟结果不代表正式优劣

当前仅训练了 `2` 个 epoch，因此：

1. 结果只能说明框架可运行
2. 不能说明 Transformer 已经显著优于 LSTM 或 GRU

### 5.2 仍未处理训练样本不平衡问题

当前训练框架仍采用原始窗口分布：

1. 没有做按轴承均衡采样
2. 没有做样本加权
3. 没有做目标值归一化

这些都属于后续正式实验时需要评估的策略，而不应在冒烟阶段提前引入太多变量。

## 6. 当前结论

第3章步骤2已经完成，当前可以进入下一步：

1. 设计正式基线实验脚本
2. 在三种工况上分别运行 Transformer、LSTM、GRU
3. 记录 MAE、RMSE、MAPE 结果
4. 在基线实验完成后再进入 Optuna 调参
