# 第3章样本构造与实验协议（冻结版）

## 1. 协议目标

本协议用于冻结第3章的数据集构造、模型比较和手动调参口径。协议一旦确认，第3章和第4章均以此为准，不再回退到旧版“FPT后切窗 + 直接RUL标签 + 动态特征筛选”的方案。

## 2. 冻结输入协议

第3章统一采用以下输入定义：

1. 协议名：`buaa_paper`
2. 振动通道：仅使用水平振动信号
3. 候选特征：`13` 个时域统计特征
4. 固定建模特征：剔除 `mean_abs` 后的 `12` 个累积时域特征
5. 特征缩放：仅在训练集上执行 `minmax` 归一化

## 3. 标签与切窗

第3章冻结采用寿命比值 `P` 作为训练标签：

$$
P_t = \frac{actRUL_t}{actRUL_0}
$$

样本构造规则如下：

1. 使用全寿命序列切窗，而不是从 FPT 之后再切窗
2. `window_size = 10`
3. `step_size = 1`
4. 每个窗口监督目标取窗口末端时刻的寿命比值 `P`
5. Transformer 的 decoder 输入为右移后的标签历史，起始值固定为 `1.0`

## 4. 数据划分

数据划分必须按轴承进行，固定为：

1. `35Hz12kN`
   - train: `Bearing1_1`, `Bearing1_2`, `Bearing1_3`
   - val: `Bearing1_4`
   - test: `Bearing1_5`
2. `37.5Hz11kN`
   - train: `Bearing2_1`, `Bearing2_2`, `Bearing2_3`
   - val: `Bearing2_4`
   - test: `Bearing2_5`
3. `40Hz10kN`
   - train: `Bearing3_1`, `Bearing3_2`, `Bearing3_3`
   - val: `Bearing3_4`
   - test: `Bearing3_5`

## 5. 模型比较协议

正式比较只保留三种模型：

1. `paper_transformer`
2. `lstm`
3. `gru`

统一部分包括：

1. 输入特征矩阵
2. 数据划分
3. 窗口大小与步长
4. 优化器类型
5. batch size
6. early stopping 规则
7. 随机种子集合

## 6. 手动调参与自动调参边界

### 6.1 手动调参

手动调参只针对 Transformer，且选择依据只看验证集，不看测试集。执行顺序固定为：

1. 先在 `35Hz12kN`、`seed=13` 上做有边界的 pilot 搜索
2. 取 pilot 前 `2` 个配置，推广到三工况 × 三随机种子验证
3. 按平均验证集 RMSE、验证集 RMSE 标准差和参数量三条规则选择唯一全局配置

### 6.2 Optuna

Optuna 只用于 Transformer，不用于 LSTM 与 GRU，并遵守：

1. 搜索时只用训练集和验证集
2. 测试集只在最优参数冻结后评估一次
3. 不允许看到测试结果后再回改搜索空间

## 7. 评价指标

第3章和第4章统一报告：

1. `MAE`
2. `RMSE`
3. `MAPE`
4. `score s`

其中 `MAPE` 分母下限取 `1`，`score s` 与当前代码中的 `phm_score` 对应。
