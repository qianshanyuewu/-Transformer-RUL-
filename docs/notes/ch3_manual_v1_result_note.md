# 第3章人工敏感调参阶段结果说明（正式版）

## 1. 当前正式实验目录

[fixed12_manual_search_v1](results/chapter3_search/fixed12_manual_search_v1)

正式基线目录：

[fixed12_final_v1_e20_p5_s3](results/chapter3_baseline/fixed12_final_v1_e20_p5_s3)

主汇总文件：

[baseline_summary.json](results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_summary.json)

## 2. 当前实验口径

1. 水平振动单通道
2. 固定 12 个累积时域统计特征
3. 寿命比值 `P` 训练标签
4. 编码器-解码器 Transformer
5. Transformer 手动调参后配置：`d_model=48`、`num_heads=2`、`num_layers=2`、`ffn_dim=96`、`dropout=0.2`
6. Transformer 优化器参数：`learning_rate=1e-3`、`weight_decay=1e-4`
7. 多随机种子重复：`13`、`42`、`3407`

## 3. 当前正式结果

### 35Hz12kN

1. Transformer：`test RMSE = 19.047634`
2. LSTM：`test RMSE = 15.282075`
3. GRU：`test RMSE = 15.500526`

### 37.5Hz11kN

1. Transformer：`test RMSE = 94.760698`
2. LSTM：`test RMSE = 50.405171`
3. GRU：`test RMSE = 103.359974`

### 40Hz10kN

1. Transformer：`test RMSE = 62.633451`
2. LSTM：`test RMSE = 34.885595`
3. GRU：`test RMSE = 54.240165`

## 4. 当前结论

1. 按验证集指标冻结的 Transformer 配置并未在正式测试集中整体压过 LSTM。
2. Transformer 在 `37.5Hz11kN` 工况下优于 GRU，但仍明显弱于 LSTM。
3. Transformer 在 `35Hz12kN` 和 `40Hz10kN` 工况下均未优于 LSTM。
4. 因此，第3章当前最多只能表述为“Transformer 在部分验证场景中表现出优势，但测试集整体尚未超过 LSTM”。

## 5. 后续动作

当前已转入第4章 Optuna 自动调参阶段。若 Optuna 找到更优配置，将以新配置补充 chapter4 专用结果，而不是篡改第3章正式 baseline 结论。
