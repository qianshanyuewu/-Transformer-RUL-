# 第3章步骤3：正式基线实验检查

## 1. 实验配置

本次正式基线实验采用以下统一配置：

1. 工况：`35Hz12kN`、`37.5Hz11kN`、`40Hz10kN`
2. 模型：`paper_transformer`、`LSTM`、`GRU`
3. 随机种子：`13`、`42`、`3407`
4. `max_epochs = 20`
5. `patience = 5`
6. Transformer 配置来自验证集手动搜索结果：`d_model=48`、`num_heads=2`、`num_layers=2`、`ffn_dim=96`、`dropout=0.2`

输出目录：

- `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3`

结果文件：

1. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_runs.json`
2. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_summary.json`
3. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_manifest.json`

## 2. 正式基线结果

### 2.1 35Hz12kN

Transformer:

- 参数量：`96961`
- val RMSE：`5.185315 ± 1.492475`
- test RMSE：`19.047634 ± 0.898121`

LSTM:

- 参数量：`55361`
- val RMSE：`13.118185 ± 1.055510`
- test RMSE：`15.282075 ± 1.917203`

GRU:

- 参数量：`42049`
- val RMSE：`13.645636 ± 0.565562`
- test RMSE：`15.500526 ± 0.619255`

### 2.2 37.5Hz11kN

Transformer:

- 参数量：`96961`
- val RMSE：`9.482406 ± 0.465196`
- test RMSE：`94.760698 ± 5.971245`

LSTM:

- 参数量：`55361`
- val RMSE：`10.558235 ± 1.381663`
- test RMSE：`50.405171 ± 10.298663`

GRU:

- 参数量：`42049`
- val RMSE：`9.927191 ± 0.265635`
- test RMSE：`103.359974 ± 8.853458`

### 2.3 40Hz10kN

Transformer:

- 参数量：`96961`
- val RMSE：`213.234045 ± 45.843214`
- test RMSE：`62.633451 ± 6.114413`

LSTM:

- 参数量：`55361`
- val RMSE：`281.147123 ± 2.696364`
- test RMSE：`34.885595 ± 3.323335`

GRU:

- 参数量：`42049`
- val RMSE：`276.586757 ± 10.867360`
- test RMSE：`54.240165 ± 7.066120`

## 3. 客观观察

### 观察1：Transformer 的验证集优势并未稳定转化为测试集优势

在 `35Hz12kN` 和 `40Hz10kN` 工况下，Transformer 取得了最低的验证集 RMSE，但对应测试集 RMSE 仍高于 LSTM。这说明当前 Transformer 具备较强的拟合能力，但泛化优势尚未真正建立。

### 观察2：LSTM 仍然是当前测试集上最稳的模型

在三工况正式测试结果中：

1. `35Hz12kN` 上 LSTM 最优，`test RMSE = 15.282075`
2. `37.5Hz11kN` 上 LSTM 最优，`test RMSE = 50.405171`
3. `40Hz10kN` 上 LSTM 仍最优，`test RMSE = 34.885595`

因此，第3章不能把 Transformer 写成整体最优模型。

### 观察3：Transformer 的优势目前主要体现在验证集拟合和部分工况比较上

平均 best epoch：

1. `35Hz12kN`
   - Transformer：`16.00`
   - LSTM：`5.00`
   - GRU：`7.33`
2. `37.5Hz11kN`
   - Transformer：`1.00`
   - LSTM：`8.00`
   - GRU：`1.00`
3. `40Hz10kN`
   - Transformer：`1.00`
   - LSTM：`5.67`
   - GRU：`3.00`

Transformer 在 `37.5Hz11kN` 上优于 GRU，在 `35Hz12kN` 和 `40Hz10kN` 上则表现出更强的验证集拟合能力。因此，它并非完全失效，而是尚未形成能够覆盖三工况的稳定测试集优势。

## 4. 当前结论

### 4.1 可以确认的结论

1. 第3章正式基线实验链路已经完全跑通。
2. `paper_transformer` 已经能够在统一协议下与 LSTM、GRU 完成公平比较。
3. 当前验证集驱动的手动调参可以提升 Transformer 的拟合能力，但尚未使其在测试集上整体超过 LSTM。

### 4.2 不能直接下的结论

当前还不能直接写：

1. “Transformer 在三工况下全面优于 LSTM 和 GRU”
2. “Transformer 已经在测试集上建立稳定优势”
3. “第3章当前正式结果已经足以单独支撑全文核心创新点”

## 5. 下一步建议

下一步更合理的动作是继续执行第4章 Optuna 自动调参，在不改变第3章协议的前提下检验 Transformer 是否还有超参数层面的提升空间。

## 6. 当前结论

正式基线实验已经完成，当前适合进入第4章 Optuna 自动调参阶段。
