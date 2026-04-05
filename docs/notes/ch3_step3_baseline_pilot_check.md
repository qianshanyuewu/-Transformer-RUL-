# 第3章步骤3：基线实验脚本能力检查

## 1. 检查目标

验证正式基线实验脚本是否满足以下要求：

1. 主实验只包含 `paper_transformer`、`LSTM`、`GRU`
2. 能够在统一协议下复现编码器-解码器 Transformer 的正式训练口径
3. 能够按工况、模型、随机种子循环运行
4. 能够输出单次结果、汇总结果和实验清单文件
5. 能够读取手动搜索选出的 Transformer 配置并用于正式 baseline

## 2. 当前采用的模型口径

本次基线实验脚本已经冻结为：

1. `paper_transformer`: 编码器-解码器 Transformer，编码器接收特征窗口，解码器接收右移标签历史
2. `LSTM`: 以窗口序列最后时刻隐藏状态回归 RUL
3. `GRU`: 以窗口序列最后时刻隐藏状态回归 RUL

## 3. 新增代码

### 3.1 基线实验工具

- `modeling/experiment.py`

功能：

1. 统一单次实验运行逻辑
2. 统计参数量
3. 汇总多 seed 结果
4. 保存 `baseline_runs.json` 与 `baseline_summary.json`

### 3.2 正式基线运行脚本

- `run_chapter3_baseline.py`

功能：

1. 支持选择工况
2. 支持选择模型
3. 支持选择随机种子
4. 支持覆盖 epoch、patience、batch size 等参数
5. 支持读取 `--paper-transformer-search-report`
6. 自动生成实验清单 `baseline_manifest.json`

## 4. 当前验证结果

当前脚本已经完成以下两类实际验证：

1. 手动搜索阶段的 pilot 和 formal validation
2. 第3章正式 baseline：`fixed12_final_v1_e20_p5_s3`

其中正式 baseline 输出目录为：

- `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3`

主要输出文件：

1. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_runs.json`
2. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_summary.json`
3. `results/chapter3_baseline/fixed12_final_v1_e20_p5_s3/baseline_manifest.json`

## 5. 客观结论

当前可以确认：

1. 正式基线实验脚本已经可运行。
2. 三个模型都能在同一数据协议下完成训练、验证、测试与结果汇总。
3. `paper_transformer` 已经能够通过手动搜索报告自动注入正式配置，无需手工抄参。
4. 结果文件结构已经稳定，后续可以直接服务于第3章和第4章正文回填。

## 6. 当前保留意见

### 6.1 当前风险不在脚本链路，而在模型表现

脚本链路已经验证通过，当前真正需要解决的问题是 Transformer 能否在测试集上建立稳定优势，而不是“脚本能不能跑”。

## 7. 当前结论

第3章步骤3已经从“脚本级通过”进入“正式结果已生成”阶段，下一步应继续进入第4章 Optuna 自动调参。
