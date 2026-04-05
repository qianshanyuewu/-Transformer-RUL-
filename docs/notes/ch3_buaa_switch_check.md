# 第3章切换为北航论文式的检查记录

## 1. 检查目标

确认 `thesis_rebuild` 的第3章默认口径是否已经从“FPT 后直接 RUL 回归”切换为“北航论文式的全寿命寿命比值 P 建模”，并且与第2章固定 `12` 特征协议保持一致。

## 2. 代码检查

已完成切换的文件：

1. 配置：[config.py](config.py)
2. 数据集构造：[dataset_builder.py](modeling/dataset_builder.py)
3. 数据加载：[data.py](modeling/data.py)
4. 模型定义：[models.py](modeling/models.py)
5. 训练逻辑：[trainer.py](modeling/trainer.py)
6. 实验入口：[experiment.py](modeling/experiment.py)

## 3. 数据构造检查

重跑入口：

[run_chapter3_dataset.py](run_chapter3_dataset.py)

产物：

[chapter3_dataset_manifest.json](results/chapter3_datasets/chapter3_dataset_manifest.json)

对 `35Hz12kN/train.npz` 的检查结果：

1. `protocol = buaa_paper`
2. `use_fpt_start = false`
3. `target_mode = life_ratio`
4. `feature_scaler_mode = minmax`
5. `X.shape = (415, 10, 12)`
6. `decoder_input.shape = (415, 10, 1)`
7. `start_indices` 从 `0` 开始，说明不再从 FPT 起窗

说明当前样本已经符合“全寿命 + 10 步窗口 + P 标签 + 右移 decoder 输入”的主线。

## 4. 模型检查

当前默认 Transformer 已切换为编码器-解码器结构。第3章正式手动调参后冻结的配置为：

1. `num_heads = 2`
2. `num_layers = 2`
3. `ffn_dim = 96`
4. `d_model = 48`
5. `dropout = 0.2`

对应实现：

[models.py](modeling/models.py)

## 5. 训练冒烟检查

试运行入口：

[run_chapter3_baseline.py](run_chapter3_baseline.py)

正式结果目录：

[fixed12_final_v1_e20_p5_s3](results/chapter3_baseline/fixed12_final_v1_e20_p5_s3)

当前说明：

1. `paper_transformer`
2. `lstm`
3. `gru`

三者已经可以在新的数据协议上正常训练、验证、测试并输出正式汇总指标。

## 6. 当前状态

1. `score s` 已纳入正式基线汇总。
2. 第3章数据集、模型训练和正式 baseline 已全部切换到 `P` 标签主线。
3. 当前仍在执行第4章 Optuna 自动调参，用于检验 Transformer 是否还能在相同协议下进一步提升。
