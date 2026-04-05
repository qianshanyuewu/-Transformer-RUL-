# 摘 要

滚动轴承是旋转机械中的关键部件，其剩余使用寿命（Remaining Useful Life, RUL）预测对设备安全运行至关重要。针对早期退化信号微弱、多维特征冗余及传统序列模型长距离依赖建模不足等问题，本文以 XJTU-SY 加速寿命试验数据集为基础，提出了一种基于多特征融合与编码器-解码器 Transformer 的轴承 RUL 预测方法。

首先，对水平振动信号进行 db4 小波去噪，提取 19 个候选统计特征（含 13 个时域特征和 6 个频域特征），经初步筛选剔除趋势表征能力较弱的频域特征后，通过 Savitzky-Golay 平滑、单调性-相关性评分排序及冗余分析，最终确定 12 个关键退化特征，并经累积变换增强趋势性；在此基础上构建多特征融合健康指标以验证特征体系的退化表征能力。

其次，建立编码器-解码器 Transformer 预测模型，以 LSTM 和 GRU 作为对比基线，在三种工况下开展系统对比实验。实验结果表明，在固定超参数配置下，Transformer 在数据充裕工况下表现出显著优势，但在小样本工况下不及循环神经网络。

进一步通过 Optuna 贝叶斯超参数优化，Transformer 在全部三种工况上均取得最优 RMSE 表现，三工况平均测试集 RMSE 从 19.00 降至 9.25，逐工况较最优基线模型降低 6.9%~83.7%，表明编码器-解码器 Transformer 在滚动轴承 RUL 预测任务中具备良好的适用性。

**关键词：** 滚动轴承；剩余使用寿命预测；多特征融合；健康指标构建；编码器-解码器 Transformer；XJTU-SY 数据集

# ABSTRACT

Rolling bearings are critical components in rotating machinery, and their remaining useful life (RUL) prediction is essential for ensuring equipment safety. To address challenges including weak early degradation signals, multi-dimensional feature redundancy, and insufficient long-range dependency modeling in traditional sequential models, this thesis proposes a bearing RUL prediction method based on multi-feature fusion and an Encoder-Decoder Transformer, using the XJTU-SY accelerated life dataset.

First, horizontal vibration signals are denoised via db4 wavelet transform, and 19 candidate statistical features (13 time-domain and 6 frequency-domain) are extracted. After preliminary screening to remove frequency-domain features with unstable trend representation, followed by Savitzky-Golay smoothing, monotonicity-correlation scoring, and redundancy analysis, 12 key degradation features are selected and enhanced through cumulative transformation; a multi-feature fused health indicator is then constructed to validate the degradation characterization capability of the feature system. Second, an Encoder-Decoder Transformer model is established for RUL prediction, with LSTM and GRU serving as comparison baselines across three operating conditions. Experimental results show that under fixed hyperparameter configurations, the Transformer exhibits significant advantages in data-rich conditions but underperforms recurrent networks in small-sample scenarios. Through further Optuna Bayesian hyperparameter optimization, the Transformer achieves the best RMSE performance across all three conditions, reducing the average test RMSE from 19.00 to 9.25, outperforming the best baseline model per condition by 6.9%--83.7%, demonstrating the applicability of the Encoder-Decoder Transformer for bearing RUL prediction.

**Keywords:** rolling bearing; remaining useful life prediction; multi-feature fusion; health indicator construction; Encoder-Decoder Transformer; XJTU-SY dataset

# 第1章 绪论

## 1.1 课题来源

本课题来源于机械设备状态监测与智能运维领域的研究需求，研究对象为滚动轴承全寿命退化数据，目标是面向轴承运行过程中的性能衰退问题，建立一套能够实现健康状态识别与剩余使用寿命预测的分析方法。课题以 XJTU-SY 轴承数据集为实验基础，结合信号处理、特征工程与深度学习方法，对滚动轴承退化特征提取、健康指标构建以及剩余使用寿命预测模型进行研究，并在此基础上完成模型训练、对比实验和结果分析。

## 1.2 研究背景与意义

滚动轴承是旋转机械系统中的关键基础部件，广泛应用于电机、齿轮箱、机床主轴和风电设备等工业场景。轴承一旦发生退化或失效，不仅会降低设备运行效率，还可能引发连锁故障，造成停机损失甚至安全事故。因此，如何在轴承发生严重损伤前及时识别其健康状态，并对其剩余使用寿命进行准确预测，已成为状态监测与预测性维护领域的重要研究内容[1]。

传统定期检修方式存在被动性强、维护成本高和资源利用率低等问题，难以满足现代工业对高可靠性与低停机成本的要求。与之相比，基于运行数据的剩余使用寿命预测方法能够根据设备实际退化状态动态制定维护策略，具有更强的针对性和经济性。对于滚动轴承这类退化过程复杂、工况变化明显的部件而言，构建稳定有效的退化表征方式，并在此基础上建立预测模型，具有重要的工程应用价值和现实意义。

## 1.3 国内外研究现状

为解决上述轴承退化表征与寿命预测问题，国内外学者从退化机理建模、性能退化指标构建以及数据驱动预测模型三个层面开展了大量研究。张金豹等[1]的综述指出，滚动轴承运行环境复杂、噪声干扰明显、工况变化频繁，同时还存在样本不平衡和标签不足等问题，这些因素共同增加了 RUL 预测的难度，也推动了研究方法从物理机理建模向数据驱动方向的持续演进。

在研究方法层面，早期工作多采用基于物理机理或统计退化模型的方法，通过建立磨损演化、疲劳损伤或可靠性分布模型来估计剩余使用寿命。这类方法具备一定可解释性，但通常依赖较强的先验假设，对模型参数和工况一致性要求较高，因此在复杂工况下的适应性受到限制。随着预测与健康管理研究的发展，公开数据集和全寿命试验平台逐步完善，尤其是 XJTU-SY 滚动轴承加速寿命试验数据集[2]的发布，为滚动轴承 RUL 预测研究提供了高质量实验基础。

在数据驱动方法中，特征提取与健康指标构建是实现准确预测的关键。现有研究通常从振动信号中提取时域、频域或时频域特征，再通过特征降维、融合或健康指标构造来表征退化过程[1]。近期研究进一步强调，健康指标不仅应能够反映退化趋势，还应具备识别退化起始阶段的能力。Zhao 等[3]提出的健康指标构建方法兼顾了退化阶段识别与 RUL 预测，为健康状态识别与寿命预测的一体化研究提供了新的思路。

在预测模型方面，循环神经网络及其改进模型长期是滚动轴承 RUL 预测的重要手段。Hochreiter 等[9]提出的长短期记忆网络（LSTM）通过门控机制有效缓解了梯度消失问题，成为时间序列建模的基础架构之一；Cho 等[10]提出的门控循环单元（GRU）进一步简化了门控结构，在参数量更少的情况下保持了相近的建模能力。在轴承 RUL 预测领域，Ren 等[4]提出的多尺度密集门控循环单元网络实现了端到端的特征学习与寿命预测，说明基于 GRU 的深度时序模型在轴承 RUL 任务中具有较强应用价值。

针对不同故障模式和工况差异，多工况深度学习方法也逐渐受到关注。Zhao 等[5]基于 XJTU-SY 数据集提出了分类与回归结合的多工况 RUL 预测方案，在提升预测精度和适应复杂工况方面取得了较好效果。周欢等[6]基于 PHM2012 数据集对 CNN、LSTM 与 CNN+LSTM 模型进行了对比研究，结果表明 CNN+LSTM 相较单一 LSTM 平均 MAE 进一步降低 15.7%，说明局部特征提取与时序建模融合能够提升轴承 RUL 预测效果。

近年来，基于注意力机制的模型为滚动轴承 RUL 预测提供了新的研究方向。Transformer 结构最早由 Vaswani 等[7]提出，其核心优势在于利用自注意力机制建模全局依赖关系，并具备较好的并行计算能力。在此基础上，周哲韬等[8]将编码器-解码器 Transformer 引入滚动轴承 RUL 预测任务，并结合小波去噪、时域特征和累积变换验证了该方法在轴承寿命预测中的有效性。上述结果表明，相较于传统循环结构，Transformer 在复杂退化序列的全局建模方面具备较大潜力。

总体来看，现有研究已经在特征构建、健康指标设计和深度学习预测模型方面取得了较多成果，但仍存在若干不足：一是 Transformer 在滚动轴承 RUL 预测中的应用研究尚不充分，尤其缺乏与 LSTM、GRU 等循环神经网络在相同实验协议下的系统对比；二是 Transformer 模型的性能对超参数配置较为敏感，现有研究多采用固定配置，缺乏对超参数优化策略的系统探索；三是在训练数据有限的工况下，Transformer 面临过拟合风险，如何通过模型结构和超参数调整来缓解这一问题值得进一步研究。

## 1.4 本文主要研究内容

针对上述问题，本文围绕滚动轴承剩余使用寿命预测任务，以 XJTU-SY 加速寿命试验数据集为实验基础开展研究，主要工作如下：

第一，基于水平振动信号建立退化特征构建流程。对原始信号进行 db4 小波去噪，提取 19 个候选统计特征（含 13 个时域特征和 6 个频域特征）。经初步筛选剔除趋势表达不稳定的频域特征，再通过 Savitzky-Golay 平滑、单调性-相关性联合评分分析及冗余检验，最终确定 12 个关键退化特征，并经累积变换增强退化趋势。在此基础上，构建多特征融合健康指标，验证所构建特征体系的退化表征能力。

第二，建立基于编码器-解码器 Transformer 的剩余使用寿命预测模型，以寿命比值作为训练标签，并以 LSTM 和 GRU 作为对比基线，在三种工况下开展系统对比实验。

第三，引入 Optuna 贝叶斯超参数优化对 Transformer 进行系统搜索，验证其在不同数据规模工况下的预测潜力。

## 1.5 论文结构安排

本文共分为五章，各章内容安排如下：

第1章为绪论，介绍研究背景、意义及国内外研究现状，并提出本文的研究内容与章节安排。

第2章围绕健康状态识别，阐述信号预处理、特征筛选、累积变换与多特征融合健康指标的构建流程。

第3章建立编码器-解码器 Transformer 预测模型，并以 LSTM 和 GRU 作为基线，在三种工况下开展系统对比实验。

第4章引入 Optuna 自动超参数优化，对 Transformer 进行系统搜索并分析最优配置特征。

第5章总结全文主要结论，讨论研究局限并展望后续研究方向。

## 参考文献

[1] 张金豹, 邹天刚, 王敏, 等. 滚动轴承剩余使用寿命预测综述[J]. 机械科学与技术, 2023, 42(1): 1-23. DOI:10.13433/j.cnki.1003-8728.20200489.

[2] 雷亚国, 韩天宇, 王彪, 等. XJTU-SY滚动轴承加速寿命试验数据集解读[J]. 机械工程学报, 2019, 55(16): 1-6. DOI:10.3901/JME.2019.16.001.

[3] ZHAO Y S, LI P, KANG Y, et al. A health indicator enabling both first predicting time detection and remaining useful life prediction: Application to rotating machinery[J]. Measurement, 2024, 235: 114994.

[4] REN L, CHENG X, WANG X, et al. Multi-scale Dense Gate Recurrent Unit Networks for bearing remaining useful life prediction[J]. Future Generation Computer Systems, 2019, 94: 601-609.

[5] ZHAO B, YUAN Q. A novel deep learning scheme for multi-condition remaining useful life prediction of rolling element bearings[J]. Journal of Manufacturing Systems, 2021, 61: 450-460.

[6] 周欢, 陈彩霞. 基于深度学习的轴承使用寿命的预测研究[J]. 应用数学进展, 2025, 14(9): 165-177. DOI:10.12677/aam.2025.149410.

[7] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[J/OL]. arXiv preprint arXiv:1706.03762, 2017.

[8] 周哲韬, 刘路, 宋晓, 等. 基于Transformer模型的滚动轴承剩余使用寿命预测方法[J]. 北京航空航天大学学报, 2023, 49(2): 430-443. DOI:10.13700/j.bh.1001-5965.2021.0247.

[9] HOCHREITER S, SCHMIDHUBER J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.

[10] CHO K, VAN MERRIENBOER B, GULCEHRE C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[C]//Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha: Association for Computational Linguistics, 2014: 1724-1734.

[11] LOSHCHILOV I, HUTTER F. Decoupled weight decay regularization[C]//Proceedings of the 7th International Conference on Learning Representations (ICLR). New Orleans, 2019.

[12] AKIBA T, SANO S, YANASE T, et al. Optuna: A next-generation hyperparameter optimization framework[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019: 2623-2631.

[13] BERGSTRA J, BARDENET R, BENGIO Y, et al. Algorithms for hyper-parameter optimization[C]//Advances in Neural Information Processing Systems 24 (NIPS 2011). Granada, 2011: 2546-2554.
