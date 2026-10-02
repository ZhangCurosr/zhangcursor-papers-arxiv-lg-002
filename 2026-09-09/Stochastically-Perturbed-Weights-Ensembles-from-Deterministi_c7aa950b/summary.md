---
title: "Stochastically-Perturbed-Weights-Ensembles-from-Deterministi"
source: https://arxiv.org/pdf/2609.08412v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-02 08:02:06"
field: "机器学习气象预报"
keywords: ["集合预报", "ML 天气模型", "权重扰动", "概率预报", "SPW", "WeatherBench-2"]
innovations: ["提出零训练成本的推理时权重扰动方法 SPW", "揭示权重噪声与初值扰动在不同预报时效下的主导分工", "提出粗尺度钩子和 Refresh 变体以缓解过分散域均值问题"]
benchmarks: ["WeatherBench-2", "ERA5", "IFS-ENS"]
---

# 论文速读：Stochastically-Perturbed-Weights-Ensembles-from-Deterministic-ML-Weather-Models

## 一句话总结
本文提出 **SPW（Stochastically Perturbed Weights）** 方法，通过在推理时对确定性 ML 天气模型（MLWM）的权重张量注入乘性高斯噪声，以零额外训练成本生成概率集合预报；在 240 h 预见期内，其 CRPSS 仅比最佳训练概率基线低 0.04～0.13，但边际训练成本为零。

## 研究问题与动机
- 确定性 ML 天气模型（如 Aurora、GraphCast、SFNO、AIFS）推理时只能产生单条预测，无法量化预测不确定性，而气象预报对概率预报（集合预报）有刚性需求。
- 现有概率 ML 气象模型（如 AIFS-ENS、FourCastNet 3、GenCast、Atlas-ERA5）均需重新训练或引入额外组件（CRPS 损失、扩散过程、SDE 积分），训练成本高。
- 物理集合预报中已有成熟先例：ECMWF 的 SPPT（Stochastically Perturbed Parametrised Tendencies）通过在每步随机扰动参数化趋势来生成集合不确定性，类比到 ML 模型是否可通过扰动权重达到类似目的？
- 核心开放问题：从已有确定性 checkpoint 能提取多少校准不确定性？噪声应注入哪个张量组、以何种尺度（σ）？在哪些情形下会失败（如相干整场偏移）？

## 核心贡献（创新点）
1. **提出 SPW 框架**：在推理时对确定性模型权重施加乘性高斯噪声（$\tilde{W}_m = W \odot (1 + \sigma \xi_m)$）以生成集合成员，零额外训练开销。与已有方法的本质区别在于不修改训练流程，直接复用已发布的确定性 checkpoint。
2. **系统性消融研究**：在三阶段消融（全权重扫描 → 单张量组 → 粗尺度钩子）下验证了四款不同架构（Aurora、GraphCast、SFNO、AIFS）的最优噪声注入位点，揭示了"无统一注入位点，因架构而异"的关键发现。
3. **提出 Refresh 变体**：每 N 步重新采样噪声，并将 σ 按 $\sqrt{T/N}$ 缩放，将相干空间均值偏移转为部分抵消的随机游走，有效缓解过分散域均值问题。
4. **IC-Augmented 混合策略**：在权重扰动基础上叠加 IFS-ENS 的 50 成员初值扰动分析，在热带气旋案例（Hurricane Milton）中展示协同效应。
5. **开源资源**：提供 github.com/MeteoSwiss/ai-models-ensembles（release v1.0.0），复现所有主干模型及概率配置，推动社区基准对齐。

## 方法详解
- **扰动公式**：$\tilde{\mathbf{W}}_m = \mathbf{W} \odot (1 + \sigma \boldsymbol{\xi}_m)$，其中 $\xi_{m,i} \overset{i.i.d.}{\sim} N(0,1)$，为乘性高斯噪声（非加性），各标量权重的方差为 $\sigma^2 |w|^2$，σ 为无量纲相对强度。
- **两自由度**：选择噪声注入的张量组 $S$（如 Encoder / Backbone / Decoder 或 g2m / m2m / m2g）和幅度 σ。
- **方差预算约束**：组内 $\sigma_{\text{group}} = \sigma_{\text{full}} \sqrt{N_{\text{total}} / N_{\text{group}}}$，保证 Phase 1/2 在同一噪声预算下可比。
- **三阶段消融**：
  - Phase 1：全权重扫描 $\sigma \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$
  - Phase 2：单架构张量组（Encoder / Backbone / Decoder 或 g2m / m2m / m2g）
  - Phase 3：仅在同步及更大尺度（粗尺度）注入噪声——Aurora enc_2（最低分辨率潜层，$\lambda \gtrsim 5000$ km）；GraphCast n_coarse_42（前 42 个 mesh 节点，$\gtrsim 3300$ km）；SFNO Lcut_10（最低 10 个球谐模态，$\gtrsim 4000$ km）。
- **Refresh 变体**：每 $N$ 步重新采样噪声，$\sigma_N = \sigma_{\text{frozen}} \sqrt{T/N}$，将相干空间均值偏移转为部分抵消的随机游走。
- **IC-Augmented 变体**：在权重扰动基础上叠加 IFS-ENS 的 50 成员扰动初值分析，用于压力测试（Hurricane Milton 案例）。

## 实验与结果
- **四个确定性骨干**：Aurora（3D Swin Transformer + Perceiver，1.3 B）、GraphCast（GNN，二十面体网格，37 M）、SFNO（球面傅里叶算子，573 M）、AIFS（graph enc./transformer/graph dec.，255 M）。
- **对比基线**：AIFS-ENS（229 M）、FourCastNet 3、Atlas-ERA5、ECMWF IFS-ENS（cycles 47r3/48r1）。
- **验证设置**：消融格（4 个中间季初值 × 10 成员，240 h 预见期）；生产格（112 个初值，M=10，最长 360 h）；评估指标含 CRPSS、RMSE、Spread-Skill Ratio (SSR)、Δmbr、Δmean。
- **核心结果**：在 240 h（10 天）预见期，SPW 集合的 CRPSS 比最佳训练概率基线低 0.04～0.13，边际训练成本为零。
- **关键表 D3（CRPSS 与 RMSE 变化）**：
  | Backbone | 24h CRPSS | 120h CRPSS | 240h CRPSS | 240h Δmean |
  |---|---|---|---|---|
  | aurora_encoder | 0.787 | 0.499 | 0.151 | -19.9% |
  | graphcast_all | 0.755 | 0.487 | 0.190 | -15.8% |
  | sfno_modes10 | 0.727 | 0.396 | 0.145 | -22.4% |
  | aifs_perturbed | 0.793 | 0.537 | 0.204 | -14.8% |
- **集合方差归因（表 D1，240 h）**：
  - Aurora：weight-only 93%，IC-only 76%，additivity 0.594
  - SFNO：weight-only 97%，IC-only 67%，additivity 0.610
  - AIFS：weight-only 92%，IC-only 57%，additivity 0.671
- **结论**：短期（24 h）IC 扰动主导（IC-only 占 90%+），权重噪声贡献有限；长期（240 h）权重噪声主导（weight-only 占 90%+），IC 影响衰减。
- **失败模式**：相干整场偏移（coherent whole-field offset）导致过分散域均值；限制噪声到粗尺度或扰动初值可修复。Aurora 与 IFS-ENS 在 refresh 后几乎无改善；SFNO 在 budget-rescaled $\sigma_N = 0.35$ 时表现最佳。
- **Hurricane Milton 案例**：所有基线重现了热带气旋暖心结构（500–850 hPa thickness anomaly），峰值振幅与 ERA5 分析相当或更高；Track error 从 0–24 h 的 <100 km 增长至 ~230 h 的 ~2000 km，aurora IC 与 graphcast all IC 保持最低均值误差。

## 相关工作脉络
1. **训练概率模型（TP）**：如 AIFS-ENS（CRPS 损失 + latent 高斯噪声）、FourCastNet 3（CRPS + 球面扩散）、GenCast（去噪扩散）、Atlas-ERA5（SDE 积分）。SPW 与其本质区别是不需重新训练，直接复用确定性 checkpoint。
2. **初值扰动**：ECMWF IFS-ENS 等传统方法通过扰动初始条件生成集合。SPW 与之互补：在长期预报中权重噪声贡献更大（240 h 占 90%+），而短期 IC 扰动占主导。
3. **已有权重扰动工作**：Scher & Messori (2021) 使用 dropout + 重训；Graubner et al. (2022) 提出 SWAG；Mahesh et al. (2024) 做 deep ensemble。SPW 的区别在于零重训、纯推理时扰动。
4. **Model Soups / Noisy Deep Ensemble**：Wortsman et al. (2022) 的 Model Soups 通过多模型权重平均提升精度但不增加推理时间；Sakai et al. (2025) 的 Noisy Deep Ensemble 通过噪声注入加速深度集成。SPW 与前者共享"不增推理成本"的理念，但与后者不同：SPW 在单个 checkpoint 上做推理时扰动，而非多模型集成。
5. **WeatherBench-2 基准**：本文在 WeatherBench-2 和 SwissClim Evaluations pipeline 上进行评估，并追加 Clausius-Clapeyron 和地转平衡检验，为社区提供可复现的概率基准。
6. **物理集合先例**：ECMWF 的 SPPT 通过在每步随机扰动参数化趋势生成集合，本文将其思想迁移到 ML 权重空间，是类物理启发的方法类比。

## 局限性与未来方向
- **无统一注入位点**：最优张量组因架构而异，SPW 目前是调参过程而非即插即用方案，需对每种模型进行消融搜索。
- **相干整场偏移**：全权重扰动易导致过分散域均值，需依赖粗尺度限制或 Refresh 变体缓解，但 Aurora 等模型在此类修正后改善有限。
- **短期预测中权重噪声贡献有限**：24 h 内 IC 扰动主导，SPW 的收益主要来自中长期，短期集合质量提升有限。
- **未来方向**：探索自适应 σ 调度（按预报时效或变量动态调整）、结合自动超参优化以消除调参依赖、将 SPW 推广至更多架构（如 WeatherGenerator 等新兴基础模型）。

## 研究启发与可借鉴点
1. **零成本概率化范式**：SPW 展示了"不重训即得集合预报"的可行路径，对资源受限团队极具吸引力；可迁移到其他确定性 ML 模型（如视觉、语音）的概率化扩展。
2. **粗尺度钩子设计**：将噪声限制在最低分辨率潜层或最低球谐模态，有效缓解了过分散域均值问题；这一思路可应用于其他空间序列模型的不确定性注入。
3. **Refresh 变体的随机游走思想**：通过定期重采样噪声并缩放幅度，将系统性偏差转为随机游走，这一机制可推广至长程迭代推理任务（如视频生成、多步规划）。
4. **IC-Augmented 混合策略**：权重扰动与初值扰动在不同时效下分别主导，两者叠加可协同提升整体集合质量；提示未来工作可按预报时效分层设计不确定性来源。
5. **开源可复现性**：github.com/MeteoSwiss/ai-models-ensembles 提供完整 release，为本团队后续基准对齐和扩展实验提供了良好起点。

## 关键术语表
- **SPW（Stochastically Perturbed Weights）**：在推理时对确定性模型权重施加随机噪声以生成概率集合的方法。
- **CRPSS（Continuous Ranked Probability Skill Score）**：衡量概率预报相对于参考预报技能提升的指标，正值表示优于参考。
- **乘性高斯噪声**：噪声幅度与权重绝对值成比例的随机扰动，方差为 $\sigma^2 |w|^2$，而非固定方差的加性噪声。
- **Refresh 变体**：每 N 步重新采样噪声并缩放幅度，将相干偏移转为随机游走的改进策略。
- **IC-Augmented**：在权重扰动基础上叠加初值（Initial Condition）扰动分析的混合集合策略。
- **Spread-Skill Ratio (SSR)**：集合离散度与观测误差标准差的比值，理想值为 1，偏离表明过分散或欠分散。
- **WeatherBench-2**：下一代数据驱动全球天气模型基准测试平台，用于公平比较不同 ML 气象模型。
- **SPPT（Stochastically Perturbed Parametrised Tendencies）**：ECMWF 物理集合预报中随机扰动参数化趋势的经典方法，本文为其 ML 类比灵感来源。

## 可复现要素
- **数据集**：ERA5 再分析资料（0.25°）；IC 扰动来自 IFS-ENS 50 成员。
- **代码/权重**：开源代码位于 github.com/MeteoSwiss/ai-models-ensembles（release v1.0.0）；四个主干模型权重可从上开源版本获取。
- **关键超参**：σ 扫描范围 {0.001, 0.003, 0.01, 0.03, 0.1}；Refresh 步数 N；预算缩放公式 $\sigma_{\text{group}} = \sigma_{\text{full}} \sqrt{N_{\text{total}} / N_{\text{group}}}$。
- **评估框架**：WeatherBench-2、SwissClim Evaluations pipeline；追加 Clausius-Clapeyron 和地转平衡检验。
