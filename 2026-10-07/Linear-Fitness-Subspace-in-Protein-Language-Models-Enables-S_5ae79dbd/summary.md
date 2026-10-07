---
title: "Linear-Fitness-Subspace-in-Protein-Language-Models-Enables-S"
source: https://arxiv.org/pdf/2610.07607v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:10:22"
---

# 论文速读：Linear-Fitness-Subspace-in-Protein-Language-Models-Enables-S

## 一句话总结
本文提出“线性适应度子空间（LFS）”假设，证明突变诱导的残基层PLM表征变化中存在一个低维、任务特异的线性可解坐标；基于此构建的SGES框架在LFS内进行代理建模与不确定采集，在有限oracle预算下显著提升了蛋白质定向进化的搜索效率与预测精度。

## 研究问题与动机
- **核心瓶颈**：模型引导的定向进化需在极有限的实验/计算评估预算下，从高维组合序列空间中高效挖掘高适应度变体。
- **Zero-shot失效**：预训练PLM的似然或masked-marginal分数缺乏任务对齐，固定的一维投影常与目标assay的真实适应度方向存在较大偏差。
- **全维监督低效**：直接在完整高维PLM嵌入空间做代理拟合与不确定性估计，受维度灾难影响，样本效率低且不确定性易被无关表征维度污染。
- **表征几何问号**：assay特异性突变效应信息究竟“隐藏”在冻结PLM的何处？能否通过少量标签恢复出局促且线性可解的低维子空间，从而指导高效搜索？

## 核心贡献（创新点）
- **LFS假设的形式化与实证**：首次论证在突变局部位点增量（site-delta）坐标中，适应度变化可由少数监督线性方向近似，区别于对全局适应度地形或PLM几何的强线性宣称。
- **零样本分数的几何操作化**：将zero-shot评分拟合为site-delta空间中的固定无任务方向，并量化其与监督LFS方向的余弦对齐度，为解释零样本优劣提供统一几何视角。
- **SGES子空间引导进化框架**：将PLS子空间发现、低维深度集成代理建模、UCB不确定采集与多样性过滤耦合，实现预算受限下的高效迭代搜索。
- **严格归因与多维验证**：通过匹配维度的PCA/随机投影/标签打乱PLS对照、组件消融及10/87/18/5四组角色分离的ProteinGym评测，清晰分离了“Fitness对齐”与“通用降维或采集策略”的贡献。

## 方法详解
- **Site-delta特征构造**：冻结PLM $h$，对位点 $p$ 的突变 $s$，定义局部残基位移 $\delta(s) = h(s)[p] - h(s_{\mathrm{wt}})[p] \in \mathbb{R}^d$，剔除野生型与突变体共享的背景噪声，聚焦突变诱导的表征扰动。
- **LFS子空间学习（Phase 1）**：用初始标记集 $D_0$（$N_{\mathrm{init}}=32$）通过偏最小二乘（PLS）拟合投影矩阵 $W \in \mathbb{R}^{d \times k}$（主实验 $k=8$），使 $z(s)=W^\top \delta(s)$ 成为与assay标签协方差最大的低维坐标；PLS保证所选方向直接对齐适应度而非仅解释表征方差。
- **子空间引导搜索（Phase 2）**：候选变体投影至LFS后，由 $M$ 个深度集成网络分别预测 $(\mu_m, \log\sigma_m^2)$；总不确定性合并数据噪声与模型分歧：$\sigma^2(s) = \frac{1}{M}\sum \sigma_m^2 + \frac{1}{M}\sum(\mu_m - \mu)^2$。
- **采集与动态更新**：采用UCB采集函数 $a(s) = \mu(s) + \beta \sigma(s)$；多突变采用位点加法聚合 $z(s) = \sum_{p \in \mathcal{M}(s)} W^\top \delta_p(s)$；每50次oracle评估重估计 $W$ 以适应搜索漂移；配合Hamming距离多样性过滤避免冗余采样。
- **零样本对齐度量**：在同一site-delta特征上对zero-shot分数拟合一维线性探针 $q_{\mathrm{zs}}(s) \approx b_{\mathrm{zs}} + w_{\mathrm{zs}}^\top \delta(s)$，计算 $\cos(w_{\mathrm{zs}}, w_{\mathrm{lfs}})$ 量化预训练方向与任务方向的先天对齐程度。

## 实验与结果
- **数据集与协议**：ProteinGym v2，划分为10核心assay（表征/消融）、87扩展assay（静态预测）、18预算搜索assay（Wilcoxon显著性）、5多突变基准。统一 $B_{\max}=500$，$N_{\mathrm{init}}=32$，5次随机种子，匹配搜索空间与proposal机制。
- **主要基线**：Random Search、AdaLead、GP-BO、EVOLVEpro、AlphaDE、BOES、LatentDE、TreeNeuralUCB、PCNN，以及ESM2/Tranception/ProSST等zero-shot PLM。
- **线性可访问性**：Ridge on $\delta$ 平均Spearman $\rho=0.763$，逼近MLP的 $0.825$（线性比0.921），显著优于Mean-pooled的 $0.574$，证明信号源自突变感知坐标而非 backbone 强度。
- **预算搜索核心结果**：SGES平均最佳适应度 $1.471$、平均排名 $1.38$，全面超越GP-BO（$1.328/2.38$）与EVOLVEpro（$1.183/5.00$，$p=0.002/0.008$）；在18个assay的AUC、Best Fitness、Budget-to-Top-5%三项指标上均呈统计显著优势。
- **归因对照（$k=8$）**：PLS-LFS（$\rho=0.804$, Best=1.45）> PCA（0.624）> 随机投影（0.512）>
