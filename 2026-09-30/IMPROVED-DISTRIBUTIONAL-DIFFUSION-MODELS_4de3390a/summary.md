---
title: "IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS"
source: https://arxiv.org/pdf/2609.37147v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:49:33"
field: "生成模型与扩散模型加速"
keywords: ["diffusion models", "distributional denoising", "few-step generation", "scoring rules", "flow matching", "latent diffusion"]
innovations: ["延迟粒子扩展：在transformer后期层展开多粒子以降低O(m)计算开销", "时间依赖评分规则调度：基于扩散动力学制度动态调整(λ,β)超参数", "局部核评分：将全局能量评分改为空间位置独立计算后平均"]
benchmarks: ["ImageNet-256²", "MS-COCO (T2I)"]
---

# 论文速读：IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS

## 一句话总结
本文提出改进的分布扩散模型（iDDM），通过**延迟粒子扩展**和**时间依赖评分规则调度**两项创新，解决了原DDM在多粒子训练计算开销和超参数固定两大瓶颈，实现了在ImageNet-256²上仅用单个DiT-XL/2模型从头训练即可在4-50步采样下获得优异FID（4.48/2.38）的随机少步生成器。

## 研究问题与动机
- **核心问题**：分布扩散模型(DDM)通过训练随机去噪器来近似条件分布 $p(x_1|x_t)$ 而非仅条件均值，但在现代图像生成规模下难以实用。
- **瓶颈一（计算开销）**：DDM对每个训练样本需要 $m$ 个粒子，每个粒子需独立前向传播，导致 $O(m)$ 计算开销，而大部分计算（早期transformer层）重复处理相同的 $x_t$ 特征。
- **瓶颈二（超参数固定）**：原DDM使用全局固定的 $(\lambda, \beta)$ 超参数， forcing 单一的保真度-多样性权衡贯穿整个扩散轨迹；当 $t \approx 1$ 时后验已集中，无需多样性，但固定设置无法自适应。
- **动机**：将DDM推向实际可用规模，同时保留其建模完整条件分布而非仅均值的优势，实现无需蒸馏、无需CFG训练的单阶段少步生成。

## 核心贡献（创新点）
1. **延迟粒子扩展（Deferred Population Expansion）**：将 $x_t$ 的前 $L-\ell_{start}$ 层shared处理，仅在最后 $\ell_{start}$ 层才展开为 $m$ 个粒子并注入噪声 $\xi$，将训练开销从 $4\times$ FM降至约 $1.5\times$。
2. **时间依赖评分规则调度（Time-Dependent Scoring Rule Schedules）**：将评分规则超参数 $(\lambda(t), \beta(t))$ 设计为随时间变化的调度，结合Biroli等人的扩散动力学制度（speciation/collapse时间），使模型在低SNR时强调多样性、高SNR时倾向回归。
3. **优化的 $\xi$ 条件注入机制**：提出并系统比较了channel concat、residual addition、AdaNorm、register tokens等多种注入方式，确定fixed-inner-width concatenation为最优方案。
4. **改进的训练时间采样策略**：发现将训练时间偏向噪声端（jit采样）能提升少步性能，并与调度设计协同优化。

## 方法详解
### 3.1 延迟粒子扩展与 $\xi$ 条件注入
- **核心思想**：对batch size $B$ 的样本，前 $\ell_{start}-1$ 层以batch size $B$ 运行一次；然后在这些层后将隐藏状态复制 $m$ 份，注入不同的 $\xi_j$，剩余层以 batch size $B \times m$ 运行。
- **计算节省**：训练步计算量降至 $(\ell_{start} + (L-\ell_{start})m)/(Lm)$，对于DiT-XL ($L=28, \ell_{start}=24, m=4$) 约为36%。
- **$\xi$ 注入机制比较**：
  - Fixed-inner-width concatenation（默认）：将 $\xi_j$ 拼接到残差流，但保持attention/MLP内部宽度不变，参数增加最小且4步FID最优。
  - AdaNorm/Token条件：少步表现严重下降。

### 3.2 时间依赖评分规则调度
- **理论基础**：基于Biroli et al. (2024) 的 Ornstein-Uhlenbeck 扩散动力学制度分析，确定三个 regimes：
  - **Regime I-II**（$\rho < \rho_c$）：prior localization，偏好distributional目标。
  - **Regime IIIa**（$\rho_c \leq \rho < \rho_{sep}$）：collapse后但未分离，保持spread同时向sharp目标过渡。
  - **Regime IIIb**（$\rho \geq \rho_{sep}$）：unambiguous数据端，倾向regression目标。
- **调度公式**：
  $$\lambda(t) = \lambda_{max} \cdot s_\lambda(t), \quad \beta(t) = 2 - (2-\beta_{min}) \cdot s_\beta(t)$$
  其中 shape function 默认采用线性 $s(t) = 1-t$，$\beta_{min}=0.1$。
- **时间采样优化**：采用 `jit` 分布（mode at $t \approx 0.25$）替代uniform，将72.5%训练时间分配到低SNR区域。

### 3.3 局部核评分
- 将全局能量评分（对整个图像flatten计算）改为**local kernel**（每个空间位置独立计算后平均），显著提升性能（Table 3d）。

## 实验与结果
### 数据集与设置
- **主实验**：ImageNet-256²，使用REPA-E (SD-VAE) latent空间，DiT-B/XL backbone，400k/200k steps训练。
- **评估指标**：FID@50k，CFG sweep优化。

### 主要结果（Table 4b，DiT-XL/2 scale）
| 方法 | 50-step FID | 4-step FID | 训练epoch数 |
|------|-------------|------------|-------------|
| Shortcut-XL/2 | 4.688 | 7.8* | 162 |
| MeanFlow-XL/2 | 3.29 | 2.93 | 240 |
| iMF-XL/2† | 1.43 | 1.51 | 800 |
| **iDDM-XL/2（ours）** | **2.38** | **4.48** | **200** |
| DDM-XL/2 (naïve) | 4.71 | 23.25 | 200 |

- **关键结论**：
  - iDDM在4步FID上比naïve DDM提升 **5.2×**（23.25→4.48），50步提升 **2.0×**（4.71→2.38）。
  - 相比flow matching基准，在匹配训练计算下，4步FID从26.36降至13.13（约减半），50步仅略降（4.21→4.61）。
  - 训练开销仅比FM高1.41×，远低于iMF（6.20×）。

### T2I迁移（Table 13）
- 在MS-COCO上，4步FID从78.20降至41.05，8步从26.33降至19.40。

## 相关工作脉络
- **DDM原始工作**（De Bortoli et al., 2025b）：本文直接扩展，解决其计算开销和超参数固定问题，定位为"practical scaling of DDMs"。
- **MeanFlow/iMF/IMM**（Geng et al., 2025; Zhou et al., 2025a）：同为few-step生成方法，但学习确定性map；iDDM与之定位差异在于**stochastic**、**single-stage from scratch**、**FID不随步数增加而退化**。
- **Consistency Models**（Song et al., 2023）：self-distillation方法；iDDM无需teacher和第二阶段训练。
- **MFM**（Potaptchik et al., 2026）：同样学习目标分布，但基于consistency loss；iDDM基于proper scoring rule，侧重多步生成。
- **Diffusion-GAN/UFOGen**：通过adversarial training学习多模态条件生成器；iDDM使用scoring rule，无对抗训练。
- **Biroli et al. (2024)**：提供动力学制度理论框架，本文将其用于指导调度设计。

## 局限性与未来方向
- 动力学制度锚点（speciation/collapse时间）基于mean-field分析，在有限规模下是近似的；平滑调度比硬步骤更鲁棒。
- 在1-2步极端少步场景下，蒸馏和fast-forward方法仍更强。
- 理论最优的 $\lambda(t), \beta(t)$ 调度仍是开放问题。
- 未来方向：探索更优调度形状、与其他加速方法（如Universal Inverse Distillation）的结合、扩展到更大分辨率/视频生成。

## 研究启发与可借鉴点
1. **延迟计算复用策略**：在multi-particle training中，"shared trunk + late expansion"的设计可迁移至其他需要多样本输出的生成模型（如ensemble learning、uncertainty quantification）。
2. **动力学制度指导训练设计**：将理论分析（如相变时间）转化为可操作的调度策略，是一种"theory-to-practice"的有效范式，可推广至其他扩散相关任务。
3. **固定内宽条件注入**：fixed-inner-width concatenation在保持参数效率的同时实现噪声注入，对设计efficient conditional generation有参考价值。
4. **局部vs全局评分权衡**：local kernel的提升表明在high-dimensional图像生成中，空间解耦的评分可能比全局评分更稳定，值得在其他distributional learning任务中验证。
5. **单一checkpoint多步兼容**：FID随步数非递增的特性（4.48→2.38）使得部署更灵活，这一设计原则可推广至其他fast sampling场景。

## 关键术语表
- **Distributional Diffusion Model (DDM)**：用随机去噪器 $\hat{x}_\theta(t, x_t, \xi)$ 替代条件均值预测，通过广义能量评分训练以逼近完整条件分布 $p(x_1|x_t)$ 的扩散模型。
- **Deferred Population Expansion**：将粒子展开延迟至transformer后期层，共享前期特征提取的计算开销。
- **Generalized Energy Score**：参数化为 $(\lambda, \beta)$ 的评分规则，包含 fidelity项 $\|X-y\|^\beta$ 和 diversity项 $\|X-X'\|^\beta$。
- **Speciation Time ($\tau_s$)**：反向扩散过程中，信号前缀特征从噪声中可分辨的时刻。
- **Collapse Time ($\tau_c$)**：后验分布从多模态流形结构坍缩为孤立训练样本主导的时刻。
- **Flow Matching (FM)**：学习从噪声到数据的time-dependent velocity field，通过ODE积分生成样本。
- **CFG (Classifier-Free Guidance)**：推理时的条件强化技术，通过无条件预测缩放条件预测。
- **NFE (Number of Function Evaluations)**：采样过程中模型前向传播的次数，对应采样步数。

## 可复现要素
- **数据集**：ImageNet-256²（公开），RECA-captioned COYO（用于T2I，公开）。
- **代码/权重**：开源，https://github.com/CompVis/iDDM。
- **关键超参**：
  - DiT-B: depth=12, width=768, $m=4$, $\ell_{start}=10$, 400k steps
  - DiT-XL: depth=28, width=1152, $m=4$, $\ell_{start}=24$, 200 epochs
  - $\lambda_{max}=1, \beta_{min}=0.1$, linear schedule $s(t)=1-t$
  - Optimizer: AdamW, lr=1e-4, batch=256 (B-scale)
  - Autoencoder: REPA-E (SD-VAE, f8d4)
- **硬件**：4×H200 (B-scale), 8×H200 (XL-scale)。
