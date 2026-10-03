---
title: "IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS"
source: https://arxiv.org/pdf/2609.37147v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:49:26"
field: "生成模型加速"
keywords: ["Diffusion Models", "Few-step Generation", "Distributional Denoising", "Flow Matching", "Energy Score"]
innovations: ["延迟粒子展开降低DDM计算开销", "时间依赖评分规则调度适应不同噪声水平"]
benchmarks: ["ImageNet-256²", "MS-COCO"]
---

# 论文速读：IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS

## 一句话总结
本文改进了分布扩散模型（DDMs），通过延迟粒子展开降低计算开销、引入时间依赖的评分规则调度，首次使DDMs在DiT骨干网络下实现实用的少数步数图像生成，在ImageNet-256²上4步和50步FID分别达4.48和2.38，且无需蒸馏或自训练。

## 研究问题与动机
- **DDMs难以扩展到现代图像生成规模**：原始DDMs每训练样本需抽取m个噪声样本进行多粒子训练，导致计算开销随粒子数O(m)线性增长，且大部分计算在重复处理相同的x_t特征。
- **固定超参数无法适应不同采样预算**：原始DDMs在整个扩散轨迹中使用全局固定的评分规则超参数(λ, β)，但p(x_1|x_t)在不同噪声水平下集中程度不同——粗粒度采样需要更多多样性，细粒度采样则需要更高保真度。
- **少数步数下条件分布方差大**：当采样步数较少时（如4步），条件分布p(x_1|x_t)呈现较大方差，仅学习条件均值的传统去噪器无法充分概括后验分布。
- **现有少数步数方法依赖蒸馏**：主流方法通过教师蒸馏或一致性模型实现加速，但需要多阶段训练和预训练教师模型。

## 核心贡献（创新点）
- **延迟粒子展开（Deferred Population Expansion）**：将x_t先通过Transformer的大部分层共享计算，仅在最后几层展开为m个粒子并注入辅助噪声ξ，将训练开销从4×降至约1.5×流匹配成本。与已有DDM相比，本质区别是从输入端全层展开改为后期局部展开，复用早期特征提取。
- **时间依赖评分规则调度**：将固定超参数(λ, β)替换为随时间变化的调度(λ(t), β(t))，根据Biroli等人提出的动力学相（speciation/collapse）将训练目标从分布导向平滑过渡到回归导向。与已有工作的本质区别在于将物理洞察直接映射到训练目标的设计上。
- **优化的ξ条件注入机制**：系统比较了通道拼接、残差加法、自适应归一化和寄存器token四种噪声注入方式，发现固定内宽拼接效果最佳。这是首次在大尺度Transformer架构下系统探索该设计空间。
- **统一的单阶段训练框架**：通过上述改进，实现了一个从头训练的单阶段少数步数生成器，FID随采样步数增加单调下降（4步4.48→50步2.38），无需蒸馏、CFG训练或自引导。

## 方法详解
- **延迟粒子展开架构**：对于批大小为B的样本，前ℓ_start层以batch size B运行一次，hidden state被复制m份后，在第ℓ_start层注入独立噪声ξ_j，剩余L-ℓ_start层以batch size B×m运行。计算开销降为(ℓ_start + (L-ℓ_start)×m)/(L×m)，对于DiT-XL（L=28, ℓ_start=24, m=4）约为36%。
- **ξ条件注入机制**：采用固定内宽通道拼接方式，将ξ拼接为额外256维通道到残差流中，保持注意力和MLP的原始内宽，速度预测仍从原始d个通道读取。引入t-dependent门控机制控制ξ的幅度。
- **时间依赖评分规则**：基于Biroli等人的动力学相分析，定义三个区域：Regime I-II（低SNR，分布需多样化）和Regime III（高SNR，可趋向回归）。使用线性调度s_λ(t)=s_β(t)=1-t，其中λ(t)=λ_max·s_λ(t)，β(t)=2-(2-β_min)·s_β(t)，默认β_min=0.1。
- **改进的训练时间采样**：将时间采样从均匀分布改为jit分布（mode在t≈0.25，偏向噪声端），使更多训练时间集中在分布性损失更关键的低SNR区域。
- **局部核评分**：将广义能量评分从全局向量空间改为逐位置局部评分，对每个空间位置独立计算后平均，强调局部结构。

## 实验与结果
- **数据集**：ImageNet-256²（类别条件图像生成），以及MS-COCO用于文本到图像迁移验证。
- **评估基线**：Flow Matching (FM)、原始DDM、MeanFlow、iMF、IMM、MFM等少数步数生成方法。
- **主要结果**（DiT-B scale, 400k steps）：
  - 4-step FID: iDDM 13.13 vs. FM 26.53 vs. 原始DDM 43.68
  - 50-step FID: iDDM 4.57 vs. FM 4.97 vs. 原始DDM 10.04
- **最强结果**（DiT-XL/2, 200 epochs）：4-step FID 4.48，50-step FID 2.38，从单模型单阶段训练获得。
- **计算匹配比较**：在相同训练计算预算下，iDDM将4-step FID从26.70降至14.33（接近减半），同时保持50-step FID在4.61（与FM的4.21差距仅0.40）。
- **文本到图像迁移**：在1.6B参数模型上，iDDM将MS-COCO 4-step FID从78.20降至41.05（几乎减半）。

## 相关工作脉络
- **Flow Matching (Lipman et al., 2023)**：学习确定性速度场映射，是本文的比较基线；DDMs扩展为学习条件分布而非仅均值。
- **原始DDMs (De Bortoli et al., 2025b)**：本文直接建立在广义能量评分框架上，解决其可扩展性和固定超参数问题。
- **MeanFlow/iMF/IMM (Geng et al., 2025, 2026; Zhou et al., 2025a)**：学习平均速度场或矩匹配的少数步方法，训练成本更高且为确定性映射；iDDM是随机生成器且训练成本更低。
- **Consistency Models (Song et al., 2023)**：通过自蒸馏学习自洽的少数步生成器；iDDM无需蒸馏且从scratch训练。
- **Meta Flow Maps (MFM, Potaptchik et al., 2026)**：使用辅助ODE的一致性损失学习随机流映射，侧重奖励对齐；iDDM聚焦直接评分规则训练。
- **Dynamical Regimes (Biroli et al., 2024)**：分析Ornstein-Uhlenbeck扩散过程的反向动力学相，本文为其应用到flow-matching框架并指导调度设计。

## 局限性与未来方向
- ** regime锚点是近似值**：基于平均场分析推导，在有限规模下不够精确；平滑调度比硬阶跃更鲁棒。
- **1-2步极限性能有限**：在极步数下，基于蒸馏的方法（如sCD、ADD）仍具有更优性能。
- **最优调度形状待探索**：理论上的最优λ(t)、β(t)组合仍是开放问题，当前使用简单线性调度。
- **仅验证了ImageNet和T2I**：方法的有效性主要在ImageNet-256²和文本到图像任务上验证，在其他模态的泛化性待考察。

## 研究启发与可借鉴点
- **计算-质量权衡的架构设计**：延迟展开策略可迁移到其他需要多粒子采样的生成模型，显著降低计算开销而不损失质量。
- **物理洞察指导训练目标设计**：利用动力学相分析来设计训练调度，而非仅依赖经验调参，为其他生成模型提供了方法论参考。
- **噪声注入机制的系统化比较**：对ξ条件注入方式的全面消融实验（concatenation、addition、AdaNorm、tokens）为后续工作提供了设计指南。
- **时间采样策略的重要性**：jit采样显著提升性能，表明训练时的时间分布设计对少数步数训练同样关键。
- **单模型多步数泛化**：一个checkpoint在4-50步范围内性能单调下降，为实际部署提供了便利，可启发其他 Few-Step 方法追求类似特性。

## 关键术语表
- **Distributional Diffusion Model (DDM)**：用分布去噪器替代均值去噪器的扩散模型，通过广义能量评分训练以学习条件分布而非仅条件均值。
- **Generalized Energy Score**：参数化为(λ, β)的评分函数，通过保真项和交互项平衡预测准确性与样本多样性。
- **Deferred Population Expansion**：将多粒子展开推迟到Transformer后期层，减少重复计算的开销。
- **Speciation Time / Collapse Time**：反向扩散过程中的两个特征时间，分别标志数据粗结构可分辨和轨迹趋向单个训练样本的转折点。
- **Signal-to-Noise Ratio (SNR)**：在flow-matching框架中，SNR = t²/(1-t)²，表征当前时刻信号与噪声的相对强度。
- **Conditional Velocity Field**：连接数据点和噪声点的速度场，flow matching通过学习该场实现从噪声到数据的转换。
- **Classifier-Free Guidance (CFG)**：一种提升生成质量的技巧，通过条件与无条件预测的差异放大来实现。
- **Posterior Diversity**：模型对同一输入x_t产生不同输出的能力，通过固定x_t重复采样评估。

## 可复现要素
- **数据集**：ImageNet-256²（公开），MS-COCO（公开用于T2I实验）
- **代码开源**：是，https://github.com/CompVis/iDDM
- **预训练权重**：是，代码页面提供
- **关键超参**：DiT-B（depth 12, width 768），DiT-XL/2（depth 28, width 1152），粒子数m=4，ℓ_start=24（XL）/10（B），学习率1×10⁻⁴，训练步数400k（B）/200 epochs（XL），自变量t采样为jit分布
