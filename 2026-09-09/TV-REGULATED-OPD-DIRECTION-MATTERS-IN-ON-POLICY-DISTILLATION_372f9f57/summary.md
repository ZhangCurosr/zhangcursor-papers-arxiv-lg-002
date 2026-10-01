---
title: "TV-REGULATED-OPD-DIRECTION-MATTERS-IN-ON-POLICY-DISTILLATION"
source: https://arxiv.org/pdf/2609.08341v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:04:27"
field: "大语言模型后训练与知识蒸馏"
keywords: ["On-Policy Distillation", "Total Variation", "Knowledge Distillation", "LLM Post-training", "Variance Reduction", "Gradient Coefficient"]
innovations: ["证明OPD token级sign-only信号等价于条件TV下降方向，无需细粒度幅度重加权", "提出TV-OPD方法：用有界TV估计同时提供局部方向和EMA衰减全局强度", "在1.5B和8B双scale上验证后期训练稳定性显著提升（PeakDrop降低33%）"]
benchmarks: ["AIME 2024", "AIME 2025"]
---

# 论文速读：TV-REGULATED OPD: DIRECTION MATTERS IN ON-POLICY DISTILLATION

## 一句话总结
本文系统剖析了在线策略蒸馏（OPD）中 token 级监督信号的内在信息构成，发现仅保留 teacher-student log-probability 差值的符号方向即可达到与标准 Raw OPD 相当的性能；在此基础上提出 TV-OPD 方法，利用条件总变差（Total Variation, TV）距离同时提供有界低方差的局部方向信号与可衰减的全局强度调节，显著提升训练稳定性和后期性能保持。

## 研究问题与动机
- **OPD 监督信号高方差与训练不稳定的根因未明**：主流 OPD（如 DeepSeek、Qwen 内部采用的方法）以 teacher-student 的 log-probability 差值作为 token 级奖励系数，该信号缺乏统一上界，极端值导致训练不稳定并限制性能上限。
- **已有方法假设"细粒度幅度重加权必有必要"**：现有改进（vOPD、PowerOPD、TIDE、OPD+ 等）通过裁剪、幂变换、控制变量、散度修正等方式精细调控梯度权重，但其核心前提——token 级幅度分配携带关键信息——尚未得到验证。
- **缺乏对信号构成要素的系统性解耦分析**：OPD 系数可分解为方向（sign）、相对幅度（relative magnitude）、全局幅度（global magnitude）三部分，三者各自的贡献程度仍不清楚。
- **后训练阶段知识蒸馏工程化需求强烈**：OPD 已在工业界广泛落地（Qwen、MiMo、GLM、DeepSeek 系列），但训练不稳定已成为实际部署中的明显痛点。

## 核心贡献（创新点）
- **系统性解耦 OPD 监督信号并给出受控实验证据**：首次在 JustRL-1.5B 上通过 Sign / Group-Constant / Permuted 等对照变体证明，细粒度 token 级幅度重加权并不能带来一致收益，方向信息本身已足够。
- **建立 sign-only 信号与条件 TV 下降方向的理论等价性**：证明保留符号的更新等价于优化 teacher-student 条件分布间的 TV 距离（Proposition 1），获得有界 $[-1,1]$ 系数的理论基础，消除了 Raw 系数无一致上界的根本缺陷。
- **提出基于采样 token 的无偏 TV 估计器与全局调节器联合框架（TV-OPD）**：利用概率质量守恒导出单侧有界估计 $\hat{d}(s,a) = [1 - e^{\Delta}]_+$，并通过 EMA 构建随分布收敛而衰减的全局系数 $c_k$，形成"局部方向由 sign 提供、全局强度由 TV 值调控"的双层架构。
- **揭示训练动力学规律并给出量化验证**：在 JustRL 1.5B 设置下，证明 TV-OPD 在后期（steps 500–625）LateMean 从 40.87% 提升至 43.06%（+2.19pp），PeakDrop 从 3.51pp 降至 2.36pp（-33%），并提供 $\alpha$ 敏感性扫描。
- **在 1.5B 与 8B 两个 scale 上均验证有效性**：Qwen3-8B 设置中 TV-OPD 在 AIME 2024 达到 70.00±1.17（较 Raw OPD 的 68.33 提升 1.67pp），两项平均 64.17%，超越所有对比方法。

## 方法详解
- **信号分解框架**：将原始 OPD 系数分解为 $A_i^{\text{Raw}} = z_i m_i$，其中 $z_i = \text{sign}(\Delta_i)$ 为方向，$m_i = |\Delta_i|$ 为幅度。通过 Sign（$z_i \cdot 1$）、Group-Constant（$z_i \kappa_{z_i}$，同符号组共享均值）、Permuted（$z_i m_{\sigma(i)}$，组内打乱幅度对应关系）等变体进行受控实验，隔离幅度信息的影响。
- **从 sign 监督到 TV 优化的理论推导**：在固定状态 $s$ 下，sign-only 更新方向满足 $-2\nabla_\theta D_{\text{TV}}(p, q_\theta) = \mathbb{E}_{a\sim q_\theta}[\text{sign}(\Delta(s,a))\nabla_\theta \log q_\theta(a|s)]$（Proposition 1），即 sign-only 梯度等价于条件 TV 下降方向。TV 有界于 $[0,1]$，sign 系数天然有界于 $[-1,1]$。
- **采样 token 的 TV 估计器**：利用归一化条件 $ \sum_a(p(a)-q_\theta(a))=0$ 即"超量概率总和 = 不足概率总和"，得到单侧无偏估计 $\hat{d}(s,a) = [1 - p(a|s)/q_\theta(a|s)]_+ = [1 - e^{\Delta(s,a)}]_+$。该估计始终落在 $[0,1]$，条件方差至多为 $1/4$。数值上采用 $-\text{expm1}(\min\{\Delta,0\})$ 避免大正指数溢出。
- **全局强度调节器（EMA-based）**：在每步 $k$ 对所有 microbatch 和 data-parallel worker 上的 $\hat{d}$ 分子与有效 token 数做 pooled 计算得到 $\widehat{D}_k$，经 EMA 平滑：$\bar{D}_k = \beta \bar{D}_{k-1} + (1-\beta)\widehat{D}_k$（$\beta=0.95$）。全局系数 $c_k = \text{clip}[(\bar{D}_{k-1}+\epsilon)/(D_{\text{ref}}+\epsilon)]^\alpha, c_{\min}, 1]$，第一步冻结 $D_{\text{ref}}$ 为其观测值。$\alpha$ 控制衰减灵敏度，$c_{\min}=0.1$ 设置下限。最终更新形式为 $A_{i,k}^{\text{TVR}} = c_k \cdot \text{sign}(\Delta_{i,k})$，$c_k$ 在采样前固定并通过 detached 方式应用于整个 batch。

## 实验与结果
- **数据集与评估基准**：数学推理基准 AIME 2024 和 AIME 2025（validation set），训练使用 DeepMath-103K（Qwen 设置）与 DAPO-Math-17K（JustRL 设置） prompts。
- **模型对设置**：
  - 1.5B：JustRL-DeepSeek-1.5B（teacher）→ DeepSeek-R1-Distill-Qwen-1.5B（student），AdamW lr=$10^{-6}$，batch=64，每 prompt 1 rollout，8×GPU，bfloat16。
  - 8B：Qwen3-8B（teacher）→ Qwen3-8B-SFT（student，400K OpenThoughts SFT 初始化），相同训练配置。
- **主要结果（Table 1，mean@4，best checkpoint 由两 benchmark 均值选取）**：

| 设置 | 基准 | Initial | Raw OPD | Sign-TV | TV-OPD |
|---|---|---|---|---|---|
| JustRL-1.5B | AIME 2024 | 30.83 | 50.83±0.83 | 51.25±0.42 | **51.67±1.18** |
| JustRL-1.5B | AIME 2025 | 28.33 | 37.92±0.42 | 37.08±0.42 | **38.75±2.08** |
| Qwen3-8B | AIME 2024 | 60.83 | 68.33±0.00 | 65.00±0.00 | **70.00±1.17** |
| Qwen3-8B | AIME 2025 | 50.83 | 57.08±2.95 | 57.08±2.08 | **58.33±1.67** |

- **最强结果**：Qwen3-8B 设置下 AIME 2024 达到 70.00%，较 Raw OPD 提升 1.67pp；两项平均 64.17%，较 Raw OPD 平均（62.71%）提升 1.46pp。
- **后期动力学（Table 2 & LateMean/P eakDrop）**：在 JustRL 设置 late stage（steps 500–625），TV-OPD LateMean 43.06±0.10 对比 Raw OPD 40.87±0.83（+2.19pp），PeakDrop 2.36±0.49 对比 3.51±1.13（-33%）。Early stage Sign-TV 略优，中后期 TV-OPD 明显超越。
- **α 敏感性（Table 3）**：JustRL 对 α∈{0.5,1.0,2.0,4.0} 扫描，α=0.5 最佳（平均 38.13%），α=4.0 最差（33.33%），呈单调下降，证明过强衰减会抑制优化进度。

## 相关工作脉络
- **On-Policy Distillation 原始方法（Lu & Thinking Machines Lab, 2025）**：本文研究的基线，使用 teacher-student log-ratio 作为策略梯度系数，缺乏系数上界，是 TV-OPD 的直接对比对象。
- **vOPD（Oh et al., 2026, arXiv:2605.07865）**：引入 detach control-variate baseline 降低方差，本质是对估计器的修正；TV-OPD 则从目标函数层面重构为 TV 距离，两者正交。
- **PowerOPD（Zhao et al., 2026, arXiv:2606.17199）**：对 likelihood ratio 做有界幂次变换保持符号一致；与 TV-OPD 均追求系数有界，但 PowerOPD 保留幅度信息，TV-OPD 证明幅度并非必要。
- **TIDE（Yu et al., 2026, arXiv:2608.09836）**：用 Bregman divergence 区分 student-excess / teacher-deficit token 施加不同 shaping；TV-OPD 不做 token 类型区分，强调 sign 方向本身的充分性。
- **OPD+（Zhao et al., 2026, arXiv:2606.01039）**：修正 f-divergence 目标下的策略梯度系数；关注估计偏差修正，与 TV-OPD 的信号解耦视角互补。
- **MiniLLM / GKD（Agarwal et al., 2023; Gu et al., 2024）**：基于 reverse KL 或灵活散度的分布匹配蒸馏路线；TV-OPD 属于 sampled-token OPD 分支下的目标重构，而非直接替代 distribution-matching 框架。

## 局限性与未来方向
- **teacher-relative 方向不保证策略单调改进**：若 teacher 本身较弱或与任务目标错配，sign 信号仍可能引导 student 走向不良行为（论文自述）。
- **TV 等价性为 conditional surrogate 且有 stopped occupancy 假设**：未对 rollout 分布求导，不保证 sequence-level return 的单调性，也不直接转化为完整响应分布的 TV 单调下降。
- **实验覆盖有限**：仅在 JustRL-1.5B 和 Qwen3-8B 两对同 family 模型、AIME 两个数学基准、各两个 random seed 上验证，无法排除特定设置偶然性。
- **缺少对缺失 token coverage 的讨论**：top-k / nucleus truncation 会改变支持集，TV 系数的有界性优势在此设定下需重新推导 importance-weighting 分析（论文在 Appendix A.6 承认此点）。
- **未探索非数学任务的泛化性**：当前评估集中在数学推理，未涉及代码生成、对话或多模态等场景。
- **方向提示**：将 TV-OPD 的 sign+EMA 机制扩展到 sequence-level objectives、探索非对称 teacher quality、以及在 top-k 截断下重新设计有界系数均是合理延伸方向。

## 研究启发与可借鉴点
- **信号解耦实验范式具有通用价值**：将监督系数拆解为方向、相对幅度、全局幅度三要素并逐一切除/置换，这种受控干预设计可直接迁移到 RLHF / DPO / REINFORCE 等梯度系数分析场景。
- **有界替代非有界系数的思路可复用**：TV 提供理论有界性（$[-1,1]$）和方差上界（≤1/4），这一"用有界距离度量替换无界 log-ratio"的策略可移植到 PPO-clip、REINFORCE+baseline 等对梯度幅度敏感的算法中。
- **EMA 式全局衰减调节器的简洁性值得借鉴**：只用一个 EMA 平滑标量即可替代逐 token 幅度，参数极少（仅 α, β, $c_{\min}$），工程实现成本低；可与任何基于 log-ratio 的方法结合形成通用"strength regulator"模块。
- **Sign-only 训练的稳定性启示**：在 RL 训练中对 advantage 取 sign（类似 REINFORCE with sign advantage）可显著降低方差，配合全局调度可在不损失性能的前提下提高 training throughput。
- **条件 TV 估计器的无偏性证明技术**：利用概率质量守恒导出单侧估计 $\hat{d}=[1-e^\Delta]_+$，这一技巧可推广至其他散度（如 Hellinger、JSD）的采样估计设计中。

## 关键术语表
- **On-Policy Distillation (OPD)**：将 teacher 分布知识蒸馏到 student 的在线策略方法，监督信号基于 student 自身生成的 prefix 上计算，已广泛用于 Qwen、DeepSeek 等大模型后训练。
- **Token-level Advantage（$\Delta$）**：OPD 核心系数，定义为 teacher 与 student 在生成 token 处的 log-probability 之差，正值鼓励该 token、负值抑制。
- **Conditional Total Variation (TV) Distance**：在固定状态 $s$ 下 teacher 与 student 条件 next-token 分布间的 $L_1$ 距离的一半，取值 $[0,1]$，比 KL 更具数值稳定性。
- **Sign-TV**：仅保留 OPD 系数符号的简化版本，对应条件 TV 下降方向的子梯度更新，作为 TV-OPD 的理论基础。
- **TV-OPD**：本文提出的方法，用 sign(Δ) 提供局部方向，用 EMA 平滑的 TV 估计 $\bar{D}_k$ 调控全局更新强度 $c_k$，使系数有界且随收敛衰减。
- **Detached Surrogate Loss**：OPD 训练中仅对 student log-probability 求梯度、teacher 及系数视为常数的代理损失函数形式，确保 on-policy 点处梯度恒等。
- **EMA Regulator**：指数移动平均调节器，$\bar{D}_k=\beta\bar{D}_{k-1}+(1-\beta)\widehat{D}_k$，用于将无偏 TV 估计平滑为稳定的全局系数输入。
- **Mean@k**：对每道题目采样 $k$ 个回答取均值是否正确的评估指标，文中 $k=4$，反映模型在多个样本上的稳定正确率。

## 可复现要素
- **数据集**：训练用 DeepMath-103K（Qwen 设置）和 DAPO-Math-17K（JustRL 设置）；评测用 AIME 2024 和 AIME 2025 validation sets。论文未明确说明各数据集的公开链接，需自行查找。
- **代码/权重**：论文未声明代码仓库或权重开源；模型基于 DeepSeek、Qwen 系列开源权重，训练数据部分公开（OpenThoughts、DAPO-Math、DeepMath）。
- **关键超参**：AdamW lr=$1\times10^{-6}$，weight decay=0.01，gradient clipping=1.0，EMA $\beta=0.95$，$\alpha=0.5$，$c_{\min}=0.1$，batch size=64 prompts，每 prompt 1 rollout（temp=1.0, top-p=1.0），最大响应长度 16384 tokens，precision=bfloat16，8×GPU。
- **评估协议**：每问题采样 4 个回答（temp=0.6, top-p=0.95, top-k=20, max len=31744），报告 mean@4；best checkpoint 由两 benchmark 均值选取，每个方法 2 个 random seed。
- **硬件**：8×GPU（具体型号论文未提及）。
