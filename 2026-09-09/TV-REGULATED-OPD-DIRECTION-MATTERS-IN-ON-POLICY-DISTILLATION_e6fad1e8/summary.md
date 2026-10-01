---
title: "TV-REGULATED-OPD-DIRECTION-MATTERS-IN-ON-POLICY-DISTILLATION"
source: https://arxiv.org/pdf/2609.08341v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:04:34"
field: "大语言模型后训练与知识蒸馏"
keywords: ["On-Policy Distillation", "Total Variation", "Knowledge Distillation", "LLM Post-training", "Variance Reduction", "Policy Gradient"]
innovations: ["证明 sign-only OPD 系数等价于条件 TV 下降方向，无需精细 token 级幅度分配", "提出基于采样 token 的单侧有界 TV 估计器与 EMA 全局强度调节机制 TV-OPD", "通过受控幅度消融分离方向与幅度信息，揭示后期训练稳定性来源"]
benchmarks: ["AIME 2024", "AIME 2025", "DAPO-Math-17K", "DeepMath-103K"]
---

# 论文速读：TV-REGULATED-OPD-DIRECTION-MATTERS-IN-ON-POLICY-DISTILLATION

## 一句话总结
本文系统探究了 On-Policy Distillation (OPD) 训练不稳定的根本原因，发现仅保留 token 级监督信号的符号即可达到与标准 OPD 相当的性能；在此基础上提出 TV-OPD 方法，利用总变差 (TV) 距离同时提供有界的局部方向信号与可衰减的全局强度调节，实现了训练动态更稳定、后期性能更优的蒸馏效果。

## 研究问题与动机
- **核心问题**：主流 OPD 方法使用的 token 级 log-probability difference 作为监督信号存在高方差与极端值，导致训练不稳定并限制最终性能上限。
- **现有方法不足**：既有工作（如 clipping、power transformation、control variate 等）均假设 token-wise 精细幅度分配是必要的，但本文通过受控消融实验表明，精确的相对幅度分配并未带来一致收益。
- **被忽视的关键洞察**：该监督信号可分解为方向（sign）、相对幅度与全局幅度三个成分，其中方向与全局缩放是有效监督的两个关键因素，而 token 间精细幅度配比并非必需。

## 核心贡献（创新点）
1. **提出“仅保留符号”的监督假设并给出实证支撑**：将 OPD 系数分解为 sign 与 magnitude，证明仅保留 sign 即可实现与 Raw OPD 相近甚至更优的训练稳定性与最终性能。
2. **建立 sign-only 信号与条件 TV 下降方向的理论等价性**：推导表明 $\text{sign}(\Delta(s,a))$ 加权梯度恰好估计条件 TV 距离的负梯度，从而将蒸馏目标从 reverse KL 转向有界、低方差的 TV 优化。
3. **设计基于采样 token 的无偏 TV 估计器**：利用概率质量守恒，构造单侧估计 $\hat{d}(s,a) = [1 - e^{\Delta(s,a)}]_+$，无需额外 teacher 打分即可得到有界且条件方差 $\le 1/4$ 的 TV 估计。
4. **提出 TV-REGULATED 全局强度调节机制**：通过 EMA 平滑累积的 TV 估计构建共享系数 $c_k$，使更新强度随师生分布收敛而衰减，避免过度抑制或放大梯度。
5. **系统验证 TV-OPD 在后期训练阶段的稳定性优势**：在 1.5B 与 8B 两套同族师生模型对、AIME 2024/2025 基准上，TV-OPD 均取得最高或最具竞争力的后期保留性能（LateMean 提升约 2–3 个百分点，PeakDrop 降低约 1 个百分点）。

## 方法详解
- **信号分解**：将原始优势 $A_i^{\text{Raw}} = \Delta_i = \log p(a|s) - \log q_{\bar\theta}(a|s)$ 分解为 $z_i = \text{sign}(\Delta_i)$ 与 $m_i = |\Delta_i|$。
- **Sign-TV 基础目标**：仅保留方向，采用 surrogate loss $\widehat{\mathcal{L}}_{\text{TV}} = -\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}} \text{sg}[\text{sign}(\Delta_i)]\log q_\theta(a_i|s_i)$，该目标在无偏意义下对应条件 TV 下降方向，且系数绝对值 $\le 1$，天然有界。
- **采样 TV 估计**：利用 $\sum_a(q_\theta(a)-p(a))=0$ 的性质，构造单侧估计 $\hat{d}(s,a)=[1-e^{\Delta(s,a)}]_+ \in [0,1]$，其条件期望等于 $D_{\text{TV}}(q_\theta(\cdot|s),p(\cdot|s))$，条件方差 $\le 1/4$。
- **全局调节系数**：在每步 $k$ 对所有 active token 的 $\hat{d}$ 求池化得 $\widehat{D}_k$，经 EMA $\bar D_k=\beta\bar D_{k-1}+(1-\beta)\widehat{D}_k$ 平滑后，计算共享缩放系数 $c_k = \text{clip}\!\left[\!\left(\frac{\bar D_{k-1}+\epsilon}{D_{\text{ref}}+\epsilon}\right)^\alpha,\, c_{\min},\,1\right]$，最终系数为 $A_{i,k}^{\text{TVR}}=c_k\,\text{sign}(\Delta_{i,k})$。
- **实现细节**：初始系数为 1；首次有效步冻结 $D_{\text{ref}}$；EMA 保持 $\beta=0.95$；$\alpha$ 控制衰减速率；$c_{\min}=0.1$ 防止过度抑制；系数在步内采样前确定并 detach，仅缩放全局强度而不改变 token 间相对方向。

## 实验与结果
- **数据集与模型对**：
  - 1.5B 对：JustRL-DeepSeek-1.5B → DeepSeek-R1-Distill-Qwen-1.5B，提示集 DAPO-Math-17K。
  - 8B 对：Qwen3-8B → Qwen3-8B-SFT (400K OpenThoughts)，提示集 DeepMath-103K。
- **评估基准**：AIME 2024、AIME 2025 validation，每题 4 次采样报告 mean@4。
- **训练设置**：batch=64 prompts，每 prompt 1 条 on-policy rollout，AdamW $\eta=10^{-6}$，gradient clip=1.0，max length=16384，bfloat16，8 GPU。
- **主要结果**（Table 1，以 AIME24+AIME25 均值选 checkpoint）：
  - **1.5B**：TV-OPD 在 AIME 2024 达 $\mathbf{51.67\pm1.18}$（Raw 50.83，Sign-TV 51.25），AIME 2025 达 $\mathbf{38.75\pm2.08}$（Raw 37.92，Sign-TV 37.08），两榜平均 45.21%。
  - **8B**：TV-OPD 在 AIME 2024 达 $\mathbf{70.00\pm1.17}$（Raw 68.33，Sign-TV 65.00），AIME 2025 达 $\mathbf{58.33\pm1.67}$（Raw 57.08，Sign-TV 57.08），两榜平均 64.17%，分别超越最强基线 1.67/1.25 分。
- **后期稳定性**（Table 2, E 节）：在 JustRL 对 late stage (step 500–625)，TV-OPD 较 Raw OPD 在 AIME 2024 提升 3.47 分、AIME 2025 提升 0.90 分；aggregate LateMean 从 $40.87\pm0.83$ 升至 $43.06\pm0.10$，PeakDrop 从 $3.51\pm1.13$ 降至 $2.36\pm0.49$。
- **超参敏感性**（Table 3）：$\alpha$ 从 0.5 增至 4.0 时性能单调下降，两榜平均从 38.13 降至 33.33（降幅 4.80 分），验证过强衰减会抑制学习。

## 相关工作脉络
- **Raw OPD / Sampled-token OPD**（Lu & Thinking Machines Lab, 2025）：本文出发点；使用 teacher-student log-ratio 作为代理梯度系数，存在高方差与极端值问题。
- **vOPD**（Oh et al., 2026）：通过 detached control-variate baseline 降低方差，但与本文不同——其修改的是估计方式而非目标散度形式。
- **PowerOPD**（Zhao et al., 2026）：对似然比做有界幂变换，保持符号一致性；本文进一步剥离幅度信息，仅保留符号并映射到 TV 目标。
- **OPD+**（Zhao et al., 2026）：针对一般 f-divergence 推导校正系数；本文聚焦 sampled-token 系数自身携带的信息，分离方向与全局强度。
- **TIDE**（Yu et al., 2026）：结合有界 Hellinger  shaping 与 teacher top-k 注入；本文方法更简洁，完全通过 TV 估计实现统一调节。
- **Demystifying OPD**（Wang et al., 2026）：系统分析 OPD 角色与病态；本文在此基础上通过受控幅度干预实验明确“方向优于精细幅度”的结论。

## 局限性与未来方向
- **教师质量依赖**：teacher-relative 方向不能保证策略改进，弱教师或错配教师仍可能引导错误行为。
- **条件代理性质**：TV 等价性是 stopped-state occupancy 下的条件近似，未对 rollout 分布求导，不能保证序列级回报单调提升。
- **实验范围有限**：仅在两个同族师生对、两类数学推理基准上验证，未覆盖多领域或更大规模设置。
- **未来方向**：① 探索 TV 目标与其他散度（如 reverse KL、f-divergence）的结合；② 将局部 TV  closeness 的行为保留性质扩展到序列级；③ 在更多任务（代码、对话、多步推理）与更大模型（>8B）上验证泛化性；④ 研究自适应 $\alpha$ 调度或结合任务奖励的混合目标。

## 研究启发与可借鉴点
- **信号分解实验范式**：将监督系数拆分为方向/相对幅度/全局幅度并施加置换、分组常值、取消对齐等受控干预，为诊断强化/蒸馏信号的信息含量提供清晰因果框架。
- **有界替代目标的价值**：用 TV 代替 reverse KL 不仅降低方差，还天然提供单调的行为保留保障（Prop. 2），可推广至其他基于采样 token 的蒸馏或 RLHF 场景。
- **全局缩放与局部权重解耦**：通过单一 EMA 调节全局强度而保持 token 间相对方向不变，设计简洁且避免重新引入 magnitude 噪声，可作为通用稳定器模块。
- **单侧估计器的数值技巧**：利用 $[1-e^\Delta]_+$ 形式结合 `expm1(min{Δ,0})` 实现避免大指数与相消的稳定计算，值得在其他似然比估计中复用。
- **Late-stage 稳定性指标**：引入 LateMean 与 PeakDrop 量化训练后期保留能力，弥补仅看最终 checkpoint 的不足，适用于任何易出现过拟合或崩溃的 post-training 流程。

## 关键术语表
- **On-Policy Distillation (OPD)**：在 post-training 阶段让 student 沿自身策略生成 token，再用 teacher-student log-probability difference 作为 token 级监督信号进行策略梯度更新的知识蒸馏方法。
- **Total Variation (TV) distance**：两个概率分布逐元素绝对差之和的一半，取值 $[0,1]$，刻画条件 next-token 分布的整体差异。
- **Sign-TV**：仅保留 $\text{sign}(\Delta(s,a))$ 作为系数的 OPD 变体，其梯度无偏估计条件 TV 的负梯度方向。
- **TV-OPD**：本文提出的 TV-regularized OPD，以 Sign-TV 提供局部方向、以 EMA 平滑的采样 TV 估计提供全局衰减系数。
- **Detached surrogate loss**：仅对 student log-probability 求梯度，teacher 概率、采样前缀与系数均被 detach 固定的代理损失。
- **Control variate**：从优势估计中减去一个与策略无关的基线以降低方差，vOPD 所用技术。
- **Power transformation**：对 likelihood ratio 施加有界幂函数以保持符号一致同时压缩极端值（PowerOPD 核心）。
- **LateMean / PeakDrop**：后期窗口平均性能与训练期峰值性能之差，分别度量稳定保留能力与退火程度。

## 可复现要素
- **数据集**：OpenThoughts-400K（SFT）、DeepMath-103K、DAPO-Math-17K；AIME 2024/2025 validation。**论文未明确声明公开状态**，但相关模型与提示集在文献中多有开源版本。
- **代码/权重**：论文**未声明**代码或模型权重开源；实验使用 Qwen3-8B、DeepSeek-R1-Distill-Qwen-1.5B、JustRL-DeepSeek-1.5B 等公开或半公开权重。
- **关键超参**：
  - 学习率 $1\times10^{-6}$，AdamW，weight decay 0.01，gradient clip 1.0
  - batch size 64 prompts，每 prompt 1 rollout，temperature 1.0、top-p 1.0
  - 最大 prompt/response 长度 1024/16384，bfloat16，8 GPU FSDP
  - EMA $\beta=0.95$，$\alpha=0.5$，$c_{\min}=0.1$，$D_{\text{ref}}$ 冻结于首次有效步
  - 评估：每题 4 次采样，temperature 0.6、top-p 0.95、top-k 20、max length 31744
- **其他**：随机种子数 2；checkpoint 选取策略为 AIME24+AIME25 均值最大者；训练曲线覆盖 step 0–625。
