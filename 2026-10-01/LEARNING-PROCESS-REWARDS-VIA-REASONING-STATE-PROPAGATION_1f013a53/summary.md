---
title: "LEARNING-PROCESS-REWARDS-VIA-REASONING-STATE-PROPAGATION"
source: https://arxiv.org/pdf/2609.39220v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:57:47"
field: "过程奖励建模与推理增强"
keywords: ["process reward model", "reasoning state propagation", "reinforcement learning", "test-time scaling", "outcome supervision"]
innovations: ["提出 RSP 通过 break/repair 概率显式建模推理状态转移，将过程标注与结果标注联合利用", "推导结果监督对中间状态转移的后验重加权梯度，证明终端标签可反向传播学习信号", "在 beam search、Best-of-N 和 RL 三场景统一验证，平均较 Qwen2.5-Math-PRM 提升 5.6%（beam search）和 2.1%（RL）"]
benchmarks: ["MATH500", "Gaokao", "ProcessBench", "DAPO-MATH-17K", "GSM8K", "AIME", "AMC"]
---

# 论文速读：LEARNING-PROCESS-REWARDS-VIA-REASONING-STATE-PROPAGATION

## 一句话总结
论文提出 Reasoning State Propagation (RSP)，通过 break 和 repair 概率显式建模推理状态在步骤间的转移，将稀缺的过程标注与可扩展的结果标注联合利用，使过程奖励模型在 beam search、Best-of-N 选择和强化学习中均优于现有 PRM 基线。

## 研究问题与动机
- **过程标注成本高、结果标注易获取**：PRM 训练依赖昂贵的步骤级人工标注（如 PRM800K），而 outcome label 只需判断最终答案正确性，易于规模化。如何有效融合两类信号尚未被探索。
- **现有 PRM 将推理前缀独立分类**：主流判别式 PRM（如 Supervised PRM、Math-Shepherd）对每个推理前缀独立预测有效性，缺乏跨步骤的状态关联机制。
- **首次错误假设过于简化**：PQM、CRM 等方法以"第一个错误步骤"为中心组织推理状态，认为后续推理保持无效，无法刻画 LRM 在后续步骤中纠错（error recovery）的现象。
- **全错组 RL 信号缺失**：RLVR 中 GRPO 等组相对方法在全错组内给出相同零 advantage，导致策略优化无学习信号，需要更细粒度的过程反馈。

## 核心贡献（创新点）
- **提出 RSP 的形式化建模**：用二元有效状态（G/B）表示推理前缀，通过 break 概率（G→B）和 repair 概率（B→G）显式传播状态，连接轨迹内所有中间步骤与最终状态。
- **联合利用过程与结果监督**：过程标注监督中间状态，结果标签约束最终状态并通过状态传播反向提供学习信号，形成互补而非替代的监督组合。
- **多场景实验验证有效性**：在 beam search、Best-of-N 选择和 RL 训练三个任务上，RSP 持续优于代表性 PRM 基线；平均较 Qwen2.5-Math-PRM 提升 5.6%（beam search）和 2.1%（RL）。
- **理论分析揭示结果监督的梯度机制**：推导 outcome loss 对 break/repair logits 的偏导数，证明结果标签通过后验重加权间接监督中间状态转移，而非简单在每个位置施加相同标签。

## 方法详解
- **状态定义**：每个推理前缀 $\mathbf{s}_{\leq t}$ 对应二元状态 $z_t \in \{\text{G}, \text{B}\}$，G 表示当前推导有效无未解决错误，B 表示存在未解决错误。
- **转移概率建模**：引入 break 概率 $\alpha_t = p_\theta(z_t=\text{B} \mid z_{t-1}=\text{G}, x, \mathbf{s}_{<t}, s_t)$ 和 repair 概率 $\beta_t = p_\theta(z_t=\text{G} \mid z_{t-1}=\text{B}, x, \mathbf{s}_{<t}, s_t)$，构成局部状态转移矩阵 $\mathbf{M}_t$。
- **状态传播公式**：初始状态 $\mathbf{p}_0 = [1, 0]$（起始为 G），通过 $\mathbf{p}_t = \mathbf{p}_{t-1} \mathbf{M}_t = \mathbf{p}_0 \prod_{j=1}^t \mathbf{M}_j$ 传播，$p_t^\text{G}$ 作为过程得分。
- **参数化方式**：在每个推理边界 $t$ 插入特殊 token 对 $\langle \text{BREAK}\rangle_t, \langle \text{REPAIR}\rangle_t$，用骨干模型 $f_\phi$ 编码后提取边界表示 $\mathbf{h}_t^{\text{br}}$ 和 $\mathbf{h}_t^{\text{rp}}$，通过两个独立 MLP 头计算：$\alpha_t = \sigma(g_{\text{br}}(\mathbf{h}_{t-1}^{\text{br}} \parallel \mathbf{h}_t^{\text{br}}))$，$\beta_t = \sigma(g_{\text{rp}}(\mathbf{h}_{t-1}^{\text{rp}} \parallel \mathbf{h}_t^{\text{rp}}))$。
- **损失函数**：
  - 过程损失：$\mathcal{L}_{\text{step}} = -\frac{1}{|\mathcal{A}_i|}\sum_{t \in \mathcal{A}_i}[y_{i,t}^{\text{step}} \log p_{i,t}^\text{G} + (1-y_{i,t}^{\text{step}})\log p_{i,t}^\text{B}]$
  - 结果损失：$\mathcal{L}_{\text{out}} = -\frac{1}{N_{\text{out}}}\sum_i[y_{i}^{\text{out}} \log p_{i,T_i}^\text{G} + (1-y_{i}^{\text{out}})\log p_{i,T_i}^\text{B}]$
  - 总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{step}} + \mathcal{L}_{\text{out}}$
- **结果监督的梯度性质**：推导显示 $\frac{\partial \ell^{\text{out}}}{\partial \log\text{it}(\alpha_t)} = \frac{p_{t-1}^\text{G}\alpha_t(1-\alpha_t)}{Z_{\text{out}}}(h_t(\text{G})-h_t(\text{B}))$，结果标签通过 $h_t$（从当前状态到达终态的概率差）调节中间转移的学习方向。

## 实验与结果
- **数据集**：
  - 过程标注：PRM800K（随机选取 20%）
  - 结果标注：AceMath-RM 过滤后的正确性标签（保留高置信样本）
  - 评估基准：MATH500、Gaokao、ProcessBench、DAPO-MATH-17K（RL 训练）
- **模型架构**：基于 Qwen3 系列（默认 Qwen3-4B 骨干，部分实验用 Qwen3-1.7B/8B/32B）
- **基线方法**：Supervised PRM、OVM、Qwen2.5-Math-PRM（7B）、CRM、Pseudo-Label PRM、Joint-Supervised PRM、CRM†
- **Beam Search（Table 1）**：
  - Qwen3-4B 骨干 + e=12 时：RSP 在 MATH500 达 79.8%，Gaokao 达 67.9%
  - 较 OVM 提升 4.9%（e=12）
  - 较 Qwen2.5-Math-PRM 7B 在 MATH500+Gaokao 平均提升约 5.6%
- **Best-of-N（Table 2）**：
  - 平均 across N∈{8,16,32,64,128}，RSP 达 74.4%，超越 Qwen2.5-Math-PRM（74.1%），后者额外使用了大规模过程标注（√+）
  - CRM† 加结果监督后从 72.1% 提升至 73.4%，RSP 进一步至 74.4%
- **强化学习（Table 3）**：
  - RSP 平均 Avg@16 达 65.4%，较 Qwen2.5-Math-PRM 的 63.3% 提升 2.1%
  - AIME 上优势最大：RSP 38.9% vs Qwen2.5-Math-PRM 31.1%（+7.8%）
  - RSP 是唯一在平均上超越 GRPO（63.0%）的 PRM 引导方法
- **ProcessBench F1（Table 5）**：
  - RSP 达 65.8%，较 Supervised PRM 的 56.6% 提升 9.2%
  - 较 Qwen2.5-Math-PRM 的 70.5% 略低（后者额外使用大规模过程标注）
- **消融实验（Table 4）**：
  - Process-Only 在 beam search 下降 12.5%，Outcome-Only 在 ProcessBench 下降 9.6%
  - 移除 Repair 模块（No Repair）beam search 下降 3.3%
  - 阻断 Outcome Propagation 导致 beam search 下降 6.7%

## 相关工作脉络
- **Supervised PRM / PRM800K（Lightman et al., 2024）**：独立对每个推理前缀做二元分类，依赖人工过程标注，不建模步骤间依赖；RSP 通过状态传播显式关联中间状态。
- **OVM（Yu et al., 2024）**：用结果标注训练 value model，将 outcome 作为每步目标；RSP 不仅利用结果监督，还通过状态传播将结果信号反馈到中间转移。
- **PQM（Li & Li, 2025）**：将 PRM 建模为 Q-value ranking，捕捉轨迹内步骤的相对关系，但仍以首个错误步骤为中心；RSP 允许状态在 B 与 G 之间灵活切换。
- **CRM（Zhang et al., 2025a）**：用条件奖励建模第一个无效状态，通过概率链式法则链接过程与结果；RSP 不再假设"首次出错后全程无效"，显式建模 repair。
- **VeriGate（Agrawal et al., 2026）**：verifier-gated 策略在 outcome 失败时激活 PRM 的 token-level 监督；本文 RSP 本身即支持过程+结果联合训练，无需额外 gating 机制。
- **Math-Shepherd（Wang et al., 2024）**：通过采样扩展推断步骤正确性以减少标注成本；RSP 通过结果标注直接提供替代监督信号。

## 局限性与未来方向
- **过程标注仍不可或缺**：Outcome-Only 在 ProcessBench 上下降 9.6%，表明细粒度过程错误识别仍需过程监督。
- **状态二元简化**：G/B 二元划分无法刻画多类错误（如计算错误 vs. 逻辑错误 vs. 表述不清），可能丢失细粒度信息。
- **仅验证数学推理领域**：当前实验集中在数学任务（MATH500、Gaokao、GSM8K 等），在其他推理领域（代码生成、科学推理）的有效性待验证。
- **传播强度因步而异**：κ_t 在不同推理步骤差异较大（79% 满足 |κ_t|<0.2），早期步骤的结果反馈较弱，可能存在梯度衰减。
- **特殊 token 引入**：需要在每个推理边界插入 ⟨BREAK⟩/⟨REPAIR⟩ token，增加序列长度，但未报告计算开销。

## 研究启发与可借鉴点
- **状态传播作为监督信号通道**：通过显式状态转移将终端监督反向传播到中间步骤的思路可迁移到其他需要细粒度反馈的场景（如代码生成、工具调用）。
- **Separate Break/Repair Tokens 设计**：使用不同特殊 token 区分概念语义（break vs. repair），避免单一 token 的语义干扰，这一设计可借鉴于其他需要多类型转移建模的任务。
- **结果监督的后验重加权解释**：论文推导的 outcome-conditioned targets（$\tilde{\alpha}_t, \tilde{\beta}_t$）揭示了结果标签如何根据"后续推理能力"动态调整中间学习信号，这一分析框架可推广到其他 sequential decision 场景。
- **训练配比灵活性**：过程:结果数据比从 1:1 到 1:9 均可工作，beam search 和 RL 随结果数据增加持续提升，为低资源过程标注场景提供实用指导。
- **评估体系完整性**：同时覆盖 search（beam）、selection（Best-of-N）、optimization（RL）和 diagnosis（ProcessBench）四个维度，评估设计可作为 PRM 研究的参考模板。

## 关键术语表
**Process Reward Model (PRM)**：对推理轨迹中每个中间步骤（reasoning prefix）输出有效性评分的奖励模型，用于细粒度过程反馈。
**Reasoning State Propagation (RSP)**：本文提出的方法，将每个推理前缀表示为二元状态（G/B），通过 break/repair 概率建模状态转移并沿轨迹传播。
**Break Probability (α_t)**：上一步有效状态在当前推理步骤后变为无效的概率，刻画错误引入。
**Repair Probability (β_t)**：上一步无效状态在当前推理步骤后恢复为有效的概率，刻画错误修复。
**State Propagation Matrix (M_t)**：描述单个推理步骤引起的状态转移概率的 2×2 矩阵。
**Outcome Supervision**：仅标注推理轨迹最终答案正确性的监督信号，成本低但缺乏步骤级细粒度。
**Process Supervision**：对每个推理步骤标注有效/无效的细粒度监督信号，成本高但提供精准过程反馈。
**κ_t (Propagation Strength)**：$κ_t = 1 - α_t - β_t$，衡量第 t 步传播状态对前一步状态的依赖强度，也决定结果监督向后传播的强度。

## 可复现要素
- **数据集**：PRM800K（公开，https://github.com/google-research/PRM800K）、AceMath-RM（公开，https://github.com/SWHL/AceMath）、DAPO-MATH-17K（公开）、MATH500（来自 PRM800K 发布时的 held-out set）、Gaokao（公开）、ProcessBench（公开）
- **代码**：论文声明提供在 supplementary material，URL 未明确给出（需查阅 arxiv 页面附件）
- **权重**：基于 Qwen3-4B 开源权重初始化
- **关键超参**：
  - 学习率：$5 \times 10^{-6}$（AdamW）
  - 训练 epoch：2
  - 过程:结果数据比：1:3（default）
  - MLP head：$2d \to d \to 1$，GELU 激活，dropout 0.1
  - RL 训练：batch size 128，micro-batch 2，actor lr $10^{-6}$ cosine schedule，temperature 1.0，top-p 1.0，max prompt 512，max generation 4096 tokens
  - 硬件：8× NVIDIA H100，bfloat16
