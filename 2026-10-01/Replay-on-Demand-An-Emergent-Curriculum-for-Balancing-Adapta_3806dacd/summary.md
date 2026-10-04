---
title: "Replay-on-Demand-An-Emergent-Curriculum-for-Balancing-Adapta"
source: https://arxiv.org/pdf/2609.40089v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-04 12:59:00"
field: "语言模型持续预训练"
keywords: ["Continual Pretraining", "Replay", "Data Selection", "Forgetting Mitigation", "Reducible Loss", "Curriculum Learning"]
innovations: ["提出RoD通过源特定可减少损失让适配与回放样本联合竞争共享训练预算，动态生成在线课程", "证明跨模型/域/尺度（8.75× scale gap）的有效性，小模型构建的课程可迁移至大模型"]
benchmarks: ["Legal adaptation", "German adaptation", "CapTrack capability taxonomy", "Nemotron family", "Qwen3.5 family"]
---

# 论文速读：Replay on Demand: An Emergent Curriculum for Balancing Adaptation and Forgetting in Continued Pretraining

## 一句话总结
本文提出 Replay on Demand (RoD)，一种在线数据选择方法，通过让适应样本和回放样本基于源特定的可减少损失联合竞争共享训练预算，动态生成适应-遗忘平衡的课程，无需预先指定回放比例。

## 研究问题与动机
- **核心矛盾**：持续预训练（CPT）使语言模型适应新领域时，常以遗忘原有预训练能力为代价，形成适应与保留的固有权衡。
- **固定回放的局限**：现有方法依赖固定比例混合原始预训练数据，忽视遗忘在不同能力维度、数据源间的异质性，以及随训练进程动态演变的特性。
- **资源分配效率问题**：固定比例可能浪费计算于已良好保留的知识，同时未能充分保护更易遗忘的部分。
- **缺少自适应机制**：缺乏根据模型实际状态自动调整回放分配量与组成内容的机制。

## 核心贡献（创新点）
1. **提出RoD动态回放框架**：通过源特定可减少损失评分，让适应与回放样本联合竞争训练预算，自动生成在线课程。与已有工作本质区别在于不复用固定比例，而是从模型学习动态中涌现回放分配。
2. **设计双向可减少损失度量**：适配样本用适应参考模型估计剩余学习潜力，回放样本用预训练模型测量遗忘程度，统一尺度实现联合选择。区别于仅优化单一目标的方法。
3. **证明跨模型/域/尺度的泛化性**：在Nemotron和Qwen家族、Legal和German领域验证有效性，并展示小模型构建的课程可迁移至更大模型（8.75× scale gap）。突破单场景验证的局限。
4. **揭示遗忘驱动的 replay 分配规律**：分析表明RoD系统性地将回放集中到脆弱知识类别，避免对稳定类别的过度保护，固定回放仅整体下移遗忘曲线而保持相对分布。

## 方法详解
**问题设定**：给定预训练模型 $\theta_0$，在适应分布 $\mathcal{D}_A$ 上微调的同时保持对预训练分布 $\mathcal{D}_R$ 的性能。候选集 $C_A \subset \mathcal{D}_A$ 和 $C_R \subset \mathcal{D}_R$，联合选择 top-k 构成训练批次 $B_t$。

**源特定可减少损失（Equation 3）**：
$$
\rho(x; \theta_t) = 
\begin{cases}
\rho_A(x; \theta_t) = \ell_{\theta_t}(x) - \ell_{\theta_{ref}^A}(x), & x \in C_A \\
\rho_R(x; \theta_t) = \ell_{\theta_t}(x) - \ell_{\theta_0}(x), & x \in C_R
\end{cases}
$$

- **适配端**：$\theta_{ref}^A$ 是在 $\mathcal{D}_A$ 上专门训练至收敛的适配专家模型，衡量剩余学习潜力。
- **回放端**：$\theta_0$ 作为参考，直接测量相对于预训练状态的损失退化。

**联合选择（Equation 4）**：
$$
B_t = \text{TopK}_{x \in C_A \cup C_R} \rho(x; \theta_t)
$$

候选池大小由倍数 $m$ 控制，$|C_A| + |C_R| = mk$，默认 $m=2$。每个训练步包含一次候选池前向评分（无梯度），加上选定批次的前向-反向更新。参考模型损失离线预计算并缓存。

**计算开销**：额外引入 $m$ 倍候选的前向推理开销，但无需梯度或优化器更新。

## 实验与结果
**数据集**：Legal（NVIDIA Nemotron预训练法律语料）、German（Soofi-Team构造的德语语料），回放数据使用Nemotron通用预训练数据（14类细粒度来源）。

**模型**：Nemotron-Nano-12B-v2、Qwen3.5-9B、Qwen3.5-4B、Nemotron-3-Nano-30B-A3B、Qwen3.5-35B-A3B。

**评估指标**：适配验证损失、验证损失遗忘（分布级）、任务级遗忘（基于CapTrack四类能力组的多项选择评测）。

**主要结果**：
- **Nemotron-Legal**：RoD在10%固定回放相近适配损失（0.967 vs 0.967）下，验证损失遗忘从0.125降至0.071（↓43%），任务遗忘从6.7降至4.9点。
- **Nemotron-German**：RoD以21% emergent回放比例达到适配损失1.788，验证遗忘0.073，优于17%固定回放（0.122遗忘）和32%固定回放（0.077遗忘）的权衡。
- **Qwen3.5-Legal**：匹配10%固定回放适配损失（0.988），验证遗忘从0.075降至0.031（↓59%）。
- **Qwen3.5-German**：适配损失1.790 vs 17%固定回放1.796，验证遗忘0.028 vs 0.035。
- **跨尺度迁移**：4B模型构建的课程用于训练9B、30B、35B模型，8.75× scale gap下仍保持有竞争力的适应-遗忘权衡。
- **最强结果**：Nemotron-Legal RoD（0.071验证遗忘，43%改善）；Qwen3.5-Legal RoD（0.031验证遗忘，59%改善）。

## 相关工作脉络
1. **Continual Pretraining with Replay**：Ibrahim et al. (2024)、Roth et al. (2024) 等研究回放缓解遗忘，但采用固定比例或简单调度；RoD从模型状态动态生成回放分配。
2. **Scaling Laws for Source-Target Mixtures**：Gu et al. (2024)、Que et al. (2024) 探索最优混合比例；RoD无需预设比例，通过竞争涌现分配。
3. **Reducible Holdout Loss (RHO)**：Mindermann et al. (2022) 提出基于参考模型的损失可减少性数据选择；RoD将其扩展到双源联合选择场景。
4. **Adaptive Data Mixing**：Chen et al. (2025)、Luo et al. (2025)、Yang et al. (2026) 等方法动态调整域权重；RoD统一适配与回放于同一竞争框架。
5. **Replay-based Continual Learning**：Atreya et al. (2026)、Feng et al. (2026)、Tong et al. (2025) 分别研究间隔重复、遗忘曲线记忆、Coreset选择；RoD将回放选择嵌入预训练而非增量学习范式。
6. **Model Merging**：Wortsman et al. (2022) 线性插值参数；RoD通过数据选择实现类似目标但不依赖后处理。

## 局限性与未来方向
- **回放分布依赖**：RoD只能保护回放数据集中存在的知识，若相关能力缺失则无法检测和保护（Section D.3）。
- **代理回放的信号偏差**：当使用非原始预训练数据作为回放时（如Qwen实验），验证损失遗忘与任务级能力遗忘的对齐度下降，尤其在推理密集型能力上（Section C.2）。
- **计算开销**：每步需额外 $mk$ 样本的前向评分，虽无需梯度但增加推理成本；当前实现未针对效率优化（Section D.1）。
- **缺乏相对权重调节**：当前框架将适配与回放损失置于同等地位，未提供显式控制适应-保留偏向的超参数（Section D.2）。
- **未来方向**：引入适配-回放相对权重调节机制、探索更轻量级的参考模型构建、研究非代表性代理回放下的改进信号。

## 研究启发与可借鉴点
1. **源特定参考模型设计**：适配用专属参考、回放用预训练参考的双轨策略，避免单一参考隐含固定权衡，可迁移至其他多目标训练场景。
2. **可缓存参考损失**：参考模型离线训练后损失可预计算缓存，线上仅需当前模型前向评分，大幅降低在线开销，适用于任何参考模型指导的选择任务。
3. **跨尺度课程迁移**：小模型构建的数据选择课程可Transfer至大模型训练，为大规模模型的高效持续预训练提供可行路径（8.75× scale gap验证）。
4. **动态 replay 分配的可解释分析**：通过遗忘曲线斜率、采样比例相关性（r=0.88）等量化分析揭示方法机制，为评估其他自适应训练方法提供分析范式。
5. **Proxy Replay的适用边界**：系统分析代理回放与原生回放的信号对齐差异，为数据不可用场景下的方法应用提供决策依据。

## 关键术语表
**Continued Pretraining (CPT)**：在已有预训练模型基础上继续训练新数据，以适配特定领域或任务，同时需缓解遗忘问题。
**Replay**：在适配训练中混入原始预训练数据以维持已学能力，是缓解遗忘的常见策略。
**Reducible Holdout Loss (RHO)**：基于参考模型估计样本当前损失中仍可被进一步学习减少的部分，用于数据选择优先级排序。
**Adaptation Reference Model ($\theta_{ref}^A$)**：在适配数据上专门训练至收敛的模型，作为适配样本的参考基准以度量剩余学习潜力。
**Emergent Curriculum**：通过模型状态驱动的动态数据选择自动形成的训练课程，而非预先设计的静态混合比例。
**Validation-Loss Forgetting**：适配后模型在泛化验证数据上的损失相对于预训练基线的增量，衡量分布级遗忘。
**Task-Based Forgetting**：基于CapTrack四类能力组（参数知识、推理、常识、多语言）的任务精度下降均值，衡量能力级遗忘。
**Proxy Replay**：当原始预训练数据不可用时，使用其他模型的预训练数据作为回放代理，其信号对齐度弱于原生回放。

## 可复现要素
- **数据集**：Nemotron-Pretraining-Legal-v1（公开）、German语料（基于Soofi-Team工作构建）、Nemotron通用预训练数据（用于回放）；均为公开数据。
- **代码开源**：GitHub仓库 https://github.com/bethgelab/replay-on-demand 提供RoD实现及实验复现代码。
- **模型权重**：Nemotron-Nano-12B-v2、Qwen3.5-9B/4B、Nemotron-3-Nano-30B-A3B、Qwen3.5-35B-A3B均为开源权重模型。
- **关键超参**：序列长度4096、batch size 1024、learning rate peak $1.2 \times 10^{-4}$ / min $1.2 \times 10^{-5}$、Adam ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$, weight decay 0.1)、gradient clipping 1.0、candidate multiplier $m=2$、seed=42。
- **训练预算**：约11.8-12.5B trained tokens（~2824-2980 updates）。
