---
title: "On-Policy-Distillation-with-Negative-Policy-Rollouts"
source: https://arxiv.org/pdf/2610.07874v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:53:22"
field: "大语言模型后训练与知识蒸馏"
keywords: ["on-policy distillation", "negative policy", "knowledge distillation", "RL post-training", "LLM reasoning", "preference optimization"]
innovations: ["在rollout阶段引入负策略提供负向信号，不修改蒸馏奖励设计", "证明负策略token期望reward为负，理论上保证学生远离负策略", "提出NSP和NOR两个可量化指标验证负信号机制的有效性"]
benchmarks: ["AIME24", "AIME25", "AMC23", "MATH500", "LiveCodeBench", "GPQA", "SciBench", "OlympiadBench", "SuperGPQA", "HMMT25", "RGMath", "RGAlgo", "CodeForces"]
---

# 论文速读：On-Policy Distillation with Negative-Policy Rollouts

## 一句话总结
本文提出 Negative-Policy On-Policy Distillation（NP-OPD），通过在 rollout 阶段引入一个低于学生模型能力的"负策略"来补充正向教师蒸馏信号，使学生在保持原有 OPD 正反馈的同时获得明确的负向参照，从而在数学、代码和科学推理任务上稳定提升多种 OPD 变体的性能。

## 研究问题与动机
- **OPD 仅有正向信号**：现有 on-policy distillation 以教师为唯一参照，仅告诉学生"应该接近谁"，缺乏"应该远离谁"的明确负向指示，当师生分布重叠不足时学习信号不充分。
- **既有改进集中在奖励设计**：ExOPD、OPD² 等通过修改 token 级 reward 来提升性能，但将负策略通过 reward 注入会与这些已有奖励改进产生兼容性问题。
- **DPO 的成功启发**：Direct Preference Optimization 同时使用偏好/非偏好样本提供双向信号，且负样本常从弱模型采样；OPD 尚未系统探索类似机制。
- **负策略 rollout 的可复用性**：预生成的负策略 rollout 可在整个训练过程中重复使用，降低在线 rollout 生成成本。

## 核心贡献（创新点）
1. **在 rollout 阶段而非 reward 阶段引入负策略**：将负策略作为 rollout 来源之一（混合比例 α），保持原有 OPD 蒸馏奖励不变，因此天然兼容 ExOPD、OPD² 等最新变体。
2. **给出负策略何以产生负向信号的理论解释**：证明在假设 $D_{KL}(\pi_n\|\pi^*)>D_{KL}(\pi_n\|\pi_\theta)$ 下，负策略生成的 token 具有负的期望 OPD reward，从而平均意义上推动学生远离负策略。
3. **提出两个可量化验证的分析指标 NSP 与 NOR**：NSP（negative-policy suppression precision）衡量被压制 token 中负策略偏好 token 的比例；NOR（negative-policy overlap reduction）衡量学生与负策略在教师优先 token 上的概率偏移，两者均验证 NP-OPD 按预期工作。
4. **大规模实验验证跨尺度、跨域、跨变体的通用增益**：在 Qwen3（1.7B/4B/8B）和 Gemma-4-E4B 上，覆盖 thinking/non-thinking 模式及 13 个推理基准，NP-OPD 在所有设置下均稳定优于 OPD 基线。
5. **揭示近期 OPD 变体的隐含负信号视角**：发现 ExOPD 和 OPD² 即使使用 on-policy rollout 也已有更高的 NSP，说明它们通过相对参考 implicitly 利用了一定负信号，NP-OPD 在此基础上进一步放大该效应。

## 方法详解
**核心设计**：在标准 OPD 框架中，每步训练中按 Bernoulli(α) 采样 rollout 来源 $z_x$：
- 若 $z_x=0$，rollout 由学生策略 $\pi_\theta$ 生成（on-policy）；
- 若 $z_x=1$，rollout 由负策略 $\pi_n$ 生成（$\pi_n$ 取自与学生同族但更小的模型）。

**Token 级 distillation reward 不变**，仍采用标准 OPD reward：
$$R_t^{\text{OPD}} = \log \pi^*(y_t | x, y_{<t}) - \log \pi_\theta(y_t | x, y_{<t})$$
对该 reward 直接应用于两种 rollout 来源生成的所有 token，得到更新方向：
$$\nabla_\theta J_{\text{NP}}(\theta) = \mathbb{E}_{x, z_x, y}\left[\sum_t R_t^{\text{OPD}} \nabla_\theta \log \pi_\theta(y_t | x, y_{<t})\right]$$

**理论机制**（第 3.2 节）：相比标准 OPD，NP-OPD 对负策略轨迹的梯度贡献增加了比例 $\alpha$ 的权重。在假设 $D_{KL}(\pi_n\|\pi^*) > D_{KL}(\pi_n\|\pi_\theta)$ 成立（负策略弱于学生）下，负策略生成 token 的期望 reward：
$$\mathbb{E}_{y_t \sim \pi_n}[R_t^{\text{OPD}}] = -D_{KL}(\pi_n\|\pi^*) + D_{KL}(\pi_n\|\pi_\theta) < 0$$
因此负策略生成的 token 平均获得负 reward，驱动学生降低这些 token 的概率，实现"远离负策略"的效果。且该机制不依赖具体 reward 形式，可直接扩展至 ExOPD、OPD² 等奖励变体。

**关键超参**：$\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$，控制负策略 rollout 的混合比例。

## 实验与结果
**数据集与基准**：训练集 30K 问题（Math:Science:Code = 1:1:1）；评测涵盖 13 个基准（7 个数学、3 个代码、3 个科学），包括 AIME24/25、AMC23、HMMT25、MATH500、OlympiadBench、RGMath、CodeForces、LCBv5、RGAlgo、GPQA、SuperGPQA、SciBench。

**主要结果（非-thinking 模式，α=1.0）**：
- **Qwen3-1.7B**：Math 平均 48.3→56.3（+8.0），Code 平均 27.2→31.1（+3.9），Science 平均 37.2→41.5（+4.3）；AIME24 从 31.7 提升至 43.1。
- **Qwen3-4B**：Math 平均 62.2→70.5（+8.3），Code 平均 40.2→51.1（+10.9），Science 平均 47.8→51.2（+3.4）。
- **Qwen3-8B**：Math 平均 65.1→69.4（+4.3），Code 平均 44.0→45.0，Science 平均 50.1→52.6。

**与 OPD 变体的结合**（thinking 模式）：
- Qwen3-1.7B：NP-OPD 与 OPD/ExOPD/OPD² 结合后 All 平均分别提升 +1.98 / +0.30 / +1.76。
- Qwen3-4B：分别提升 +1.11 / +1.86 / +2.10。

**Gemma-4 泛化验证**：Gemma-4-E4B（α=1.0）Math 63.0→66.4（+3.4），Science 49.1→49.8（+0.7）。

**训练效率**：α=1.0 时 Qwen3-1.7B 思考模式训练时间从 9h19m 降至 3h34m（约 2.6× 加速），因负策略 rollout 可预生成复用。

**最强结果**：Qwen3-4B + OPD² + NP-OPD 在思考模式下 Math 平均 76.8（AIME24: 77.9, LCBv5: 62.9）。

## 相关工作脉络
- **OPD（Agarwal et al., 2024; Lu, 2025）**：本文直接扩展的基础方法，OPD 仅以教师为正向参照，NP-OPD 在此基础上注入负向 rollout 信号。
- **DPO（Rafailov et al., 2023）**：同时使用偏好/非偏好响应对提供双向学习信号；NP-OPD 受其启发，但在 OPD 框架中以负策略 rollout 实现类似作用，且不与 reward 设计冲突。
- **ExOPD（Yang et al., 2026）**：通过引入参考策略并进行 reward 外推改进 OPD；NP-OPD 可在 ExOPD 之上叠加使用（不修改其 reward），二者正交。
- **OPD²（Heo et al., 2026）**：利用 teacher-base delta reward 并做分布中心化处理；论文分析发现 OPD² 本身已具有较高 NSP（隐含负信号），NP-OPD 可进一步放大。
- **Lightning OPD（Wu et al., 2026）**：预计算 teacher supervision 实现离线高效训练；NP-OPD 关注的是 rollout 来源的负信号角色，与训练效率优化方向不同。
- **SeqKD / SFT distillation（Kim & Rush, 2016; Guha et al., 2023）**：基于 teacher 生成序列的蒸馏；OPD 的优势在于保留 on-policy 分布，NP-OPD 在此基础上进一步补充负向参照。

## 局限性与未来方向
- **负策略的选取依赖同族小模型**：当前实验主要使用同族更小参数模型（如 Qwen3-0.6B 对 1.7B 学生），跨架构负策略的有效性未充分验证。
- **α 的最优值因任务而异**：不同模型规模和推理领域下最优 α 存在差异（如 Math 偏好较大 α，Code 有时偏好中间值），缺乏统一的自动调参策略。
- **全负策略 rollout（α=1.0）在非-thinking 模式下对 8B 模型 Code 领域略有下降**：说明负信号过强可能损害特定领域的表现，需要更好的混合平衡。
- **尝试从学生自身合成负 rollout（升温、persona prompt、layer drop）效果远低于独立负策略**：表明简单退化不足以替代真正能力更弱的独立负策略模型，如何高效构建高质量负样本仍是开放问题。
- **模型合并（α=0 ⊕ α=1 等权平均）可进一步提升性能**：但合并策略的有效性和泛化性仍需更多探索。

## 研究启发与可借鉴点
1. **rollout 来源作为负信号注入点**：不修改 reward 设计而仅在 rollout 阶段混入负策略轨迹，即可为 OPD 提供负向参考——这一思路可迁移至其他蒸馏/对齐方法中，实现与既有 reward 改进的解耦叠加。
2. **NSP 与 NOR 分析框架**：提出的两个可量化指标为验证"模型是否在学习远离负参考"提供了简洁且可复用的评估工具，可应用于其他偏好/对比学习场景的分析。
3. **负策略 rollout 可预生成的工程价值**：NP-OPD 中负策略轨迹可离线预生成并全训练周期复用，这一技巧对减少 RL-style 训练中 rollout 生成开销具有直接借鉴意义。
4. **模型合并策略**：论文展示对 α=0 和 α=1 模型进行参数平均可在多个领域同时提升性能，为多目标/多设置训练的模型集成提供了可行思路。
5. **对 OPD 变体的新解读视角**：将 ExOPD/OPD² 理解为已含有隐含负信号的观点，启发了从"负信号强度"角度统一分析和改进不同 OPD 变体的可能性。

## 关键术语表
**On-Policy Distillation (OPD)**：学生模型用自己的 rollout 序列进行知识蒸馏，token 级监督来自更强教师，保留 on-policy 分布特性。
**Negative-Policy OPD (NP-OPD)**：本文提出方法，在 OPD 的 rollout 阶段混入低能力负策略轨迹，为学生显式提供负向参照。
**Negative Policy ($\pi_n$)**：能力低于学生的辅助策略，用于生成负向 rollout，使学生在训练中持续接触并远离其输出 token。
**Mixing Ratio (α)**：控制负策略 rollout 在训练批次中所占比例的超参，α=0 为纯 OPD，α=1 为全负策略 rollout。
**Negative-Policy Suppression Precision (NSP)**：被学生降低概率的 token 中，同时被负策略偏好超过教师的 token 所占比例，衡量负向压制目标的精确度。
**Negative-Policy Overlap Reduction (NOR)**：衡量学生训练后相对于教师优先 token 与负策略优先 token 的概率 gap 变化，正值表示学生更接近教师方向。
**ExOPD**：引入参考策略并对 teacher-reference reward 进行外推的 OPD 变体，通过 $\lambda$ 控制外推程度。
**OPD²**：使用 teacher 与其 base 模型的 delta reward，并在学生分布下做中心化处理，仅在 delta 信号与 OPD 方向一致时应用。

## 可复现要素
- **数据集**：训练数据为 Math/Science/Code 混合的 30K 问题（来源包括 OpenScienceReasoning-2、OpenCoderReasoning、Aimo-2 等）；评测使用 AIME24/25、AMC23、HMMT25、MATH500、OlympiadBench、RGMath、CodeForces、LiveCodeBench、RGAlgo、GPQA、SuperGPQA、SciBench。
- **代码**：论文声明代码将开源至 https://github.com/naver-ai/np-opd。
- **模型权重**：使用 Qwen3（1.7B/4B/8B/0.6B/30B-A3B）和 Gemma-4（E2B/E4B/12B）开源模型。
- **关键超参**：batch size=256，学习率=5×10⁻⁶，max rollout length=16K，training steps=100，rollout 温度=1.0，eval 温度=0.7，eval max length=32K，KL 正则化系数 β=0（Qwen）/0.04（Gemma），α∈{0.0, 0.25, 0.5, 0.75, 1.0}。
- **硬件**：Qwen 实验在单节点 8×H100 上完成，使用 vLLM 生成 rollout，ZeRO-3 分布式训练。
