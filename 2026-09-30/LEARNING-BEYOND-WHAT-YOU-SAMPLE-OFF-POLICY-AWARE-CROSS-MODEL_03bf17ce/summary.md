---
title: "LEARNING-BEYOND-WHAT-YOU-SAMPLE-OFF-POLICY-AWARE-CROSS-MODEL"
source: https://arxiv.org/pdf/2609.37868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:52:28"
field: "大语言模型强化学习"
keywords: ["RLVR", "GRPO", "cross-model learning", "off-policy", "trajectory exchange", "complementary success"]
innovations: ["互补且平衡的全失败组分组合交换策略", "源计算优势保留与序列级兼容性门控双重控制", "peer-last训练顺序激活token级裁剪机制"]
benchmarks: ["MATH500", "AIME2024", "AIME2025", "AMC23", "Minerva"]
---

# 论文速读：LEARNING BEYOND WHAT YOU SAMPLE: OFF-POLICY-AWARE CROSS-MODEL TRAJECTORY EXCHANGE FOR RLVR

## 一句话总结
本文提出 GRAFT 框架，通过异质模型间互补轨迹交换解决 RLVR 训练中“全失败组”缺乏策略梯度信号的问题；在保持相同 rollout 预算下，利用兼容性门控与 token 级重要性裁剪控制跨模型不匹配，较 GRPO 平均提升 2.1 分。

## 研究问题与动机
- **全失败组导致 RLVR 学习信号中断**：GRPO 等方法依赖自生成轨迹的奖励方差计算优势函数，当某 prompt 的所有响应均失败时（advantage 全为零），该组无法提供有效的策略梯度更新信号。
- **增加 rollout 预算成本高昂且存在瓶颈**：虽然增大组大小（n）可提高采样到成功轨迹的概率，但线性增加计算成本，且 RLVR 本质受限于 base model 自身可解的 prompt 集合边界。
- **现有跨模型方法存在设计缺陷**：HACPO 在接收方成功的 prompt 上也共享 peer rollout，引入不必要噪声；SGT 仅传输单一成功响应且使用固定权重的监督损失，缺乏基于兼容性的自适应权重控制。
- **异质模型存在互补成功模式**：不同 pretraining history 的模型在独立 GRPO 运行中表现出显著的互补性（如 SmolLM3-3B 解决 Qwen3-1.7B 全失败的 47.9% prompts），为免 teacher 的相互学习提供了机会。

## 核心贡献（创新点）
- **提出互补且平衡的分组选择策略**：仅当接收方全失败且 peer 组包含成功与失败响应时才触发交换，并通过 min(|C_AB|, |C_BA|) 平衡双向交换量，避免单向信息过载。
- **设计源计算优势保留机制**：用 peer 组的完整响应替换接收方失败组，保留 peer 内部计算的奖励对比优势（而非 pooling 跨模型 reward），维持组内正负信号对比。
- **引入序列级兼容性门控与 token 级裁剪的双重控制**：通过平均 token log-likelihood 差值定义兼容性分数 s(o|q)，结合阈值 δ 过滤低兼容轨迹，并用 token-level importance ratio clipping 限制接收方策略更新幅度。
- **提出 peer-last 训练顺序**：将包含 peer 轨迹的 minibatch 置于接收方自有 on-policy minibatch 之后，确保首次优化时 clipping 机制生效，使 receiver 数据始终优先。
- **验证存储轨迹复用可行性**：证明从已完成独立 GRPO 运行中存储的 peer 轨迹仍可有效利用，无需同步 co-training，平均保持在线增益的 84%。

## 方法详解
**整体框架**：两个异质策略 π_θ^A 和 π_θ^B 在同一 prompt 分布上同时训练，共享验证二元奖励，仅交换采样响应、生成 log-probability 和 reward 信息。

**互补分组选择（Section 4.1）**：
- 定义候选集 C_{A→B} = {q ∈ Q: k_B(q)=0 ∧ 1≤k_A(q)<n}，其中 k_M(q) 为模型 M 在 prompt q 上的成功响应数。
- 平衡策略：取 m = min(|C_{A→B}|, |C_{B→A}|)，按源模型成功数降序排名并保留前 m 个（含边界并列），双向对称处理。
- 分组替换：对选中 prompt，用源模型整个 rollout 组 G_A(q) 替换接收方的失败组，每个响应携带源计算的 advantage â_i^A。

**离线感知 peer 更新（Section 4.2）**：
- **兼容性门控**：定义平均 token log-likelihood l̄(π, o^M|q) = (1/|o^M|) Σ_t log π(o_t^M|q, o_{<t}^M)，兼容性分数 s(o|q) = exp(l̄(π_θ^B_old, o^B|q) - l̄(π_φ, o^A|q))，权重 w(o|q) = 1[s(o|q)>δ]·min{s(o|q), 1}，阈值 δ=0.8 过滤低兼容响应。
- **Token 级重要性比裁剪**：对 grafted 响应，用接收方 tokenizer 重 tokenize，计算 ρ_{j,t}(θ^B) = π_θ^B(o_{j,t}^B|q_j, o_{j,<t}^B)/π_θ^B_old(o_{j,t}^B|q_j, o_{j,<t}^B)，分母始终为接收方行为策略。
- **优化目标**：I(θ^B) = E_B[(1/Σ|o_j^B|) Σ_j w(o_j) Σ_t min(ρ_{j,t}â_j, clip(ρ_{j,t}, 1-ε_low, 1+ε_high)â_j)]，advantage 和权重在优化中固定。
- **Peer-last 更新顺序**：含 peer 组的 minibatch 排在 receiver 自有 minibatch 之后，确保首次优化时 ρ≈1 使 clipping 生效。

## 实验与结果
**实验设置**：
- 三对异质模型：Pair 1 (SmolLM3-3B ↔ Qwen3-1.7B)、Pair 2 (OctoThinker-3B ↔ Qwen3-1.7B)、Pair 3 (SmolLM3-3B ↔ OctoThinker-3B)。
- 五个数学推理基准：MATH500、AIME2024、AIME2025、AMC23、Minerva，报告 pass@1 及五基准均值。
- 训练配置：verl+FSDP+vLLM，n=8 rollouts/prompt，learning rate 10⁻⁶，epochs=3，compatibility threshold δ=0.8。

**主要结果（Table 1）**：
- GRAFT 在所有六组模型对比中均优于 GRPO(n=8)，平均提升 +2.1 分，最高 Pair 1 SmolLM3 提升 +4.46 分。
- Pair 1 两模型均超过 GRPO(n=32)（4倍 rollout 预算），Pair 2 与 GRPO(n=32) 相当。
- 相比基线：平均优于 HACPO 4.0 分、SGT 1.5 分。
- **最强结果**：SmolLM3-3B (Pair 1) 达到 37.06 分（MATH500:76.86, AIME2024:14.42, AIME2025:14.50, AMC23:50.87, Minerva:28.68）。

**计算效率（Section 6.1）**：
- Pair 1 总 GPU-hours 40.9 小时达到 pair-mean 35.25 分，比 GRPO(n=32) 高 1.18 分且成本仅 45%。

**存储轨迹复用（Section 6.2）**：
- 使用已完成 GRPO(n=8) 运行的存储 peer 轨迹，六组均优于 GRPO(n=8)，平均 +1.78 分（在线增益的 84%）。

**消融实验（Table 2a）**：
- 移除 compatibility gate 导致 SmolLM3 下降 8.36 分（最大贡献）。
- Peer-last 顺序优于 peer-first 和 uniform。
- δ=0.8 为最优阈值。

## 相关工作脉络
- **GRPO 及零方差处理**：DAPO 等通过动态重采样或 entropy-guided advantage shaping 恢复信号，但仍依赖接收方自身探索，不扩展可解 prompt 集合。
- **HACPO**：广泛复用 peer rollouts 并用 sequence-level importance sampling 控制 mismatch，但未限制于接收方失败 prompt，本文证明其在不平衡场景下表现劣于 GRAFT。
- **SGT (Mutual RL)**：仅在接收方失败时传输单一成功 peer 响应并用固定权重 λ=0.1 的 SFT loss 学习，缺乏兼容性加权与完整组优势对比，本文 ablation 显示其平均落后 1.5 分。
- **F-TIS**：同家族模型协作 GRPO，使用共享 vocabulary 下的 truncated importance sampling，本文面向异质 tokenizer 场景，兼容性分数为 empirical proxy 而非精确 density ratio。
- **LUFFY-style off-policy correction**：用 shaping function f(x)=x/(x+γ) 替代 clipping，本文消融显示其落后 GRAFT 4.55/2.03 分（Pair 1 两模型）。
- **交叉模型 RL 趋势**：从 teacher-student distillation 转向 peer-to-peer mutual learning，本文定位为解决“何处交换”与“如何学习”的联合设计问题。

## 局限性与未来方向
- **增益依赖模型互补性**：Pair 3（SmolLM3 ↔ OctoThinker）增益最小（+0.66/+1.42），互补性弱时效果有限。
- **兼容性分数为代理指标**：跨 tokenizer 场景下 s(o|q) 是基于平均 token log-likelihood 的经验代理，非精确的 cross-tokenizer importance ratio。
- **仅验证两模型对**：未研究多 peer 协同交换场景（>2 models）。
- **仅数学领域验证**：局限于可验证奖励的数学推理任务，未拓展至无 verifiable reward 领域（如开放生成、对话）。
- **模型规模限制**：仅测试 ≤3B base models，大模型（>7B）下的行为未知。
- **未来方向**：扩展至多 peer 协作、无验证奖励任务、更大规模模型及更通用领域。

## 研究启发与可借鉴点
- **全失败组作为天然交换触发器**：将 receiver all-fail 且 peer 有方差作为交换条件，既保证学习信号必要性，又避免干扰已有 on-policy 信号的 prompt。
- **源计算优势保留 vs pooling**：跨模型交换时保留源组内 reward contrast 优于合并计算 advantage，避免跨模型 reward 尺度不一致引入噪声。
- **序列级过滤+Token 级裁剪的双重控制**：将跨模型 mismatch 分解为序列兼容性问题（门控）和策略更新幅度问题（clipping），分离处理比单一机制更有效。
- **存储轨迹离线复用模式**：在线 co-training 后可用存储轨迹进行单侧学习，节省 27-76% GPU-hours，适用于资源受限场景或事后分析。
- **Peer-last 训练顺序的工程价值**：简单的 minibatch 排序即可激活 clipping 机制，无需修改优化器或损失函数，易于集成到现有 RLVR 框架。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：基于可验证奖励的强化学习，通过精确答案验证（如数学推导）提供二元 reward 信号。
- **GRPO (Group Relative Policy Optimization)**：组相对策略优化，利用组内响应的相对奖励计算优势函数，无需 value network。
- **All-fail group**：全失败组，指某 prompt 的所有 n 个 rollout 响应均失败，导致组内 reward 方差为零、advantage 全零。
- **Cross-model mismatch**：跨模型不匹配，指不同模型因参数、tokenizer、pretraining 差异导致的策略分布偏移。
- **Compatibility gate**：兼容性门控，基于接收方与源模型对 peer 响应的平均 token log-likelihood 差值定义的序列级过滤机制。
- **Token-level importance ratio clipping**：token 级重要性比率裁剪，对接收方 tokenizer 重 tokenize 后的每个 token 计算策略比并施加 clipping bounds。
- **Peer-last update**：Peer-last 更新，将含 peer 轨迹的 minibatch 安排在接收方自有 minibatch 之后的训练顺序策略。
- **Stored peer trajectories**：存储的 peer 轨迹，指从已完成独立 GRPO 运行中保存的响应、log-probability 和 reward 记录，可离线复用。

## 可复现要素
- **数据集**：MATH training split (7,500 problems)，MATH500、AIME2024、AIME2025、AMC23、Minerva 验证集；论文未明确声明数据集公开状态，但均为公开基准。
- **代码开源**：论文未提及代码开源。
- **模型权重**：使用开源 base models（SmolLM3-3B-Base、Qwen3-1.7B-Base、OctoThinker-3B-Hybrid-Base），可从 HuggingFace 获取。
- **关键超参**：n=8 rollouts/prompt，δ=0.8，ε_low=0.2/ε_high=0.28，learning rate=10⁻⁶，epochs=3，minibatch size=32 prompts，max response length=4,096 tokens。
