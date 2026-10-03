---
title: "MARCO-Multi-Round-Agentic-Reinforcement-for-Conditional-Mole"
source: https://arxiv.org/pdf/2609.36683v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:58:10"
field: "分子生成与优化"
keywords: ["molecular optimization", "reinforcement learning", "instruction following", "multi-turn RL", "chemical AI", "policy optimization"]
innovations: ["验证器接地多轮轨迹强化学习框架 MARCO", "Same-1/Same-5 双预算评估协议统一测试单次与多轮能力", "分层形状奖励聚合属性进展、相似度范围与趋势改进"]
benchmarks: ["MuMOInstruct", "BDP", "BDQ", "BPQ", "HMPQ", "BDPQ"]
---

# 论文速读：MARCO-Multi-Round-Agentic-Reinforcement-for-Conditional-Mole

## 一句话总结
论文将条件分子优化建模为受限的提议-反馈-修订轨迹决策过程，提出 MARCO 框架，通过验证器引导的多轮强化学习训练分子编辑器，使其在单次响应（Same-1）和多轮交互（Same-5）两种预算下均能显著提升属性成功率与结构相似性的综合指标 SR×Sim。

## 研究问题与动机
- 分子优化本质是迭代过程，但现有指令遵循模型均采用单次生成范式，无法在结构内利用验证反馈进行修正。
- 现有方法（如 GRPO、RePO）仅评估单次响应，未利用轨迹级多轮反馈信息优化初始生成行为。
- 迭代任务的验证器反馈信号被浪费：推理时可用多轮反馈，但训练时缺乏对应的轨迹学习方法。
- 需要统一框架：既提升首次响应质量，又保留多轮修订能力，在同一策略下测试两种操作模式。

## 核心贡献（创新点）
1. 将条件分子优化重构为受限反馈条件决策过程，提出验证器接地轨迹奖励聚合方法，综合属性进展、结构相似度、有效性和修订质量四个维度。
2. 提出 Same-1/Same-5 双预算评估协议，在相同策略下分别测试单次响应与多轮修订能力，揭示轨迹训练对初始编辑行为的迁移增益。
3. 设计分层形状奖励函数：属性质量（ clipped directional progress）、相似度质量（range-aware）、成功门控bonus、趋势贡献（improvement/regression terms）。
4. 实验证明轨迹训练可同时提升 Same-1 首次响应质量和 Same-5 多轮交互表现，提供初始化、训练周期、目标数量、公开权重适应四类消融证据。

## 方法详解
**环境反馈机制**：在每轮 t，策略 π_θ 从历史 h_t 采样分子响应 o_t，经解析函数 ψ 提取候选分子 x_t；验证器返回结构化反馈 z_t = (validity, property deltas, similarity, similarity acceptance, invalid info)。

**轨迹回报构造**：
- 属性质量：Q_prop = (1/|P|) Σ clip(Δ_p, 0, c_p)，衡量方向性属性进展
- 相似度质量：Q_sim 为分段函数，接受范围 [δ_low, δ_high) 给予线性奖励，低于下限惩罚低相似度漂移，高于上限惩罚近复制
- 状态质量：q_t = w_prop·Q_prop + w_sim·Q_sim + b_succ·Succ·A_sim
- 趋势贡献：φ_t 从第2轮起计算，[q_t - max_{j<t} q_j - η_imp]_+ 奖励改进，-[q_{t-1} - q_t - η_reg]_+ 惩罚倒退
- 形状回合奖励：r_t = q_t + φ_t - λ_len·1[ℓ_t > ℓ_max]（无效响应固定惩罚 r_inv）
- 无折扣轨迹回报：R(τ) = Σ r_t

**策略优化**：对每个指令采样 K 条轨迹，计算组内相对优势 A_i = (R_i - μ)/(σ + ε)，采用 clip PPO 目标函数 J_MARCO，对模型生成的 token 计算策略比率 ρ_{i,k}， masked 环境反馈文本，KL 正则化到参考策略。

## 实验与结果
- **数据集**：MuMOInstruct 基准，包含 BDP/BDQ/BPQ 三个三目标任务和 HMPQ/BDPQ 四个四目标扩展，含 seen/unseen 指令分割
- **模型**：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-4B-Instruct-2507
- **基线**：Base、SFT、GRPO、RePO
- **关键结果（Same-1，Qwen3-4B，BPQ seen）**：MARCO* SR×Sim=0.315，显著优于 RePO (0.213)、GRPO (0.137)、SFT (0.279)
- **最强结果**：MARCO* 在所有 backbone/split/objective 组合中均获得最高 SR×Sim；Same-5 进一步将 Qwen2.5-3B BPQ seen 从 0.232 提升至 0.358
- **提升幅度**：同轨对比显示，5-turn 训练相比 1-turn 训练在 Same-5 下获得 21.7% 相对提升，Same-1 下 5.2% 相对提升
- **SWS 审计**：MARCO 在所有三元组任务-splits 中获得最高 success-weighted similarity，排除复制行为解释

## 相关工作脉络
1. **传统分子优化方法**（Cheminformatics/RL-based）：关注属性-相似度权衡，但缺乏 LLM 接口和验证器交互范式。
2. **Training-free LLM 方法**（Speak-to-Structure、ChemCrow 等）：推理时使用工具/检索，但不利用轨迹更新策略。
3. **Instruction-tuning 方法**（DrugAssist、Mol-Instructions）：提供 SFT 基础，但未引入多轮反馈强化学习。
4. **GRPO/RePO**：单轮组内对比策略优化，无轨迹累积，MARCO 扩展至多轮并聚合轨迹回报。
5. **MolAct**：两阶段 curriculum + 工具增强 RL，MARCO 聚焦单一分子编辑器且无外部工具调用。
6. **多轮 RL Agent**（RAGEN、ArCHer）：通用交互式环境，MARCO 针对分子优化的特殊反馈结构和化学约束定制奖励。

## 局限性与未来方向
- 验证器依赖预训练属性预测器（BBBP、DRD2 等为黑盒模型），存在预测误差传播风险。
- 训练预算固定为 5 轮，最优交互深度可能因任务而异，缺乏自适应停止机制。
- 仅测试 Qwen 系列 backbone，未验证架构无关性（如 LLaMA、Gemini）。
- 公开权重实验仅在 GeLLM³O 一个 checkpoint 上验证，泛化性待更多基准检验。
- 相似度区间 [δ_low, δ_high) 为固定超参，未讨论不同源分子的最优设置。

## 研究启发与可借鉴点
1. **双预算评估设计**：Same-1/Same-5 分离测试策略的"学习"与"应用"能力，可作为多轮代理训练的通用评估范式。
2. **形状奖励的分层构造**：属性进展 clipped、相似度 range-aware、趋势 improvement/regression 分解，可迁移至其他需要平衡多目标的生成任务。
3. **无效动作处理**：固定惩罚 + 轨迹延续机制，避免单次错误阻断学习信号，适用于容错性要求高的决策过程。
4. **组内相对优势**：沿袭 GRPO 思想但扩展到轨迹级别，减少价值网络依赖，适合资源受限场景。
5. **公开权重适应实验**：验证方法在非本训练分布上的迁移能力，为社区提供可复现基准。

## 关键术语表
- **MARCO**：Multi-Round Agentic Reinforcement for Conditional Molecular Optimization，本文提出的验证器接地多轮强化学习框架
- **SR×Sim**：Success Rate × Similarity，属性成功率与 Tanimoto 相似度的乘积，作为主评估指标
- **Same-1 / Same-5**：单次响应 / 最多五次响应的评估协议，测试同一策略在不同交互预算下的表现
- **Tanimoto 相似度**：基于 Morgan 指纹的分子结构相似性度量，本文用于约束编辑距离
- **Clipped directional progress**：有界属性方向进展，clip(Δ_p, 0, c_p) 防止极端值主导奖励
- **Group-relative advantage**：组内相对优势，(R_i - μ)/σ 消除绝对回报尺度依赖
- **SWS**：Success-Weighted Similarity，仅对有效且非复制的成功候选计算相似度均值
- **Verifer feedback**：验证器结构化反馈，包含有效性、属性增量、相似度及修改建议

## 可复现要素
- **数据集**：MuMOInstruct（由 GeLLM³O 引入，已公开）
- **代码**：https://github.com/euReka025/MARCO-release（已开源）
- **权重**：使用 Qwen2.5/3 官方 checkpoint 及 GeLLM³O-P(6)_Mistral 公开权重
- **关键超参**：actor lr=1e-6，group size K=8，rollout horizon=5 turns，KL coefficient=0.05，w_prop/w_sim=1.0/1.5，α_low/α_copy=1.0/2.0，δ_low/δ_high 未明确给出（论文表格仅标注符号）
