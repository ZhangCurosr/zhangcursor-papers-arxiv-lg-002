---
title: "LOOPED-ACTOR-DEPTH-RECURRENT-REASONINGMODELS-FOR-REINFORCEME"
source: https://arxiv.org/pdf/2609.37432v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:55:16"
field: "强化学习中的自适应计算与规划"
keywords: ["reinforcement learning", "looped transformer", "adaptive computation", "depth-recurrent policy", "fixed-point reasoning", "multistep planning", "goal-conditioned RL"]
innovations: ["理论证明状态自适应计算在MDP族上渐近优于固定运行时策略", "提出Looped Actor将FPRM适配至RL策略，基于KL散度停止准则实现参数效率与自适应计算结合", "揭示循环迭代逐步精炼多步计划表征的机制并验证计算分配的结构性"]
benchmarks: ["Boxoban", "Rush Hour", "OGBench"]
---

# 论文速读：LOOPED-ACTOR-DEPTH-RECURRENT-REASONINGMODELS-FOR-REINFORCEMENT-LEARNING

## 一句话总结
本文提出Looped Actor，一种将深度循环Transformer适配到强化学习策略的自适应计算框架，通过重复应用共享参数块并基于KL散度停止准则动态决定循环次数，在不增加模型参数的情况下显著提升多步规划任务性能，理论上证明了状态自适应计算优于固定运行时策略。

## 研究问题与动机
- 强化学习中不同状态所需的决策计算量差异很大（如复杂规划vs常规执行），但现有策略通常采用固定计算预算。
- 传统显式规划方法（如MCTS）计算成本高，而简单循环模型难以根据状态动态分配计算。
- 循环Transformer在语言推理任务中已取得显著进展，但在强化学习序列决策中的应用尚未充分探索。
- 前作Ghugare et al. (2026) 证明了增加计算可提升性能，但状态依赖计算分配仍是开放问题。

## 核心贡献（创新点）
- **理论证明状态自适应计算优势**：构造一族MDP，证明在相同描述长度约束下，状态自适应策略可实现相同最优回报但渐近更低的期望计算量，优势随规模无界增长。
- **提出Looped Actor框架**：首次将Fixed-Point Reasoning Model (FPRM) 适配到强化学习动作选择，共享Transformer块重复迭代直至输出分布稳定，实现参数效率与自适应计算的结合。
- **设计KL散度停止准则**：以相邻循环动作分布的KL散度低于阈值作为停止条件，而非原FPRM的隐状态残差收敛，使策略能根据置信度动态决定计算量。
- **系统性实验验证**：在22个任务（6个环境，含在线/离线、离散/连续动作）上验证，Looped Actor以1/16参数匹配或超越Iso-FLOPs基线，Boxoban成功率约两倍，且>95%轨迹在16圈前停止，实现60%+推理加速。
- **揭示迭代计算的规划机制**：Latent probe显示 successive loops 逐步细化最优计划表征，近端动作预测提升更显著，且成功episode中循环数与剩余推箱子数单调相关。

## 方法详解
- **基础架构**：基于FPRM，给定嵌入输入x，共享Transformer块 $f_\theta$ 迭代更新隐状态至不动点 $\mathbf{z}^\star = f_\theta(\mathbf{z}^\star; \mathbf{x})$。
- **稳定迭代设计**：Stack of $K$ Transformer blocks，每子层应用 $\mathbf{h}^k = \alpha_1 \mathbf{h}^{k-1} + \beta_1 g_{\theta^k}^k(\text{Norm}(\mathbf{h}^{k-1}))$，输入嵌入通过 $\alpha_2 \mathbf{h}^{2K} + \beta_2 \mathbf{x}$ 重注入，可学习系数满足 $\beta_2 = 1 - \alpha_2 \alpha_1^{2K}$，保证迭代有界。
- **停止准则**：第$\ell$圈输出分布 $p_\ell$ 与第$\ell-1$圈KL散度 $D_{\text{KL}}(p_{\ell-1}\|p_\ell) < \tau$ 时停止，$\tau=10^{-3}$。
- **训练方式**：无截断BPTT，不依赖deep supervision；在线RL用PPO，离线goal-conditioned RL用GCIQL+DDPG+BC。
- **策略输出**：在停止圈$\ell_\theta(s_t,g)$处的输出分布作为动作采样分布 $\pi_\theta(a_t|s_t,g)=p_{\ell_\theta}(a_t)$，停止本身不可微但梯度通过已执行的循环传播。

## 实验与结果
- **环境**：Boxoban（10×10推箱子，900k训练）、Rush Hour（6×6谜题，2M训练）、OGBench（4个tabletop manipulation环境，20个离线目标条件任务）。
- **基线**：Iso-Parameters（单次应用同块）、Iso-FLOPs（16次堆叠不共享，参数16倍）、DRC(3,3)（仅Boxoban/Rush Hour）。
- **主要结果**：
  - Boxoban：Looped Actor成功率约Iso-FLOPs的2倍，远超Iso-Parameters（几乎无学习）。
  - OGBench整体平均成功率：Looped Actor 69.7% vs Iso-FLOPs 68.5% vs Iso-Parameters 33.0%，在Cube triple single pnp等任务上显著提升。
  - 自适应计算：>95%轨迹在16圈前停止，相对固定16圈推理加速60%+；固定4/8圈性能大幅下降。
  - Latent probe：从loop 1到16，下一步最优动作解码准确率从0.50升至0.75，第4步从0.30升至0.45，表明迭代逐步精炼多步计划。
  - 计算分配：成功episode中循环数与剩余推箱子数几乎单调相关（1~19步从6圈增至>10圈），失败episode无此相关性。

## 相关工作脉络
- **Universal Transformer (Dehghani et al., 2018)**：首个循环Transformer，针对可分解为子程序的算法任务设计，本文将其思想拓展至RL策略。
- **FPRM (Movahedi et al., 2026) / Equilibrium Reasoners (Huang et al., 2026)**：基于吸引子动力学的循环推理模型，本文取其框架但将停止准则改为输出分布稳定并适配RL。
- **Deep Repeated ConvLSTM (Guez et al., 2019) / DRC**：深度循环策略展现涌现规划，但未解决状态依赖计算分配问题，本文用Transformer替代ConvLSTM并实现自适应深度。
- **Ghugare et al. (2026)**：证明更多计算提升性能且泛化至更长 horizon，指出状态依赖计算分配是开放问题，本文直接回应该问题。
- **Thinker (Chung et al., 2023) / Value Iteration Networks (Tamar et al., 2016)**：显式规划或可微价值迭代，需预设算法结构；Looped Actor无需预设规划算法，隐式学习迭代计算。
- **Looped World Models (Lu et al., 2026)**：将自适应循环应用于转移预测而非动作选择，本文聚焦策略学习。

## 局限性与未来方向
- 理论构造不能保证Looped Actor在实践中学到计算最优策略，计算-奖励Pareto前沿仍有优化空间。
- 仅考虑步内计算，未探索跨环境步骤的内部推理记忆，限制了部分可观测性场景的应用。
- 最大循环数16为硬上限，实际分布虽右偏长尾但极端困难状态可能仍需更多迭代。
- 未系统研究halting threshold τ、残差缩放系数等超参对计算分配与性能的影响。

## 研究启发与可借鉴点
- **KL散度停止准则**可迁移至其他需动态计算的任务（如持续推理、多阶段决策），作为轻量级置信度度量。
- **理论+实证双验证**范式值得借鉴：先构造理论动机再设计算法，增强方法说服力。
- **Latent probe解码最优计划**的分析方法可用于诊断循环模型的规划能力形成过程。
- **参数共享+自适应深度**的设计可在视觉-语言动作控制、机器人操作等需多步推理的RL任务中复现。
- **将推理模型（FPRM）与RL策略学习结合**的思路可扩展至world model learning或offline RL的actor-critic架构。

## 关键术语表
- **Looped Actor**：本文提出的基于深度循环Transformer的强化学习策略，通过重复应用共享计算块直至动作分布稳定实现自适应计算。
- **Fixed-Point Reasoning Model (FPRM)**：基于吸引子动力学的循环Transformer，迭代更新隐状态至不动点以完成推理。
- **State-adaptive policy**：根据输入状态动态分配计算量的策略，简单状态少计算、困难状态多计算。
- **Iso-FLOPs baseline**：计算量与Looped Actor相当但不共享参数的基线，参数量为后者的16倍。
- **KL Halting Criterion**：以相邻循环输出分布的KL散度低于阈值作为停止迭代准则。
- **Multistep planning**：需前瞻多步才能做出正确动作的决策过程，如Boxoban中的推箱子路径规划。
- **OGBench**：离线目标条件强化学习基准，包含多个tabletop manipulation环境用于评估策略泛化。
- **Depth recurrence**：通过重复应用同一模块增加有效深度而不增加参数的设计模式。

## 可复现要素
- **数据集**：Boxoban（900k train/100k val/1k test）、Rush Hour（2M/10K/10K）、OGBench（state-based play datasets），公开可用。
- **代码**：已开源，URL https://github.com/camail-official/LoopedActor。
- **关键超参**：隐藏维度128，每块2层Transformer，4个注意力头，MLP扩展比4，最大循环数16，halting阈值τ=10⁻³，学习率10⁻⁴（PPO）/3×10⁻⁴（GCIQL），梯度裁剪全局范数1.0，batch size 1024。
- **框架**：JAX实现，在线RL用PPO，离线RL用GCIQL+DDPG+BC。
