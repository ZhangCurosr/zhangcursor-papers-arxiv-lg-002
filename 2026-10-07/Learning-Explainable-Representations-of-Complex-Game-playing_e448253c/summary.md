---
title: "Learning-Explainable-Representations-of-Complex-Game-playing"
source: https://arxiv.org/pdf/2610.07638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:08:50"
field: "可解释强化学习与程序合成"
keywords: ["strategy synthesis", "decision transformer", "inductive logic programming", "interpretable policy", "programmatic reinforcement learning", "chess tactics", "Karel"]
innovations: ["将策略合成建模为MDP并用决策Transformer生成离散DSL程序", "把国际象棋战术用一阶逻辑/ILP从新手对局中学习", "对DT做动作掩码/交叉熵/GRU状态嵌入的离散化改造"]
benchmarks: ["Karel tasks (cleanHouse/fourCorners/harvester/randomMaze/stairClimber/topOff)", "Chess beginner positions vs Stockfish 14 and Maia-1600"]
---

# 论文速读：Learning-Explainable-Representations-of-Complex-Game-playing

## 一句话总结
本文提出一种类似人类认知的方式，训练强化学习智能体将学到的策略和政策合成为可执行程序，并展示了该方法在国际象棋和基于网格的任务环境中自动学习这些程序的能力。

## 研究问题与动机
- 人类玩家在玩复杂游戏时会形成抽象概念和策略来解释自己和对手的行为，但掌握优秀策略需要大量练习或专家知识，不易获取。
- 现有工作多关注计算学习算法和策略表示，对学习者能否理解所学策略、哪些因素促进可解释性关注不足。
- 难以有意义地评估学到的策略与玩家心理模型之间的对齐程度，需要借助可解释AI的计算指标作为替代。
- 策略合成缺乏兼顾有效性与高样本效率、且能生成玩家可理解的可执行程序的方法。

## 核心贡献（创新点）
- 提出认知启发的国际象棋策略模型，用一阶逻辑/Prolog表达战术规则，并用覆盖率和发散度评估其有效性。
  - 与以往仅用约束满足或遗传算法生成非可解释策略不同，这里显式建模人类常用的“战术”概念以提升可理解性。
- 将国际象棋策略学习形式化为归纳逻辑编程问题，使用Popper从新手对局数据中学习逻辑规则。
  - 区别在于把策略学成逻辑霍恩子句而非黑盒策略网络，便于玩家对照棋理理解。
- 将程序化策略合成建模为RL问题，并提出基于决策Transformer（Decision Transformer）的离散动作适配方案。
  - 以往神经程序合成多依赖搜索/梯度，这里用序列建模直接把轨迹当条件输入策略。
- 在决策Transformer中加入动作掩码、交叉熵损失、GRU状态嵌入等改造，使其适用于离散DSL程序生成。
  - 与原始处理连续动作的DT不同，改动重点在离散采样与合法性约束。
- 在Karel域上证明该方法与SOTA程序策略学习方法LEAPS性能相当，且搜索/使用的候选程序数显著更少。
  - 强调样本效率与程序空间探索量的优势，而不只是最终回报。

## 方法详解
- 策略合成视角：把策略模型当作政策参数化θ，合成算法即优化过程；并用可解释AI的指标（如认知块数量、模型规模）作为“对齐”的替代度量。
- 国际象棋策略表示：用一阶逻辑谓词词汇表P表达位置特征与移动，形式为`tactic(Position, From, To) ← feature_1(...), ..., feature_n(...)`。
- 评估指标：
  - Coverage：策略可应用的局面比例 `|P_A| / |P|`。
  - Divergence：策略输出与参考动作之间的加权距离，`Div_E(σ, P) = (1/|P_A|) Σ_{(s,a1)∈P_A} Σ_{a2∈A(s)} σ(a2|s) d_E(s, a1, a2)`。
- ILP学习流程：给定正例E+、负例E-和背景知识B，学习假设H使`H ∪ B`尽量蕴含正例、不蕴含负例；使用Popper并约束产生合法着法、剪枝永不适用或召回低于阈值的规则。
- 程序化策略合成的RL形式化：DSL `G=(V,Σ,R,S)`，合成MDP的状态为部分展开的程序序列，动作是应用产生式规则，终态奖励等于执行环境奖励，否则为0；折现因子设为1。
- 决策Transformer改造：
  - 轨迹表示为`(R̂1,s1,a1,...,R̂T,sT,aT)`，reward用回报-to-go。
  - 动作掩码：非法动作logit置为−∞后softmax。
  - 损失：离散动作改用交叉熵而非L2。
  - 采样：从动作分布中采样而非回归层直接输出。
  - 状态嵌入：用GRU对状态序列编码，提升对MDP顺序结构的表征。

## 实验与结果
- 国际象棋：
  - 数据：1297个由新手对局采样的⟨position, move⟩对； held-out测试集100对。
  - 参考oracle：Stockfish 14、Maia-1600。
  - 结论：以Stockfish 14作为距离度量时，所学策略比随机基线更能模仿训练集。
- Karel程序策略：
  - 训练：50,000个程序样本。
  - 基线：LEAPS。
  - 主要结果（Mean Return [0,1.1]与Unique Programs explored）：
    - cleanHouse：LEAPS 0.16(0.13)/3627，DT 0.23/59
    - fourCorners：LEAPS 0.35(0.00)/9872，DT 0.35/55
    - harvester：LEAPS 0.61(0.21)/11708，DT 0.66/28
    - randomMaze：LEAPS 0.97(0.04)/295，DT 1.0/63
    - stairClimber：LEAPS 0.74(0.49)/298，DT 1.1/49
    - topOff：LEAPS 0.80(0.11)/30278，DT 0.66/63
  - 最强结果与提升幅度：stairClimber上DT达到1.1，超过LEAPS均值0.74（+0.37），同时仅探索49个独特程序（远少于LEAPS的298）；randomMaze上DT达1.0略优于LEAPS的0.97，探索程序63 vs 295。

## 相关工作脉络
- 程序合成领域（形式规则、I/O对、演示、自然语言、MDP奖励等多种规范）中，本文聚焦用MDP奖励+Transformer做神经程序合成。
- LEAPS（Trivedi et al., 2021）：程序策略学习SOTA；本文在Karel上与其对比，强调样本效率与探索量优势。
- 可解释RL/代理模型路线（Speith, 2022; Orfanos & Lelis, 2023; Qiu & Zhu, 2022; Inala et al., 2020; Verma et al., 2018, 2019）：本文用可执行程序作为解释，并与这些工作定位不同，更强调序列建模的训练范式。
- 游戏策略学习：Butler et al. (2017)用约束满足学Nonograms策略；Canaan et al. (2018)用遗传算法学Hanabi策略；Mariño等用DSL学实时策略——本文用DT学习更通用的程序化策略。
- 决策Transformer（Chen et al., 2021）：原文面向连续动作；本文将其改造到离散动作与程序合成场景。
- 归纳逻辑编程：用Popper学逻辑规则，区别于直接神经网络策略拟合。

## 局限性与未来方向
- 仅在国际象棋新手数据和Karel小尺度网格环境上验证，泛化到更大/更复杂游戏环境未充分验证。
- 策略“可理解性/对齐性”仍用计算指标代理，缺少对人类用户实际理解与性能提升的直接测量。
- 国际象棋策略只逼近新手水平，未展示对更强人类或引擎策略的拟合能力。
- 未来计划：在更广泛的游戏环境上基准化决策Transformer与其他程序策略学习方法；测量所学策略的可解释性及其对玩家表现的可测量提升。

## 研究启发与可借鉴点
- 将程序合成明确建模为MDP，并用GPT-style自回归生成程序语句，可在离散动作/代码生成任务中复用。
- 决策Transformer改造三件套（动作掩码、交叉熵、分布采样）适合任意离散策略合成环境。
- 用GRU对状态序列进行嵌入，有助于提升离散/符号环境中时序结构的表征。
- 把“覆盖率+发散度”作为策略可用性与接近参考策略的综合度量，可作为策略学习的轻量评估。
- 与团队方向结合：若需生成可解释游戏/AI行为规则，可先用ILP学逻辑原型，再用DT学更大规模程序化策略。

## 关键术语表
- **Strategy synthesis**：从示例或奖励信号中自动学习可执行的策略表示。
- **Mental model**：人对系统行为与状态的内在理解模型。
- **Inductive logic programming (ILP)**：基于背景知识与正负例学习一阶逻辑规则的符号学习方法。
- **Popper**：基于Prolog的ILP系统，支持从失败中学习并剪枝无效假设。
- **Decision Transformer**：将轨迹作为条件输入、用Transformer做序列建模的策略学习架构。
- **Coverage**：策略能够应用的局面比例。
- **Divergence**：策略输出分布与参考动作之间的距离期望。
- **Karel**：面向教学的网格世界编程语言与环境。

## 可复现要素
- 数据集：
  - 国际象棋：1297个新手对局⟨position, move⟩样本；held-out 100对（论文未明示是否公开）。
  - Karel：50,000个程序（来自Trivedi et al., 2021）。
- 代码/权重：论文未明确声明开源。
- 关键超参：论文未详细列出；提到沿用Chen et al. (2021)的超参与训练流程。
- 基线与工具：Stockfish 14、Maia-1600、Popper、LEAPS。
