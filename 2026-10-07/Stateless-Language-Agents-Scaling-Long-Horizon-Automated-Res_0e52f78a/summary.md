---
title: "Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res"
source: https://arxiv.org/pdf/2610.07625v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:26:36"
field: "自动化科学研究（AutoResearch）"
keywords: ["自动化研究", "无状态智能体", "长周期搜索", "多智能体协调", "LLM agent", "token 效率"]
innovations: ["提出有状态搜索+无状态智能体的框架原则，将研究状态与上下文严格分离", "Harness 控制的上下文重建：Advisor 接收跨方向证据摘要，Worker 仅见分配与局部反馈", "证据驱动的工作分配机制，防止并行 Worker 重复劳动且协调开销<0.6% token"]
benchmarks: ["FrontierSWE", "Anthropic VLIW SIMD Kernel Optimization", "SOL-ExecBench", "FrontierCS Structured-LWE"]
---

# 论文速读：Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res

## 一句话总结
本文提出无状态语言智能体（SLA）框架，核心思想是"有状态搜索、无状态智能体"：研究状态（候选解与评估结果）由外部 harness 持有，每次调用时为 Advisor 和 Worker 重建独立、角色特定的上下文，从而解决长周期自动化研究中上下文膨胀、工作重复和执行停滞等问题。在高达 10 亿 token 的预算下，SLA 在所有任务上均取得最佳结果，且以 84%–93% 更少的 token 达到最强基线性能。

## 研究问题与动机
1. **长周期推理不等于持续进步**：现有自动研究系统（如 CORAL、SwarmResearch）在多步编码智能体中累积越来越长的对话历史，导致 token 消耗增加但有效实验减少；例如 CORAL 在 GPU kernel 任务中，最后改进出现在 231M tokens 之后，98% 的最终 1500 次会话未做任何工具调用。
2. **现有评测预算过短，基准过早饱和**：大多数 AutoResearch 评测以迭代或模型调用次数衡量预算，且基准（如 26-circle packing）在 5M tokens 内即趋于饱和，无法检验上下文管理、实验选择等设计选择在长周期中的复利效应。
3. **历史记录复用与上下文控制之间的矛盾**：候选解、失败尝试是可复用的证据，但反复回放增长的历史会消耗 token 并 degrade 智能体行为；如何在保留可用证据的同时控制每个智能体的输入上下文是关键挑战。
4. **积累的证据未能转化为 productive 探索**：并行智能体往往重复实现相同功能（如 CORAL 中两个智能体从同一父代实现同一 feature），已有记录不决定下一步该做什么，缺乏协调机制。

## 核心贡献（创新点）
1. **提出"有状态搜索、无状态智能体"框架原则**：将研究状态与智能体上下文严格分离，harness 拥有候选解和评估结果，每次调用时从零重建角色特定的上下文，而非让智能体累积对话历史。
2. **Harness 控制的上下文重建机制**：Advisor 接收跨搜索方向的结构化证据摘要（区分"已测量的未改进"与"实现/运行时/正确性不确定的失败"），每个 Worker 仅接收其分配任务、保留候选和局部反馈，二者均不可访问完整历史记录或其他 Worker 的工作空间。
3. **基于证据驱动的工作分配机制**：Advisor 利用全局证据将实验转化为具体分配，指导并行 Worker 探索不同方向或集中攻关瓶颈，Worker 间不再依赖共享日志自行协调；消融实验表明移除分配后 Worker 会重复实现相同 feature。
4. **系统性地论证了短周期评测的误导性**：在 FrontierSWE 上，SLA 在 175M tokens（25% 预算）时落后于最强基线，但在 700M tokens（全预算）时反超；移除 Worker 隔离的成本从 100M-token 延续的 5–11 个周期放大至全 1B-token 运行的 224 个周期，证明评测需要多预算点报告。
5. **证明显式协调开销极低**：Advisor 消耗不到总 token 的 0.6% 和模型成本的 2.3%，远少于 SwarmResearch Shepherd 的 8.4%–10.7% token 和 6.1%–7.4% 成本，且 Worker 模型可与 Advisor 模型解耦独立选择。

## 方法详解
1. **状态与上下文的分离**：引入两个概念——**state**（持久化的研究状态，由 harness 持有：候选解、评估结果、全局最优）和**context**（单次调用中智能体可见的内容）。每次 Advisor epoch 和每次 Worker trial 均为一次独立调用，启动全新会话，无任何跨调用状态传递。
2. **Advisor 上下文重建**：每个 epoch，harness 从研究状态构建结构化证据摘要，包括：当前全局最优代码与分数、按搜索方向分组的最近尝试与结果、Worker 状态、最近进展趋势，以及可用时的任务特定诊断。摘要明确区分"已测量的非改进"（negative performance evidence）与"实现/运行时/正确性失败"（inconclusive failure），避免将执行失败误判为某方向已穷尽。
3. **Worker 上下文隔离**：每次 trial，Worker 仅收到：任务规范、当前分配、其保留的局部候选、以及上一次 trial 的状态/分数/正确性诊断。Worker 不可访问其他 Worker 工作空间或完整研究历史；来自其他方向的证据只能通过 Advisor 分配间接获取。
4. **Evidence-driven 工作分配**：Advisor 根据全局证据决定实验选择——可分散 Worker 到不同方向，也可集中多个 Worker 攻关同一瓶颈但采用不同实现策略。当 Worker 方向被重定向时，其候选重置为全局最优，首个成功评估的候选成为新局部基线。harness 保留各 Worker 局部候选的改进（即使落后于全局最优），给予替代方案成熟时间。
5. **实现保障**：使用 Codex 或 Claude Code；Advisor 每 epoch 启动新会话；Worker 以 Linux Landlock 文件系统限制、受限网络访问运行，session 在 trial 间清除；评估在 Worker 环境外运行，评估代码受保护不可被修改；harness 按角色记录 token 用量并支持 checkpoint 恢复。

## 实验与结果
- **数据集与任务**：(1) FrontierSWE 软件工程（libexpat→x86-64 Assembly、Git→Zig、Dart→Haskell、Lua Native Compiler）；(2) Anthropic VLIW SIMD kernel 优化 + NVIDIA SOL-ExecBench（#1 attention softmax/dropout/value matmul、#58 MoE token sorting、#210 fused residual add+RMS norm）；(3) FrontierCS Structured-LWE 密码学算法设计。
- **基线**：EvoX、CORAL、SwarmResearch，均使用相同编码智能体（Codex+GPT-5.5 或 Claude Code+Opus 4.8）、相同 seed 解和受保护评估器。
- **主要结果**：
  - **Anthropic Kernel（Codex）**：SLA 1B tokens 下达 1112.0±8.9 cycles，最强基线 SwarmResearch 为 1275.7±133.9；SLA 以 67.9M tokens（比 SwarmResearch 少 93.1%）达到最强基线最终性能；三组独立运行全部超越所有九组基线运行（最差 SLA=1122 vs 最好基线=1191）。
  - **FrontierSWE（Codex）**：SLA 在 700M tokens 时四个任务均领先，平均比最强基线高 4.95 分；在 175M tokens 时落后但全预算时反超。
  - **SOL-ExecBench**：SLA 平均提升 5.5%；在 #58 上以 14.5M tokens 达到最强基线目标（减少 95.9%）。
  - **FrontierCS Structured-LWE**：SLA 200M tokens 下得分 59.5，比最强基线高 2.6%。
- **消融结果（共享 checkpoint 延续）**：移除 Advisor 上下文重建、Worker 上下文隔离、或 Advisor 分配均降低各任务的平均进展；效果在长周期中复利放大——Worker 隔离的成本从 100M-token 延续的 5–11 周期增至全 1B-token 运行的 224 周期。
- **协调开销**：Advisor 消耗 0.24–0.51% tokens 和 1.2–2.3% 成本；SwarmResearch Shepherd 消耗 8.39–10.70% tokens 和 6.14–7.35% 成本。

## 相关工作脉络
1. **LLM-guided 进化搜索**：FunSearch、AlphaEvolve、EvoX 等方法将 LLM 生成的程序嵌入进化搜索循环，每个候选由一次或少数几次模型调用产生；SLA 使用多步 coding agent 替代直接 LLM 调用，且将研究状态移至 harness 而非嵌在智能体上下文中。
2. **多智能体 AutoResearch 框架**：CORAL 运行并行 agent 并通过持久化记忆和心跳触发反射共享发现，但每个 agent 维持一个长运行会话；SLA 将历史记录移出 agent 会话，由 harness 控制每个 agent 可见内容。
3. **SwarmResearch**：使用 Shepherd agent 通过亲缘选择、agent 类型选择和选择性上下文共享协调 Search Agents，但被指示不分配具体想法；SLA 的 Advisor 直接分配具体实验方向，且无长上下文累积。
4. **AI Scientist 系统**：The AI Scientist、Agent Laboratory 等自动化多个科研阶段（假设生成、文献综述、实验、写作）；SLA 聚焦于可执行方案的开放搜索，与 AI Scientist 可组合（SLA harness 可作为 AI Scientist 中的实验引擎）。
5. **内存与上下文工程**：Dynamic Cheatsheet、ACE、ReasoningBank 等方法通过记忆/上下文积累跨任务经验；SLA 关注单任务内长达 1B token 的持续搜索，经验是问题特定的候选-结果记录。
6. **Context Rot 问题**：Hong et al. 的 Context Rot 工作指出增长输入 token 损害 LLM 性能；Liu et al. 发现长 session 的 agent 最终落后于独立采样；SLA 通过将每次调用限制在紧凑上下文中直接规避此问题。

## 局限性与未来方向
1. **多数配置仅单次运行**：因长周期运行成本高昂（数百至数千美元），除 Anthropic kernel 与 checkpoint 消融外，其他任务/配置均为单次运行，统计显著性受限。
2. **仅限可执行评估器的任务**：所有任务均有快速自动评估器，模糊目标或慢评估目标的适用性未检验。
3. **消融未改变状态lessness 本质**：消融仅改变每个智能体看到的内容，而非是否跨调用保留对话；无状态性的因果贡献需更多验证。
4. **未来方向**：将 SLA 的长运行序列作为强化学习 episode 用于 post-training；拆分 Advisor 角色至多 Advisor 并行；将跨任务记忆（如 ACE playbook）作为 Advisor 上下文的种子。

## 研究启发与可借鉴点
1. **状态/上下文分离的设计范式**：将持久化研究状态与单次调用上下文严格分离，可推广至任何需要长周期运行的多智能体系统，避免因对话累积导致的性能退化。
2. **多预算点评测的重要性**：短周期评测会误判方法优劣（如 FrontierSWE 上的 SLA），后续评测应报告多个 cumulative token 预算点下的结果，揭示方法排名随预算变化的趋势。
3. **Advisor-Worker 解耦的架构灵活性**：协调与实现角色分离后，可为不同任务独立选择最优 Worker 模型（如 GPT-5.4 mini 在 Git→Zig 上优于 GPT-5.5 在同等成本下），为成本-性能优化提供维度。
4. **证据摘要的细粒度设计**：区分"已测量的非改进"与"不确定的失败"比简单记录"好/坏"更有价值，避免将实现错误误判为方向穷尽；此思路可用于其他搜索系统中的反馈处理。
5. **从长运行中抽取 RL 训练数据**：SLA 的每次 Worker trial 是 bounded context 的 episode 带 evaluator score，天然适合直接用于 post-training 的强化学习，无需训练模型处理 1B-token 级长上下文。

## 关键术语表
**Stateless Language Agent（SLA）**：每次调用启动全新会话、不跨调用保留任何对话状态的智能体，研究状态由外部 harness 持有和管理。
**Harness**：SLA 框架的外部控制系统，拥有全部研究状态（候选解与评估结果），负责重建每次调用的上下文并记录 token 用量。
**Advisor**：SLA 中的全局协调智能体，接收跨搜索方向的证据摘要，负责任务分配和搜索方向决策，每次 epoch 从零开始。
**Worker**：SLA 中的并行执行智能体，仅接收分配任务与局部反馈，在隔离工作空间中进行多轮 trial 实现改进。
**Cumulative Tokens**：以累计输入+输出 token 数衡量的搜索预算，比迭代次数或墙钟时间更可控、可复现的 scaling axis。
**Context Reconstruction**：每次调用时从研究状态重建智能体可见内容的过程，使智能体所见成为显式设计选择而非历史累积产物。
**Worker Isolation**：限制 Worker 仅可见自身分配、局部候选和反馈，不可访问其他 Worker 工作空间或完整历史记录的上下文设计。
**Evidence-Driven Assignment**：Advisor 基于跨方向证据将实验转化为具体、可执行分配的工作协调机制，防止并行 Worker 重复劳动。

## 可复现要素
- **数据集**：FrontierSWE、Anthropic VLIW SIMD kernel optimization、SOL-ExecBench、FrontierCS Structured-LWE；论文声明使用 pinned upstream release 的官方实现与默认配置。
- **代码/权重**：论文未提及开源 SLA 代码或模型权重；Baseline（EvoX、CORAL、SwarmResearch）使用官方 pinned release。
- **关键超参**：Worker 数量 W=3（默认），每 epoch 每 Worker 3 次本地 trial；Codex CLI v0.152.1 + GPT-5.5 或 Claude Code v2.1.258 + Claude Opus 4.8；reasoning effort 设为 high；上下文压缩使用各 agent 默认设置。
- **Token 预算**：Anthropic kernel 1B tokens；FrontierSWE 700M tokens/task；SOL-ExecBench 500M tokens/task；FrontierCS 200M tokens。
- **运行次数**：Anthropic kernel + Codex 3 次独立运行（均值±标准差）；其他配置均为单次运行；消融实验从 10% 和 20% checkpoint 延续各 3 次。
