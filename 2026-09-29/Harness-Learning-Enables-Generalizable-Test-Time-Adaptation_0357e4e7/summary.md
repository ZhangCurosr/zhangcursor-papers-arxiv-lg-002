---
title: "Harness-Learning-Enables-Generalizable-Test-Time-Adaptation"
source: https://arxiv.org/pdf/2609.35738v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:28:31"
---

# 论文速读：Harness-Learning-Enables-Generalizable-Test-Time-Adaptation

## 一句话总结
提出 **Harness Learning** 框架，将大模型 Agent 的编排程序（Harness）修订建模为可执行程序上的元学习，通过强化学习训练 Proposer 模型仅凭执行反馈迭代修改代码；在 Reasoning Gym 与多跳 QA 的未见任务上，4B Proposer 的平均单步修订性能超越 35B 教师模型，且单次训练策略可在测试时通过多轮反馈实现稳定自适应。

## 研究问题与动机
1. **Harness 决定 Agent 能力上限**：Agent 由模型与编排其调用、工具交互、信息流的执行代码共同定义，不同任务需不同的编排结构，但手工设计或搜索调参成本高且难以泛化。
2. **现有方法缺乏跨任务适应机制**：当前 Harness 优化多依赖固定提示词搜索、监督模仿或仅在同类数据上微调，无法将“如何从失败反馈中修订代码”提炼为可迁移的通用适应技能。
3. **测试时参数更新成本过高**：传统自适应方法通常在测试阶段更新模型权重，本文希望探索在**冻结 Solver 与 Proposer 参数**的前提下，仅凭执行反馈实现测试时程序级自适应的可行性。
4. **迭代改进的可持续性存疑**：单次修订往往受限于当前报告的质量，如何利用同一策略在多轮反馈中持续优化 Harness，同时保证对全新任务族的零样本迁移，是本文的核心科学问题。

## 核心贡献（创新点）
1. **将 Harness 修订形式化为可执行程序上的元学习**。Proposer 学习的是“从执行反馈到代码编辑”的通用适应规则，而非特定任务的答案模式，这与仅做数据拟合的监督优化或仅搜索固定工作流的启发式方法本质不同。
2. **提出基于可验证奖励的 GRPO 强化学习训练范式**。以修订后 Harness 在独立评分集上的任务得分（含编辑有效性奖励）作为组归一化优势信号，使 Proposer 学会权衡“尝试新结构”与“维持可运行性”，区别于仅依赖人类标注或粗糙 heuristics 的编辑器训练。
3. **实证 4B Proposer 在未见推理族上平均超越 35B 教师**。在 Reasoning Gym 21 个完全隔离的未见族上，RL 训练后单步修订平均分达 0.62，显著高于 Base 的 0.32 与 Teacher 的 0.56，证明小规模模型学习适应规则后可弥补参数量差距。
4. **揭示单步 RL 训练即可涌现多轮测试时迭代能力**。Multistep RL 训练连续修订状态并未带来稳定收益，表明单次修订的强化学习信号已足以让 Proposer 掌握可复用的适应性结构先验，降低了多步 credit assignment 的复杂度。

## 方法详解
- **任务与评估定义**：Task 包含描述、问题集 $Q$ 与 Grader；Harness $h$ 为冻结 Solver 的可执行程序。Harness 质量以 $J(h; Q) = \frac{1}{|Q|}\sum_{q\in Q}\text{score}(h,q)$ 衡量，评分集 $Q^{\text{score}}$ 与反馈集 $Q^{\text{fb}}$ 严格隔离。
- **测试时自适应循环（Inner Loop）**：从种子 Harness $h_0$ 出发，第 $t$ 轮输入 $x_t$ 拼接任务描述、当前 Harness $h_{t-1}$ 及执行报告；Proposer $\pi_\theta$ 采样 $G$ 个修订 $y_i$，`Apply()` 解析代码 diff 并合并到父 Harness，`Select()` 在 $Q^{\text{score}}_t$ 上打分后保留最优候选 $h_t$。全程无参数更新。
- **元学习目标（Outer Loop）**：最大化保留集上的期望最终得分 $\max_\theta \mathbb{E}[J(h_T; Q^{\text{eval}})]$，训练时用单步修订的即时表现作为目标代理。
- **训练流程**：
  - **SFT 初始化（可选）**：用 35B Teacher 在训练族上生成成功修订数据，过滤通过解析与质量校验的样本进行 LoRA 微调（rank/scale=32）。
  - **Single-step RL**：从固定上下文池（Parent Harness + 执行报告）采样，GRPO 组大小 $G=8$，奖励 $r_i = J(h'_i; Q^{\text{score}}) + v_i$，其中 $v_i$ 为编辑解析、可运行性等辅助奖励；学习率 $10^{-6}$，batch 40 contexts/step，共 1 epoch。
  - **Multistep RL**：在离线阶段间刷新状态（将父 Harness 推进至最佳修订并重新评分），或用在线 QA 设置维护连续修订轨迹；奖励公式相同，旨在训练 Proposer 适应不断变化的父状态。
- **关键设计**：失败编辑保留父分或给零分（论文主要结果保留零分以反映真实可用性）；评分集与反馈集永远不相交，防止数据泄露。

## 实验与结果
- **数据集**：Reasoning Gym（21 SFT 族 + 5 RL 族，21 未见族）；多跳 QA（训练 HotpotQA，未见 MuSiQue、2WikiMultihopQA，使用 5.23M Wikipedia abstracts 构建检索语料）。
- **模型配置**：Reasoning Gym 用 Qwen3.5-4B 作 Proposer/Solver；QA 用 Qwen3-4B 作 Proposer、Qwen3-8B 作 Solver。
- **核心数字**：
  - Reasoning Gym 未见族单步平均得分：Base 0.32 → SFT 0.41 → **RL 0.62**；Teacher（35B）0.56。RL 训练后平均提案质量已超越 Seed，且超越 Teacher。
  - 迭代增益：5 轮多步修订后，Canonical 题 Single-step RL 达 0.699，Format-varied 达 0.725；QA 10 轮迭代中首轮仅贡献 1/3–2/3 的总提升。
  - 泛化迁移：HotpotQA 训练的 Proposer 在 MuSiQue（0.27 EM）与 2WikiMultihopQA 上均持续超过 Base，且平均最终跑分超越 Oracle best-of-80 独立提议。
- **最强结果**：4B Single-step RL Proposer 在 Reasoning Gym 21 个未见推理族上单步平均得分 **0.62**（较 Base 提升 **+94%**，较 Teacher 提升 **+10.7%**）；QA 十轮迭代后最终 Harness 在未见集上的精确匹配率显著优于所有独立提议基线。

## 相关工作脉络
1. **Harness-R1 / JIT-Agent / Ornith**：同期可执行 Harness 编辑器工作。本文与之区别在于将修订明确建模为元学习适应规则，重点验证跨任务族（family-level）零样本迁移，且 Proposer 与 Solver 在测试时均冻结，不依赖在线微调或扩大 archive。
2. **Meta-Harness / AutoHarness / TTHE**：基于执行反馈优化 Agent 流水线。本文核心差异是引入可验证奖励的 GRPO 训练与严格的 Feedback/Scoring/Evaluation 三集隔离协议，使泛化评估更可信，并系统剖析单步 vs 多步训练信号的效用边界。
3. **Self-Refine / Reflexion / STOP**：自我改进方法多针对模型输出文本、记忆或权重参数。本文转向纯程序级适配，证明即使模型参数不变，通过学习“如何改代码”也能实现稳定的测试时性能爬坡。
4. **AdaptFlow / RL² / MAML**：传统元学习关注参数初始化或梯度适应。本文把适应对象推广至离散的可执行代码结构，利用环境级执行反馈替代损失函数梯度，开辟程序空间上的策略学习路径。

## 局限性与未来方向
1. **SFT 数据多样性受限**：Reasoning Gym 阶段仅依赖单一 35B 教师生成首轮修订，且集中于 interpreter loop 与直接算法解，可能限制 Proposer 早期探索的策略分布。
2. **Proposer 缺乏自主探查能力**：训练与测试均使用固定提示词与外部生成的执行报告，Proposer 不能主动运行诊断脚本、对比候选设计或审查代码，限制了其对复杂失效模式的挖掘深度。
3. **Solver 完全冻结**：未探索 Proposer 与 Solver 的联合优化；协同训练可能产生互惠增益，但也可能引发训练不稳定性。
4. **未来方向**：使用更强/多样教师或 harness 设计先验初始化；让 Proposer 在具工具环境中自主探索修订策略；拓展至更长 horizon 的多步工具使用与复杂 Agent 工作流；探索跨轮 credit assignment 以更充分捕获多步训练价值。

## 研究启发与可借鉴点
1. **程序级元学习范式**：将“代码编辑”本身作为可学习策略的输出空间，配合结构化 diff 接口与解析/运行双保险，为 LLM Agent 的自动化工作流演进提供了可复用的训练架构。
2. **单步奖励足以支撑多步迭代**：实验表明无需复杂的多步 discount 或轨迹级 credit assignment，对单次修订进行 GRPO 优化即可在测试时涌现稳定的多轮改进能力，大幅降低训练设计复杂度。
3. **严格的三分离数据协议**：Feedback / Scoring / Evaluation 集互不相交且 OOD 任务完全隔离，能有效避免执行反馈泄露导致的虚假泛化，值得在一切基于仿真/沙箱的策略学习中采纳。
4. **有效性奖励的工程实践**：引入解析成功、可运行性、完成率等软奖励（validity reward）与任务得分组合，显著降低失败提案比例（Reasoning Gym OOD 失败率从 38% 降至 8%），是代码生成 RL 训练的稳定技巧。

## 关键术语表
**Harness**：编排大模型调用、工具交互与信息流转的可执行程序，决定 Agent 的行为逻辑与控制流。
**Proposer**：接收任务描述与执行报告、生成 Harness 代码修订建议的模型，即被训练的测试时适应策略。
**Solver**：执行具体任务推理或工具调用的基础大模型，在 Harness Learning 全流程中参数保持冻结。
**Meta-Learning over Executable Programs**：将可执行代码视为被适应对象，外层学习“从反馈到编辑”的通用规则，内层在测试时执行离散代码更新。
**Execution Feedback / Report**：Harness 运行后自动生成的结构化日志，含得分统计、失败样例片段、工具调用轨迹与历史修订记录。
**Single-step RL**：每次训练样本独立，Parent Harness 均来自固定池，不涉及连续修订状态的依赖。
**Multistep RL**：训练样本来自策略自身产生的连续修订轨迹，试图让 Proposer 适应动态变化的父状态分布。
**Test-Time Adaptation**：在推理阶段不更新任何模型权重，仅通过执行反馈驱动程序代码的迭代修订以匹配新任务分布的过程。

## 可复现要素
- **数据集**：Reasoning Gym（公开）、HotpotQA、MuSiQue、2WikiMultihopQA（均公开）；训练/评估数据集为作者自行划分，未公开具体切分脚本。
- **代码/权重开源**：论文未提及代码仓库与预训练权重公开地址。
- **关键超参**：Proposer 模型 Qwen3.5-4B / Qwen3-4B；Solver Qwen3.5-4B / Qwen3-8B；GRPO 组大小 $G=8$；学习率 $10^{-6}$；Context 长度 32,768 tokens；Response 预算 16,384（Reasoning Gym）/ 8,192（QA）tokens；采样温度 0.6 / 1.0；KL 惩罚系数 0（Reasoning Gym）/ $10^{-3}$（QA
