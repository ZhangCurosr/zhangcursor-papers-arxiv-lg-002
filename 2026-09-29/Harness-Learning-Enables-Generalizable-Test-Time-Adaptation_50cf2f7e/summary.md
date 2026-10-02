---
title: "Harness-Learning-Enables-Generalizable-Test-Time-Adaptation"
source: https://arxiv.org/pdf/2609.35738v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:28:14"
field: "可执行程序的元学习与测试时适应"
keywords: ["harness learning", "test-time adaptation", "meta-learning", "executable programs", "reinforcement learning", "agent workflow"]
innovations: ["提出将harness修订视为元学习过程，proposer从执行反馈中学习可泛化的代码编辑策略", "训练后的小模型proposer在单次修订中平均超越35B教师模型", "发现单步RL训练即可使proposer在测试时多轮迭代改进harness，无需专门训练修订序列"]
benchmarks: ["Reasoning Gym (21 unseen families)", "HotpotQA", "MuSiQue", "2WikiMultihopQA"]
---

# 论文速读：Harness-Learning-Enables-Generalizable-Test-Time-Adaptation

## 一句话总结
该论文提出了**Harness Learning**框架，训练一个提议器（proposer）根据执行反馈修改求解器的可执行程序（harness），使求解器在不更新参数的情况下也能通过元学习方法适应新任务；在推理和多跳QA任务上验证了训练后的小模型（4B）能超越大模型教师（35B），且适应策略可泛化到未见过的任务并支持多轮迭代改进。

## 研究问题与动机
- **核心问题**：语言模型agent的表现不仅取决于模型本身，还取决于组织模型调用、工具使用和流程控制的harness（可执行程序）。如何学习一种可泛化的方式来自动改进harness？
- **现有方法不足**：传统harness优化多依赖人工设计或静态搜索，难以从执行反馈中持续学习；已有元学习方法多针对模型参数，而非可执行程序结构；test-time adaptation缺乏跨任务迁移能力。
- **动机**：通过元学习框架训练proposer，使其在test time仅用执行反馈调整harness（而非更新模型权重），实现可迁移的持续适应。

## 核心贡献（创新点）
1. **提出Harness Learning作为可执行程序的元学习**：proposer从执行结果中学习如何修订harness代码，将代码修订视为梯度下降中的权重更新。
2. **训练后的小模型在单次修订中超越大教师模型**：在21个未见推理任务族上，训练的4B proposer平均得分0.62，超过35B teacher的0.56。
3. **适应策略可泛化到未见任务并支持多轮迭代改进**：在QA基准（HotpotQA训练→MuSiQue/2WikiMultihopQA测试）上证明跨任务迁移能力，且即使仅在种子harness上训练的proposer也能在多轮中持续改进。
4. **揭示了单步训练即可产生多步适应能力的现象**：训练于个体修订的proposer在测试时可连续改进harness，无需专门训练修订序列。

## 方法详解
**框架**：Meta-learning over executable programs。
- **Proponent（proposer）** $\pi_\theta$：输入任务描述、当前harness $h_{t-1}$、执行报告，输出代码编辑$y$；使用GRPO进行强化学习训练。
- **Reward设计**：$r_i = J(h'_i; Q^{score}) + v_i$，其中$J$为评分集上的平均grader分数，$v_i$为编辑有效性辅助奖励（如代码可解析、运行成功等）。
- **训练流程**：可选SFT初始化（使用35B教师修订数据）+ RL优化；支持Single-step RL（独立修订上下文）和Multistep RL（从之前选定harness生成新上下文）。
- **Test-time adaptation**：proposer和solver均冻结，proposer基于每次执行反馈迭代修订harness，无需参数更新。
- **关键公式**：Meta-learning目标$\max_\theta \mathbb{E}[J(h_T; Q^{eval})]$，其中$h_T$由proposer的修订循环产生。

## 实验与结果
**数据集**：
- Reasoning Gym：21个未见任务族（maze、sudoku等），包含canonical和format-varied问题。
- Multi-hop QA：HotpotQA训练，MuSiQue和2WikiMultihopQA测试。

**基线**：Base（未训练proposer）、SFT初始化、Teacher（35B模型）、Oracle best-of-N选择。

**主要结果**：
- **Reasoning Gym未见任务**：Single-step RL使平均得分从Base的0.32提升至0.62，超过35B teacher的0.56；5轮迭代后进一步接近oracle best-of-eight。
- **QA跨任务迁移**：HotpotQA训练的proposer在MuSiQue和2WikiMultihopQA上持续改进，10轮后平均得分超过seed和Base；即使仅用种子harness训练的Single-step RL也能在多轮中提升。
- **训练对比**：Multistep RL并未始终优于Single-step RL，表明单步训练已能产生迭代改进能力。

## 相关工作脉络
- **Harness优化**：Harness-R1使用教师SFT+RL编辑harness，但评估在相同基准的 disjoint 实例上；本文关注跨任务泛化和重复修订。
- **Meta-learning**：AdaptFlow学习工作流初始化，The Last Harness演化蓝本；本文直接将可执行harness作为适应对象，proposer学习离散代码编辑规则。
- **TTHE**：用无标签执行轨迹适应harness；本文使用带graded feedback的开发问题指导修订。
- **Self-improvement**：STOP、Self-Refine等更新输出或参数；本文保持求解器冻结，仅修订程序结构。
- **JIT-Agent/Ornith**：同期工作，前者学习任务条件化harness生成与归档演化，后者联合优化scaffold和solver；本文专注proposer的泛化适应与多轮改进分析。

## 局限性与未来方向
- 训练依赖固定solver和 prescribed revision prompts，proposer无法主动探查代码或执行测试。
- 监督数据仅来自单一35B教师，可能限制修订策略的多样性。
- 多步训练（Multistep RL）未显著提升单步训练效果，跨轮次信用分配机制尚未充分探索。
- 未来方向：联合训练proposer和solver、探索更强教师或多步修订演示、让proposer具备工具使用能力以主动检查代码。

## 研究启发与可借鉴点
1. **程序作为适应对象**：将meta-learning框架应用于可执行代码（而非权重），为agent harness优化提供新视角。
2. **小模型超越大教师的可行性**：通过RL从执行反馈学习，小规模proposer可在平均修订质量上击败更大教师模型。
3. **单步训练产生多步能力**：即使仅在独立修订上训练，proposer仍能测试时迭代改进，简化了训练设计。
4. **可复用的评估协议**：分离feedback、scoring、evaluation问题集，确保泛化性评估可靠；可用于其他agent适应研究。
5. **与团队方向结合机会**：若团队研究agent workflow优化或test-time adaptation，可直接借鉴proposer训练流程、reward设计及跨任务评估方法。

## 关键术语表
- **Harness**：组织语言模型调用、工具使用和中间信息流的可执行程序。
- **Proposer**：学习如何修订harness的模型，其参数在测试时冻结。
- **Solver**：被harness调用的基础模型，参数在训练和测试中均固定。
- **Meta-learning over executable programs**：将程序（harness）视为适应对象，proposer学习修订规则的元学习框架。
- **GRPO（Group Relative Policy Optimization）**：用于训练proposer的强化学习算法，通过组内归一化优势估计奖励。
- **Execution report**：记录harness在反馈问题集上的执行结果、错误示例等摘要信息。
- **Single-step / Multistep RL**：前者训练于独立修订上下文，后者训练于由先前修订生成的连续状态。
- **Oracle best-of-N**：从N个候选中选择测试集上表现最好的，用于评估候选池质量的上界。

## 可复现要素
- **数据集**：Reasoning Gym和HotpotQA/MuSiQue/2WikiMultihopQA；论文未明确声明公开状态，但通常这些基准有公开代码。
- **代码/权重**：论文未提及开源代码或预训练权重，需作者确认。
- **关键超参**：Proposer为Qwen3.5-4B（推理）或Qwen3-4B（QA）；Solver为Qwen3.5-4B（非thinking）或Qwen3-8B；RL学习率1e-6，batch size 40 contexts/更新，温度0.6（Reasoning Gym）或1.0（QA）；SFT使用LoRA rank=32，上下文长度32,768 tokens。
