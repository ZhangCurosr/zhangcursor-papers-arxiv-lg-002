---
title: "Learning-Meta-Skills-for-Agent-Harness-Design-in-Test-Time-A"
source: https://arxiv.org/pdf/2609.38143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:56:09"
field: "Agent harness design & test-time learning"
keywords: ["AI4AI", "test-time learning", "meta-skill", "agent harness", "fixed-weight self-improvement", "Builder-Target architecture", "reusable support principles"]
innovations: ["提出三字段 meta-skill 结构（when/provide/use）使 Builder 可从 Target 反馈学习可复用支持原则", "构建 construction-execution-reflection 迭代闭环，在双模型固定权重下实现经验到可执行支持的转化", "证明 same-model self-improvement 可行：同一模型可通过学习构建更好 harness 实现自身性能提升"]
benchmarks: ["Harness-Bench", "NewtonBench"]
---

# 论文速读：Learning-Meta-Skills-for-Agent-Harness-Design-in-Test-Time-A

## 一句话总结
本文研究测试时 AI-for-AI 场景，提出让 Builder 从 Target 执行反馈中学习可复用的 **meta-skills**（支持原则），并以固定权重通过构建任务特定 harness 为 Target 提供可执行支持，在 Harness-Bench 和 NewtonBench 上相比无技能构建和直接交付知识均有显著提升。

## 研究问题与动机
1. **核心问题**：当 Builder 和 Target 模型权重均固定时，如何通过学习将执行经验转化为可复用的支持设计原则，从而提升 Target 在未见任务上的表现？
2. **现有方法不足**：已有 AI4AI 方法（如 Meta-Harness、AFlow）聚焦于搜索或优化可执行 harness 实现，但未解决如何将经验提炼为跨任务复用的支持设计原则；直接为 Target 附加 skills 忽略了"谁来执行知识"的关键差异。
3. **动机**：Agent 性能既取决于推理能力也取决于执行环境；一个强 Advisor 通过建立共享日志、可复现工具和验证工作流帮助 PhD 学生，类比可知 Builder 应学会为 Target 提供使其既有能力更有效的支持。

## 核心贡献（创新点）
1. **提出 meta-skill 概念与三字段结构**：定义 meta-skill = (when, provide, use)，区别于 Target 使用的 task skills，专门记录 Builder 设计支持的原则；本质区别在于前者指导"何时提供什么支持"，后者指导"如何完成任务"。
2. **构建 construction–execution–reflection 学习闭环**：从空 skill bank 开始，Builder 迭代构建 harness→Target 执行→review 反馈→更新 bank，每步最多一次基于证据的保留/修订/新增；区别于 prior 的单次离线总结，本研究强调迭代反思与证据 grounding。
3. **冻结 skill bank 后为每个测试任务构建全新 harness**：meta-skill 作为外部知识存入 bank，测试时冻结并指导 fresh harness construction；区别于 Meta-Harness 直接搜索 harness 实现，本工作将学习对象从 harness 结构转移到支持设计原则。
4. **在固定双模型权重下实现系统级自改进**：相同模型可同时担任 Builder 和 Target，self-improvement 无需 weight update；区别于传统 self-evolution 方法需更新模型参数，本研究探索了固定权重下的环境级改进路径。

## 方法详解
1. **Meta-Skill 定义**：每个 meta-skill s = (when, provide, use)，when 识别需要支持的可观测条件，provide 指定环境应提供的能力/资源，use 解释 Target 如何使用该支持并保留哪些判断责任。每个 meta-skill 不超过 192 tokens。

2. **Skill Learning Workflow**：
   - 从空 bank S₀ 开始
   - 对每个 development 任务 x，Builder 用当前 bank 和 neutral 环境 H₀ 构建 harness：Hₓʲ = B(H₀, x, Sⱼ)
   - Target 在 harness 内执行得到记录 eₓʲ 和分数
   - Builder review 生成程序和执行反馈 F(eₓʲ)，做出 keep/revise/add 决策：Sⱼ₊₁ = Revise_B(Sⱼ, {(Hₓʲ, F(eₓʲ))}ₓ∈Gⱼ)
   - 迭代至开发集 pass 数预算用完

3. **Test-Time Skill Selection**：
   - Bank 冻结后，每任务构建全新 harness
   - 两种模式：full bank（所有 meta-skill 入 context）或 retrieval（BM25 检索 top-2，k₁=1.5, b=0.75）
   - Hₓ* = B(H₀, x, K(x, S*))，其中 K 提供检索或全量 skills

4. **Harness 组件空间**：七个可选组件家族——instructions（指令）、memory（记忆）、context（上下文组织）、composed tools（组合工具）、execution control（执行控制）、verification & recovery（验证与恢复）、workspace（工作区）。Builder 可选择实现这些组件。

5. **约束条件**：所有模型 temperature=0，harness 生成限 16K tokens，skill update 限 8K tokens；Target 执行预算按 benchmark 固定（Harness-Bench: 30 turns/30 tool calls/96K tokens；NewtonBench: 12 turns/10 tool calls/192K tokens）。

## 实验与结果
1. **数据集**：Harness-Bench（106 tasks, 11 dev/95 test）和 NewtonBench（324 tasks, 32 dev/292 test），各保留约 10% 用于学习。

2. **模型设置**：Builder 用 GPT-5.6-Sol；Targets 用 Gemini-3.6-Flash、Qwen3.8-Flash、GPT-OSS-120B。每个组合从零 bank 开始，经过 2 个 dev pass 学习。

3. **主要结果（Table 3）**：
   - **Full-bank meta-skills（GPT-5.6-Sol Builder）**：Harness-Bench macro-average 65.31%，NewtonBench 对应最佳 Gemini 68.84%
   - 相比 no-skill Builder：Harness-Bench 平均提升 **8.95 points**，NewtonBench 平均提升 **10.96 points**
   - 相比 direct delivery 相同 bank 给 Target：平均提升 **12.02 points**（最大单设置提升 25.43 points）
   - Full-bank 在 6/6 设置中优于 BM25 top-2 retrieval，平均增益 7.19 points（NewtonBench）

4. **关键发现**：
   - NewtonBench（协调瓶颈）比 Harness-Bench（异构工作流）从经验中受益更多
   - Same-model self-improvement：相同模型担任 Builder+Target 时，meta-skills 平均比 no-skill 提升 **18.71 points**
   - Cross-Builder transfer 可行但效果依赖 receiving Builder 的实现能力

5. **消融分析（Table 4）**：controller 组件对 Gemini 提升 13.36 points（p<0.05），对 Qwen 提升 4.79 points；memory+context 效果因 Target 而异。

## 相关工作脉络
1. **AI for AI / Harness 搜索**：Meta-Harness [3] 直接搜索 harness 实现；本文学习可复用支持原则而非搜索实现，区分了"知识内容"与"执行 enactment"的价值。
2. **Agent Skills**：Evo-Harness [9] 编译经验为 solver skills 并直接给 Target；本文 skills 是给 Builder 的，强调 Translator 角色。
3. **Meta-level Learning**：Meta Context Engineering [13]、MetaSkill-Evolve [8] 学习 context/code 构造技能；本文学习支持设计原则，更关注 when/provide/use 结构。
4. **Strong-to-Weak Transfer**：作者先前工作 [5] 研究强 Builder 向弱 Target 的 harness 传递；本文扩展为学习阶段使 Builder 可积累跨任务经验。
5. **SkillRL / Voyager**：通过 RL 或存储可执行例程进化技能；本文在固定权重下学习抽象原则，不涉及 policy 训练。

## 局限性与未来方向
1. **学习历史较短**：仅 2 个 dev pass，更长 stream 下的 skill 保留/组合/选择性修订机制待研究。
2. **跨 benchmark 泛化未验证**：目前仅在一个 benchmark 内 learn-evaluate，cross-dataset reuse 留作未来工作。
3. **Transfer 不确定性**：cross-Builder 迁移的 95% CI 包含 0，需更多配置验证；recipient-side learning 表现更强。
4. **计算成本未评估**：当前评估固定 Target 预算下的 performance，未计入 Builder learning/construction 的计算开销。
5. **Component 效果因 Target 而异**：memory+context 对 GPT-OSS 有轻微负 effect，说明 support 需适配 recipient capability。

## 研究启发与可借鉴点
1. **Meta-skill 三字段结构可作为通用模板**：when/provide/use 分离"触发条件-资源供给-使用责任"，适用于任何需要外部支持的 agent 场景，可迁移到代码生成、科学发现等任务。
2. **Construction-execution-reflection 闭环值得推广**：不同于单次离线总结，迭代反思+证据 grounding 避免 overfitting 单 episode，可在更多 agent workflow 中应用。
3. **Full-bank vs retrieval 权衡的发现**：compact skill bank 中 skills 互补性可能超过 lexical retrieval 精度，提示在 small-bank 场景下 full-context 可能是更强 baseline。
4. **Same-model self-improvement 路径**：同一模型兼任 Builder/Target 的实验证明 fixed-weight 自改进可行，可与 self-play、iterative refinement 等方法结合。
5. **Harness component 消融设计**：冻结 harness 后 replay 测试任务做 ablation，隔离组件贡献，适合任何 harness/环境设计研究。

## 关键术语表
**Meta-skill**：面向 Builder 的可复用支持设计原则，包含 when（何时需要支持）、provide（提供什么资源/能力）、use（Target 如何使用）三字段，区别于 Target 使用的 task skills。

**Harness**：Builder 为 Target 构建的增强执行环境，由 instructions、memory、tools、controller 等组件组成，Target 在其中完成目标任务。

**Builder-Target 架构**：AI4AI 中的双角色设定，Builder 设计支持环境，Target 使用支持执行任务；两者模型权重均固定。

**Construction–Execution–Reflection Loop**：Builder 构建 harness→Target 执行→review 反馈→更新 skill bank 的迭代学习闭环。

**BM25 Retrieval**：基于初始任务 prompt 的 lexical retrieval 方法（k₁=1.5, b=0.75），从 skill bank 检索 top-2 相关 meta-skill。

**Outcome Transition Analysis**：对比 no-skill 与 full-bank 条件下各任务 outcome 变化（如 no-valid→correct、partial→full），揭示 meta-skill 帮助的具体机制。

**Same-Model Self-Evolution**：同一模型同时担任 Builder 和 Target 的设定，证明 fixed-weight 下可通过环境设计实现系统级自改进。

## 可复现要素
- **数据集**：Harness-Bench [12]、NewtonBench [18]（均引用开源 benchmark）
- **代码/权重**：论文声明"Upon paper acceptance, we will release the full code and all prompts"，当前未公开
- **Builder 模型**：GPT-5.6-Sol
- **Target 模型**：Gemini-3.6-Flash、Qwen3.8-Flash、GPT-OSS-120B
- **关键超参**：temperature=0，high reasoning effort；harness 生成 16K tokens；skill update 8K tokens；meta-skill 上限 192 tokens；BM25 k₁=1.5, b=0.75；dev pass=2；retrieval top-k=2
