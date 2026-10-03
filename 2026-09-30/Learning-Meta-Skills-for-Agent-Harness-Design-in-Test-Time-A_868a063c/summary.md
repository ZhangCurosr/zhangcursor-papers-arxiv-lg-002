---
title: "Learning-Meta-Skills-for-Agent-Harness-Design-in-Test-Time-A"
source: https://arxiv.org/pdf/2609.38143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:55:43"
field: "Agent harness design & test-time AI4AI"
keywords: ["AI4AI", "test-time", "meta-skill", "agent harness", "self-improvement", "reusable support principles"]
innovations: ["提出 meta-skill 三元组 (when, provide, use) 表示可复用的支持设计原则", "构建 construction-execution-reflection 循环从 Target 反馈中学习 meta-skills 并冻结部署", "验证 same-model 自进化路径：固定权重下模型通过学习构建更好 harness 实现自我提升"]
benchmarks: ["Harness-Bench", "NewtonBench"]
---

# 论文速读：Learning Meta-Skills for Agent Harness Design in Test-Time AI4AI

## 一句话总结
论文研究测试时（test-time）AI-for-AI 场景下，Builder 如何通过学习目标模型 Target 的执行反馈来积累可复用的 meta-skills（元技能），进而为不同任务构建更优的 agent harness（执行环境）。

## 研究问题与动机
1. **AI agent 性能的双重依赖**：agent 性能既取决于其推理能力，也取决于其所处的执行环境。如何让一个更强的模型（Builder）为弱模型（Target）设计更好的支持环境是一个开放问题。
2. **现有 harness 构造方法的局限**：如 AIDE、AFlow、Meta-Harness 等工作主要关注搜索可执行程序或优化 harness 实现，但未系统性地从 Target 执行反馈中学习**可复用的支持设计原则**。
3. **经验难以复用**：直接给 Target 额外的任务技能（task skills）只能帮助解决单一任务，而 Builder 需要学习的是**何时提供何种支持**的元知识（meta-skills），以便泛化到未见过的任务。
4. **训练时 vs 测试时 AI4AI**：训练时方法（如 MLE-Dojo）需要更新模型权重，而测试时 AI4AI 的目标是在**双方权重固定**的前提下，通过外部环境设计提升 Target 表现。

## 核心贡献（创新点）
1. **提出 meta-skill 概念与三元组表示**：定义 meta-skill 为 `(when, provide, use)` 三元组，明确支持需求条件、供给内容与 Target 使用方式，区别于 Task skills 的任务执行知识。
2. **构建"构造—执行—反思"循环学习框架**：Builder 从 Target 的开发集执行反馈中迭代更新 meta-skill bank，并在测试时对每个新任务冻结 bank 后重新构造专属 harness。
3. **实验验证 meta-skills 优于直接技能传递**：在 Harness-Bench 和 NewtonBench 上，full-bank meta-skills 分别比无 skill 构造和直接将同等 bank 给 Target 提升 8.95pp 和 12.02pp 的宏观平均分数。
4. **揭示 same-model 自进化路径**：当 Builder 和 Target 为同一模型实例时，meta-skills 仍能带来 18.71pp 的平均提升，证明模型可通过学习构建更好的支持环境实现系统级自我改进。

## 方法详解
**问题形式化**：给定固定 Builder B 和 Target τ，基准环境 H₀、评估器 r 和执行预算 Cₓ，harness 策略 H 将公开输入 x 映射为增强环境 Hₓ = H(x)，目标为最大化 E[r(x, τ)]，满足 cost(τ) ≤ Cₓ。

**Meta-skill 定义**：s = (when, provide, use)，其中 when 识别需支持的观测条件，provide 指定环境应供给的能力或资源，use 说明 Target 如何使用及保留的判断责任。每个 meta-skill 上限 192 tokens。

**Skill Learning Workflow**：从空 bank S₀ 开始，对每个开发任务 x，Builder 用当前 bank 构造 harness Hₓʲ = B(H₀, x, Sⱼ)，Target 执行产生 eₓʲ，Builder 根据执行反馈 F(eₓʲ) 更新 bank：Sⱼ₊₁ = Revise_B(Sⱼ, {(Hₓʲ, F(eₓʲ))}ₓ∈Gⱼ)。每轮最多一次 add/revise/keep 操作。

**Test-Time Skill Selection**：bank 冻结后，两种模式提供 skills：(1) Full bank：将所有 meta-skills 放入 context；(2) Retrieval：用 BM25 检索 top-2 相关 skill。然后 Hₓ* = B(H₀, x, K(x, S*))，Target 在专属 harness 中执行。

**Harness 七大组件族**：instructions（具体任务指导）、memory（记录格式与存取规则）、context（历史选择规则）、composed tools（工具调用可见性）、execution control（执行阶段与工具隐藏）、verification & recovery（提交检查与失败处理）、workspace（初始文件与模板）。

## 实验与结果
**数据集**：Harness-Bench（106 tasks，开发集 11，测试集 95，agent workflow 评测）和 NewtonBench（324 tasks，开发集 32，测试集 292，科学定律发现评测）。

**模型设置**：Builder 固定为 GPT-5.6-Sol；Targets 为 Gemini-3.6-Flash、Qwen3.8-Flash、GPT-OSS-120B。开发集迭代 2 轮学习，测试时冻结 bank。

**主要结果**（Table 3）：
- **Best full-bank**：65.31% macro-average（Harness-Bench + NewtonBench 平均）。
- **vs No-skill Builder**：+8.95pp（所有 6 个 setting 均提升）。
- **vs Direct Builder skills (all)**：+12.02pp（语义知识固定，检验 enactment 价值）。
- **vs Native environment**：最高提升达 25.43pp（Gemini on Harness-Bench）。
- **Full bank vs Retrieval**：6 个 setting 中 5 个 full bank 更优，平均 +7.19pp（NewtonBench）。

**关键结论**：
- Meta-skills 在 Target 执行瓶颈为**协调问题**时收益最大（NewtonBench +10.96pp vs Harness-Bench +6.95pp）。
- **Same-model self-improvement**：相同模型同时担任 Builder 和 Target，meta-skills 平均提升 18.71pp over no-skill。

## 相关工作脉络
1. **AI4AI in Agentic Systems**：AIDE/ML E-Dojo 面向模型开发（训练时）；AFlow/Darwin Godel Machine/Meta-Harness 面向测试时 agent 程序搜索，本文侧重从执行反馈学习可复用支持原则并冻结部署。
2. **Agent Skills 与 Meta-Skills**：Voyager/ExpeL/Agentic Context Engineering 积累可复用程序或文本知识；SkillRL/Evo-Harness 通过 RL 演化 skills；本文在 meta 层面学习**支持设计原则**而非任务技能。
3. **Strong-to-weak harness construction**（Qian et al., 2026）：本文在此设定上扩展，加入离线 skill learning + frozen bank test-time deployment 机制。
4. **Meta Context Engineering / MetaSkill-Evolve**：前者学习构造 context 文件的技能，后者演化改进 task skills 的策略；本文让 Builder 从自身 harness 结果中学习 how to support，而非直接改进 Target 能力。
5. **SkillsBench**：测量 skill utility 的基准，本文方法与之互补，关注 skill 如何被 Builder  enacted 为可执行环境。

## 局限性与未来方向
1. **跨 benchmark 泛化未验证**：目前仅在同一 benchmark 内不同 task 间泛化，cross-dataset 复用尚未评估。
2. **Refinement 非单调**：部分 Target（如 Qwen on Harness-Bench）在第二轮更新后性能下降，说明过度拟合近期证据，需要选择性保留机制。
3. **Cross-Builder transfer 不确定性**：置信区间包含零，不同 Builder 对相同 bank 的实现差异影响效果。
4. **Execution budget 之外不计入成本**：Builder 学习与构造的计算成本未纳入 Target 预算，实际部署需联合优化。
5. **Component ablation 样本有限**：仅基于 NewtonBench 一个 benchmark，结论需在其他环境中验证。

## 研究启发与可借鉴点
1. **Meta-skill 三元组设计**：`(when, provide, use)` 清晰分离了支持需求的识别、供给与使用责任，可作为其他 AI4AI 场景中 reusable knowledge 的表征范式。
2. **Frozen bank + per-task fresh construction**：开发集学习后冻结、测试时逐任务重构造的策略，平衡了经验复用与任务适配，适用于任何需要 offline learning + online adaptation 的系统。
3. **Outcome transition 分析**：不仅看 aggregate score，还追踪 no-skill → full-bank 的 outcome 转移矩阵（如 invalid → correct、partial → full），揭示了 skill 主要解决哪类失败模式，值得借鉴为评估框架。
4. **Same-model self-improvement 实验设计**：证明模型无需外部 teacher 也可通过 harness 设计实现自我进化，为未来 agent 自改进研究提供了简洁可行的实验范式。
5. **Seven-component harness 空间**：显式定义的 instructions/memory/context/tools/controller/verification/workspace 组件族，为系统性地结构化 harness 设计提供了可复用的分类框架。

## 关键术语表
**Meta-skill**：Builder 学习的可复用支持设计原则，三元组 (when, provide, use)，区别于 Target 使用的 task skills。
**Harness**：Builder 为 Target 构造的执行环境，包含指令、资源、工具、控制器等组件，Augments 基准环境 H₀。
**Builder–Target 分工**：Builder 负责设计与提供执行支持，Target 负责使用支持完成任务；两者权重均固定。
**Test-time AI4AI**：模型权重不更新，通过外部 harness 设计提升 agent 性能的测试时增强范式。
**Construction–Execution–Reflection loop**：Builder 构造 harness → Target 执行 → Builder 反思反馈并更新 skill bank 的迭代学习流程。
**Macro-average performance**：在 Harness-Bench 和 NewtonBench 上分别计算指标后取平均，反映跨 benchmark 综合效果。

## 可复现要素
- **数据集**：Harness-Bench（arXiv:2605.27922）、NewtonBench（ICLR 2026）；论文声明开发/测试划分按比例 10%/90% 取整分配。
- **代码/权重**：论文 Appendix B.3 声明"Upon paper acceptance, we will release the full code and all prompts"，当前未公开。
- **关键超参**：Builder=GPT-5.6-Sol，Targets=Gemini-3.6-Flash/Qwen3.8-Flash/GPT-OSS-120B；temperature=0，high reasoning effort；harness generation 16K tokens，skill update 8K tokens；每 skill ≤192 tokens；开发集迭代 2 轮；BM25 检索 k₁=1.5, b=0.75，top-2；Target 预算 Harness-Bench 30 turns/30 tool calls/96K tokens，NewtonBench 12 turns/10 tool calls/192K tokens。
