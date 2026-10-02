---
title: "GENMEM-GENERATIVE-SYMBOLIC-MEMORY-FOR-SELF-EVOLVING-HARNESS"
source: https://arxiv.org/pdf/2609.34633v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-10-02 08:11:04"
field: "具身智能与记忆增强大模型"
keywords: ["生成式符号寻址", "长期记忆管理", "Agent自我演化", "神经符号记忆", "GRPO对齐", "寻址不变性"]
innovations: ["用生成式离散符号寻址替代连续向量检索，解决记忆演化中的寻址漂移问题", "提出MemRetriever与MemEvolver双模块闭环架构，支持训练后冻结参数的在线记忆更新", "三阶段训练（SID对齐预训练+ReAct蒸馏+GRPO双奖励后训练）兼顾符号表示与策略优化"]
benchmarks: ["ALFWorld", "WebShop", "2WikiMultiHopQA", "HotpotQA", "GPQA"]
---

# 论文速读：GENMEM-GENERATIVE-SYMBOLIC-MEMORY-FOR-SELF-EVOLVING-HARNESS

## 一句话总结
将长期记忆管理从判别式检索重构为**生成式符号寻址**，通过离散符号标识符（SID）与双模块闭环架构实现无需梯度的在线自我演化，在具身规划、网页导航与多跳问答任务上全面超越现有记忆增强基线。

## 研究问题与动机
- **经验复用高度稀疏**：可复用经验冗余严重，仅有少量轨迹真正有价值；任务级反馈稀疏且延迟，信用分配困难。
- **寻址不变性挑战**：记忆内容持续演化时，基于连续嵌入的检索接口会因内容更新发生语义漂移，导致寻址失效。
- **现有方法缺乏结构化稀疏原则**：传统RAG或向量记忆未借鉴LoRA低秩、MoE稀疏激活、Engram条件知识存储等结构化先验，难以在动态演化中保持稳定寻址。

## 核心贡献（创新点）
1. **生成式符号寻址框架**：用离散多层符号元组替代连续向量检索，从根本上规避embedding漂移，保持寻址接口稳定。
2. **Context-aware RQ-KMeans代码本构建**：离线生成含48,16,8,8四级结构的SID空间，共享前缀自动组织功能相近经验，提升符号的语义可组合性。
3. **双模块闭环架构（MemRetriever + MemEvolver）**：检索模块负责从上下文生成SID并重写为任务自适应支持，演化模块负责蒸馏执行轨迹并决策insert/revise操作，形成Retrieve→Execute→Evolution闭环。
4. **三阶段训练策略**：SID对齐预训练（双向映射）+ 中期ReAct轨迹蒸馏 + GRPO双通道后训练，兼顾符号对齐、推理轨迹质量与过程/结果奖励优化。
5. **冻结参数下的在线自我演化**：训练完成后参数与代码本冻结，外部记忆更新无需反向传播，实现低开销、抗漂移的持续学习。

## 方法详解
- **Symbolic Identifier (SID)**：多级离散token元组 $\mathbf{z}_i = (s_i^1, \ldots, s_i^L) \in \mathcal{Z}$，从笛卡尔积地址空间抽取。代码本大小为 **(48, 16, 8, 8)**，共 **80 个离散符号**，编码 **49,152 个可能地址**；通过 context-aware RQ-KMeans 离线构建，前缀共享机制使功能相近经验在地址空间中邻近分布。
- **双模块架构**：
  - **MemRetriever**：从查询/推理上下文生成SID，检索并重写为任务自适应支持（task-adaptive support）。
  - **MemEvolver**：蒸馏执行轨迹，生成更新后的SID并决策 insert/revise 操作，实现记忆的增量维护。
- **闭环流程**：`Retrieve → Execute (StackPlanner) → Evolution`。
- **三阶段训练**：
  1. **SID对齐预训练**：17,465 步，batch size 48，学习查询/轨迹/示例 ↔ SID 的双向映射。
  2. **中期训练**：157 步，基于 ReAct-style CoT 轨迹，采用 rejection sampling + 教师蒸馏。
  3. **GRPO后训练**：130 步，双通道优化，联合过程奖励 $R_{process}$ 与结果奖励 $R_{out}$。
- **在线演化机制**：训练结束后参数与代码本冻结，记忆更新完全由离散符号路由完成，无需梯度回传，彻底避免连续表示漂移。

## 实验与结果
- **骨干模型**：Qwen2.5-7B-Instruct（全模块）
- **数据集**：ALFWorld（Pick/Look/Clean/Heat/Cool/Pick2）、WebShop、2WikiMultiHopQA、HotpotQA（in-domain）；Bamboogle、MuSiQue、NQ、TriviaQA（out-of-domain）；Deep Research、Code、SQL；GPQA（符号语言分析）
- **训练环境**：8× NVIDIA A100 80GB GPU，DeepSpeed ZeRO-3，vLLM rollout
- **主要结果**：
  - **ALFWorld Look**：GENMEM **100%** vs Skill0 85.8%（+14.2pp）；**Clean**：**96.8%** vs Skill0 94.6%（+2.2pp）
  - **WebShop Score**：**93.6** vs Skill0 89.8（+3.8）；**Success**：**82.8%** vs Skill0 71.9%（+10.9pp）
  - **Search QA整体F1**：**52.8%** vs SDAR 49.1%（+3.7pp）
  - **检索Top-5命中率**：**13.15%** vs SkillRouter 10.28%
  - **在线演化表现**：Candidate Hit (t=0→9) 20.25% → **25.58%**（+5.33pp）；Task ACC 30.56% → **36.20%**（+5.64pp）
  - **GPQA完整SID vs 原始查询**：54.06-54.66% vs 48.32%（+5.74-6.34pp）
- **核心结论**：GENMEM在三大类任务上平均提升 **17.0pp（训练前）** / **15.3pp（训练后）**；寻址不变性验证显示基于内容的TF-IDF/embedding基线在演化中停滞或下降，而GENMEM持续稳定提升；更新的记忆可被后续查询复用，证明跨案例迁移能力；SID可与自然语言token交织，呈现**组合性涌现**（合并多个SID可组合存储经验以应对单一条目未覆盖的新情境）。

## 相关工作脉络
- **Prompt/记忆类**：ReAct、Reflexion、Mem0、MemP、ExpeL、SimpleMem、IRCOT、TCRAG——本文区别于它们对连续文本或向量记忆的依赖，转向离散符号寻址以实现地址稳定。
- **RL/优化类**：GRPO、RLOO、AEPO、ARPO、ZeroSearch、AgenticRAG-R1——本文继承GRPO双通道奖励思想，但将其与符号记忆演化耦合，而非仅优化策略网络本身。
- **RAG/检索增强类**：传统向量RAG与SKILL系列——本文指出其面对动态记忆时存在embedding drift，需通过生成式符号地址维持寻址不变性。
- **神经符号结合类**：LoRA/MoE/Engram的结构化稀疏先验——本文首次将这些思想迁移至agent长期记忆管理，构建离散的生成式寻址空间。
- **定位差异**：从“检索-匹配”范式转向“生成-寻址-演化”范式，核心优势在于地址稳定性与跨任务经验重组能力。

## 局限性与未来方向
- **离线代码本扩展性**：context-aware RQ-KMeans需在固定数据分布上预构建，面对全新领域或超长尾经验时可能缺乏自适应扩容机制。
- **地址空间容量上限**：49,152个离散地址在极端长程任务或大规模多智能体协同场景下可能成为瓶颈。
- **推理开销**：SID与原生token交织生成会增加解码步数，实时性要求极高的场景需进一步轻量化路由设计。
- **未来方向**（合理推断）：动态代码本扩展、层级地址压缩、跨模型SID迁移对齐、以及将符号寻址推广至多模态记忆系统。

## 研究启发与可借鉴点
1. **符号寻址替代连续检索**：在记忆易漂移的动态场景中，离散多层符号地址可作为向量检索的稳健替代或补充，值得迁移至持续学习、 lifelong agent 等方向。
2. **冻结权重下的在线演化机制**：训练后参数/代码本冻结、纯离散路由更新记忆的设计，为低资源部署与防灾难性遗忘提供了工程范式。
3. **三阶段训练配方**：对齐预训练 → 轨迹蒸馏 → GRPO双奖励后训练的组合，兼顾表示对齐与策略优化，可直接复用于其他记忆增强agent模块。
4. **符号组合性利用**：通过合并多个SID构建复合经验，为跨任务迁移与少样本情境泛化提供了新的结构化解耦思路。

## 关键术语表
- **Generative Symbolic Addressing（生成式符号寻址）**：用LLM生成离散多层符号元组作为记忆地址，替代传统连续向量检索，保证寻址接口在内容演化下的稳定性。
- **Symbolic Identifier (SID)**：多级离散token元组，作为记忆的标准化地址与索引键，支持共享前缀与组合拼接。
- **Context-aware RQ-KMeans**：离线构建SID代码本的聚类算法，兼顾查询上下文感知与层次化地址分配，使功能相近经验在地址空间邻近。
- **MemRetriever / MemEvolver**：双模块架构的核心组件，前者负责从上下文生成SID并检索重写，后者负责蒸馏轨迹并决策记忆更新操作。
- **StackPlanner**：执行阶段的任务规划器，接收检索结果并生成动作轨迹，驱动Retrieve→Execute→Evolution闭环。
- **GRPO后训练**：基于组相对策略优化的训练阶段，联合过程奖励 $R_{process}$ 与结果奖励 $R_{out}$ 提升记忆检索与演化的策略质量。
- **Addressing Invariance（寻址不变性）**：记忆内容持续更新时，寻址接口（如地址映射）保持稳定不发生语义漂移的性质。
- **Neural-Symbolic Emergence（神经符号涌现）**：离散符号地址可与自然语言token无缝交织，并在组合使用时表现出超越单一条目覆盖范围的结构化泛化能力。

## 可复现要素
- **数据集**：ALFWorld、WebShop、2WikiMultiHopQA、HotpotQA、Bamboogle、MuSiQue、NQ、TriviaQA、Deep Research、GPQA（均为公开基准，论文未声明额外私有数据）
- **代码/权重开源状态**：论文未提及
- **关键超参**：代码本维度 (48, 16, 8, 8)、离散符号数 80、地址总数 49,152；预训练 17,465 步（BS=48）、中期训练 157 步、GRPO后训练 130 步；骨干模型 Qwen2.5-7B-Instruct；训练硬件 8× NVIDIA A100 80GB，DeepSpeed ZeRO-3，vLLM rollout
