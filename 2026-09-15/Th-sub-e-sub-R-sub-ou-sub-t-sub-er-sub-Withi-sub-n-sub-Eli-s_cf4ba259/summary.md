---
title: "Th-sub-e-sub-R-sub-ou-sub-t-sub-er-sub-Withi-sub-n-sub-Eli-s"
source: https://arxiv.org/pdf/2609.15982v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:57:36"
field: "LLM Agent 技能路由"
keywords: ["skill routing", "agent LLM", "frozen LLM reading", "product of experts", "SkillTraj", "progressive disclosure", "mid-rollout routing"]
innovations: ["从冻结 LLM 中间层隐状态通过两个线性投影提取路由信号，无需外部模型和上下文技能文本", "两阶段 glance-verdict 框架：glance 对比式全局扫描 + verdict 生成式/判别式深度审查，融合为产品专家", "构建 SkillTraj 基准评测 mid-rollout 环境下噪声多轮上下文中的实时技能路由能力"]
benchmarks: ["SkillRet test", "SRA-Bench", "Eval-Core", "SkillTraj", "Skill-Use"]
---

# 论文速读：ThesubRsubesubOusububRsubesubTsuberWithsubinEsubisubCsubititingNativeSkillRoutsubingfromaFrozenLLM

## 一句话总结
本文提出 Gavel，一种从冻结 LLM 自身前向传播中提取技能路由信号的新方法，仅需训练两个线性投影（共 7.9M 参数），无需在上下文中预加载技能文本即可在完整技能库上完成路由，并在三个公开基准和一个新构建的 SkillTraj 基准上超越渐进披露和检索-重排流水线。

## 研究问题与动机
1. **技能库规模扩张导致上下文拥堵**：渐进披露（Claude Code、Codex 等部署方案）将每个技能的元数据预加载到系统提示中，随技能库规模线性增长，严重挤占 agent 处理任务的上下文空间，路由准确率随技能数量对数衰减。
2. **外部检索模型与 agent 能力脱钩**：检索+重排管道将选择权交给外部嵌入模型和重排器，这些模型无法理解 agent 在 rollout 过程中的上下文需求，尤其在中途出现技能调用需求时表现不佳，且不会随 agent 本身能力的提升而改善。
3. **如何在agent自身推理能力不进入上下文的前提下实现高质量路由**：这是一个开放问题——如何利用 agent LLM 自身的前向传播作为路由接口，而不引入额外模型或上下文开销。

## 核心贡献（创新点）
1. **提出 Gavel 框架**：从冻结 agent LLM 自身前向传播中提取路由信号，无需外部模型、无需在路由前将技能文本放入上下文，且路由能力随 backbone 能力提升而自然提升。
2. **两阶段 glance-verdict 设计**：glance 阶段通过两个训练线性投影从中间层隐状态提取对比式匹配分数，对全库进行 token 级投票排序；verdict 阶段对 glancing 短名单进行完整前向传播，读取生成式似然和判别式 yes/no 判断，三者融合为专家乘积（product of experts）。
3. **构建 SkillTraj 基准**：包含 372 条模拟 agent 轨迹，覆盖四种中途触发技能需求的场景（用户直接请求、工具结果揭示、agent 自身计划、错误技能恢复），评测路由器在噪声多轮上下文中的实时路由能力。
4. **ε-cover 压缩保证**：引入理论保证的 ε-cover 对技能 key bank 进行压缩，将技能存储从 token 数级别降至可区分方向数级别，压缩后 bank 缩小约 8.5×，代价不超过 1.6 个点。

## 方法详解
**问题设定**：技能库 S = {s₁, ..., sN}，agent 为冻结 LLM M，路由点处上下文 c 包含任务（可能是用户请求或 rollout 中途的执行状态）。路由器观察 c 和 S 决定加载哪个技能。

**Glance 阶段（第 3.2-3.3 节）**：
- 在 LLM 中间层 ℓ*（矩阵熵谷底，约 70% 深度）嫁接一个新的注意力头，含查询投影 Wq 和键投影 Ws（无 value 路径），共 7.9M 参数。
- 安装时：对每个技能 s 渲染固定 prompt r(s)，运行一次前向传播，提取每层 ℓ* 的 token 隐状态，经 Ws 映射为单位键向量 dⱼˢ，存入技能 bank。
- 路由时：将任务 token 的隐状态经 Wq 映射为单位查询 qᵢ，每个任务 token 对每个技能的 bank 做 max-similarity 投票（类 late-interaction）：mᵢ(s) = maxⱼ ⟨qᵢ, dⱼˢ⟩，仅对每个 token 的 top-k 技能投票，加权平均得到 glance 分数 g(s|x)。
- 训练：对比学习损失（multi-positive InfoNCE），温度 τ=40，梯度截断在 hℓ* 处，仅更新 Wq 和 Ws。
- ε-cover 压缩：对技能 key bank 做最远点遍历，保留子集 Cₛ 使得每个丢弃的 key 距某个保留 key 的欧氏距离 ≤ ε，保证每个技能得分最多降低 ε，同时 bank 缩小约 8.5×。

**Verdict 阶段（第 3.4 节）**：
- 对 glance 短名单（约 9 个候选）中的每个技能 s，将安装时的技能前缀 r(s) 与任务 x 拼接后运行完整前向传播（因果掩码保证任务 token 不被修改）。
- 提取三种信号：(1) 生成式似然 L(s|x) = (1/|x|) Σᵢ log pM(xᵢ|r(s), x<ᵢ)，衡量技能对任务的预期能力；(2) 判别式判断 V(s|x) = log Σₜ∈Y pM(t|r(s), x, u) - log Σₜ∈N pM(t|r(s), x, u)，u 为固定问题"Do this skill provide what that task needs?"的 yes/no log-odds；(3) glance 分数 g(s|x)。
- 三种信号均可解释为 log p(s|x) 的不同估计（对比式、生成式、判别式），以统一尺度融合。

**Ruling 阶段（第 3.5 节）**：
- 产品专家融合：S(s|x) = g(s|x) + αL(s|x) + γV(s|x)，其中 α=1.0, γ=0.025 为交换率/阻尼系数。任何单一信号可否决其他两个容忍的候选。

## 实验与结果
**数据集与基线**：
- 三个公开基准：SkillRet test（4,997 查询/6,660 技能）、SRA-Bench（861 查询/26,262 技能）、Eval-Core（75 查询/78K 文档，分 easy/hard pool）。
- 新基准：SkillTraj（372 条轨迹，四种场景）。
- 基线：渐进披露（Qwen3-Embedding-8B 短名单 20 个技能）、SkillRouter（0.6B embedder + 0.6B reranker）、Qwen3-Embedding-8B + Qwen3-Reranker-8B、BM25、SKILLRET-Emb-0.6B。
- 评估指标：adjudicated Hit@1（GPT-5.6 Sol 作为 judge  adjudication）。

**主要结果**：
- **Written tasks**：Gavel 在三个基准上全面领先。较最强外部流水线（Qwen3-Emb-8B + RR-8B）提升：SkillRet +3.8、SRA-Bench **+13.4**、Eval-Core +1.3~2.7 点。Glance 单独已是 SRA-Bench 和 Eval-Core 上最强的检索阶段。
- **Mid-rollout（SkillTraj）**：Gavel 在四种场景均领先，提升幅度达 **8.6~21.9** 点。SkillRouter 在全噪声上下文中最差，Gavel 在 user request 场景达 96.55% adj Hit@1。
- **端到端部署（Skill-Use）**：Qwen3-32B + Gavel 触发正确技能率 **>90%**，远超 Codex 中更大前沿模型（GLM-5.1: 70.6%、MiniMax-M3: 86.4%、DeepSeek-V4-Pro: 65.0%）。
- **缩放实验**：Gavel 路由准确率随 backbone 增大而提升（Fig 6），即使在 0.6B 模型上也胜过 32B 模型的渐进披露；在 Qwen3.8-27B 上仍领先同 backbone 渐进披露 12.2 点。
- **Gemma 迁移**：在 Gemma-4-31B 上重建，方法形状保持，各基准得分趋势一致。

## 相关工作脉络
1. **渐进披露路由**（Anthropic 2025; OpenAI 2026a）：将全部技能元数据预加载到上下文，Gavel 与之本质区别在于完全不将技能文本放入上下文，通过 frozen LLM 自身隐状态读取路由信号。
2. **检索+重排管道**（SkillRouter Zheng et al. 2026; SKILLRET Kang et al. 2026）：使用外部嵌入模型和重排器，Gavel 的优势在于路由信号来自 agent 自身推理，随 agent 能力提升，且外部参数仅为 7.9M vs 1.2B~16B。
3. **ToolkenGPT**（Hao et al. 2023）：冻结 LLM 输出 tool token，但每个工具需单独训练 embedding，新增工具成本高；Gavel 新增技能只需一次前向传播建 bank，无需训练。
4. **LLM 隐状态读出**（Alain & Bengio 2017; Belinkov 2022; Wang et al. 2024）： probing 表明隐状态携带多于一阶 token 预测的信息，GRIT 微调整个 backbone 统一 embedding 与生成；Gavel 在不微调 backbone 的前提下，通过两个线性投影读出路由信号。
5. **Query-likelihood 重排**（Sachan et al. 2022; Zhuang et al. 2023）：用 LLM 对 query 赋予文档的似然评分；Gavel 借鉴此思路但将其与判别式 judgment 和对比式 glance 融合为 product of experts，并应用于技能路由而非文档重排。

## 局限性与未来方向
1. **Gate 设计简化**：当前 gate 为线性分类器，仅判断是否触发路由，未利用 glance 分数的时序累积模式；作者承认可改进为序列模型或 fine-tuned skill-call token。
2. **训练数据依赖合成 query-skill 对**：训练数据来自 SkillRet（Qwen3.5 生成），与真实 agent 轨迹存在分布差异，SkillTraj 部分填补但样本量有限（372 条）。
3. **Verdict 阶段仍需多次前向传播**：虽比渐进披露节省上下文，但每次路由仍需约 9 次完整前向传播（短名单），对极端低延迟场景有成本压力。
4. **未验证工具/记忆/MCP server 路由**：结论部分明确提到 "whether the same read-out also routes tools, memories, or MCP servers" 是开放问题。
5. **ε-cover 压缩的理论上界可能在某些边界场景影响排序**：虽然 worst-case  distortion 有界，但对 top-k 边界的 skill 可能有微小影响。

## 研究启发与可借鉴点
1. **Frozen LLM 隐状态作为路由信号**：通过训练轻量投影从中间层隐状态读取任务-技能匹配信号，避免了外部检索模型的部署开销，且信号随 backbone 自动增强——这一思路可迁移到工具选择、记忆检索、MCP server 路由等通用 agent 路由问题。
2. **Product of Experts 融合三种路由信号**：将对比式 glance、生成式似然、判别式 judgment 统一为 log posterior 的不同估计并以乘积专家融合，每种信号提供独立纠错能力，这一融合策略可推广到多模态路由或多源证据聚合场景。
3. **ε-cover 压缩的理论保证**：利用 farthest-first traversal 构建 ε-cover 对技能 key bank 进行有界失真的压缩，将存储从 token 线性依赖转为可区分方向数依赖，对大规模技能库部署有直接参考价值。
4. **SkillTraj 的四种 mid-rollout 场景设计**：user request/tool evidence/agent plan/wrong-skill recovery 覆盖了真实 agent 路由的关键触发模式，其轨迹生成+程序化锚点验证+blind judge 审核的流程可作为后续 agent 路由基准建设的参考模板。
5. **压缩谷（compression valley）层选择准则**：利用矩阵熵谷底（约 70% 深度）作为信息读出层 ℓ*，这一无监督准则具有模型无关性，可用于其他需要从 LLM 中间层提取结构化信息的任务。

## 关键术语表
**Gavel**：一种从冻结 LLM 自身前向传播中提取技能路由信号的两阶段方法，包含 glance（快速全局扫描）和 verdict（对短名单深度审查）两个步骤。
**Glance**：Gavel 的第一阶段，通过两个训练线性投影从 LLM 中间层隐状态读取任务-token 与技能-token 之间的对比式匹配分数，对全库进行 token 级投票排序。
**Verdict**：Gavel 的第二阶段，对 glance 短名单中的每个技能运行完整前向传播，读取生成式任务似然和判别式 yes/no judgment 两种 native 信号。
**Product of Experts**：将 glance 的对比分数、verdict 的生成式似然和判别式判断三者以加权求和（log-space）融合，任一信号可否决其他信号容忍的候选。
**ε-cover**：对技能 key bank 的压缩方法，保留一个子集使得每个丢弃 key 距某个保留 key 的欧氏距离不超过 ε，理论保证压缩后得分最多降低 ε。
**SkillTraj**：论文构建的新基准，包含 372 条模拟 agent 轨迹，评测路由器在中途技能需求触发时的路由准确率，覆盖四种噪声上下文场景。
**Progressive Disclosure**：现有部署方案（如 Claude Code、Codex）采用的路由方式，将所有技能元数据预加载到系统提示中，由 agent LLM 自行决定加载哪个技能。
**Compression Valley**：LLM 中间层表征的一个现象，指层深度约 70% 处隐状态的矩阵熵达到谷底，此处表征最具信息压缩性，适合作为路由信号读出层。

## 可复现要素
- **数据集**：SkillRet（公开，训练集 51,104 queries，测试集 4,997 queries，9,084 skills）、SRA-Bench（公开，26,262 skills）、Eval-Core/SkillsBench（公开，78K docs）、Skill-Use（公开，177 tasks/79 skills）、SkillTraj（论文构建，372 条轨迹）。
- **代码/权重**：论文未明确提及代码开源声明（截至当前版本），但方法依赖 Qwen3-32B backbone 和公开的 SkillRet 训练数据，两个投影矩阵 Wq/Ws 共 7.9M 参数可在 SkillRet 训练集上复现。
- **关键超参**：温度 τ=40，压缩半径 ε=0.83（Gemma 上 ε=0.77），交换率 α=1.0、γ=0.025，剪枝 margin Δ=0.133（约 9 个候选进入 verdict），读出层 ℓ*=44（Qwen3-32B，64 层中约 70% 深度），AdamW lr=10⁻³，weight decay=0.01，训练 24,000 steps，压缩后 Wq 再 fine-tune 3,000 steps（lr=10⁻⁴）。
