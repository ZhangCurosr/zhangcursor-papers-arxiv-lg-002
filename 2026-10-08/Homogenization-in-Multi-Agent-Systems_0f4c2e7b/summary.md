---
title: "Homogenization-in-Multi-Agent-Systems"
source: https://arxiv.org/pdf/2610.09824v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:54:16"
field: "多智能体系统安全性与公平性"
keywords: ["multi-agent systems", "homogenization", "algorithmic bias", "LLM safety", "interaction dynamics", "code generation security", "fairness in AI hiring", "peer review"]
innovations: ["提出conformity/polarization/inertia三维度同质化度量框架", "揭示采样随机性和混合模型配置无法缓解MAS同质化风险", "在代码生成/招聘/同行评审三场景中验证同质化的系统性危害"]
benchmarks: ["CodeLMSec", "ICLR 2025 submissions", "synthetic counterfactual resume dataset"]
---

# 论文速读：Homogenization in Multi-Agent Systems

## 一句话总结
本文揭示了多智能体系统（MAS）中一个被忽视的失败模式——**同质化（Homogenization）**：智能体在交互过程中趋向于收敛到相似行为，导致多样性丧失并引发系统性风险；在代码生成、招聘和同行评审三个应用中验证了该风险，并证明简单的干预手段（采样随机性、混合模型配置）均无法有效缓解。

## 研究问题与动机
- **多智能体系统存在多样性崩塌问题**：MAS 通过多智能体交互协作完成复杂任务，但交互过程可能导致智能体失去独立判断，趋同于相似行为模式，而非发挥各自独特视角。
- **现有评估体系忽视交互动态**：当前 MAS 研究主要关注最终聚合性能，缺乏对智能体间交互动态（如从众、极化、惯性）的系统性度量与分析。
- **简单干预手段不足以应对同质化风险**：直觉上认为增加随机性或混合不同模型可能提升多样性，但本文证明这些"表面多样性"无法真正缓解深层的同质化动态。
- **下游应用存在系统性风险**：同质化在关键应用场景中会放大相关性错误、传播偏见、固化偏好，造成难以察觉的系统性伤害。

## 核心贡献（创新点）
1. **首次将"同质化"形式化为 MAS 的系统性失败模式**，并定义三个正交度量指标：从众性（方差衰减）、极化（均值漂移至极端）、惯性（变化速率递减），区别于已有工作仅关注单一共识指标的局限。
2. **在三个高影响力应用场景中实证验证同质化的危害**：代码生成中的系统性安全盲点（特定漏洞跨模型放大）、招聘中的偏见持久化（ biased agent 移除后偏见仍留存）、同行评审中的领域偏见固化（研究领域的评分差距随轮次扩大）。
3. **揭示简单干预手段的失效**：证明采样随机性仅产生"表面差异"（surface-level variance），混合模型配置因跨家族盲点共享（cross-family homogenization）同样无效，为 MAS 多样性研究提供了关键反直觉结论。
4. **建立了 MAS 评估的新范式**：主张 MAS 评估应从"仅关注聚合性能"转向"同时分析交互动态"，并提出可复用的三维度评估框架。

## 方法详解
**MAS 架构**：采用编排器驱动（Orchestrator-Driven）的多智能体系统，定义为元组 $\mathcal{M} = (\mathcal{A}, O, \mathcal{R}, T)$，其中 $\mathcal{A}$ 为 worker agents，$O$ 为编排器，$\mathcal{R} = [r_{\min}, r_{\max}]$ 为评分区间，$T$ 为交互轮次。每轮执行三步：并行执行（各 agent 输出评分 $r_i^{(t)}$ 和元数据 $c_i^{(t)}$）、聚合（编排器生成下一状态 $g^{(t+1)}$）、广播（将状态广播给所有 agent）。

**三个同质化度量指标**：
- **从众性（Conformity）**：相邻轮次间 agent 评分方差递减，即 $\operatorname{Var}(r_1^{(t+1)},\ldots,r_n^{(t+1)}) < \operatorname{Var}(r_1^{(t)},\ldots,r_n^{(t)})$，表示 agent 间分歧减少。
- **极化（Polarization）**：agent 平均评分向评分区间端点漂移，即 $\min(|\bar{r}^{(t+1)} - r_{\min}|, |\bar{r}^{(t+1)} - r_{\max}|) < \min(|\bar{r}^{(t)} - r_{\min}|, |\bar{r}^{(t)} - r_{\max}|)$。
- **惯性（Inertia）**：平均评分变化速率递减，即 $|\bar{r}^{(t+2)} - \bar{r}^{(t+1)}| < |\bar{r}^{(t+1)} - \bar{r}^{(t)}|$，表示系统对变化的抵抗增强。

**三个应用场景设计**：
- **代码生成**：使用 CodeLMSec 数据集（280 个 Python/C++ 提示），包含安全相关 coding task；code generator + 多个 reviewer agent（Security、Optimality、Correctness 等角色）；最终通过 CodeQL 检测漏洞。
- **招聘**：构建合成反事实简历数据集（200 个职位-简历对，每对 40 个反事实变体，覆盖 4 种族裔×2 性别），评估人口统计学偏见；注入对抗性指令创建 biased agent。
- **同行评审**：采样 ICLR 2025 的 1950 篇投稿，采用 ReviewerToo 架构（3 个 reviewer + 1 个 meta-reviewer + 1 个 author）；按研究领域聚合评分分析偏见。

**实验设置**：使用 7 个 LLM（Qwen3-30B-A3B-Thinking-2507、Qwen3.6-35B-A3B、DeepSeek-V4-Flash/Pro、NVIDIA-Nemotron-3-Super/Ultra、Kimi-K2-Thinking）及专用代码模型（Kimi-K2.7-Code、Qwen3-Coder-Next）；每场景 $T=5$ 轮，$M=10$ 次独立试验（不同随机种子），报告 95% 置信区间。

## 实验与结果
- **同质化普遍存在**：在所有三个应用场景中，观察到的 conformity 使 agent 评分方差降至初始值的 $\sim 0.5\times$ 或更低；polarization 使系统均值偏移高达 0.5 分；inertia 使最后一轮评分变化幅度小于首轮变化的 $\sim 0.2\times$。其中 inertia 趋势最强，招聘场景的 polarization 最弱。
- **代码生成安全风险**：MAS 虽将总体漏洞率降低高达 50%，但特定漏洞类型（如 py/url-redirection、py/log-injection）在 deployability 过滤后反而放大；跨模型共享盲点（如 URL 重定向漏洞）在所有 9 个 LLM 中一致存在，形成系统性安全风险。
- **招聘偏见持久化**：对抗性 agent 仅在最初 2 轮注入偏见，但偏见迅速传播至整个 agent 小组；即使移除对抗性 agent 后，偏见仍在其他 agent 中持续存在；早期轮次的偏见影响远大于晚期轮次。
- **同行评审领域偏见**：不同研究领域的平均评分差距随交互轮次扩大；部分领域（如特定研究方向）在所有 LLM 中受到系统性不利对待或优待。
- **干预措施失效**：
  - 采样随机性（$M=10$ 次试验）仅产生"表面差异"：招聘/评审场景的跨试验均值分数方差存在，但同质化动态不变；代码场景中通过 LLM-as-a-Judge 评估功能等价性，仍发现高度趋同。
  - 混合模型配置（16 种不同模型组合）未能缓解风险：跨模型家族的盲点高度共享，代码漏洞类型、招聘偏见、领域偏见在混合设置中均持续存在。

## 相关工作脉络
1. **Agent 辩论与意见动力学**（Oh et al., 2026; Lin et al., 2025; El et al., 2026）：研究 agent 讨论主观话题时多样性减少、回声室形成等现象，但集中于游戏/辩论任务，未涉及真实世界应用。本文将其扩展到代码生成、招聘、同行评审三个高影响力场景。
2. **MAS 失败模式**（Cemri et al., 2026; Anthropic Frontier Red Team, 2026; Nakamura et al., 2026）：探索 inter-agent collusion、攻击入口、责任分配等新风险；本文补充了"homogenization"作为独立失败模式，强调交互动态导致的集体认知趋同。
3. **算法同质化/单一种植**（Kleinberg & Raghavan, 2021; Bommasani et al., 2022; Jain et al., 2025; Wu et al., 2025; Jiang et al., 2026）：讨论 isolated model 的输出趋同（algorithmic monoculture、generative monoculture、artificial hivemind）；本文首次将讨论延伸至 multi-agent interaction 语境，揭示交互本身驱动趋同的机制。
4. **预测多重性与 Rashomon 效应**（Marx et al., 2020; Fisher et al., 2019; Ganesh et al., 2025）：证明存在多个 equally optimal 的多样化模型；本文指出通过训练多个 distinct model 来利用 multiplicity 在 SOTA LLM 上计算不可行，转而评估无需额外训练的干预策略。
5. **ReviewerToo 与自动同行评审**（Sahu et al., 2025; Goyal et al., 2026; Biswas et al., 2026）：推动 AI 辅助同行评审；本文揭示该类系统中存在的领域偏见固化风险，为自动化评审系统设计提供审计视角。

## 局限性与未来方向
- **架构局限性**：实验仅覆盖 orchestrator-driven MAS 架构，其他架构（如完全去中心化、层次化、网络拓扑驱动）中的同质化动态尚未探索。
- **度量范围限制**：三个指标均基于数值评分，尚未扩展到自然语言输出层面的同质化度量（如 argument 多样性、rationale 差异性）。
- **未提出有效缓解策略**：仅证明采样随机性和混合模型无效，尚未提出能被验证的有效干预方案。
- **对抗性注入强度有限**：招聘实验中偏见 agent 仅注入 2 轮，未探索更强/更长持续的 adversarial 场景。

## 研究启发与可借鉴点
1. **三维度评估框架可直接迁移**：conformity/polarization/inertia 的数学定义简洁且正交，可应用于任何基于多轮交互产出数值评分的 MAS 场景（如金融决策、医疗诊断辅助），作为交互动态审计工具。
2. **"表面多样性 vs 实质多样性"的区分具有重要方法论价值**：采样随机性产生的是 syntactic/ surface-level 差异，而非 behavioral 多样性；这一洞见可指导后续研究设计真正的多样性干预（如 prompt engineering 引入认知偏差、结构化辩论协议）。
3. **跨模型共享盲点的发现启示安全评估策略**：代码生成中的相关性错误不依赖于特定模型架构，提示未来安全审计需从"单模型鲁棒性"转向"系统级盲区图谱"构建。
4. **早期影响 > 晚期影响的不等式**：招聘实验中早期轮次的偏见注入对最终结果影响远大于晚期，这一发现可指导人工介入时机——应在交互初期设置多样性保障机制。
5. **反事实数据构建方法可复用**：招聘数据集通过 demographic signals（extracurricular、languages、hobbies 等）隐式编码人口统计学属性，而非直接标注，为公平性研究提供了细粒度的 bias injection 范式。

## 关键术语表
- **Homogenization（同质化）**：多智能体系统中 agent 在交互过程中趋向于收敛到相似行为/输出的现象，导致多样性丧失。
- **Conformity（从众性）**：同质化的第一个度量维度，指相邻交互轮次间 agent 评分方差递减的趋势。
- **Polarization（极化）**：同质化的第二个度量维度，指系统平均评分向评分区间极端端点漂移的趋势。
- **Inertia（惯性）**：同质化的第三个度量维度，指系统平均评分变化速率随轮次递减、对变化抵抗增强的趋势。
- **Orchestrator-Driven MAS（编排器驱动的多智能体系统）**：由一个 central orchestrator 协调多个 worker agent 按轮次交互的 MAS 架构，是本文研究的标准模型。
- **Correlated Errors（相关性错误）**：多个 agent/模型在相同任务上犯相似错误的现象，在同质化环境下会被放大。
- **Cross-Family Homogenization（跨家族同质化）**：不同模型家族（如 Qwen、DeepSeek、Nemotron）共享相似盲点和失败模式的现象。
- **Deployability Filtering（可部署性过滤）**：在代码生成评估中，仅分析 reviewer panel 平均评分 ≥ 3（认为代码可部署）时的漏洞分布，用于识别"逃逸"整体 MAS 检查的安全漏洞。

## 可复现要素
- **数据集**：
  - CodeLMSec（Hajipour et al., 2024）：280 个 Python/C++ 安全相关 coding prompts，公开可用。
  - 招聘数据集：作者自行构建的合成反事实简历数据集，基于 Tan et al. (2026) 方法适配美国职位，论文提供了完整构建流程（Appendix C）。
  - 同行评审数据集：从 ICLR 2025 采样的 1950 篇论文，论文说明原始数据未公开，由作者重新采样。
- **代码/权重**：论文未提供开源代码或模型权重；使用商业/开源 LLM API（vLLM 部署）。
- **关键超参**：交互轮次 $T = 5$，独立试验次数 $M = 10$，评分区间依任务而定（代码 1-5、招聘 1-5、评审 1-10）；生成参数使用 vLLM 默认值。
