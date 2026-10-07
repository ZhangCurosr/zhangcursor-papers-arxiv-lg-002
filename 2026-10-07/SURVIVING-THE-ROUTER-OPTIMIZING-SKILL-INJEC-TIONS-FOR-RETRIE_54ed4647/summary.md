---
title: "SURVIVING-THE-ROUTER-OPTIMIZING-SKILL-INJEC-TIONS-FOR-RETRIE"
source: https://arxiv.org/pdf/2610.08098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:38:59"
field: "LLM Agent 安全与鲁棒性"
keywords: ["skill injection", "prompt injection", "LLM agent security", "retrieval attack", "agent skill", "security evaluation"]
innovations: ["揭示skill router对技能注入攻击的关键过滤作用，纠正现有工作的高估偏差", "提出CORSA聚类级路由器感知优化框架，两阶段分离优化检索与执行目标", "系统验证跨模型/路由器/Agent框架的注入技能迁移性与泛化性"]
benchmarks: ["SkillRouter", "SKILLINJECT payload taxonomy"]
---

# 论文速读：SURVIVING-THE-ROUTER-OPTIMIZING-SKILL-INJEC-TIONS-FOR-RETRIE

## 一句话总结
本文提出 CORSA（Cluster Optimization for Router-Aware Skill Attacks），一种路由器感知的技能注入攻击方法，通过**聚类级两阶段优化**使恶意技能在竞争性检索环境中既能被召回又能执行恶意载荷，将现有攻击的有效成功率提升约 2 倍，且在跨模型、跨路由器、跨 Agent 框架中具有良好的迁移性。

## 研究问题与动机
- **现有攻击评估过于乐观**：已有技能注入工作（如 SKILLJECT）默认恶意技能已被选中执行，忽略了真实多技能市场中注入技能需先通过 skill router 竞争检索的前提，导致对 ASR 高估 87%–97%。
- **朴素注入会损害检索排名**：实验显示，SKILLJECT 注入后技能的 Hit@1 从 100% 骤降至约 11%，注入技能往往因与任务语义不匹配而被路由器过滤，无法被执行。
- **检索与执行目标不一致**：检索需要技能描述与任务高度相关，而执行需要清晰的载荷触发指令，单一目标优化难以同时兼顾两者。
- **缺乏端到端攻击评估基准**：现有 benchmark 未系统评估攻击技能在包含数千个竞争技能环境中的整体生存能力。

## 核心贡献（创新点）
1. **揭示了 skill router 对技能注入攻击的关键过滤作用**：证明现有注入方法在多技能竞争场景下有效 ASR 极低，与既往结论形成本质区别。
2. **提出 CORSA 聚类级路由器感知优化框架**：首次将技能 router 纳入攻击目标，通过聚类优化而非单任务优化，显著提升注入技能的检索竞争力。
3. **设计两阶段分离优化策略**：Stage A 优化检索排名（Hit@1），Stage B 在 A 基础上优化端到端 ASR，与联合优化相比取得更优的检索-执行平衡。
4. **系统性评估跨模型/路由器/Agent 框架的迁移性**：展示优化后的注入技能在 GPT-5.4、DeepSeek-V4-Pro、GLM-5.3 等多模型及 BM25、SR-0.6B、Qwen3-8B 等多种 router 间保持一致的有效性。

## 方法详解
**威胁模型**：
- 技能表示为 $S = (n, d, s, A)$，其中 $n$ 为名称、$d$ 为自然语言描述、$s$ 为指令正文（SKILL.md）、$A$ 为可选辅助脚本。
- 注入技能 $S' = (n', d', s', A')$ 满足三条件：保持原始功能、指令 $i \subseteq s'$、载荷 $p \in A'$。
- 目标函数：定义 $\mathrm{Hit}(t,S)=\mathbf{1}[\mathrm{rank}_R(t,S,\mathcal{D})=1]$、$\mathrm{Exec}(t,S)=\mathbf{1}[p \in \mathcal{A}_\pi(t,S)]$。

**两阶段优化目标**：
- Stage A：$S_A = \arg\max_{S'} \sum_{t \in \mathcal{C}} \mathrm{Hit}(t, S')$，最大化检索命中数。
- Stage B：$S^{\star} = \arg\max_{S'} \sum_{t \in \mathcal{C}} \mathrm{Hit}(t, S') \cdot \mathrm{Exec}(t, S')$，在保持检索的同时提升端到端成功率。

**优化器 GEPA**：使用 Genetic-Pareto 反射进化搜索，维护候选池，通过任务级检索/执行反馈引导 LLM 重构 $n', d', s'$，载荷 $p$ 固定不变。

**任务聚类**：将 75 个测试任务划分为 8 个语义簇（coding-and-devops、data-science-and-analytics 等），每个簇优化 1 个注入技能以覆盖簇内所有任务。

## 实验与结果
**数据集与设置**：
- 基准：扩展 SkillRouter (Zheng et al., 2026)，75 个 prompt、约 80K 技能库 + 2000 个固定竞争技能（hard 设置加 780 个干扰技能）。
- 8 类恶意载荷：Data Exfiltration、Data Destruction、DoS、Malware/Ransomware、Phishing、Backdoors、Manipulation、Poisoning。
- 5 种路由器：BM25、OAI-Emb-3L、Qwen3-8B、SR-0.6B、R3-0.6B。
- 攻击者模型：GPT-5.4 与 DeepSeek-V4-Pro。

**主要结果（GPT-5.4）**：
- **Hit@1**：SKILLJECT 11.0% → CORSA 32.3%（提升近 3 倍）。
- **ASR**：SKILLJECT 10.0% → CORSA 22.0%（提升 120%）。
- **User Utility**：CORSA（38.8%）与 SKILLJECT（38.3%）持平。
- **自然性**：CORSA 注入文本更隐蔽，GPT-5.4 judge 评分更高。

**泛化性**：在 Paraphrase 任务上 ASR 保持 11.5%，在 Synthetic 任务上保持 5.4%，SKILLJECT 在同一条件下分别降至 1.0% 和 0.2%。

**跨模型迁移（DS-V4-Pro 攻击，不同受害者）**：CORSA ASR 在 GPT-5.4 上为 17.6%/19.4%，在 GLM-5.3-Flash 上为 15.7%/17.1%，在所有模型上均优于 SKILLJECT。

**跨 Router 迁移**：各源路由器优化的攻击在目标路由器上 ASR 仍可达 4.9%–26.4%，Qwen3-8B 最具抗性。

**防御评估**：静态扫描器全部漏检；LLM 增强模式下 SkillSpector+Qwen3.8-27B 召回 6–7/8，但仍未能完全检测。

## 相关工作脉络
- **SKILLJECT（Jia et al., 2026）**：本文主要基线，通过辅助脚本隐藏 payload，但未考虑 router 竞争，本文证明其在真实场景中 ASR 极低。
- **SKILLINJECT（Schmotz et al., 2026）**：技能注入基准，提出 8 类伤害分类，本文沿用其分类体系扩展 SkillRouter benchmark。
- **SkillRouter（Zheng et al., 2026）**：提供多技能检索基准，本文在其基础上集成恶意 payload 和攻击评估。
- **TOOLHIJACKER / TOOLTWEAK（Shi et al., 2025; Sneh et al., 2025）**：面向工具选择的检索层攻击，本文将同类思路迁移至技能场景。
- **SKILLATTACK（Duan et al., 2026）**：反向攻击——搜索触发 benign 技能执行恶意行为的 user prompt，本文则直接修改技能文件本身。
- **Cisco Skill Scanner / NVIDIA SkillSpector**：本文评估的两种检测工具，静态模式完全无效，LLM 增强模式有一定召回但不完备。

## 局限性与未来方向
- **仅评估间接攻击（helper script）**：未研究直接在 SKILL.md 中注入指令的变体，真实场景中直接注入可能更容易被发现或绕过。
- **未针对检测器做对抗优化**：论文明确说明当前方法未将 stealthiness 作为优化目标（附录 C.2 显示加入 stealthiness 反而降低 ASR），意味着面对 LLM 增强检测器时存在进一步被拦截的风险。
- **仅在 GPT-5.4 / DS-V4-Pro 两个攻击者模型上验证**：未探索更小或不同架构的 LLM 作为攻击者时的生成能力边界。
- **78 个硬干扰技能下仍有一定效果但需进一步验证**：附录 B 仅在 Hard 设置下验证，更极端的多技能竞争环境未被研究。
- **未考虑技能发布后的持续演化**：现实市场中技能会迭代更新，一次性优化能否长期存活尚不明确。

## 研究启发与可借鉴点
1. **"路由器感知"的评估视角值得迁移**：任何涉及检索+执行的 agent 安全评估（如 tool injection、retrieval-augmented agent 攻击）均应纳入检索竞争环节，避免过高的 ASR 估计。
2. **两阶段分离优化策略**：先优化检索目标再优化执行目标是解决多目标稀疏奖励问题的有效思路，可迁移至 PROMPT EVOLUTION、RED TEAM 自动化等任务。
3. **聚类级优化而非单任务优化**：将相关任务聚合为 cluster 并优化单一注入，显著提升跨任务的泛化能力，对 RED TEAM 数据生成、ADVERSARIAL EXAMPLE 构造具有借鉴价值。
4. **可复用实验设置**：SkillRouter benchmark + 8 类 payload + 5 种 router 的配置可作为后续 agent 安全研究的标准化评测协议。
5. **与防御研究的协同**：本文为 LLM-augmented skill scanner 提供了对抗样本来源，可作为防御方评估基准的输入。

## 关键术语表
**CORSA**：Cluster Optimization for Router-Aware Skill Attacks，本文提出的聚类级路由器感知技能注入攻击方法。
**Skill Router**：根据用户任务从技能库中检索并排序相关技能的检索模块，决定了哪些技能能被 agent 加载。
**ASR（Attack Success Rate）**：端到端攻击成功率，定义为注入技能被检索为第 1 名且载荷成功执行的比率。
**Hit@1**：注入技能在 router 中排名第一位的任务占比，衡量检索阶段的攻击有效性。
**GEPA**：Genetic-Pareto Advisor，一种基于反射进化搜索的 LLM 文本优化器，用于迭代改进 SKILL.md 等文本组件。
**任务簇（Task Cluster）**：共享领域词汇的 3–13 个相关任务集合，用于支持聚类级优化以提升泛化性。
**间接攻击（Indirect Attack）**：通过注入技能触发执行辅助 helper script 而非直接在 prompt 中嵌入 payload 的攻击方式。
**SkillJect**：SKILLJECT 基线方法，通过辅助脚本隐藏 payload 的自动化技能注入攻击，未考虑 router 竞争。

## 可复现要素
- **数据集**：扩展 SkillRouter benchmark（75 prompts，8 类 payload，80K 技能库），论文未声明第三方数据集公开状态。
- **代码**：已开源，见 https://github.com/compass-group-tue/CORSA-Cluster-Optimization-for-Router-Aware-Skill-Attacks
- **权重**：未开源自有模型，使用 GPT-5.4、DeepSeek-V4-Pro、Qwen3.8-27B、GLM-5.3 等商用/开源模型。
- **关键超参**：聚类大小 3–13 个任务；竞争技能数 2000；hard 设置增加 780 个干扰技能；Stage A/B 均使用 GEPA 反射进化搜索；router 共 5 种（BM25、OAI-Emb-3L、Qwen3-8B、SR-0.6B、R3-0.6B）。
- **模型配置**：攻击者（GPT-5.4 / DS-V4-Pro）、受害者（GPT-5.4 / Qwen3.8-27B / GLM-5.3 / GLM-5.3-Flash / DS-V4-Pro）、Judge（GPT-5.4）。
